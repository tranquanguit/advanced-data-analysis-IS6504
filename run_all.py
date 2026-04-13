from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.dataset_builder import create_multi_horizon_targets, split_train_val_test
from src.eda import run_eda
from src.evaluation import evaluate_horizons, outbreak_metrics, per_province_mae, significance_test
from src.feature_engineering import create_features
from src.insight_extractor import generate_insights
from src.data_loader import load_all_provinces
from src.models.naive import naive_predict, seasonal_naive_predict
from src.models.prophet_model import prophet_forecast_per_province
from src.models.tree_models import train_hgb, train_xgb
from src.models.lstm_model import LSTMModel
from src.runtime_config import load_runtime_config
from src.shap_analysis import run_shap_analysis, shap_by_province
from src.trainer import train_lstm
from src.visualization import plot_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Run dengue forecasting pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def ensure_dirs(cfg):
    data_folder = Path(cfg.paths.get("data_folder", "data/raw"))
    processed_folder = Path(cfg.paths.get("processed_folder", "data/processed"))
    output_dir = Path(cfg.paths.get("output_dir", "outputs"))

    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "shap").mkdir(parents=True, exist_ok=True)
    processed_folder.mkdir(parents=True, exist_ok=True)
    data_folder.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_folder,
        "processed": processed_folder,
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "plots": output_dir / "plots",
        "shap": output_dir / "shap",
    }


def build_feature_cols(df: pd.DataFrame, target: str, horizons: list[int]) -> list[str]:
    exclude = {"year", "month", "date", "province", target} | {f"{target}_t+{h}" for h in horizons}
    # Avoid leakage/double-counting when target is a rate derived from cases
    return [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.endswith("_cases")
    ]


def run_pipeline(config_path: str):
    cfg = load_runtime_config(config_path)
    dirs = ensure_dirs(cfg)

    exp = cfg.experiment
    model_cfg = cfg.model
    run_cfg = cfg.run

    target = exp.get("target", "Dengue_fever_rates")
    diseases = exp.get("diseases", [])
    weather_vars = exp.get("weather_vars", [])
    social_vars = exp.get("social_vars", [])
    lags = exp.get("lags", [1, 2, 3]) # keep for backward compatibility or replace
    input_sequence_length = exp.get("input_sequence_length", 12)
    predict_horizon = exp.get("predict_horizon", 3)
    horizons = list(range(1, predict_horizon + 1))
    train_end = exp.get("train_end", "2014-12-31")
    val_end = exp.get("val_end", "2015-12-31")
    test_start = exp.get("test_start", "2016-01-01")
    test_end = exp.get("test_end")

    df_raw = load_all_provinces(
        dirs["data"],
        target_col=target,
        cases_col=exp.get("cases_col"),
        compute_rate_per100k=exp.get("compute_rate_per100k", False),
    )
    run_eda(df_raw, target, weather_vars, diseases, dirs["plots"])

    df_feat = create_features(
        df_raw,
        target,
        diseases,
        weather_vars,
        social_vars,
        input_sequence_length,
        include_other_diseases=exp.get("include_other_diseases_as_features", False),
    )
    df_all = create_multi_horizon_targets(df_feat, target, horizons)
    df_all.to_csv(dirs["processed"] / "dataset_modeling.csv", index=False)

    train, val, test = split_train_val_test(df_all, train_end, val_end, test_start, test_end)
    feature_cols = build_feature_cols(df_all, target, horizons)

    scaler = StandardScaler()
    x_train_df = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols, index=train.index)
    x_val_df = pd.DataFrame(scaler.transform(val[feature_cols]), columns=feature_cols, index=val.index)
    x_test_df = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols, index=test.index)

    x_train = x_train_df.to_numpy(dtype=np.float32)
    y_train = train[[f"{target}_t+{h}" for h in horizons]].to_numpy(dtype=np.float32)
    x_val = x_val_df.to_numpy(dtype=np.float32)
    y_val = val[[f"{target}_t+{h}" for h in horizons]].to_numpy(dtype=np.float32)
    x_test = x_test_df.to_numpy(dtype=np.float32)
    y_test = test[[f"{target}_t+{h}" for h in horizons]].to_numpy(dtype=np.float32)

    results = []

    naive_preds = np.column_stack([naive_predict(df_all, target, h).loc[test.index].to_numpy() for h in horizons])
    seasonal_preds = np.column_stack([seasonal_naive_predict(df_all, target, h).loc[test.index].to_numpy() for h in horizons])

    model_preds = {
        "Naive": naive_preds,
        "SeasonalNaive": seasonal_preds,
    }

    if run_cfg.get("enable_prophet", True):
        prophet_preds_h1 = prophet_forecast_per_province(pd.concat([train, val]), test, f"{target}_t+1")
        model_preds["Prophet"] = np.column_stack([prophet_preds_h1.to_numpy() for _ in horizons])

    # --- Quick hyperparam search on val for XGB/HGB ---
    xgb_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7},
    ]
    hgb_grid = [
        {"max_iter": 250, "learning_rate": 0.05, "max_depth": 6},
        {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
        {"max_iter": 350, "learning_rate": 0.03, "max_depth": 8},
    ]

    def eval_grid(grid, trainer_fn, name):
        best = None
        for params in grid:
            m = trainer_fn(x_train, y_train, params=params)
            preds_val = m.predict(x_val)
            scores = evaluate_horizons(y_val, preds_val, horizons)
            score_key = "MAE@1"
            score = scores.get(score_key, list(scores.values())[0])
            if (best is None) or (score < best[0]):
                best = (score, params)
        return best[1] if best else {}

    best_xgb_params = eval_grid(xgb_grid, train_xgb, "XGB")
    best_hgb_params = eval_grid(hgb_grid, train_hgb, "HGB")

    # retrain on train+val with best params
    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.vstack([y_train, y_val])

    mx = train_xgb(x_trainval, y_trainval, params=best_xgb_params or model_cfg.get("xgb", {}))
    mh = train_hgb(x_trainval, y_trainval, params=best_hgb_params or model_cfg.get("hgb", {}))
    model_preds["XGBoost"] = mx.predict(x_test)
    model_preds["HistGB"] = mh.predict(x_test)
    xgb_models = [mx.estimators_[0]] if hasattr(mx, 'estimators_') else [mx] # For SHAP

    lstm_cfg = model_cfg.get("lstm", {})
    # build sequences for LSTM (seq_len configurable; default 24 months)
    seq_len = lstm_cfg.get("seq_len", 24)

    def build_sequences(x_arr, y_arr):
        Xs, Ys = [], []
        for i in range(len(x_arr) - seq_len + 1):
            Xs.append(x_arr[i : i + seq_len])
            Ys.append(y_arr[i + seq_len - 1])
        return torch.tensor(np.stack(Xs), dtype=torch.float32), torch.tensor(np.stack(Ys), dtype=torch.float32)

    x_train_t, y_train_t = build_sequences(x_train, y_train)
    x_val_t, y_val_t = build_sequences(x_val, y_val)
    x_test_t, y_test_t_lstm = build_sequences(x_test, y_test)

    lstm = LSTMModel(
        input_size=x_train.shape[1],
        hidden_size=lstm_cfg.get("hidden_size", 128),
        num_layers=lstm_cfg.get("num_layers", 2),
        out_dim=len(horizons),
        dropout=lstm_cfg.get("dropout", 0.2),
    )
    lstm = train_lstm(
        lstm,
        x_train_t,
        y_train_t,
        val_data=(x_val_t, y_val_t),
        epochs=lstm_cfg.get("epochs", 60),
        lr=lstm_cfg.get("lr", 1e-3),
        batch_size=lstm_cfg.get("batch_size", 64),
    )
    lstm.eval()
    with torch.no_grad():
        model_preds["LSTM"] = lstm(x_test_t).cpu().numpy()

    y_true_map = {"LSTM": y_test[seq_len - 1 :]}  # align with sequence reduction

    for model_name, pred in model_preds.items():
        y_true_use = y_true_map.get(model_name, y_test)
        scores = evaluate_horizons(y_true_use, pred, horizons)
        outbreak = outbreak_metrics(y_true_use[:, 0], pred[:, 0], percentile=95)
        results.append({"model": model_name, **scores, **outbreak})

        pred_df = test[["province", "date"]].iloc[-len(pred) :].copy()
        for i, h in enumerate(horizons):
            pred_df[f"actual_t+{h}"] = y_true_use[:, i]
            pred_df[f"pred_t+{h}"] = pred[:, i]
        pred_df.to_csv(dirs["predictions"] / f"pred_{model_name}.csv", index=False)

    res_df = pd.DataFrame(results).sort_values("MAE@1")
    res_df.to_csv(dirs["metrics"] / "model_comparison.csv", index=False)

    best_model = res_df.iloc[0]["model"]
    province_df = test[["province", "date"]].copy()
    province_df["actual"] = y_test[:, 0]
    province_df["pred"] = model_preds[best_model][:, 0]
    per_province_mae(province_df, "actual", "pred").to_csv(dirs["metrics"] / "province_metrics.csv", index=False)

    base_err_full = np.abs(y_test[:, 0] - seasonal_preds[:, 0])
    stats_rows = []
    for name, pred in model_preds.items():
        if name == "SeasonalNaive":
            continue
        true_arr = y_true_map.get(name, y_test)
        ref_err = base_err_full[-len(pred) :]
        model_err = np.abs(true_arr[:, 0] - pred[:, 0])
        stats_rows.append({"model": name, **significance_test(ref_err, model_err)})
    pd.DataFrame(stats_rows).to_csv(dirs["metrics"] / "significance_vs_seasonal.csv", index=False)

    top2 = res_df.head(2)["model"].tolist()
    for name in top2:
        plot_prediction(y_test, model_preds[name], dirs["plots"] / f"prediction_{name}.png", horizon_idx=0)

    # if run_cfg.get("enable_shap", True):
    #     try:
    #         run_shap_analysis(xgb_models[0], x_train_df[feature_cols], x_test_df[feature_cols], feature_cols, dirs["shap"])
    #         shap_by_province(
    #             xgb_models[0],
    #             pd.concat([test[["province"]], x_test_df[feature_cols]], axis=1),
    #             feature_cols,
    #             dirs["shap"],
    #         )
    #         generate_insights(
    #             dirs["shap"] / "top_features.csv",
    #             dirs["shap"] / "shap_by_province.csv",
    #             dirs["shap"] / "insights.txt",
    #         )
    #     except Exception as exc:
    #         warnings.warn(f"SHAP step skipped due to: {exc}")
    
    def _robust_clean_numeric(df: pd.DataFrame, cols):
        """Convert ALL values to float, handle '[...]', strings, edge cases."""
        df = df.copy()

        def _fix(x):
            if pd.isna(x):
                return None
            if isinstance(x, (int, float)):
                return x

            x = str(x).strip()

            # remove brackets
            x = re.sub(r"[\[\]]", "", x)

            # normalize null-like
            if x.lower() in {"nan", "none", ""}:
                return None

            try:
                return float(x)
            except:
                return None

        for col in cols:
            df[col] = df[col].apply(_fix)

        return df


    def _detect_non_numeric(df: pd.DataFrame, cols, name="df"):
        """Find exact bad values instead of guessing."""
        bad = {}

        for col in cols:
            mask = ~pd.to_numeric(df[col], errors="coerce").notna()
            if mask.any():
                bad[col] = df.loc[mask, col].head(5).tolist()

        if bad:
            print(f"\n[CRITICAL] Non-numeric detected in {name}:")
            for k, v in bad.items():
                print(f"  {k}: {v}")
            raise ValueError(f"{name} still contains non-numeric values")


    def _validate_no_nan(df: pd.DataFrame, cols, name="df"):
        nan_cols = df[cols].columns[df[cols].isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"{name} contains NaN after cleaning in columns: {nan_cols}")


    # ================= MAIN =================
    if run_cfg.get("enable_shap", True):
        try:
            # 1. Reset index
            x_train = x_train_df[feature_cols].reset_index(drop=True)
            x_test = x_test_df[feature_cols].reset_index(drop=True)
            test_meta = test[["province"]].reset_index(drop=True)

            # 2. Clean (robust version)
            x_train = _robust_clean_numeric(x_train, feature_cols)
            x_test = _robust_clean_numeric(x_test, feature_cols)

            # 3. Detect lỗi thật sự (QUAN TRỌNG)
            _detect_non_numeric(x_train, feature_cols, "x_train")
            _detect_non_numeric(x_test, feature_cols, "x_test")

            # 4. Validate NaN
            _validate_no_nan(x_train, feature_cols, "x_train")
            _validate_no_nan(x_test, feature_cols, "x_test")

            # 5. Run SHAP
            run_shap_analysis(
                xgb_models[0],
                x_train,
                x_test,
                feature_cols,
                dirs["shap"],
            )

            # 6. SHAP by province
            shap_input = pd.concat([test_meta, x_test], axis=1)

            shap_by_province(
                xgb_models[0],
                shap_input,
                feature_cols,
                dirs["shap"],
            )

            # 7. Insights
            generate_insights(
                dirs["shap"] / "top_features.csv",
                dirs["shap"] / "shap_by_province.csv",
                dirs["shap"] / "insights.txt",
            )

        except Exception as exc:
            warnings.warn(f"[SHAP ERROR] {type(exc).__name__}: {exc}")

            print("\n[DEBUG] Deep scan for bad values...")
            try:
                _detect_non_numeric(x_train_df[feature_cols], feature_cols, "RAW x_train_df")
                _detect_non_numeric(x_test_df[feature_cols], feature_cols, "RAW x_test_df")
            except Exception as inner_exc:
                print(f"[ROOT CAUSE FOUND] {inner_exc}")    

    print(f"Done. Results at {dirs['metrics'] / 'model_comparison.csv'}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
