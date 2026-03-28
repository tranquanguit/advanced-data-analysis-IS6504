from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from src.config import (
    DATA_FOLDER,
    HORIZONS,
    LAGS,
    METRICS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_FOLDER,
    SEQ_LEN,
    SHAP_DIR,
    SOCIAL_VARS,
    TARGET,
    TEST_START,
    TRAIN_END,
    WEATHER_VARS,
)
from src.data_loader import load_all_provinces
from src.dataset_builder import create_multi_horizon_targets, split_train_test
from src.eda import run_eda
from src.evaluation import evaluate_horizons, outbreak_metrics, per_province_mae, significance_test
from src.feature_engineering import create_features
from src.insight_extractor import generate_insights
from src.models.naive import naive_predict, seasonal_naive_predict
from src.models.prophet_model import prophet_forecast_per_province
from src.models.tree_models import train_hgb, train_xgb
from src.models.lstm_model import LSTMModel
from src.shap_analysis import run_shap_analysis, shap_by_province
from src.trainer import train_lstm
from src.visualization import plot_prediction


def ensure_dirs() -> None:
    for d in [METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, SHAP_DIR, PROCESSED_FOLDER]:
        d.mkdir(parents=True, exist_ok=True)


def build_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"year", "month", "date", "province", TARGET} | {f"{TARGET}_t+{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def run_pipeline() -> None:
    ensure_dirs()

    df_raw = load_all_provinces(DATA_FOLDER)
    run_eda(df_raw, TARGET, WEATHER_VARS, PLOTS_DIR)

    df_feat = create_features(df_raw, TARGET, WEATHER_VARS, SOCIAL_VARS, LAGS, [3])
    df_all = create_multi_horizon_targets(df_feat, TARGET, HORIZONS)
    df_all.to_csv(PROCESSED_FOLDER / "dataset_modeling.csv", index=False)

    train, test = split_train_test(df_all, TRAIN_END, TEST_START)
    feature_cols = build_feature_cols(df_all)

    x_train = train[feature_cols].to_numpy()
    y_train = train[[f"{TARGET}_t+{h}" for h in HORIZONS]].to_numpy()
    x_test = test[feature_cols].to_numpy()
    y_test = test[[f"{TARGET}_t+{h}" for h in HORIZONS]].to_numpy()

    results = []

    # Baseline: Naive / Seasonal Naive per horizon
    naive_preds = np.column_stack([naive_predict(df_all, TARGET, h).loc[test.index].to_numpy() for h in HORIZONS])
    seasonal_preds = np.column_stack(
        [seasonal_naive_predict(df_all, TARGET, h).loc[test.index].to_numpy() for h in HORIZONS]
    )

    # Prophet (one-step direct, reused for all horizons as simple baseline extension)
    prophet_preds_h1 = prophet_forecast_per_province(train, test, f"{TARGET}_t+1")
    prophet_preds = np.column_stack([prophet_preds_h1.to_numpy() for _ in HORIZONS])

    # Tree models
    preds_xgb = []
    preds_hgb = []
    xgb_models = []
    for i, h in enumerate(HORIZONS):
        mx = train_xgb(x_train, y_train[:, i])
        mh = train_hgb(x_train, y_train[:, i])
        preds_xgb.append(mx.predict(x_test))
        preds_hgb.append(mh.predict(x_test))
        xgb_models.append(mx)
    preds_xgb = np.column_stack(preds_xgb)
    preds_hgb = np.column_stack(preds_hgb)

    # LSTM pooled (tabular-to-seq simplified using seq_len=1 immediate features)
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

    lstm = LSTMModel(input_size=x_train.shape[1], hidden_size=64, num_layers=2, out_dim=len(HORIZONS))
    lstm = train_lstm(lstm, x_train_t, y_train_t, epochs=30, lr=1e-3)
    lstm.eval()
    with torch.no_grad():
        preds_lstm = lstm(x_test_t).cpu().numpy()

    model_preds = {
        "Naive": naive_preds,
        "SeasonalNaive": seasonal_preds,
        "Prophet": prophet_preds,
        "XGBoost": preds_xgb,
        "HistGB": preds_hgb,
        "LSTM": preds_lstm,
    }

    for model_name, pred in model_preds.items():
        scores = evaluate_horizons(y_test, pred, HORIZONS)
        outbreak = outbreak_metrics(y_test[:, 0], pred[:, 0], percentile=95)
        row = {"model": model_name, **scores, **outbreak}
        results.append(row)

        pred_df = test[["province", "date"]].copy()
        for i, h in enumerate(HORIZONS):
            pred_df[f"actual_t+{h}"] = y_test[:, i]
            pred_df[f"pred_t+{h}"] = pred[:, i]
        pred_df.to_csv(PREDICTIONS_DIR / f"pred_{model_name}.csv", index=False)

    res_df = pd.DataFrame(results).sort_values("MAE@1")
    res_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    # province MAE sample for best model @1
    best_model = res_df.iloc[0]["model"]
    best_pred = model_preds[best_model]
    province_df = test[["province", "date"]].copy()
    province_df["actual"] = y_test[:, 0]
    province_df["pred"] = best_pred[:, 0]
    prov_mae = per_province_mae(province_df, "actual", "pred")
    prov_mae.to_csv(METRICS_DIR / "province_metrics.csv", index=False)

    # Statistical significance vs Seasonal Naive using horizon 1 absolute error
    base_err = np.abs(y_test[:, 0] - seasonal_preds[:, 0])
    stats_rows = []
    for name, pred in model_preds.items():
        if name == "SeasonalNaive":
            continue
        err = np.abs(y_test[:, 0] - pred[:, 0])
        stats = significance_test(base_err, err)
        stats_rows.append({"model": name, **stats})
    pd.DataFrame(stats_rows).to_csv(METRICS_DIR / "significance_vs_seasonal.csv", index=False)

    # Plots for top 2 models
    top2 = res_df.head(2)["model"].tolist()
    for name in top2:
        plot_prediction(y_test, model_preds[name], PLOTS_DIR / f"prediction_{name}.png", horizon_idx=0)

    # SHAP on XGBoost horizon-1 model
    try:
        run_shap_analysis(xgb_models[0], train[feature_cols], test[feature_cols], feature_cols, SHAP_DIR)
        shap_by_province(xgb_models[0], test[["province", *feature_cols]], feature_cols, SHAP_DIR)
        generate_insights(SHAP_DIR / "top_features.csv", SHAP_DIR / "shap_by_province.csv", SHAP_DIR / "insights.txt")
    except Exception as exc:  # keep pipeline robust
        warnings.warn(f"SHAP step skipped due to: {exc}")

    print("Done. Results at outputs/metrics/model_comparison.csv")


if __name__ == "__main__":
    run_pipeline()
