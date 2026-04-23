from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.dataset_builder import create_multi_horizon_targets
from src.evaluation import evaluate_horizons, outbreak_metrics
from src.feature_engineering import create_features
from src.data_loader import load_all_provinces
from run_all import build_feature_cols
from src.models.naive import naive_predict, seasonal_naive_predict
from src.models.tree_models import train_hgb, train_xgb
from src.models.lgbm_model import train_lgbm
from src.models.lstm_model import LSTMModel
from src.runtime_config import load_runtime_config
from src.trainer import train_lstm

def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold CV for dengue forecasting pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()

def run_cv_pipeline(config_path: str):
    cfg = load_runtime_config(config_path)
    
    exp = cfg.experiment
    model_cfg = cfg.model
    
    target = exp.get("target", "Dengue_fever_rates")
    diseases = exp.get("diseases", [])
    weather_vars = exp.get("weather_vars", [])
    social_vars = exp.get("social_vars", [])
    input_sequence_length = exp.get("input_sequence_length", 12)
    predict_horizon = exp.get("predict_horizon", 6)
    horizons = list(range(1, predict_horizon + 1))
    
    data_folder = Path(cfg.paths.get("data_folder", "data/raw"))
    df_raw = load_all_provinces(
        data_folder,
        target_col=target,
        cases_col=exp.get("cases_col"),
        compute_rate_per100k=exp.get("compute_rate_per100k", False),
    )
    
    df_feat = create_features(
        df_raw,
        target,
        diseases,
        weather_vars,
        social_vars,
        input_sequence_length,
        cross_disease_map=exp.get("cross_disease_map", None),
    )
    df_all = create_multi_horizon_targets(df_feat, target, horizons)
    
    feature_cols = build_feature_cols(
        df_all, target, horizons,
        weather_vars=weather_vars,
        social_vars=social_vars,
        diseases=diseases,
        cross_disease_map=exp.get("cross_disease_map", None),
    )
    
    print(f"[INFO] Features selected: {len(feature_cols)}")
    print(f"[INFO] Target: {target}")
    
    # Define 5 folds
    test_years = [2014, 2015, 2016, 2017, 2018]
    
    cv_results = []
    
    for y in test_years:
        train_end = f"{y-2}-12-31"
        val_end = f"{y-1}-12-31"
        test_start = f"{y}-01-01"
        test_end = f"{y}-12-31"
        
        train = df_all[df_all["date"] <= train_end].copy()
        val = df_all[(df_all["date"] > train_end) & (df_all["date"] <= val_end)].copy()
        test = df_all[(df_all["date"] >= test_start) & (df_all["date"] <= test_end)].copy()
        
        if len(test) == 0:
            print(f"[WARN] No test data for fold year {y}")
            continue
            
        print(f"\n--- FOLD Test Year {y} ---")
        print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
        
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
        
        naive_preds = np.column_stack([naive_predict(df_all, target, h).loc[test.index].to_numpy() for h in horizons])
        seasonal_preds = np.column_stack([seasonal_naive_predict(df_all, target, h).loc[test.index].to_numpy() for h in horizons])

        model_preds = {
            "Naive": naive_preds,
            "SeasonalNaive": seasonal_preds,
        }
        
        # Merge Train and Val for Final Training of Boosting Models
        x_trainval = np.vstack([x_train, x_val])
        y_trainval = np.vstack([y_train, y_val])

        mx = train_xgb(x_trainval, y_trainval, params=model_cfg.get("xgb", {}))
        mh = train_hgb(x_trainval, y_trainval, params=model_cfg.get("hgb", {}))
        ml = train_lgbm(x_trainval, y_trainval, params=model_cfg.get("lgbm", {}))
        model_preds["XGBoost"] = mx.predict(x_test)
        model_preds["HistGB"] = mh.predict(x_test)
        model_preds["LightGBM"] = ml.predict(x_test)
        
        lstm_cfg = model_cfg.get("lstm", {})
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
            horizon_weights=exp.get("horizon_weights", None),
        )
        lstm.eval()
        with torch.no_grad():
            model_preds["LSTM"] = lstm(x_test_t).cpu().numpy()

        y_true_map = {"LSTM": y_test[seq_len - 1 :]}
        
        for model_name, pred in model_preds.items():
            y_true_use = y_true_map.get(model_name, y_test)
            scores = evaluate_horizons(y_true_use, pred, horizons)
            outbreak = outbreak_metrics(y_true_use[:, 0], pred[:, 0], percentile=95)
            cv_results.append({"fold_year": y, "model": model_name, **scores, **outbreak})
            
    res_df = pd.DataFrame(cv_results)
    
    # Aggregate results over folds
    agg_df = res_df.groupby("model").mean().reset_index().drop(columns=["fold_year"])
    
    output_dir = Path(cfg.paths.get("output_dir", "outputs"))
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_dir / "metrics" / "cv_results_detailed.csv", index=False)
    agg_df.to_csv(output_dir / "metrics" / "cv_results_mean.csv", index=False)
    
    print("\n--- 5-Fold Cross Validation Mean Results ---")
    print(agg_df.sort_values("MAE@1")[["model", "MAE@1", "MAE@3", "MAE@6"]].to_string(index=False))

if __name__ == "__main__":
    args = parse_args()
    run_cv_pipeline(args.config)
