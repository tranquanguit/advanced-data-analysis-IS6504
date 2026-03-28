from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def evaluate_horizons(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int]) -> dict:
    out = {}
    for i, h in enumerate(horizons):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[f"MAE@{h}"] = mean_absolute_error(yt, yp)
        # Older sklearn in the environment does not support squared=False, so compute RMSE manually
        out[f"RMSE@{h}"] = mean_squared_error(yt, yp) ** 0.5
        out[f"SMAPE@{h}"] = smape(yt, yp)
    return out


def outbreak_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 95.0) -> dict:
    threshold = np.percentile(y_true, percentile)
    actual = y_true >= threshold
    pred = y_pred >= threshold
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"outbreak_threshold": threshold, "precision": precision, "recall": recall}


def significance_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    mask = ~np.isnan(errors_a) & ~np.isnan(errors_b)
    if mask.sum() == 0:
        return {"wilcoxon_stat": float("nan"), "p_value": float("nan")}
    stat, p = wilcoxon(errors_a[mask], errors_b[mask], zero_method="wilcox", correction=False)
    return {"wilcoxon_stat": float(stat), "p_value": float(p)}


def per_province_mae(df: pd.DataFrame, actual_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for province, g in df.groupby("province"):
        rows.append({
            "province": province,
            "MAE": mean_absolute_error(g[actual_col], g[pred_col]),
        })
    return pd.DataFrame(rows).sort_values("MAE")
