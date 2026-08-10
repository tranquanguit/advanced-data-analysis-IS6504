from __future__ import annotations

import pandas as pd

PROVINCE_COL = "province"


def create_multi_horizon_targets(df: pd.DataFrame, target: str, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"{target}_t+{h}"] = out.groupby(PROVINCE_COL)[target].shift(-h)
    out = out.dropna().reset_index(drop=True)
    return out


def split_train_test(df: pd.DataFrame, train_end: str, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= train_end].copy()
    test = df[df["date"] >= test_start].copy()
    return train, test


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_start: str,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test_mask = df["date"] >= test_start
    if test_end:
        test_mask &= df["date"] <= test_end
    test = df[test_mask].copy()
    return train, val, test


def rolling_origin_folds(df: pd.DataFrame, n_folds: int = 3):
    """Yield rolling-origin (time-based) train/val indices for robustness checks.

    Splits are uniform over the date range; test set is left to caller.
    """
    dates = df["date"].sort_values().unique()
    if len(dates) < n_folds + 1:
        return []
    fold_size = len(dates) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        cutoff = dates[fold_size * i]
        train_idx = df[df["date"] < cutoff].index
        val_idx = df[(df["date"] >= cutoff) & (df["date"] < cutoff + (dates[-1] - dates[0]) / (n_folds + 1))].index
        folds.append((train_idx, val_idx))
    return folds
