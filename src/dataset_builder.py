from __future__ import annotations

import pandas as pd

from src.config import PROVINCE_COL


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
