from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROVINCE_COL


def create_features(
    df: pd.DataFrame,
    target: str,
    weather_vars: list[str],
    social_vars: list[str],
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    out = df.copy()

    for col in [target] + weather_vars + social_vars:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(PROVINCE_COL)[col].shift(lag)

    for w in rolling_windows:
        out[f"{target}_rollmean_{w}"] = (
            out.groupby(PROVINCE_COL)[target].shift(1).rolling(window=w).mean().reset_index(level=0, drop=True)
        )
        out[f"{target}_rollstd_{w}"] = (
            out.groupby(PROVINCE_COL)[target].shift(1).rolling(window=w).std().reset_index(level=0, drop=True)
        )

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out = out.dropna().reset_index(drop=True)
    return out
