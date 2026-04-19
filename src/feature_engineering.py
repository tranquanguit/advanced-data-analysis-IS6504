from __future__ import annotations

import numpy as np
import pandas as pd

PROVINCE_COL = "province"


def create_features(
    df: pd.DataFrame,
    target: str,
    diseases: list[str],
    weather_vars: list[str],
    social_vars: list[str],
    input_sequence_length: int,
    include_other_diseases: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    # build base list of columns to lag
    extra_disease_cols = [d for d in diseases if d != target] if include_other_diseases else []
    for col in [target, *extra_disease_cols, *weather_vars, *social_vars]:
        if col not in out.columns:
            continue
        for lag in range(1, input_sequence_length + 1):
            out[f"{col}_lag{lag}"] = out.groupby(PROVINCE_COL)[col].shift(lag)

    for w in [3, 6]:
        if w <= input_sequence_length:
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
