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
    cross_disease_map: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    # build base list of columns to lag for target and exogenous vars
    for col in [target, *weather_vars, *social_vars]:
        if col not in out.columns:
            continue
        for lag in range(1, input_sequence_length + 1):
            out[f"{col}_lag{lag}"] = out.groupby(PROVINCE_COL)[col].shift(lag)

    # Selective cross-disease features
    if cross_disease_map:
        for extra_col, lags in cross_disease_map.items():
            if extra_col not in out.columns or extra_col == target:
                continue
            for lag in lags:
                if lag == 0:
                    continue  # lag 0 is the raw column itself, we keep it as is
                out[f"{extra_col}_lag{lag}"] = out.groupby(PROVINCE_COL)[extra_col].shift(lag)

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


    return out
