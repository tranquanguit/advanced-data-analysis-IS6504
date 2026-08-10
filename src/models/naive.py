from __future__ import annotations

import pandas as pd

PROVINCE_COL = "province"

# NOTE (2026-08, review fix):
# Targets are built as y_{t+h} = target.shift(-h) (see dataset_builder). For a
# forecast made at origin t, the h-step-ahead prediction must be aligned to the
# SAME origin row t and compared against the y_{t+h} column. Hence:
#   - Naive (persistence):  y_hat_{t+h} = y_t         -> shift(0)
#   - Seasonal Naive:       y_hat_{t+h} = y_{t+h-12}  -> shift(12 - h)
# The previous code used shift(h) and shift(12+h-1) (Seasonal Naive was off by one
# month). Fixed below. Baseline comparisons must be re-run after this change.


def naive_predict(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    # Persistence: predict every horizon with the value at the forecast origin t.
    return df.groupby(PROVINCE_COL)[target].shift(0)


def seasonal_naive_predict(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    # Same calendar month one year before the target month t+h: y_{t+h-12}.
    return df.groupby(PROVINCE_COL)[target].shift(12 - horizon)
