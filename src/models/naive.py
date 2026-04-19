from __future__ import annotations

import pandas as pd

PROVINCE_COL = "province"


def naive_predict(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    # t+h dự báo bằng t
    return df.groupby(PROVINCE_COL)[target].shift(horizon)


def seasonal_naive_predict(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    # t+h dự báo bằng cùng tháng năm trước
    return df.groupby(PROVINCE_COL)[target].shift(12 + horizon - 1)
