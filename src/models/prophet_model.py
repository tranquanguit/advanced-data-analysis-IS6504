from __future__ import annotations

import pandas as pd


def prophet_forecast_per_province(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> pd.Series:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet is not installed. Please install prophet.") from exc

    preds = []
    for province, g_test in test_df.groupby("province"):
        g_train = train_df[train_df["province"] == province].copy()
        if len(g_train) < 24:
            p = pd.Series([g_train[target].iloc[-1]] * len(g_test), index=g_test.index)
            preds.append(p)
            continue

        fit_df = g_train[["date", target]].rename(columns={"date": "ds", target: "y"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(fit_df)

        future = g_test[["date"]].rename(columns={"date": "ds"})
        forecast = model.predict(future)
        p = pd.Series(forecast["yhat"].values, index=g_test.index)
        preds.append(p)

    return pd.concat(preds).sort_index()
