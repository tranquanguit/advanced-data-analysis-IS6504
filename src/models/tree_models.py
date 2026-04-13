from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_xgb(X_train, y_train, params: dict | None = None):
    params = params or {}
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    defaults.update(params)
    model = XGBRegressor(**defaults)
    wrapper = MultiOutputRegressor(model)
    wrapper.fit(X_train, y_train)
    return wrapper


def train_hgb(X_train, y_train, params: dict | None = None):
    params = params or {}
    defaults = {"max_iter": 300, "random_state": 42}
    defaults.update(params)
    model = HistGradientBoostingRegressor(**defaults)
    wrapper = MultiOutputRegressor(model)
    wrapper.fit(X_train, y_train)
    return wrapper
