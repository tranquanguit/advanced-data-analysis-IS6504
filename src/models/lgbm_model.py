from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

def train_lgbm(X_train, y_train, params: dict | None = None):
    params = params or {}
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    defaults.update(params)
    model = LGBMRegressor(**defaults)
    wrapper = MultiOutputRegressor(model)
    wrapper.fit(X_train, y_train)
    return wrapper
