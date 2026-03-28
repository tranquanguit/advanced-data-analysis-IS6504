from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE


def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_hgb(X_train, y_train):
    model = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model
