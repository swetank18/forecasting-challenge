from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


class Forecaster:
    name = "forecaster"

    def fit(self, y) -> "Forecaster":
        self.y_ = np.asarray(y, dtype=float)
        if len(self.y_) == 0:
            raise ValueError("cannot fit on an empty series")
        return self

    def predict(self, h: int) -> np.ndarray:
        raise NotImplementedError

    def _flat(self, h: int, value: float | None = None) -> np.ndarray:
        return np.full(h, self.y_[-1] if value is None else value, dtype=float)


class Persistence(Forecaster):
    name = "persistence"

    def predict(self, h: int) -> np.ndarray:
        return self._flat(h)


class MovingAverage(Forecaster):
    def __init__(self, k: int):
        self.k = k
        self.name = f"mean_{k}"

    def predict(self, h: int) -> np.ndarray:
        return self._flat(h, self.y_[-self.k:].mean())


class Drift(Forecaster):
    name = "drift"

    def predict(self, h: int) -> np.ndarray:
        y = self.y_
        slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        return y[-1] + slope * np.arange(1, h + 1)


class SeasonalNaive(Forecaster):
    def __init__(self, period: int):
        self.period = period
        self.name = f"seasonal_naive_{period}"

    def predict(self, h: int) -> np.ndarray:
        if len(self.y_) < self.period:
            return self._flat(h)
        last = self.y_[-self.period:]
        return last[np.arange(h) % self.period]
