from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .baselines import Drift, Forecaster, MovingAverage, Persistence, SeasonalNaive
from .features import design_row, min_history


def _is_flat(y: np.ndarray) -> bool:
    return np.ptp(y) < 1e-12


class HoltDamped(Forecaster):
    name = "holt_damped"

    def fit(self, y):
        super().fit(y)
        self.res_ = None
        if len(self.y_) >= 10 and not _is_flat(self.y_):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res_ = ExponentialSmoothing(
                    self.y_, trend="add", damped_trend=True, initialization_method="estimated"
                ).fit()
        return self

    def predict(self, h):
        return self._flat(h) if self.res_ is None else np.asarray(self.res_.forecast(h))


class SeasonalETS(Forecaster):
    def __init__(self, period: int):
        self.period = period
        self.name = f"ets_seasonal_{period}"

    def fit(self, y):
        super().fit(y)
        self.res_ = None
        if len(self.y_) >= 2 * self.period + 2 and not _is_flat(self.y_):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res_ = ExponentialSmoothing(
                    self.y_, seasonal="add", seasonal_periods=self.period,
                    initialization_method="estimated",
                ).fit()
        return self

    def predict(self, h):
        return self._flat(h) if self.res_ is None else np.asarray(self.res_.forecast(h))


class AutoARIMA(Forecaster):
    name = "arima_aic"

    def __init__(self, max_p: int = 2, max_d: int = 1, max_q: int = 2):
        self.grid = list(itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)))

    def fit(self, y):
        super().fit(y)
        self.res_, self.order_ = None, None
        if len(self.y_) < 12 or _is_flat(self.y_):
            return self
        best = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, d, q in self.grid:
                try:
                    res = ARIMA(self.y_, order=(p, d, q), trend="c" if d == 0 else "n").fit()
                except Exception:
                    continue
                if np.isfinite(res.aic) and res.aic < best:
                    best, self.res_, self.order_ = res.aic, res, (p, d, q)
        return self

    def predict(self, h):
        return self._flat(h) if self.res_ is None else np.asarray(self.res_.forecast(h))


class LevelSeason(Forecaster):
    """Mean of the last cycle plus the average within-cycle profile."""

    def __init__(self, period: int):
        self.period = period
        self.name = f"level_season_{period}"

    def fit(self, y):
        super().fit(y)
        k = len(self.y_) // self.period
        self.profile_ = np.zeros(self.period)
        if k >= 2:
            cycles = self.y_[len(self.y_) - k * self.period:].reshape(k, self.period)
            self.profile_ = (cycles - cycles.mean(axis=1, keepdims=True)).mean(axis=0)
        return self

    def predict(self, h):
        level = self.y_[-self.period:].mean()
        return level + self.profile_[np.arange(h) % self.period]


class DirectRidge(Forecaster):
    """One ridge model across horizons, predicting the gap to the recent level."""

    def __init__(self, period: int, horizon: int, alpha: float = 100.0):
        self.period, self.horizon, self.alpha = period, horizon, alpha
        self.name = "ridge_direct"

    def fit(self, y):
        super().fit(y)
        self.model_ = None
        n, start = len(self.y_), min_history(self.period)
        if n - start < 2 * self.period or _is_flat(self.y_):
            return self
        X, t = [], []
        for o in range(start, n):
            level = self.y_[o - self.period:o].mean()
            for s in range(1, min(self.horizon, n - o) + 1):
                X.append(design_row(self.y_, o, s, self.period))
                t.append(self.y_[o - 1 + s] - level)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X), t)
        return self

    def predict(self, h):
        if self.model_ is None:
            return self._flat(h)
        n = len(self.y_)
        level = self.y_[-self.period:].mean()
        X = [design_row(self.y_, n, s, self.period) for s in range(1, h + 1)]
        return level + self.model_.predict(self.scaler_.transform(X))


def candidates(period: int, horizon: int) -> dict:
    makers = [
        Persistence,
        lambda: MovingAverage(period // 2),
        lambda: MovingAverage(period),
        Drift,
        lambda: SeasonalNaive(period),
        lambda: LevelSeason(period),
        HoltDamped,
        lambda: SeasonalETS(period),
        AutoARIMA,
        lambda: DirectRidge(period, horizon),
    ]
    return {m().name: m for m in makers}
