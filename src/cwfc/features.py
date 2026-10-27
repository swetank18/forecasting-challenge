from __future__ import annotations

import numpy as np

N_LAGS = 6


def origin_features(y: np.ndarray, origin: int, period: int) -> np.ndarray:
    """Features known at `origin`, i.e. computed from y[:origin] only."""
    hist = np.asarray(y[:origin], dtype=float)
    level = hist[-period:].mean()
    lags = hist[-N_LAGS:][::-1] - level
    recent = hist[-period:]
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0
    return np.concatenate([
        lags,
        [hist[-period // 2:].mean() - level, recent.std(), hist[-period // 2:].std(), slope],
    ])


def step_features(y: np.ndarray, origin: int, step: int, period: int) -> np.ndarray:
    hist = np.asarray(y[:origin], dtype=float)
    level = hist[-period:].mean()
    back = period * int(np.ceil(step / period))
    seasonal_lag = hist[origin - 1 + step - back] - level
    phase = np.zeros(period)
    phase[(origin - 1 + step) % period] = 1.0
    return np.concatenate([[seasonal_lag, step], phase])


def design_row(y: np.ndarray, origin: int, step: int, period: int) -> np.ndarray:
    return np.concatenate([origin_features(y, origin, period), step_features(y, origin, step, period)])


def min_history(period: int) -> int:
    return max(period, N_LAGS) + 1
