from __future__ import annotations

import pandas as pd


def train_holdout_split(series: pd.DataFrame, h: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    if h <= 0:
        raise ValueError("h must be positive")
    if len(series) <= h:
        raise ValueError(f"need more than {h} timestamps, got {len(series)}")
    series = series.sort_values("timestamp").reset_index(drop=True)
    return series.iloc[:-h].copy(), series.iloc[-h:].copy()


def rolling_origins(n_train: int, h: int, min_train: int, step: int) -> list[int]:
    if h <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("h, min_train and step must be positive")
    return list(range(min_train, n_train - h + 1, step))


def backtest_plan(n_train: int, h: int, min_train: int, step: int) -> tuple[int, list[int]]:
    """Shrinks min_train and the horizon on short series instead of failing."""
    min_train = min(min_train, max(n_train // 2, 2))
    h_bt = min(h, n_train - min_train)
    if h_bt < 1:
        return 0, []
    return h_bt, rolling_origins(n_train, h_bt, min_train, step)
