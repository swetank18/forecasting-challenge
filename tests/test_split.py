import pandas as pd
import pytest

from cwfc.split import backtest_plan, rolling_origins, train_holdout_split


def series(n):
    return pd.DataFrame({"timestamp": [i * 300 for i in range(n)], "y": range(n)})


def test_holdout_is_last_h_rows():
    s = series(134)
    train, hold = train_holdout_split(s.sample(frac=1, random_state=1), h=24)
    assert len(train) == 110 and len(hold) == 24
    assert hold["timestamp"].tolist() == s["timestamp"].iloc[-24:].tolist()
    assert train["timestamp"].max() < hold["timestamp"].min()
    assert set(train["timestamp"]).isdisjoint(hold["timestamp"])
    assert train["timestamp"].is_monotonic_increasing


def test_too_short_raises():
    with pytest.raises(ValueError):
        train_holdout_split(series(24), h=24)


def test_origins_stay_inside_train():
    origins = rolling_origins(110, h=24, min_train=48, step=3)
    assert origins[0] == 48
    assert all(o + 24 <= 110 for o in origins)


def test_plan_falls_back_on_short_series():
    h_bt, origins = backtest_plan(16, h=24, min_train=48, step=3)
    assert 0 < h_bt < 24
    assert origins and all(o + h_bt <= 16 for o in origins)
    assert backtest_plan(1, h=24, min_train=48, step=3) == (0, [])
