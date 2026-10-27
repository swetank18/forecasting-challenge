import numpy as np
import pandas as pd
import pytest

from cwfc.backtest import run_backtest
from cwfc.features import design_row, origin_features
from cwfc.models import candidates


@pytest.fixture
def y():
    rng = np.random.default_rng(3)
    t = np.arange(90)
    return 10 + np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.2, len(t))


def corrupt_after(y, o):
    bad = y.copy()
    bad[o:] = np.random.default_rng(9).uniform(-1e3, 1e3, len(y) - o)
    return bad


@pytest.mark.parametrize("origin", [20, 47, 70])
def test_features_ignore_the_future(y, origin):
    bad = corrupt_after(y, origin)
    assert np.array_equal(origin_features(y, origin, 12), origin_features(bad, origin, 12))
    for step in range(1, 25):
        assert np.array_equal(design_row(y, origin, step, 12), design_row(bad, origin, step, 12))


def test_backtest_forecasts_ignore_the_future(y):
    o, h = 48, 24
    models = candidates(12, h)
    clean = run_backtest(pd.Series(y), models, h, [o])
    dirty = run_backtest(pd.Series(corrupt_after(y, o)), models, h, [o])
    assert np.allclose(clean["forecast"], dirty["forecast"])
    assert not np.allclose(clean["actual"], dirty["actual"])


def test_every_candidate_handles_short_and_flat_series():
    for name, make in candidates(12, 24).items():
        for series in (np.full(30, 7.0), np.array([3.0, 4.0, 5.0])):
            pred = make().fit(series).predict(24)
            assert pred.shape == (24,) and np.isfinite(pred).all(), name
