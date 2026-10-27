from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.9, 0.95)


def residual_quantiles(errors: pd.DataFrame, h: int, probs=QUANTILES, pool: int = 2) -> pd.DataFrame:
    """Per-step quantiles of backtest errors (actual - forecast), pooled over nearby steps."""
    steps = np.arange(1, h + 1)
    if errors.empty:
        warnings.warn("no backtest errors available, intervals collapse to the point forecast")
        return pd.DataFrame(0.0, index=steps, columns=list(probs))
    rows = {}
    max_step = int(errors["step"].max())
    for s in steps:
        s_eff = min(s, max_step)
        near = errors.loc[(errors["step"] - s_eff).abs() <= pool, "error"]
        rows[s] = [float(near.quantile(p)) for p in probs]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(probs))


def add_bands(point: np.ndarray, quantiles: pd.DataFrame, lo: float = 0.0, hi: float = np.inf) -> pd.DataFrame:
    out = pd.DataFrame({"forecast": point})
    for p in quantiles.columns:
        out[f"p{int(round(p * 100))}"] = np.clip(point + quantiles[p].to_numpy(), lo, hi)
    return out
