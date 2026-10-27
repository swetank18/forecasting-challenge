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


def policy_stats(actual, capacity, window=None) -> dict:
    gap = np.asarray(capacity, dtype=float) - np.asarray(actual, dtype=float)
    out = {
        "breach_rate": float(np.mean(gap < 0)),
        "mean_headroom": float(np.mean(gap)),
        "worst_shortfall": float(max(-gap.min(), 0.0)),
    }
    if window is not None:
        out["window_breach_rate"] = float(pd.Series(gap < 0).groupby(np.asarray(window)).any().mean())
    return out


def backtest_policies(results: pd.DataFrame, model: str, h: int, p: float = 0.95) -> pd.DataFrame:
    """Capacity rules scored on backtest folds: per-step band, flat 2h block, static margin on persistence."""
    m = results[results["model"] == model].sort_values(["origin", "step"])
    base = results[results["model"] == "persistence"].sort_values(["origin", "step"])
    q = residual_quantiles(m, h, probs=(p,))[p]
    per_step = m["forecast"].to_numpy() + m["step"].map(q).to_numpy()
    block = pd.Series(per_step).groupby(m["origin"].to_numpy()).transform("max").to_numpy()
    static = base["forecast"].to_numpy() + base["error"].quantile(p)
    actual, window = m["actual"].to_numpy(), m["origin"].to_numpy()
    rules = {"per_step_band": per_step, "block_max_band": block, "persistence_static_margin": static}
    return pd.DataFrame({k: policy_stats(actual, v, window) for k, v in rules.items()}).T
