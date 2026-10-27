from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .baselines import Forecaster
from .metrics import mae, mase, peak_mae, rmse

Factory = Callable[[], Forecaster]


def run_backtest(
    y: pd.Series,
    models: dict[str, Factory],
    h: int,
    origins: list[int],
) -> pd.DataFrame:
    values = y.to_numpy(dtype=float)
    index = y.index.to_numpy()
    rows = []
    for o in origins:
        train, actual = values[:o], values[o:o + h]
        scale = np.mean(np.abs(np.diff(train))) if len(train) > 1 else np.nan
        for name, make in models.items():
            pred = make().fit(train).predict(len(actual))
            rows.append(pd.DataFrame({
                "model": name,
                "origin": o,
                "step": np.arange(1, len(actual) + 1),
                "timestamp": index[o:o + h],
                "actual": actual,
                "forecast": pred,
                "scale": scale,
            }))
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["actual"] - out["forecast"]
    return out


def summarise(results: pd.DataFrame, peak_threshold: float, baseline: str = "persistence") -> pd.DataFrame:
    rows = []
    for name, g in results.groupby("model", sort=False):
        folds = g.groupby("origin")
        fold_mase = folds.apply(lambda f: _fold_mase(f), include_groups=False)
        rows.append({
            "model": name,
            "mae": mae(g["actual"], g["forecast"]),
            "rmse": rmse(g["actual"], g["forecast"]),
            "mase": float(fold_mase.mean()),
            "peak_mae": peak_mae(g["actual"], g["forecast"], peak_threshold),
            "worst_fold_mae": float(folds.apply(lambda f: mae(f["actual"], f["forecast"]), include_groups=False).max()),
        })
    table = pd.DataFrame(rows).set_index("model")
    if baseline in table.index:
        base = table.loc[baseline, "mae"]
        table["mae_vs_baseline_pct"] = 100 * (1 - table["mae"] / base) if base > 0 else 0.0
    return table


def _fold_mase(f: pd.DataFrame) -> float:
    scale = f["scale"].iloc[0]
    err = mae(f["actual"], f["forecast"])
    if not np.isfinite(scale) or scale == 0:
        return 0.0 if err == 0 else float("inf")
    return err / scale


def pick_model(summary: pd.DataFrame) -> str:
    # summary rows are ordered simplest first, so ties go to the simpler model
    ranked = summary.assign(_order=range(len(summary))).sort_values(["mase", "rmse", "_order"])
    return str(ranked.index[0])
