from __future__ import annotations

import numpy as np
import pandas as pd

from .io import infer_cadence


def build_targets(readings: pd.DataFrame, cadence: int | None = None) -> pd.DataFrame:
    # duplicates are averaged per VM first; empty grid slots are interpolated and flagged
    valid = readings[np.isfinite(readings["avg_cpu"])]
    per_vm = valid.groupby(["timestamp", "vm_id"], sort=True)["avg_cpu"].mean().clip(0, 100)
    by_ts = per_vm.groupby(level="timestamp")
    out = pd.DataFrame({
        "active_vm_count": by_ts.size().astype(float),
        "aggregate_cpu_utilization": by_ts.mean(),
    })

    cadence = cadence or infer_cadence(readings["timestamp"])
    grid = np.arange(readings["timestamp"].min(), readings["timestamp"].max() + 1, cadence)
    out = out.reindex(grid)
    out.index.name = "timestamp"
    out["imputed"] = out["aggregate_cpu_utilization"].isna()
    if out["imputed"].any():
        out[["active_vm_count", "aggregate_cpu_utilization"]] = (
            out[["active_vm_count", "aggregate_cpu_utilization"]].interpolate(limit_direction="both")
        )
    return out.reset_index()


def segment_cpu(readings: pd.DataFrame, meta: pd.DataFrame, by: str) -> pd.DataFrame:
    df = readings[["timestamp", "vm_id", "avg_cpu"]].merge(meta[["vm_id", by]], on="vm_id", how="left")
    df[by] = df[by].fillna("missing")
    return df.groupby(["timestamp", by])["avg_cpu"].mean().unstack(by)


def cpu_quantile(readings: pd.DataFrame, q: float = 0.95) -> pd.Series:
    return readings.groupby("timestamp")["avg_cpu"].quantile(q).rename(f"p{int(q * 100)}_cpu")
