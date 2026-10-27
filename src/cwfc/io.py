from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import METADATA_FILE, READINGS_FILE

READING_COLS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / READINGS_FILE)
    missing = set(READING_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"readings file is missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise").astype("int64")
    df["vm_id"] = df["vm_id"].astype(str)
    for c in ("min_cpu", "max_cpu", "avg_cpu"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["timestamp", "vm_id"], kind="stable").reset_index(drop=True)


def load_metadata(data_dir: str | Path) -> pd.DataFrame | None:
    path = Path(data_dir) / METADATA_FILE
    if not path.exists():
        return None
    meta = pd.read_csv(path)
    meta["vm_id"] = meta["vm_id"].astype(str)
    for c in ("virtual_core_bucket", "memory_gb_bucket"):
        if c in meta:
            meta[c] = meta[c].astype(str)
    return meta


def infer_cadence(timestamps, default: int = 300) -> int:
    ts = np.unique(np.asarray(timestamps, dtype="int64"))
    if len(ts) < 2:
        return default
    return int(np.min(np.diff(ts)))


def check_data(readings: pd.DataFrame, meta: pd.DataFrame | None = None) -> dict:
    ts = np.unique(readings["timestamp"].to_numpy())
    diffs = np.diff(ts)
    cpu = readings[["min_cpu", "max_cpu", "avg_cpu"]]
    per_ts = readings.groupby("timestamp")["vm_id"].nunique()
    per_vm = readings.groupby("vm_id")["timestamp"].agg(["min", "max", "count"])

    report = {
        "rows": len(readings),
        "vms": readings["vm_id"].nunique(),
        "timestamps": len(ts),
        "first_ts": int(ts[0]),
        "last_ts": int(ts[-1]),
        "cadence_values": sorted(set(diffs.tolist())),
        "duplicate_rows": int(readings.duplicated(["timestamp", "vm_id"]).sum()),
        "missing_values": readings.isna().sum().to_dict(),
        "cpu_out_of_range": int(((cpu < 0) | (cpu > 100)).any(axis=1).sum()),
        "min_gt_avg": int((readings["min_cpu"] > readings["avg_cpu"] + 1e-9).sum()),
        "avg_gt_max": int((readings["avg_cpu"] > readings["max_cpu"] + 1e-9).sum()),
        "vms_per_ts_min": int(per_ts.min()),
        "vms_per_ts_max": int(per_ts.max()),
        "obs_per_vm_min": int(per_vm["count"].min()),
        "obs_per_vm_max": int(per_vm["count"].max()),
        "all_vms_share_timestamps": bool(per_ts.nunique() == 1 and per_vm["count"].min() == len(ts)),
    }
    if meta is not None:
        rv, mv = set(readings["vm_id"]), set(meta["vm_id"])
        report["vms_without_metadata"] = len(rv - mv)
        report["metadata_without_readings"] = len(mv - rv)
        report["duplicate_metadata"] = int(meta["vm_id"].duplicated().sum())
    return report


def print_report(report: dict) -> None:
    width = max(len(k) for k in report)
    for k, v in report.items():
        print(f"{k:<{width}}  {v}")
