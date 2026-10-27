import numpy as np
import pandas as pd
import pytest


def make_readings(n_vms=5, n_ts=40, cadence=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vms):
        base = rng.uniform(2, 30)
        for t in range(n_ts):
            avg = float(np.clip(base + 3 * np.sin(t / 6) + rng.normal(0, 1), 0, 100))
            rows.append((t * cadence, f"vm{v}", max(avg - 1, 0), min(avg + 2, 100), avg))
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


@pytest.fixture
def smoke_dir(tmp_path):
    readings = make_readings()
    readings.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    meta = pd.DataFrame({
        "vm_id": readings["vm_id"].unique(),
        "vm_category": ["Interactive", "Delay-insensitive", "Unknown", "Interactive", "Delay-insensitive"],
        "virtual_core_bucket": ["2", "2", "4", "8", "2"],
    })
    meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
    return tmp_path
