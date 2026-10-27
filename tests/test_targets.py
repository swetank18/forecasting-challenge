import numpy as np
import pandas as pd

from cwfc.io import check_data, load_readings
from cwfc.targets import build_targets


def frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_counts_each_vm_once_and_averages():
    df = frame([
        (0, "a", 0, 0, 10.0),
        (0, "a", 0, 0, 20.0),
        (0, "b", 0, 0, 30.0),
        (300, "a", 0, 0, 40.0),
    ])
    out = build_targets(df).set_index("timestamp")
    assert out.loc[0, "active_vm_count"] == 2
    assert out.loc[0, "aggregate_cpu_utilization"] == 22.5
    assert out.loc[300, "active_vm_count"] == 1
    assert out.loc[300, "aggregate_cpu_utilization"] == 40.0


def test_invalid_and_out_of_range_readings():
    df = frame([
        (0, "a", 0, 0, 150.0),
        (0, "b", 0, 0, -5.0),
        (0, "c", 0, 0, np.nan),
    ])
    out = build_targets(df).iloc[0]
    assert out["active_vm_count"] == 2
    assert out["aggregate_cpu_utilization"] == 50.0


def test_gap_is_flagged_and_filled():
    df = frame([(0, "a", 0, 0, 10.0), (600, "a", 0, 0, 30.0)])
    out = build_targets(df, cadence=300)
    assert out["timestamp"].tolist() == [0, 300, 600]
    assert out["imputed"].tolist() == [False, True, False]
    assert out.loc[1, "aggregate_cpu_utilization"] == 20.0


def test_check_data_on_smoke(smoke_dir):
    report = check_data(load_readings(smoke_dir))
    assert report["vms"] == 5
    assert report["duplicate_rows"] == 0
    assert report["cadence_values"] == [300]
    assert report["all_vms_share_timestamps"]
