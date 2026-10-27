from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc.config import Config
from cwfc.predict import main, run

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def check_schema(preds: pd.DataFrame, readings: pd.DataFrame, h: int = 24):
    assert list(preds.columns) == COLUMNS
    assert len(preds) == h
    expected = np.sort(readings["timestamp"].unique())[-h:]
    assert preds["timestamp"].tolist() == expected.tolist()
    assert preds["timestamp"].is_unique
    values = preds[COLUMNS[1:]].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (preds["active_vm_count"] >= 0).all()
    assert preds["aggregate_cpu_utilization"].between(0, 100).all()


def test_cli_on_smoke_data(smoke_dir, tmp_path):
    out = tmp_path / "predictions.csv"
    main(["--data", str(smoke_dir), "--output", str(out), "--outputs-dir", str(tmp_path / "o"), "--quiet"])
    preds = pd.read_csv(out)
    check_schema(preds, pd.read_csv(smoke_dir / "vm_cpu_readings.csv.gz"))
    assert (tmp_path / "o" / "model_choice.json").exists()


def test_holdout_readings_do_not_change_predictions(smoke_dir):
    clean, *_ = run(smoke_dir, Config(data_dir=smoke_dir), verbose=False)

    readings = pd.read_csv(smoke_dir / "vm_cpu_readings.csv.gz")
    late = readings["timestamp"] >= clean["timestamp"].min()
    readings.loc[late, "avg_cpu"] = np.random.default_rng(1).uniform(0, 100, late.sum())
    readings = readings[~(late & (readings["vm_id"] == "vm0"))]
    readings.to_csv(smoke_dir / "vm_cpu_readings.csv.gz", index=False)

    dirty, *_ = run(smoke_dir, Config(data_dir=smoke_dir), verbose=False)
    pd.testing.assert_frame_equal(clean, dirty)


def test_committed_predictions_match_contract():
    path, data = Path("predictions.csv"), Path("challenge_data/vm_cpu_readings.csv.gz")
    if not (path.exists() and data.exists()):
        pytest.skip("needs challenge data and a generated predictions.csv")
    readings = pd.read_csv(data, usecols=["timestamp"])
    check_schema(pd.read_csv(path), readings)
