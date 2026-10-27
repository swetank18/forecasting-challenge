from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGETS, Config
from .intervals import add_bands, residual_quantiles
from .metrics import coverage, mae, rmse
from .models import candidates
from .train import Selection, prepare, save, select

LIMITS = {"active_vm_count": (0.0, np.inf), "aggregate_cpu_utilization": (0.0, 100.0)}


def forecast_target(train: pd.DataFrame, sel: Selection, cfg: Config) -> pd.DataFrame:
    make = candidates(cfg.season, cfg.horizon)[sel.model]
    point = make().fit(train[sel.target].to_numpy(dtype=float)).predict(cfg.horizon)
    lo, hi = LIMITS[sel.target]
    bands = add_bands(np.clip(point, lo, hi), residual_quantiles(sel.errors, cfg.horizon), lo, hi)
    return bands


def run(data_dir: Path, cfg: Config, verbose: bool = True):
    cfg, train, holdout = prepare(data_dir, cfg, verbose)
    selections = select(train, cfg)
    preds = pd.DataFrame({"timestamp": holdout["timestamp"].to_numpy()})
    bands = {}
    for target in TARGETS:
        bands[target] = forecast_target(train, selections[target], cfg)
        preds[target] = bands[target]["forecast"].to_numpy()
    preds["active_vm_count"] = preds["active_vm_count"].round().astype(int)
    return preds, bands, selections, train, holdout


def holdout_report(preds, bands, selections, train, holdout) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        actual = holdout[target].to_numpy(dtype=float)
        naive = np.full(len(actual), train[target].iloc[-1])
        b = bands[target]
        rows.append({
            "target": target,
            "model": selections[target].model,
            "mae": mae(actual, preds[target]),
            "rmse": rmse(actual, preds[target]),
            "persistence_mae": mae(actual, naive),
            "persistence_rmse": rmse(actual, naive),
            "coverage_p90": coverage(actual, b["p90"]),
            "coverage_p95": coverage(actual, b["p95"]),
        })
    return pd.DataFrame(rows).set_index("target")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Forecast the final holdout window and write predictions.csv.")
    ap.add_argument("--data", type=Path, default=Config.data_dir)
    ap.add_argument("--output", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--horizon", type=int, default=Config.horizon)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    preds, bands, selections, train, holdout = run(
        args.data, Config(data_dir=args.data, horizon=args.horizon), not args.quiet
    )
    preds.to_csv(args.output, index=False)
    print(f"\nwrote {len(preds)} rows to {args.output}")
    for target, sel in selections.items():
        print(f"  {target}: {sel.model}")

    save(selections, args.outputs_dir)
    for target, b in bands.items():
        b.assign(timestamp=preds["timestamp"]).to_csv(args.outputs_dir / f"forecast_{target}.csv", index=False)

    # scored after the forecast is written; nothing here feeds back into fitting or selection
    report = holdout_report(preds, bands, selections, train, holdout)
    report.to_csv(args.outputs_dir / "holdout_eval.csv")
    print("\nholdout evaluation (not used for selection):")
    print(report.round(4).to_string())


if __name__ == "__main__":
    main()
