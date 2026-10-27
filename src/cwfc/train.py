from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .backtest import pick_model, run_backtest, summarise
from .config import TARGETS, Config
from .io import check_data, infer_cadence, load_metadata, load_readings, print_report
from .models import candidates
from .split import backtest_plan, train_holdout_split
from .targets import build_targets


@dataclass
class Selection:
    target: str
    model: str
    horizon: int
    origins: list[int]
    peak_threshold: float
    results: pd.DataFrame
    summary: pd.DataFrame

    @property
    def errors(self) -> pd.DataFrame:
        return self.results[self.results["model"] == self.model]


def prepare(data_dir: Path, cfg: Config, verbose: bool = True):
    readings = load_readings(data_dir)
    cfg = replace(cfg, cadence_s=infer_cadence(readings["timestamp"], cfg.cadence_s))
    if verbose:
        print_report(check_data(readings, load_metadata(data_dir)))
    series = build_targets(readings, cfg.cadence_s)
    train, holdout = train_holdout_split(series, cfg.horizon)
    return cfg, train, holdout


def select(train: pd.DataFrame, cfg: Config) -> dict[str, Selection]:
    h_bt, origins = backtest_plan(len(train), cfg.horizon, cfg.min_train, cfg.origin_step)
    models = candidates(cfg.season, cfg.horizon)
    out = {}
    for target in TARGETS:
        y = train.set_index("timestamp")[target]
        threshold = float(y.quantile(cfg.peak_quantile))
        if not origins:
            empty = pd.DataFrame(columns=["model", "origin", "step", "timestamp", "actual", "forecast", "error"])
            out[target] = Selection(target, "persistence", 0, [], threshold, empty, pd.DataFrame())
            continue
        results = run_backtest(y, models, h_bt, origins)
        summary = summarise(results, threshold)
        out[target] = Selection(target, pick_model(summary), h_bt, origins, threshold, results, summary)
    return out


def save(selections: dict[str, Selection], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    choice = {}
    for target, sel in selections.items():
        sel.results.assign(target=target).to_csv(out_dir / f"backtest_{target}.csv", index=False)
        sel.summary.to_csv(out_dir / f"summary_{target}.csv")
        choice[target] = {"model": sel.model, "horizon": sel.horizon, "origins": sel.origins}
    (out_dir / "model_choice.json").write_text(json.dumps(choice, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run rolling-origin backtests on the training window.")
    ap.add_argument("--data", type=Path, default=Config.data_dir)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--horizon", type=int, default=Config.horizon)
    args = ap.parse_args(argv)

    cfg, train, _ = prepare(args.data, Config(data_dir=args.data, horizon=args.horizon))
    selections = select(train, cfg)
    for target, sel in selections.items():
        print(f"\n{target}: {len(sel.origins)} origins, h={sel.horizon}, chosen={sel.model}")
        print(sel.summary.round(4).to_string())
    save(selections, args.out)


if __name__ == "__main__":
    main()
