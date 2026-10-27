from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("challenge_data")
    horizon: int = 24
    cadence_s: int = 300
    min_train: int = 48
    origin_step: int = 3
    peak_quantile: float = 0.8
    interval_levels: tuple[float, ...] = (0.8, 0.95)
    seed: int = 0

    @property
    def season(self) -> int:
        return 3600 // self.cadence_s
