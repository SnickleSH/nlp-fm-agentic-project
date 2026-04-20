from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    architecture: str
    domain: str
    difficulty: Literal["easy", "hard"]
    num_runs: int = 5
    max_iterations: int = 10
    temperature: float = 0.7


def load_experiment_configs(path: str | Path = "configs/experiments.yaml") -> list[ExperimentConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [ExperimentConfig(**entry) for entry in raw["experiments"]]
