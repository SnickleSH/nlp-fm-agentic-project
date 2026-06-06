from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class ExperimentConfig(BaseModel):
    architecture: str
    domain: str
    difficulty: Literal["easy", "hard"]
    num_runs: int = 5
    max_iterations: int = 10
    max_critic_iterations: int = 3
    temperature: float = 0.7
    max_tokens: int = 4096
    enable_thinking: bool = True  # kept for gridworld backward compat
    thinking_token_budget: int | None = None  # swept variable (2k/4k/6k); triggers reasoning config
    request_timeout: float = 1800.0

    @model_validator(mode="after")
    def _derive_max_tokens(self) -> "ExperimentConfig":
        if self.thinking_token_budget is not None:
            self.max_tokens = self.thinking_token_budget + 2000
        return self


def load_experiment_configs(path: str | Path = "configs/experiments.yaml") -> list[ExperimentConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [ExperimentConfig(**entry) for entry in raw["experiments"]]
