from __future__ import annotations

from src.architectures.base import BaseArchitecture

ARCHITECTURES: dict[str, type[BaseArchitecture]] = {}


def _register_architectures() -> None:
    from src.architectures.level1_baseline import Level1Baseline
    from src.architectures.level2a_planner_executor import Level2APlannerExecutor

    ARCHITECTURES["level1"] = Level1Baseline
    ARCHITECTURES["level2a"] = Level2APlannerExecutor


_register_architectures()


def get_architecture(name: str) -> BaseArchitecture:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name!r}. Available: {list(ARCHITECTURES)}")
    return ARCHITECTURES[name]()
