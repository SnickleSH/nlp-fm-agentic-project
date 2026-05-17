from __future__ import annotations

from src.domains.base import BaseDomain

DOMAINS: dict[str, type[BaseDomain]] = {}


def _register_domains() -> None:
    from src.domains.gridworld.domain import GridworldDomain
    from src.domains.logic_puzzles.domain import LogicPuzzlesDomain

    DOMAINS["gridworld"] = GridworldDomain
    DOMAINS["logic_puzzles"] = LogicPuzzlesDomain


_register_domains()


def get_domain(name: str) -> BaseDomain:
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain: {name!r}. Available: {list(DOMAINS)}")
    return DOMAINS[name]()
