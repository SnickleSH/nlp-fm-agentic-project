"""Analysis package: load results, classify failures, aggregate, plot.

Typical use (notebook or script):

    from src.analysis import load_all, add_failure_mode
    from src.analysis import aggregate as agg
    from src.analysis import plots

    df = add_failure_mode(load_all("results/logic_final.jsonl",
                                   "results/results_gridworld.jsonl"))
    agg.capability_matrix_with_ci(df, "logic_puzzles")
    plots.plot_capability(df, "gridworld")
"""
from __future__ import annotations

from src.analysis.failure_modes import add_failure_mode, classify_failure
from src.analysis.loader import (
    ARCH_LABELS,
    ARCH_ORDER,
    LEVEL_ORDER,
    derive_level,
    load_all,
    load_jsonl,
)

__all__ = [
    "load_all",
    "load_jsonl",
    "derive_level",
    "add_failure_mode",
    "classify_failure",
    "LEVEL_ORDER",
    "ARCH_ORDER",
    "ARCH_LABELS",
]
