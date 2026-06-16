"""Analyse the logic pilot (results/logic_pilot.jsonl): budget zone + H2 gate."""
from __future__ import annotations

import warnings
from pathlib import Path

from src.analysis.aggregate import bootstrap_ci
from src.analysis.failure_modes import add_failure_mode
from src.analysis.loader import load_jsonl

PILOT_PATH = Path("results/logic_pilot.jsonl")


def cell(scores):
    mean, lo, hi = bootstrap_ci(scores)
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}] n={len(scores)}"


def main():
    if not PILOT_PATH.exists():
        raise SystemExit(f"{PILOT_PATH} not found; run configs/logic_pilot.yaml first.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = add_failure_mode(load_jsonl(PILOT_PATH))

    print("L1 by budget:")
    l1 = df[df.architecture == "level1"]
    for b in sorted(l1.thinking_token_budget.dropna().unique()):
        c = l1[l1.thinking_token_budget == b]
        print(f"  {int(b):>5}  {cell(c.score.tolist())}")

    for b in (1500, 4000):
        print(f"\nbudget {b}:")
        for arch in ("level1", "level2a", "level2b"):
            c = df[(df.architecture == arch) & (df.thinking_token_budget == b)]
            if c.empty:
                continue
            trunc = int(c.any_call_truncated.sum())
            print(f"  {arch:<8} {cell(c.score.tolist())}  trunc={trunc}")


if __name__ == "__main__":
    main()
