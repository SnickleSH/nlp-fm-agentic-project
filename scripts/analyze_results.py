"""CLI: load both domains' results, print every analysis table, save every figure.

    poetry run python scripts/analyze_results.py \
        --logic results/logic_final.jsonl \
        --gridworld results/gridworld_final.jsonl \
        --figdir figures

Reproducible batch path for the analysis. The notebook is the interactive twin;
both call the same ``src.analysis`` functions, so they cannot diverge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import pandas as pd

from src.analysis import add_failure_mode, load_all
from src.analysis import aggregate as agg
from src.analysis import plots

DOMAINS = ["logic_puzzles", "gridworld"]


def _print(title, frame):
    print(f"\n{title}")
    with pd.option_context("display.width", 140, "display.max_columns", 30):
        print(frame.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", default="results/logic_final.jsonl")
    ap.add_argument("--gridworld", default="results/gridworld_final.jsonl")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    df = add_failure_mode(load_all(args.logic, args.gridworld))
    if df.empty:
        print("No results loaded — nothing to analyse.\n"
              f"  checked --logic={args.logic}\n"
              f"  checked --gridworld={args.gridworld}\n"
              "Point these at the uploaded JSONL result files and re-run.")
        return

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    _print("MATRIX COVERAGE (n runs per cell — empty cells are unrun)",
           agg.coverage_report(df))

    summary = {}
    for domain in DOMAINS:
        if df[df["domain"] == domain].empty:
            print(f"\n[skip] no rows for {domain}")
            continue
        cap = agg.capability_matrix_with_ci(df, domain)
        _print(f"[{domain}] CAPABILITY (mean score, 95% CI, n)", cap)
        _print(f"[{domain}] EFFICIENCY — total_tokens",
               agg.efficiency_matrix(df, domain, "total_tokens"))
        _print(f"[{domain}] EFFICIENCY — num_llm_calls",
               agg.efficiency_matrix(df, domain, "num_llm_calls"))
        _print(f"[{domain}] DISPERSION — num_llm_calls (H2)",
               agg.dispersion_table(df, domain, "num_llm_calls"))
        _print(f"[{domain}] FAILURE RATES", agg.failure_rate_table(df, domain))

        for name, fig in [
            (f"{domain}_capability", plots.plot_capability(df, domain)),
            (f"{domain}_efficiency_tokens", plots.plot_efficiency(df, domain, "total_tokens")),
            (f"{domain}_efficiency_calls", plots.plot_efficiency(df, domain, "num_llm_calls")),
            (f"{domain}_efficiency", plots.plot_efficiency_pair(df, domain)),
            (f"{domain}_failures", plots.plot_failure_stack(df, domain)),
            (f"{domain}_dispersion", plots.plot_dispersion(df, domain, "num_llm_calls")),
        ]:
            path = figdir / f"{name}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            print(f"  saved {path}")

        summary[domain] = cap.to_dict(orient="records")

    adv = agg.planner_advantage(df)
    if not adv.empty:
        _print("H3 — PLANNER ADVANTAGE (mean L2A - mean L1, per domain/level)", adv)
        summary["planner_advantage"] = adv.to_dict(orient="records")

    gap = agg.l3_gap_vs_best(df)
    if not gap.empty:
        _print("H4 — L3 GAP vs BEST SIMPLER ARCHITECTURE (mean L3 - best non-L3 mean)", gap)
        summary["l3_gap_vs_best"] = gap.to_dict(orient="records")

    l3 = agg.l3_memory_diagnostics(df)
    if not l3.empty:
        _print("H5 — L3 MEMORY/ToT DIAGNOSTICS (descriptive, not a causal test)", l3)
        summary["l3_diagnostics"] = l3.to_dict(orient="records")

    (figdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {figdir/'summary.json'} and figures to {figdir}/")


if __name__ == "__main__":
    main()
