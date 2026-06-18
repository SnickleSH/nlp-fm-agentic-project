"""Aggregation helpers: capability matrices, CIs, efficiency, failure rates.

All functions take the tidy frame from ``loader.load_all`` (after
``failure_modes.add_failure_mode``) and return small DataFrames ready to print
or hand to ``plots``. Nothing here calls the LLM endpoint — it is pure pandas.

Pre-committed rules honoured (see audit trail / docs/analysis_design.md):
  R1  primary metric = mean cell-level score +/- 95% CI (not success rate)
  R2  efficiency = total_tokens, num_llm_calls; runtime is reported but caveated
  D7  truncated (and infra_error) runs excluded from the capability mean
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.failure_modes import CAPABILITY_MODES, MODE_ORDER
from src.analysis.loader import ARCH_ORDER, LEVEL_ORDER

_RNG = np.random.default_rng(0)  # fixed seed -> reproducible bootstrap CIs


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi).

    Bootstrap (not normal-theory) because cell scores are bounded in [0, 1] and
    often skewed/bimodal at constrained budgets, where a t-interval misbehaves.
    """
    vals = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    boot = _RNG.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(vals.mean()), float(lo), float(hi))


def _capability_subset(df: pd.DataFrame, include_parse_failures: bool = True) -> pd.DataFrame:
    """Rows eligible for the capability mean (excludes infra_error + truncated)."""
    modes = set(CAPABILITY_MODES)
    if not include_parse_failures:
        modes.discard("parse_failure")
    return df[df["failure_mode"].isin(modes)]


def capability_matrix(
    df: pd.DataFrame, domain: str, value: str = "score", include_parse_failures: bool = True
) -> pd.DataFrame:
    """Architecture x level matrix of mean ``value`` over capability-eligible runs.

    Returns a DataFrame indexed by architecture, columns = LEVEL_ORDER. Cells
    that have no runs come out as NaN (e.g. an L3 row before L3 has been run).
    """
    sub = _capability_subset(df[df["domain"] == domain], include_parse_failures)
    if sub.empty:
        return pd.DataFrame(index=ARCH_ORDER, columns=LEVEL_ORDER, dtype=float)
    pivot = (
        sub.groupby(["architecture", "level"], observed=True)[value]
        .mean()
        .unstack("level")
    )
    return pivot.reindex(index=ARCH_ORDER, columns=LEVEL_ORDER)


def capability_matrix_with_ci(
    df: pd.DataFrame, domain: str, include_parse_failures: bool = True
) -> pd.DataFrame:
    """Long-form table: one row per (architecture, level) with mean, lo, hi, n.

    This is the table the report's main results figure is built from.
    """
    sub = _capability_subset(df[df["domain"] == domain], include_parse_failures)
    records = []
    for arch in ARCH_ORDER:
        for level in LEVEL_ORDER:
            cell = sub[(sub["architecture"] == arch) & (sub["level"] == level)]
            mean, lo, hi = bootstrap_ci(cell["score"].tolist())
            records.append(
                {
                    "architecture": arch,
                    "level": level,
                    "n": int(len(cell)),
                    "mean_score": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "success_rate": cell["success"].mean() if len(cell) else np.nan,
                }
            )
    return pd.DataFrame(records)


def efficiency_matrix(
    df: pd.DataFrame, domain: str, value: str = "total_tokens"
) -> pd.DataFrame:
    """Architecture x level matrix of mean efficiency metric (all runs, incl. truncated).

    Efficiency is measured over ALL runs that produced token counts — truncation
    is part of the cost story, so it is not excluded here (unlike capability).
    Use value='total_tokens' or 'num_llm_calls' (R2). Avoid 'runtime_seconds'.
    """
    sub = df[(df["domain"] == domain) & (df["error"].isna())]
    if sub.empty:
        return pd.DataFrame(index=ARCH_ORDER, columns=LEVEL_ORDER, dtype=float)
    pivot = (
        sub.groupby(["architecture", "level"], observed=True)[value]
        .mean()
        .unstack("level")
    )
    return pivot.reindex(index=ARCH_ORDER, columns=LEVEL_ORDER)


def dispersion_table(df: pd.DataFrame, domain: str, value: str = "num_llm_calls") -> pd.DataFrame:
    """Per-condition mean, std and coefficient of variation of ``value``.

    Drives H2 (linear L2A should show low CV in call count; cyclic L2B high CV).
    """
    sub = df[(df["domain"] == domain) & (df["error"].isna())]
    g = sub.groupby(["architecture", "level"], observed=True)[value].agg(
        ["mean", "std", "count"]
    )
    g["cv"] = g["std"] / g["mean"].replace(0, np.nan)
    return g.reset_index()


def planner_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """H3: does committing to an explicit plan help or hurt, per (domain, level)?

    One row per (domain, level) with the capability mean of L1, L2A, L2B and the
    planner advantage = mean(L2A) - mean(L1). A positive advantage means the
    planner helps; negative means it hurts (expected on the dynamic domain).
    Means are taken over capability-eligible runs only (``is_capability``).
    """
    sub = df[df.get("is_capability", False)]
    if sub.empty:
        return pd.DataFrame()
    means = (
        sub.groupby(["domain", "architecture", "level"], observed=True)["score"]
        .mean()
        .unstack("architecture")
    )
    out = pd.DataFrame(index=means.index)
    for arch in ("level1", "level2a", "level2b"):
        out[f"score_{arch}"] = means.get(arch)
    out["adv_vs_l1"] = out["score_level2a"] - out["score_level1"]
    out["adv_vs_l2b"] = out["score_level2a"] - out["score_level2b"]
    out = out.reset_index()
    # order rows by domain then the difficulty ladder
    out["_lvl"] = out["level"].map({l: i for i, l in enumerate(LEVEL_ORDER)})
    return out.sort_values(["domain", "_lvl"]).drop(columns="_lvl").reset_index(drop=True)


def l3_gap_vs_best(df: pd.DataFrame) -> pd.DataFrame:
    """H4: does the adaptive L3 beat the best simpler architecture, per level?

    One row per (domain, level) with: L3 capability mean and its n, the best
    non-L3 architecture and its mean, and ``gap = mean(L3) - best_other_mean``.
    The bar is the best *architecture mean* among {L1, L2A, L2B} — not the best
    single run — so the gap answers "is adding adaptivity worth it here?".
    H4 predicts gap <= 0 at easy/medium (the critic over-corrects) and > 0 at
    the hardest levels. Means use capability-eligible runs only.
    """
    sub = df[df.get("is_capability", False)]
    if sub.empty or (sub["architecture"] == "level3").sum() == 0:
        return pd.DataFrame()
    means = sub.groupby(["domain", "architecture", "level"], observed=True)["score"].mean()
    l3_n = (
        sub[sub["architecture"] == "level3"]
        .groupby(["domain", "level"], observed=True)["score"].size()
    )
    records = []
    for (domain, level), l3_mean in means.xs("level3", level="architecture").items():
        others = means.drop("level3", level="architecture")
        try:
            col = others.xs((domain, level), level=("domain", "level"))
        except KeyError:
            continue
        col = col.dropna()
        if col.empty:
            continue
        best_arch = col.idxmax()
        best_mean = float(col.max())
        records.append(
            {
                "domain": domain,
                "level": level,
                "l3_mean": float(l3_mean),
                "l3_n": int(l3_n.get((domain, level), 0)),
                "best_other_arch": best_arch,
                "best_other_mean": best_mean,
                "gap": float(l3_mean) - best_mean,
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["_lvl"] = out["level"].map({l: i for i, l in enumerate(LEVEL_ORDER)})
    return out.sort_values(["domain", "_lvl"]).drop(columns="_lvl").reset_index(drop=True)


def failure_rate_table(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Per-condition fraction of runs in each failure mode (rows sum to 1)."""
    sub = df[df["domain"] == domain]
    if sub.empty:
        return pd.DataFrame()
    counts = (
        sub.groupby(["architecture", "level", "failure_mode"], observed=True)
        .size()
        .unstack("failure_mode")
        .reindex(columns=MODE_ORDER)
        .fillna(0)
    )
    rates = counts.div(counts.sum(axis=1), axis=0)
    return rates.reset_index()


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """How full is the experimental matrix? One row per (domain, arch, level) with n.

    Use this to see at a glance which cells are still empty (e.g. L3 before B3).
    """
    if df.empty:
        return pd.DataFrame()
    cov = (
        df.groupby(["domain", "architecture", "level"], observed=True)
        .size()
        .rename("n_runs")
        .reset_index()
    )
    return cov.sort_values(["domain", "architecture", "level"]).reset_index(drop=True)


def l3_memory_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive L3 ToT/memory diagnostics (R4: descriptive, not a formal test)."""
    sub = df[df["architecture"] == "level3"]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["domain", "level"], observed=True)[
            ["branch_count", "mem_retrievals", "mem_reuse_hits"]
        ]
        .mean()
        .reset_index()
    )
