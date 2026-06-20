"""Plot functions for the analysis. Matplotlib only (no seaborn dependency).

Each function returns a matplotlib Figure so the notebook can display it and the
CLI can savefig it. Colours/labels are centralised so figures are consistent.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.aggregate import (
    capability_matrix_with_ci,
    efficiency_matrix,
    failure_rate_table,
)
from src.analysis.failure_modes import MODE_COLORS, MODE_ORDER
from src.analysis.loader import ARCH_LABELS, ARCH_ORDER, LEVEL_ORDER

_ARCH_BAR_COLORS = {
    "level1": "#90a4ae",
    "level2a": "#42a5f5",
    "level2b": "#7e57c2",
    "level3": "#ef5350",
}


def plot_capability(df: pd.DataFrame, domain: str):
    """Grouped bars: mean score per architecture across the difficulty ladder,
    with bootstrap 95% CI whiskers. The main results figure."""
    tbl = capability_matrix_with_ci(df, domain)
    fig, ax = plt.subplots(figsize=(9, 5))
    archs = [a for a in ARCH_ORDER if a in tbl["architecture"].unique()]
    x = np.arange(len(LEVEL_ORDER))
    width = 0.8 / max(len(archs), 1)
    for i, arch in enumerate(archs):
        sub = tbl[tbl["architecture"] == arch].set_index("level").reindex(LEVEL_ORDER)
        means = sub["mean_score"].values
        lo = sub["mean_score"].values - sub["ci_lo"].values
        hi = sub["ci_hi"].values - sub["mean_score"].values
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            label=ARCH_LABELS.get(arch, arch),
            color=_ARCH_BAR_COLORS.get(arch, None),
            yerr=[np.nan_to_num(lo), np.nan_to_num(hi)],
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_ylabel("Mean cell-level score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Capability by architecture x difficulty — {domain}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_efficiency_pair(df: pd.DataFrame, domain: str):
    """Two panels side by side: total_tokens (log y) and num_llm_calls (linear),
    so the compute cost and the model-call overhead are read together."""
    panels = [("total_tokens", True), ("num_llm_calls", False)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(LEVEL_ORDER))
    for ax, (value, log_y) in zip(axes, panels):
        mat = efficiency_matrix(df, domain, value=value)
        for arch in ARCH_ORDER:
            if arch not in mat.index:
                continue
            y = mat.loc[arch].reindex(LEVEL_ORDER).values.astype(float)
            if np.all(np.isnan(y)):
                continue
            ax.plot(x, y, marker="o", label=ARCH_LABELS.get(arch, arch),
                    color=_ARCH_BAR_COLORS.get(arch))
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_ORDER)
        ax.set_ylabel(value)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(value)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Efficiency — {domain}")
    fig.tight_layout()
    return fig


def plot_failure_stack(df: pd.DataFrame, domain: str):
    """Stacked bars of failure-mode composition per (architecture, level)."""
    rates = failure_rate_table(df, domain)
    if rates.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {domain}", ha="center")
        return fig
    rates = rates.copy()
    rates["cond"] = (
        rates["architecture"].astype(str) + "\n" + rates["level"].astype(str)
    )
    # order conditions arch-major then level
    rates["arch_ord"] = rates["architecture"].map({a: i for i, a in enumerate(ARCH_ORDER)})
    rates["lvl_ord"] = rates["level"].map({l: i for i, l in enumerate(LEVEL_ORDER)})
    rates = rates.sort_values(["arch_ord", "lvl_ord"])
    fig, ax = plt.subplots(figsize=(max(10, len(rates) * 0.5), 5))
    bottom = np.zeros(len(rates))
    x = np.arange(len(rates))
    for mode in MODE_ORDER:
        if mode not in rates.columns:
            continue
        vals = rates[mode].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=mode, color=MODE_COLORS[mode])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(rates["cond"], fontsize=6, rotation=0)
    ax.set_ylabel("Fraction of runs")
    ax.set_ylim(0, 1)
    ax.set_title(f"Failure-mode composition — {domain}")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


_LEVEL_DOT_COLORS = {
    "easy": "#90caf9",
    "medium": "#4caf50",
    "hard": "#ff9800",
    "extra_hard": "#e53935",
}


def plot_dispersion(df: pd.DataFrame, domain: str, value: str = "num_llm_calls"):
    """Per-run scatter of ``value`` by architecture, split by difficulty (H2).

    Each dot is one run; the dots for a given architecture are split into the
    four difficulty sub-columns so the spread that matters — the variation
    *within* one (architecture, difficulty) condition — is legible rather than
    pooled across the whole ladder. A short horizontal tick marks each
    condition's mean. A tight vertical cluster means a near-deterministic call
    count (linear L2A); a tall spread means variable looping (cyclic L2B, or the
    gridworld tool loop). The matching coefficients of variation are in
    ``dispersion_table``.
    """
    sub = df[(df["domain"] == domain) & (df["error"].isna())]
    archs = [a for a in ARCH_ORDER if a in sub["architecture"].unique()]
    levels = [l for l in LEVEL_ORDER if l in sub["level"].unique()]
    fig, ax = plt.subplots(figsize=(10, 5))
    step = 0.8 / max(len(levels), 1)
    rng = np.random.default_rng(0)
    for ai, arch in enumerate(archs):
        for li, level in enumerate(levels):
            vals = sub[(sub["architecture"] == arch) & (sub["level"] == level)][value].dropna().values
            if len(vals) == 0:
                continue
            center = ai + (li - (len(levels) - 1) / 2) * step
            jitter = (rng.random(len(vals)) - 0.5) * step * 0.7
            ax.scatter(np.full(len(vals), center) + jitter, vals, alpha=0.55, s=16,
                       color=_LEVEL_DOT_COLORS.get(level),
                       label=level if ai == 0 else None)
            ax.plot([center - step * 0.4, center + step * 0.4], [vals.mean()] * 2,
                    color="black", lw=1.4)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in archs], fontsize=9)
    ax.set_ylabel(f"{value} (one dot = one run)")
    ax.set_title(f"Per-run {value} spread by architecture x difficulty — {domain}")
    ax.legend(title="difficulty", fontsize=8, ncol=len(levels))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
