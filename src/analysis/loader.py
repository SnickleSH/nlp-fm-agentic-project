"""Unified loader: both domains' JSONL -> one tidy DataFrame for analysis.

Both ``results/logic_final.jsonl`` and ``results/results_gridworld.jsonl`` are
written by ``runner.RunResult.model_dump_json`` (one JSON object per line). This
module loads either (or both) and flattens the nested ``token_usage`` /
``evaluation_details`` / ``state_metadata`` blobs into flat columns.

Difficulty levels
-----------------
``difficulty`` IS the conceptual 4-level ladder (easy / medium / hard /
extra_hard) for both domains, so the analysis keys directly on it (exposed as
the ``level`` column). For logic puzzles, ``easy`` draws from the 3x3 pool and
medium/hard/extra_hard all draw from the same 5x5 pool, separated by
``thinking_token_budget`` (None / 4000 / 1500); the experiment configs set the
conceptual ``difficulty`` explicitly and the engine routes any non-easy
difficulty to the 5x5 pool, so the level is recorded in the data rather than
reconstructed.

Older logic result files (written before this was fixed) stored ``difficulty:
hard`` for every 5x5 row and coded the level only in the budget; those need the
difficulty migration before they will label correctly here.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

# Canonical order used everywhere (matrix columns, plot x-axes, sort).
LEVEL_ORDER = ["easy", "medium", "hard", "extra_hard"]
ARCH_ORDER = ["level1", "level2a", "level2b", "level3"]

ARCH_LABELS = {
    "level1": "L1 Baseline",
    "level2a": "L2A Planner+Executor",
    "level2b": "L2B Solver+Critic",
    "level3": "L3 Adaptive (ToT+Mem)",
}

def derive_level(domain: str, difficulty: str, budget=None) -> str:
    """Conceptual difficulty level for the analysis ladder.

    ``difficulty`` is now the conceptual level for both domains, so this just
    validates and returns it. ``budget`` is accepted for backward-compatible
    call sites and is no longer used. A difficulty that is not a known level is
    passed through with a warning — most likely a stale logic results file from
    before the difficulty migration, whose budget-coded levels need migrating.
    """
    if difficulty not in LEVEL_ORDER:
        warnings.warn(
            f"Unknown difficulty={difficulty!r} (domain={domain}); passing through. "
            "A pre-migration logic file would store 'hard' for every 5x5 row — "
            "migrate it so difficulty carries the conceptual level."
        )
    return difficulty


def _flatten(record: dict) -> dict:
    """One RunResult JSON object -> one flat row dict."""
    usage = record.get("token_usage") or {}
    details = record.get("evaluation_details") or {}
    state_meta = record.get("state_metadata") or {}
    domain = record.get("domain")
    difficulty = record.get("difficulty")
    budget = record.get("thinking_token_budget")

    return {
        # identity
        "domain": domain,
        "architecture": record.get("architecture"),
        "difficulty": difficulty,                       # raw code enum
        "level": derive_level(domain, difficulty, budget),  # conceptual ladder
        "thinking_token_budget": budget,
        "task_id": record.get("task_id"),
        "run_id": record.get("run_id"),
        "max_critic_iterations": record.get("max_critic_iterations", 3),
        # outcome
        "success": bool(record.get("success", False)),
        "score": float(record.get("score", 0.0)),
        # efficiency (cross-domain comparable)
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "num_llm_calls": usage.get("llm_call_count", record.get("num_llm_calls")),
        "max_per_call_completion": max(
            usage.get("per_call_completion_tokens") or [0], default=0
        ),
        "runtime_seconds": record.get("runtime_seconds"),  # caveated, see R2
        # failure-diagnosis flags (raw; classify_failure() turns these into a label)
        "parse_failure": bool(record.get("parse_failure", False)),
        "any_call_truncated": bool(record.get("any_call_truncated", False)),
        "budget_saturated": bool(record.get("budget_saturated", False)),
        "revision_count": record.get("revision_count", 0),
        "error": record.get("error"),
        # domain-specific
        "num_steps": details.get("steps_taken"),          # gridworld only
        "reached_goal": details.get("reached_goal"),       # gridworld only
        "correct_cells": details.get("correct_cells"),     # logic only
        "total_cells": details.get("total_cells"),         # logic only
        "puzzle_id": details.get("puzzle_id"),             # logic only
        "answer_schema": details.get("answer_schema"),     # logic only
        # L3 diagnostics (empty for L1/L2A/L2B)
        "branch_count": state_meta.get("branch_count"),
        "mem_retrievals": state_meta.get("mem_retrievals"),
        "mem_reuse_hits": state_meta.get("mem_reuse_hits"),
        # keep a short answer sample for qualitative failure inspection
        "final_answer": record.get("final_answer", ""),
    }


def load_jsonl(path: str | Path, domain_filter: str | None = None) -> pd.DataFrame:
    """Load one JSONL results file into a tidy DataFrame.

    domain_filter: if given, keep only rows for that domain (a single combined
    file holding both domains is supported).
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Results file not found: {path}. Returning empty frame.")
        return pd.DataFrame()
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(f"Skipping malformed JSON on line {i + 1} of {path}.")
                continue
            if domain_filter and rec.get("domain") != domain_filter:
                continue
            rows.append(_flatten(rec))
    return pd.DataFrame(rows)


def load_all(
    logic_path: str | Path = "results/logic_final.jsonl",
    gridworld_path: str | Path = "results/results_gridworld.jsonl",
) -> pd.DataFrame:
    """Load both domains and concatenate into one analysis frame.

    Either path may be missing (a warning is emitted) so the pipeline runs
    while one domain's sweep is still in flight or while L3 is gated.
    """
    frames = [
        load_jsonl(logic_path, domain_filter="logic_puzzles"),
        load_jsonl(gridworld_path, domain_filter="gridworld"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        warnings.warn("No results loaded from either domain.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Order categoricals so matrices/plots come out in the intended order.
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df["architecture"] = pd.Categorical(
        df["architecture"], categories=ARCH_ORDER, ordered=True
    )
    return df
