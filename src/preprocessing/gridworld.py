"""Load gridworld run results into a tidy DataFrame for analysis.

Emits one row per run with the columns the analysis notebook needs:
domain, architecture, level, score, total_tokens, num_llm_calls, num_steps,
failure_mode, task_id, run_id.

`failure_mode` is derived by priority — the first matching condition wins —
so each row gets exactly one label.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COMMON_COLUMNS = [
    "domain", "architecture", "level", "score",
    "total_tokens", "num_llm_calls", "num_steps",
    "failure_mode", "task_id", "run_id",
]


def _failure_mode(record: dict) -> str:
    if record.get("parse_failure"):
        return "parse"
    if record.get("any_call_truncated"):
        return "truncated"
    if record.get("budget_saturated"):
        return "exhausted"
    if record.get("error"):
        return "reasoning"
    return "none"


def _row(record: dict) -> dict:
    usage = record.get("token_usage", {}) or {}
    eval_details = record.get("evaluation_details", {}) or {}
    return {
        "domain": record["domain"],
        "architecture": record["architecture"],
        "level": record["difficulty"],
        "score": record["score"],
        "total_tokens": usage.get("total_tokens"),
        "num_llm_calls": usage.get("llm_call_count", record.get("num_llm_calls")),
        "num_steps": eval_details.get("steps_taken"),
        "failure_mode": _failure_mode(record),
        "task_id": record["task_id"],
        "run_id": record["run_id"],
    }


def load_gridworld(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("domain") != "gridworld":
                continue
            rows.append(_row(record))
    return pd.DataFrame(rows, columns=COMMON_COLUMNS)
