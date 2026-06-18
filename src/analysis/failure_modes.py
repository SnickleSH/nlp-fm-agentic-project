"""Single canonical failure-mode classifier for BOTH domains.

Why this file exists
--------------------
The pre-committed taxonomy lives in the audit trail (decision D8). The shipped
``src/domains/preprocessing/gridworld.py::_failure_mode`` implements a *different*
taxonomy, and it is gridworld-only. Two divergent classifiers would make the
cross-domain failure analysis incoherent, so all failure labelling goes through
``classify_failure`` here. ``preprocessing/gridworld.py`` should be updated to
call this (or be retired) — see docs/analysis_design.md.

Differences from the shipped gridworld helper (state these at the defense):

1. Order. D8 puts **truncation before parse_failure**; the shipped helper put
   parse first. We follow D8.
2. "exhausted". The shipped helper used ``budget_saturated`` (a call hit the
   ``max_tokens`` ceiling). D8's mode-3 is **revision exhaustion** — the L2B/L3
   critic looped to ``max_critic_iterations`` without converging. These are
   different signals; we report ``budget_saturated`` separately as a flag and
   reserve the *exhausted* label for revision exhaustion.
3. "reasoning". The shipped helper labelled a row "reasoning" only when an
   ``error`` string was present. D8's mode-4 is a clean run that produced a
   parseable-but-wrong answer (score < 1.0, no truncation/parse/error). We add a
   distinct ``infra_error`` bucket for exceptions/timeouts so an endpoint
   timeout is never mistaken for a reasoning failure.

Priority order (first match wins) — one label per run:

    infra_error        error is not None (exception / endpoint timeout)
    truncated          any_call_truncated  (margin/budget artifact; excl. from capability mean)
    parse_failure      parse_failure       (output not parseable as a solution)
    exhausted          L2B/L3: revision_count >= max_critic_iterations AND score < 0.9
    reasoning          none of the above AND score < 1.0  (clean run, wrong answer)
    success            score == 1.0 / success True

Capability mean (R1/D7): computed over runs whose mode is in
``CAPABILITY_MODES`` — i.e. excluding infra_error and truncated. Parse failures
ARE included (they score 0.0 and reflect the architecture's inability to emit a
valid answer); flip ``include_parse_failures`` in the aggregator if you want
them out too.
"""
from __future__ import annotations

import pandas as pd

# Modes that count toward the capability mean (infra_error + truncated excluded).
CAPABILITY_MODES = {"parse_failure", "exhausted", "reasoning", "success"}

# Stable order for stacked-bar plots / tables.
MODE_ORDER = [
    "success",
    "reasoning",
    "exhausted",
    "parse_failure",
    "truncated",
    "infra_error",
]

MODE_COLORS = {
    "success": "#2e7d32",
    "reasoning": "#f9a825",
    "exhausted": "#ef6c00",
    "parse_failure": "#c62828",
    "truncated": "#6a1b9a",
    "infra_error": "#455a64",
}


def classify_failure(row) -> str:
    """Return the single failure-mode label for one run row (dict or Series)."""
    get = row.get if isinstance(row, dict) else (lambda k, d=None: row.get(k, d))

    if get("error") is not None:
        return "infra_error"
    if get("any_call_truncated", False):
        return "truncated"
    if get("parse_failure", False):
        return "parse_failure"

    arch = get("architecture")
    if arch in ("level2b", "level3"):
        rev = get("revision_count", 0) or 0
        cap = get("max_critic_iterations", 3) or 3
        score = float(get("score", 0.0) or 0.0)
        if rev >= cap and score < 0.9:
            return "exhausted"

    score = float(get("score", 0.0) or 0.0)
    if get("success", False) or score >= 1.0:
        return "success"
    return "reasoning"


def add_failure_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``failure_mode`` column (and ``is_capability`` boolean)."""
    if df.empty:
        df = df.copy()
        df["failure_mode"] = pd.Series(dtype="object")
        df["is_capability"] = pd.Series(dtype="bool")
        return df
    out = df.copy()
    out["failure_mode"] = out.apply(classify_failure, axis=1)
    out["is_capability"] = out["failure_mode"].isin(CAPABILITY_MODES)
    out["failure_mode"] = pd.Categorical(
        out["failure_mode"], categories=MODE_ORDER, ordered=True
    )
    return out
