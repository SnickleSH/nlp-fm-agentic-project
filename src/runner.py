from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.architectures import get_architecture
from src.config import ExperimentConfig
from src.domains import get_domain
from src.metrics import MetricsCallback, TokenUsage


class RunResult(BaseModel):
    architecture: str
    domain: str
    difficulty: str
    task_id: int
    run_id: int
    thinking_token_budget: int | None = None
    # Logged so dedup key and exhaustion classifier can read it from the result row
    # without needing the original config. Default 3 matches pre-sweep pilot behaviour.
    max_critic_iterations: int = 3
    success: bool
    score: float
    token_usage: TokenUsage
    runtime_seconds: float
    num_iterations: int
    num_llm_calls: int
    final_answer: str
    evaluation_details: dict = {}
    error: str | None = None
    # True when parse_llm_answer returned None — a different failure mode from a
    # wrong-but-parseable answer.  Must be excluded from score means.
    parse_failure: bool = False
    # Number of critic calls made in a Level 2B run (0 for L1/L2A).
    revision_count: int = 0
    # True when any single call hit finish_reason=length (hard completion cap).
    any_call_truncated: bool = False
    # True when any single call's completion_tokens ≈ max_tokens ceiling
    # (thinking_token_budget + 2000), meaning output was near the hard cap.
    budget_saturated: bool = False


def run_single(config: ExperimentConfig, task_id: int, run_id: int) -> RunResult:
    domain = get_domain(config.domain)
    arch = get_architecture(config.architecture)
    task = domain.generate_task(config.difficulty, task_id)
    metrics_cb = MetricsCallback()
    graph = arch.build_graph(domain, config)

    system_prompt = domain.format_system_prompt(task)
    task_prompt = domain.format_task_prompt(task)

    initial_state = {
        "task": task.model_dump(),
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt),
        ],
        "final_answer": "",
        "iteration": 0,
        "max_iterations": config.max_iterations,
        "metadata": {},
    }

    try:
        start = time.time()
        final_state = graph.invoke(
            initial_state,
            config={"callbacks": [metrics_cb], "recursion_limit": config.recursion_limit},
        )
        elapsed = time.time() - start

        # Extract final answer from last AI message if not explicitly set
        final_answer = final_state.get("final_answer", "")
        if not final_answer:
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break

        evaluation = domain.evaluate(task, final_answer)
        usage = metrics_cb.get_usage()
        # budget_saturated: was any call near the hard completion ceiling?
        # Threshold is max_tokens (thinking_token_budget + 2000), not the
        # thinking budget alone — the earlier threshold was always True.
        budget_saturated = (
            config.thinking_token_budget is not None
            and usage.max_per_call_completion_tokens
            >= int(config.max_tokens * 0.95)
        )
        any_call_truncated = "length" in usage.per_call_finish_reasons
        parse_failure = evaluation.details.get("error") == "parse_failure"

        return RunResult(
            architecture=config.architecture,
            domain=config.domain,
            difficulty=config.difficulty,
            task_id=task_id,
            run_id=run_id,
            thinking_token_budget=config.thinking_token_budget,
            max_critic_iterations=config.max_critic_iterations,
            success=evaluation.success,
            score=evaluation.score,
            token_usage=usage,
            runtime_seconds=round(elapsed, 3),
            num_iterations=final_state.get("iteration", 0),
            num_llm_calls=usage.llm_call_count,
            final_answer=final_answer[:500],
            evaluation_details=evaluation.details,
            parse_failure=parse_failure,
            revision_count=final_state.get("critic_iterations", 0),
            any_call_truncated=any_call_truncated,
            budget_saturated=budget_saturated,
        )
    except Exception as e:
        elapsed = time.time() - start
        usage = metrics_cb.get_usage()
        return RunResult(
            architecture=config.architecture,
            domain=config.domain,
            difficulty=config.difficulty,
            task_id=task_id,
            run_id=run_id,
            thinking_token_budget=config.thinking_token_budget,
            max_critic_iterations=config.max_critic_iterations,
            success=False,
            score=0.0,
            token_usage=usage,
            runtime_seconds=round(elapsed, 3),
            num_iterations=0,
            num_llm_calls=usage.llm_call_count,
            final_answer="",
            error=f"{type(e).__name__}: {e}",
            any_call_truncated="length" in usage.per_call_finish_reasons,
        )


def save_result(result: RunResult, path: str | Path = "results/results.jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(result.model_dump_json() + "\n")


def load_completed_keys(path: str | Path) -> set[tuple]:
    """Return the set of (architecture, domain, difficulty, task_id, run_id, thinking_token_budget)
    tuples already present in the results file.  Used to skip re-runs on restart.
    """
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add((
                    r.get("architecture"),
                    r.get("domain"),
                    r.get("difficulty"),
                    r.get("task_id"),
                    r.get("run_id"),
                    r.get("thinking_token_budget"),
                    r.get("max_critic_iterations", 3),  # default 3 matches pre-sweep pilot rows
                ))
            except json.JSONDecodeError:
                pass
    return keys
