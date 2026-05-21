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
    success: bool
    score: float
    token_usage: TokenUsage
    runtime_seconds: float
    num_iterations: int
    num_llm_calls: int
    final_answer: str
    evaluation_details: dict = {}
    error: str | None = None


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
        final_state = graph.invoke(initial_state, config={"callbacks": [metrics_cb]})
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

        return RunResult(
            architecture=config.architecture,
            domain=config.domain,
            difficulty=config.difficulty,
            task_id=task_id,
            run_id=run_id,
            success=evaluation.success,
            score=evaluation.score,
            token_usage=usage,
            runtime_seconds=round(elapsed, 3),
            num_iterations=final_state.get("iteration", 0),
            num_llm_calls=usage.llm_call_count,
            final_answer=final_answer[:500],
            evaluation_details=evaluation.details,
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
            success=False,
            score=0.0,
            token_usage=usage,
            runtime_seconds=round(elapsed, 3),
            num_iterations=0,
            num_llm_calls=usage.llm_call_count,
            final_answer="",
            error=f"{type(e).__name__}: {e}",
        )


def save_result(result: RunResult, path: str | Path = "results/results.jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(result.model_dump_json() + "\n")
