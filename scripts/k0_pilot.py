"""K0 pilot: find the puzzle grade where unlimited ≈ solves 5x5 but 1500 clearly degrades.

Tests CANDIDATE_GRADES × BUDGETS on PUZZLES_PER_GRADE puzzles each using L1 only.
Prints a summary table to help pick the discrimination-zone grade.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.architectures.level1_baseline import Level1Baseline
from src.config import ExperimentConfig
from src.domains.base import Task
from src.domains.logic_puzzles.domain import LogicPuzzlesDomain
from src.domains.logic_puzzles.engine import (
    LogicPuzzle,
    _load_raw_records,
    _parse_solution,
)
from src.metrics import MetricsCallback

CANDIDATE_GRADES = ["level2", "level3", "level4"]
PUZZLES_PER_GRADE = 2
BUDGETS: list[int | None] = [None, 1500]
OUTPUT = Path("results/k0_pilot.jsonl")


def _get_puzzle(grade: str, idx: int) -> LogicPuzzle:
    records = _load_raw_records()
    pool = sorted(
        [r for r in records if "5x5" in r["ID"] and f"_{grade}-" in r["ID"]],
        key=lambda r: r["ID"],
    )
    if idx >= len(pool):
        raise ValueError(f"No puzzle at index {idx} for grade={grade}")
    rec = pool[idx]
    sol = _parse_solution(rec["SolutionGrid"])
    return LogicPuzzle(
        puzzle_id=rec["ID"],
        clues=rec["Clues"],
        solution=sol,
        num_positions=len(next(iter(sol.values()))),
        num_attributes=len(sol),
        difficulty="hard",
    )


def _run_one(puzzle: LogicPuzzle, budget: int | None) -> dict:
    domain = LogicPuzzlesDomain()

    task = Task(
        task_id=0,
        description=(
            f"Solve a logic grid puzzle. "
            f"{puzzle.num_positions} positions, {puzzle.num_attributes} attribute categories."
        ),
        rules=[
            "Positions are numbered from 1 to N, left to right.",
            "Each attribute category has exactly one value per position.",
            "Use all clues to deduce the full assignment.",
            "Return a JSON object mapping each attribute to a list of values in position order.",
            "Return only JSON, with no extra text or markdown.",
        ],
        ground_truth=puzzle.solution,
        difficulty="hard",
        metadata={
            "puzzle_id": puzzle.puzzle_id,
            "num_positions": puzzle.num_positions,
            "num_attributes": puzzle.num_attributes,
            "clues": puzzle.clues,
        },
    )

    from src.llm import UNLIMITED_MAX_TOKENS
    cfg = ExperimentConfig.model_construct(
        architecture="level1",
        domain="logic_puzzles",
        difficulty="hard",
        num_runs=1,
        max_iterations=10,
        max_critic_iterations=3,
        num_branches=3,
        recursion_limit=100,
        temperature=0.7,
        thinking_token_budget=budget,
        max_tokens=(budget + 2000) if budget is not None else UNLIMITED_MAX_TOKENS,
        request_timeout=1800.0,
    )

    arch = Level1Baseline()
    graph = arch.build_graph(domain, cfg)
    metrics_cb = MetricsCallback()

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
        "max_iterations": 10,
        "metadata": {},
    }

    start = time.time()
    try:
        final_state = graph.invoke(
            initial_state,
            config={"callbacks": [metrics_cb], "recursion_limit": 100},
        )
        elapsed = time.time() - start

        final_answer = final_state.get("final_answer", "")
        if not final_answer:
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break

        evaluation = domain.evaluate(task, final_answer)
        usage = metrics_cb.get_usage()

        return {
            "puzzle_id": puzzle.puzzle_id,
            "budget": budget,
            "success": evaluation.success,
            "score": evaluation.score,
            "total_tokens": usage.total_tokens,
            "elapsed_s": round(elapsed, 1),
            "any_truncated": "length" in usage.per_call_finish_reasons,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "puzzle_id": puzzle.puzzle_id,
            "budget": budget,
            "success": False,
            "score": 0.0,
            "total_tokens": 0,
            "elapsed_s": round(elapsed, 1),
            "any_truncated": False,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for grade in CANDIDATE_GRADES:
        for idx in range(PUZZLES_PER_GRADE):
            try:
                puzzle = _get_puzzle(grade, idx)
            except ValueError as e:
                print(f"SKIP {grade}[{idx}]: {e}")
                continue

            for budget in BUDGETS:
                label = "unlimited" if budget is None else f"budget={budget}"
                print(f"{puzzle.puzzle_id}  {label} ... ", end="", flush=True)
                row = _run_one(puzzle, budget)
                rows.append(row)
                with open(OUTPUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                status = "OK" if row["success"] else f"score={row['score']:.2f}"
                if row["error"]:
                    status = f"ERROR: {row['error'][:60]}"
                trunc = " [TRUNCATED]" if row["any_truncated"] else ""
                print(f"{status}  tokens={row['total_tokens']}  {row['elapsed_s']}s{trunc}")

    from collections import defaultdict
    grade_stats: dict[str, dict] = defaultdict(lambda: {"unlimited": [], "budget": []})
    for r in rows:
        grade = re.search(r"level\d+", r["puzzle_id"]).group()
        key = "unlimited" if r["budget"] is None else "budget"
        grade_stats[grade][key].append(r["score"])

    print("\ngrade  unlimited  1500  delta")
    for grade in CANDIDATE_GRADES:
        s = grade_stats[grade]
        u = sum(s["unlimited"]) / len(s["unlimited"]) if s["unlimited"] else float("nan")
        b = sum(s["budget"]) / len(s["budget"]) if s["budget"] else float("nan")
        print(f"  {grade}  {u:.2f}  {b:.2f}  {u - b:.2f}")


if __name__ == "__main__":
    main()
