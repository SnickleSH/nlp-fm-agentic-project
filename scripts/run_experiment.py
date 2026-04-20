"""Run a single experiment condition from the command line."""
from __future__ import annotations

import argparse
import sys

from src.config import ExperimentConfig
from src.runner import run_single, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment condition")
    parser.add_argument("--architecture", required=True, help="Architecture name (e.g. level1, level2a)")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. gridworld)")
    parser.add_argument("--difficulty", required=True, choices=["easy", "hard"])
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs per task")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks to generate")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="results/results.jsonl")
    args = parser.parse_args()

    config = ExperimentConfig(
        architecture=args.architecture,
        domain=args.domain,
        difficulty=args.difficulty,
        num_runs=args.num_runs,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    total = args.num_tasks * args.num_runs
    completed = 0

    print(f"Running: {config.architecture} / {config.domain} / {config.difficulty}")
    print(f"Tasks: {args.num_tasks}, Runs per task: {args.num_runs}, Total: {total}")
    print("-" * 60)

    for task_id in range(args.num_tasks):
        for run_id in range(args.num_runs):
            completed += 1
            print(f"[{completed}/{total}] task={task_id} run={run_id} ... ", end="", flush=True)
            result = run_single(config, task_id, run_id)
            save_result(result, args.output)
            status = "OK" if result.success else "FAIL"
            if result.error:
                status = f"ERROR: {result.error[:60]}"
            print(f"{status} (score={result.score:.2f}, {result.runtime_seconds:.1f}s, "
                  f"{result.token_usage.total_tokens} tokens)")

    print("-" * 60)
    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
