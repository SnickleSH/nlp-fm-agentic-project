"""Run a single experiment condition from the command line."""
from __future__ import annotations

import argparse
import sys

from src.architectures.memory import RecentSuccessMemory
from src.config import ExperimentConfig
from src.runner import load_completed_keys, run_single, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment condition")
    parser.add_argument("--architecture", required=True, help="Architecture name (e.g. level1, level2a)")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. gridworld)")
    parser.add_argument("--difficulty", required=True, choices=["easy", "medium", "hard", "extra_hard"])
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs per task")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks to generate")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking-token-budget", type=int, default=None,
                        help="Per-call thinking token budget (triggers reasoning_effort=medium)")
    parser.add_argument("--output", default="results/results.jsonl")
    args = parser.parse_args()

    config = ExperimentConfig(
        architecture=args.architecture,
        domain=args.domain,
        difficulty=args.difficulty,
        num_runs=args.num_runs,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        thinking_token_budget=args.thinking_token_budget,
    )

    total = args.num_tasks * args.num_runs
    done_count = 0

    budget_str = f", budget={config.thinking_token_budget}" if config.thinking_token_budget else ""
    print(f"Running: {config.architecture} / {config.domain} / {config.difficulty}{budget_str}")
    print(f"Tasks: {args.num_tasks}, Runs per task: {args.num_runs}, Total: {total}")
    print("-" * 60)

    completed_keys = load_completed_keys(args.output)
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} run(s) already in {args.output}, will skip.")

    # Per-condition episodic memory for L3 (S2 decision: reset between (domain, difficulty)).
    memory = RecentSuccessMemory() if config.architecture == "level3" else None

    for task_id in range(args.num_tasks):
        for run_id in range(args.num_runs):
            done_count += 1
            key = (
                config.architecture, config.domain, config.difficulty,
                task_id, run_id, config.thinking_token_budget,
                config.max_critic_iterations,
            )
            if key in completed_keys:
                print(f"[{done_count}/{total}] task={task_id} run={run_id} ... SKIP (already logged)")
                continue
            print(f"[{done_count}/{total}] task={task_id} run={run_id} ... ", end="", flush=True)
            result = run_single(config, task_id, run_id, memory=memory)
            save_result(result, args.output)
            completed_keys.add(key)
            status = "OK" if result.success else "FAIL"
            if result.error:
                status = f"ERROR: {result.error[:60]}"
            print(f"{status} (score={result.score:.2f}, {result.runtime_seconds:.1f}s, "
                  f"{result.token_usage.total_tokens} tokens)")

    print("-" * 60)
    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
