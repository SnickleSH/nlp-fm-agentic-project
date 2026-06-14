"""Run all experiments defined in the config file."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.architectures.memory import RecentSuccessMemory
from src.config import load_experiment_configs
from src.runner import load_completed_keys, run_single, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments from config")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks per condition")
    parser.add_argument("--output", default="results/results.jsonl")
    parser.add_argument("--domain", default=None, help="Only run configs for this domain (e.g. logic_puzzles)")
    parser.add_argument("--architecture", default=None, help="Only run configs for this architecture (e.g. level1)")
    args = parser.parse_args()

    configs = load_experiment_configs(args.config)
    if args.domain:
        configs = [c for c in configs if c.domain == args.domain]
    if args.architecture:
        configs = [c for c in configs if c.architecture == args.architecture]
    print(f"Loaded {len(configs)} experiment conditions from {args.config}")

    completed = load_completed_keys(args.output)
    if completed:
        print(f"Resuming: {len(completed)} run(s) already in {args.output}, will skip.")

    for i, config in enumerate(configs, 1):
        label = f"{config.architecture}/{config.domain}/{config.difficulty}"
        total_runs = args.num_tasks * config.num_runs
        print(f"\n{'='*60}")
        print(f"Condition {i}/{len(configs)}: {label} ({total_runs} runs)")
        print(f"{'='*60}")

        # Per (domain, difficulty) episodic memory bank for L3 rows (S2 decision).
        # Bank lives for the duration of this condition and is dropped between rows.
        # Resume caveat: a partial L3 row restarted from disk gets a fresh empty bank
        # — the resumed runs are NOT equivalent to the originals. Delete the partial
        # rows from the JSONL before resuming if strict comparability matters.
        memory = RecentSuccessMemory() if config.architecture == "level3" else None

        for task_id in range(args.num_tasks):
            for run_id in range(config.num_runs):
                key = (
                    config.architecture, config.domain, config.difficulty,
                    task_id, run_id, config.thinking_token_budget,
                    config.max_critic_iterations,
                )
                if key in completed:
                    print(f"  task={task_id} run={run_id} ... SKIP (already logged)")
                    continue
                print(f"  task={task_id} run={run_id} ... ", end="", flush=True)
                result = run_single(config, task_id, run_id, memory=memory)
                save_result(result, args.output)
                completed.add(key)
                status = "OK" if result.success else "FAIL"
                if result.error:
                    status = f"ERROR: {result.error[:50]}"
                print(f"{status} (score={result.score:.2f}, {result.runtime_seconds:.1f}s)")

    print(f"\nAll experiments complete. Results: {args.output}")


if __name__ == "__main__":
    main()
