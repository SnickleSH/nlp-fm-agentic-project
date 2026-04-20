"""Run all experiments defined in the config file."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_experiment_configs
from src.runner import run_single, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments from config")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks per condition")
    parser.add_argument("--output", default="results/results.jsonl")
    args = parser.parse_args()

    configs = load_experiment_configs(args.config)
    print(f"Loaded {len(configs)} experiment conditions from {args.config}")

    for i, config in enumerate(configs, 1):
        label = f"{config.architecture}/{config.domain}/{config.difficulty}"
        total_runs = args.num_tasks * config.num_runs
        print(f"\n{'='*60}")
        print(f"Condition {i}/{len(configs)}: {label} ({total_runs} runs)")
        print(f"{'='*60}")

        for task_id in range(args.num_tasks):
            for run_id in range(config.num_runs):
                print(f"  task={task_id} run={run_id} ... ", end="", flush=True)
                result = run_single(config, task_id, run_id)
                save_result(result, args.output)
                status = "OK" if result.success else "FAIL"
                if result.error:
                    status = f"ERROR: {result.error[:50]}"
                print(f"{status} (score={result.score:.2f}, {result.runtime_seconds:.1f}s)")

    print(f"\nAll experiments complete. Results: {args.output}")


if __name__ == "__main__":
    main()
