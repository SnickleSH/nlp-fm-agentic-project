"""Analyze experiment results and generate summary plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Flatten token_usage
    if "token_usage" in df.columns:
        usage_df = pd.json_normalize(df["token_usage"])
        usage_df.columns = [f"token_{c}" for c in usage_df.columns]
        df = pd.concat([df.drop(columns=["token_usage"]), usage_df], axis=1)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    grouped = df.groupby(["architecture", "domain", "difficulty"])
    summary = grouped.agg(
        n=("success", "count"),
        success_rate=("success", "mean"),
        avg_score=("score", "mean"),
        std_score=("score", "std"),
        avg_runtime=("runtime_seconds", "mean"),
        avg_tokens=("token_total_tokens", "mean"),
        avg_llm_calls=("num_llm_calls", "mean"),
    ).round(3)

    print(summary.to_string())
    print()


def plot_success_rates(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = df.groupby(["architecture", "difficulty"])["success"].mean().reset_index()
    sns.barplot(data=grouped, x="architecture", y="success", hue="difficulty", ax=ax)
    ax.set_title("Success Rate by Architecture and Difficulty")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=150)
    print(f"Saved: {output_dir / 'success_rates.png'}")


def plot_token_usage(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="architecture", y="token_total_tokens", hue="difficulty", ax=ax)
    ax.set_title("Token Usage by Architecture and Difficulty")
    ax.set_ylabel("Total Tokens")
    fig.tight_layout()
    fig.savefig(output_dir / "token_usage.png", dpi=150)
    print(f"Saved: {output_dir / 'token_usage.png'}")


def plot_runtime(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="architecture", y="runtime_seconds", hue="difficulty", ax=ax)
    ax.set_title("Runtime by Architecture and Difficulty")
    ax.set_ylabel("Runtime (seconds)")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime.png", dpi=150)
    print(f"Saved: {output_dir / 'runtime.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--input", default="results/results.jsonl")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    print(f"Loaded {len(df)} results from {args.input}")

    print_summary(df)
    plot_success_rates(df, output_dir)
    plot_token_usage(df, output_dir)
    plot_runtime(df, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
