"""Ad-hoc tests for the episodic memory backend.

Matches the style of scripts/test_logic_pipeline.py — no pytest dependency.
Run with: poetry run python scripts/test_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures.memory import NoOpMemory, RecentSuccessMemory


def _ep(domain: str, difficulty: str, sig: str, outcome: str = "ACCEPT") -> dict:
    return {
        "domain": domain,
        "difficulty": difficulty,
        "puzzle_signature": sig,
        "strategy_summary": f"strategy for {sig}",
        "outcome": outcome,
        "score": 1.0 if outcome == "ACCEPT" else 0.0,
    }


def test_roundtrip_returns_most_recent() -> None:
    """write+retrieve roundtrip: 3 same-key episodes, k=2 returns the 2 most recent."""
    mem = RecentSuccessMemory()
    mem.write(_ep("logic", "easy", "p0"))
    mem.write(_ep("logic", "easy", "p1"))
    mem.write(_ep("logic", "easy", "p2"))

    hits = mem.retrieve({"domain": "logic", "difficulty": "easy"}, k=2)
    sigs = [h["puzzle_signature"] for h in hits]
    assert sigs == ["p2", "p1"], f"expected most-recent ordering, got {sigs}"


def test_accept_takes_priority_over_recency() -> None:
    """2 REJECTs then 1 ACCEPT: retrieve k=2 puts ACCEPT first."""
    mem = RecentSuccessMemory()
    mem.write(_ep("logic", "hard", "p0", outcome="REJECT"))
    mem.write(_ep("logic", "hard", "p1", outcome="REJECT"))
    mem.write(_ep("logic", "hard", "p2", outcome="ACCEPT"))

    hits = mem.retrieve({"domain": "logic", "difficulty": "hard"}, k=2)
    assert hits[0]["outcome"] == "ACCEPT", f"ACCEPT should be first, got {hits[0]}"
    # Second slot should be the most recent REJECT (p1).
    assert hits[1]["puzzle_signature"] == "p1", f"expected p1 as runner-up, got {hits[1]}"


def test_domain_difficulty_filter() -> None:
    """Episodes from other (domain, difficulty) keys must not leak in."""
    mem = RecentSuccessMemory()
    mem.write(_ep("logic", "easy", "p0"))
    mem.write(_ep("gridworld", "easy", "g0"))
    mem.write(_ep("logic", "hard", "h0"))
    mem.write(_ep("logic", "easy", "p1"))

    hits = mem.retrieve({"domain": "logic", "difficulty": "easy"}, k=10)
    sigs = sorted(h["puzzle_signature"] for h in hits)
    assert sigs == ["p0", "p1"], f"filter leak: got {sigs}"


def test_empty_retrieve() -> None:
    """Empty bank returns []."""
    mem = RecentSuccessMemory()
    assert mem.retrieve({"domain": "logic", "difficulty": "easy"}, k=3) == []


def test_noop_memory() -> None:
    """NoOpMemory contract still works."""
    mem = NoOpMemory()
    mem.write(_ep("logic", "easy", "p0"))  # must not raise
    assert mem.retrieve({"domain": "logic", "difficulty": "easy"}, k=5) == []


def test_empty_bank_not_replaced_by_noop_in_build_graph() -> None:
    """Regression: build_graph must not swap an empty RecentSuccessMemory for a NoOpMemory.

    The bug was `self.memory = memory or NoOpMemory()` — because an empty
    RecentSuccessMemory has __len__==0, it's falsy, so `or` would swap it for
    a temporary NoOp at the start of each run and episodes would never accumulate.
    """
    from src.architectures import get_architecture
    from src.config import ExperimentConfig
    from src.domains import get_domain

    mem = RecentSuccessMemory()
    assert bool(mem) is False, "empty RecentSuccessMemory must be falsy for this test to be meaningful"

    arch = get_architecture("level3")
    domain = get_domain("logic_puzzles")
    cfg = ExperimentConfig(architecture="level3", domain="logic_puzzles", difficulty="easy", num_runs=1)
    arch.build_graph(domain, cfg, memory=mem)

    assert arch.memory is mem, "build_graph swapped the empty bank for NoOpMemory"


def main() -> int:
    tests = [
        test_roundtrip_returns_most_recent,
        test_accept_takes_priority_over_recency,
        test_domain_difficulty_filter,
        test_empty_retrieve,
        test_noop_memory,
        test_empty_bank_not_replaced_by_noop_in_build_graph,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
