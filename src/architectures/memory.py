from __future__ import annotations

from typing import Protocol


class EpisodicMemory(Protocol):
    """Interface for an episodic memory bank used by L3.

    The bank is injected via build_graph(memory=...) and outlives a single invocation.
    Per the B3 design, writes happen live during measured runs (see _finalize_node);
    the backend decides what to retain and how to rank retrieval.
    """

    def retrieve(self, task: dict, k: int) -> list[dict]:
        """Return top-k similar episodes given a task dict.

        Episodes have shape: {domain, difficulty, puzzle_signature, strategy_summary, outcome, score}.
        The task dict is expected to carry at least `domain` and `difficulty` keys so the
        backend can filter on the same fields it stores.
        """
        ...

    def write(self, episode: dict) -> None:
        """Append an episode to the bank. Called by L3's _finalize_node after every run."""
        ...


class NoOpMemory:
    """Stub memory: returns empty hits, ignores writes. Default when no bank is injected."""

    def retrieve(self, task: dict, k: int) -> list[dict]:
        return []

    def write(self, episode: dict) -> None:
        pass


class RecentSuccessMemory:
    """In-memory episodic bank: (domain, difficulty) filter, ACCEPT first, then most recent.

    Lifetime is one experiment row (per (domain, difficulty) reset, per the S2 decision).
    Storage is an append-only list — small N (tens of episodes at most per condition),
    so a linear scan on retrieve is fine.
    """

    def __init__(self) -> None:
        self._episodes: list[dict] = []

    def retrieve(self, task: dict, k: int) -> list[dict]:
        domain = task.get("domain")
        difficulty = task.get("difficulty")

        matching = [
            (i, ep)
            for i, ep in enumerate(self._episodes)
            if ep.get("domain") == domain and ep.get("difficulty") == difficulty
        ]
        # ACCEPT episodes first, then by insertion order descending (most recent first).
        matching.sort(key=lambda x: (0 if x[1].get("outcome") == "ACCEPT" else 1, -x[0]))
        return [ep for _, ep in matching[:k]]

    def write(self, episode: dict) -> None:
        self._episodes.append(dict(episode))

    def __len__(self) -> int:
        return len(self._episodes)
