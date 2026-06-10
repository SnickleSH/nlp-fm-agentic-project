from __future__ import annotations

from typing import Protocol


class EpisodicMemory(Protocol):
    """Interface for an episodic memory bank used by L3.

    The bank is injected via build_graph(memory=...) and outlives a single invocation.
    During measured runs, the bank is read-only (writes happen only in a pre-pass).
    """

    def retrieve(self, task: dict, k: int) -> list[dict]:
        """Return top-k similar episodes given a task dict.

        Episodes have shape: {domain, difficulty, puzzle_signature, strategy_summary, outcome, score}.
        """
        ...

    def write(self, episode: dict) -> None:
        """Persist an episode to the bank (used only in pre-pass, not during measured runs)."""
        ...


class NoOpMemory:
    """Stub memory: returns empty hits, ignores writes. Used until B3 lands."""

    def retrieve(self, task: dict, k: int) -> list[dict]:
        return []

    def write(self, episode: dict) -> None:
        pass
