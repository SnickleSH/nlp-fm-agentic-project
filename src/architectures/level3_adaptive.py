from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from src.architectures.base import BaseArchitecture
from src.config import ExperimentConfig
from src.domains.base import BaseDomain


class Level3Adaptive(BaseArchitecture):
    """ToT + episodic memory. Planner → Critic → Executor loop.

    Graph built in K2; this stub registers the architecture key.
    """

    def build_graph(self, domain: BaseDomain, config: ExperimentConfig) -> CompiledStateGraph:
        raise NotImplementedError("Level3 graph not yet implemented (K2 pending)")
