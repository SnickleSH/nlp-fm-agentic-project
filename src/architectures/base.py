from __future__ import annotations

from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph

from src.config import ExperimentConfig
from src.domains.base import BaseDomain


class BaseArchitecture(ABC):
    @abstractmethod
    def build_graph(self, domain: BaseDomain, config: ExperimentConfig) -> CompiledStateGraph:
        """Return a compiled LangGraph ready to invoke."""
        ...
