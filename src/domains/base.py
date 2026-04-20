from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Task(BaseModel):
    task_id: int
    description: str
    rules: list[str]
    ground_truth: Any
    difficulty: str
    metadata: dict = {}


class EvaluationResult(BaseModel):
    success: bool
    score: float
    details: dict = {}


class BaseDomain(ABC):
    @abstractmethod
    def generate_task(self, difficulty: str, task_id: int) -> Task: ...

    @abstractmethod
    def format_system_prompt(self, task: Task) -> str: ...

    @abstractmethod
    def format_task_prompt(self, task: Task) -> str: ...

    @abstractmethod
    def evaluate(self, task: Task, answer: str) -> EvaluationResult: ...

    def get_tools(self) -> list:
        """Return LangChain tools for interactive domains. Default: none."""
        return []
