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

    def format_critic_prompt(self, task: Task, solution: str) -> str:
        """Prompt for the Level 2B critic to evaluate a proposed solution.

        The critic is an LLM-based self-review; it must NOT receive the ground
        truth. The default checks the solution against the task's stated rules.
        Override to inject domain-specific context (e.g. environment state).
        """
        rules_text = "\n".join(f"- {rule}" for rule in task.rules)
        return (
            f"Task:\n{task.description}\n\n"
            f"Rules and constraints:\n{rules_text}\n\n"
            f"Proposed solution:\n{solution}\n\n"
            "Carefully verify whether the proposed solution fully and correctly "
            "satisfies every rule and constraint. Check for logical errors, "
            "rule violations, and formatting problems.\n\n"
            "Respond in this exact format:\n"
            "VERDICT: ACCEPT  (or)  VERDICT: REJECT\n"
            "If REJECT, then give specific, actionable feedback describing each "
            "problem and how to fix it. If ACCEPT, briefly state why it is correct."
        )
