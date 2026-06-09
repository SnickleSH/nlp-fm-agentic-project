from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    task: dict
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str
    iteration: int
    max_iterations: int
    # Solver + Critic (Level 2B) fields. Default via .get() in nodes, so they
    # need not be present in the initial state.
    critic_iterations: int
    critic_verdict: str
    critic_feedback: str   # latest feedback only (for backward compat)
    critique_history: list[str]  # accumulated per-revision feedbacks shown to solver
    metadata: dict
    # Level 3 (ToT + episodic memory) transient fields.
    # All read via .get() defaults so L1/L2A/L2B states are unaffected.
    branches: list[dict]           # each {"content": str, "score": float}
    selected_branch: str | None    # committed branch after critic pass
    retrieved_episodes: list[dict] # episodes from EpisodicMemory.retrieve()
