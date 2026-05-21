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
    critic_feedback: str
    metadata: dict
