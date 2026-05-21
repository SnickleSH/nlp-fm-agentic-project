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
    metadata: dict
