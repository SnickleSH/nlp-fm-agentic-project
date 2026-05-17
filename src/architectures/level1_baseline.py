from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
try:
    from langgraph.prebuilt import ToolNode
except ImportError:
    from langgraph.prebuilt.tool_node import ToolNode

from src.architectures.base import BaseArchitecture
from src.config import ExperimentConfig
from src.domains.base import BaseDomain
from src.llm import create_llm
from src.state import AgentState


class Level1Baseline(BaseArchitecture):
    def build_graph(self, domain: BaseDomain, config: ExperimentConfig) -> CompiledStateGraph:
        self.domain = domain
        self.config = config
        self.llm = create_llm(temperature=config.temperature)
        tools = domain.get_tools()

        graph = StateGraph(AgentState)

        if tools:
            # Interactive domain: ReAct agent loop with tools
            self.llm_with_tools = self.llm.bind_tools(tools)
            graph.add_node("agent", self._agent_node)
            graph.add_node("tools", ToolNode(tools))

            graph.add_edge(START, "agent")
            graph.add_conditional_edges("agent", self._should_use_tool, {
                "tools": "tools",
                "end": END,
            })
            graph.add_edge("tools", "agent")
        else:
            # Non-interactive domain: single LLM call
            graph.add_node("solve", self._solve_node)
            graph.add_edge(START, "solve")
            graph.add_edge("solve", END)

        return graph.compile()

    def _agent_node(self, state: AgentState) -> dict:
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

    def _should_use_tool(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if state.get("iteration", 0) >= state.get("max_iterations", 10) * 3:
            return "end"
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    def _solve_node(self, state: AgentState) -> dict:
        response = self.llm.invoke(state["messages"])
        return {"messages": [response], "final_answer": response.content}
