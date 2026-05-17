from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.architectures.base import BaseArchitecture
from src.config import ExperimentConfig
from src.domains.base import BaseDomain
from src.llm import create_llm
from src.state import AgentState


class Level2APlannerExecutor(BaseArchitecture):
    def build_graph(self, domain: BaseDomain, config: ExperimentConfig) -> CompiledStateGraph:
        self.domain = domain
        self.config = config
        self.llm = create_llm(temperature=config.temperature)
        self.tools = domain.get_tools()

        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)

        if self.tools:
            # Interactive domain: executor with tool-calling loop
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            graph.add_node("executor", self._executor_agent_node)
            graph.add_node("tools", ToolNode(self.tools))

            graph.add_edge(START, "planner")
            graph.add_edge("planner", "executor")
            graph.add_conditional_edges("executor", self._should_use_tool, {
                "tools": "tools",
                "end": END,
            })
            graph.add_edge("tools", "executor")
        else:
            # Non-interactive domain: single executor call
            graph.add_node("executor", self._executor_simple_node)

            graph.add_edge(START, "planner")
            graph.add_edge("planner", "executor")
            graph.add_edge("executor", END)

        return graph.compile()

    def _planner_node(self, state: AgentState) -> dict:
        planning_prompt = (
            "You are a strategic planner. Analyze the task and create a detailed "
            "step-by-step plan. Be specific about each action. Output ONLY the plan, "
            "numbered step by step."
        )
        messages = [
            SystemMessage(content=planning_prompt),
            HumanMessage(content=self._extract_task_description(state)),
        ]
        response = self.llm.invoke(messages)
        plan = response.content

        # Store plan in metadata and add to message history for executor
        return {
            "messages": [
                AIMessage(content=f"[PLAN]\n{plan}"),
                HumanMessage(content="Now execute this plan step by step using the available tools."),
            ],
            "metadata": {**state.get("metadata", {}), "plan": plan},
        }

    def _executor_agent_node(self, state: AgentState) -> dict:
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

    def _executor_simple_node(self, state: AgentState) -> dict:
        execution_prompt = (
            "You are a precise executor. Follow the plan provided and solve the task. "
            "Provide your final answer clearly."
        )
        # Rebuild messages: system + original task + plan + execution instruction
        messages = [SystemMessage(content=execution_prompt)] + list(state["messages"])
        response = self.llm.invoke(messages)
        return {"messages": [response], "final_answer": response.content}

    def _should_use_tool(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if state.get("iteration", 0) >= state.get("max_iterations", 10) * 3:
            return "end"
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    def _extract_task_description(self, state: AgentState) -> str:
        # Extract the human message with the task description
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                return msg.content
        return str(state.get("task", {}))
