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
from src.domains.base import BaseDomain, Task
from src.llm import create_llm
from src.state import AgentState

CRITIC_SYSTEM_PROMPT = (
    "You are a meticulous critic. Your job is to rigorously verify a proposed "
    "solution against the task's rules. You do not solve the task yourself — you "
    "judge correctness and give actionable feedback. Be strict: only accept a "
    "solution that fully satisfies every rule."
)


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse a critic response into (verdict, feedback).

    verdict is 'ACCEPT' or 'REJECT' (defaults to REJECT when ambiguous, so the
    loop errs toward another solver pass rather than accepting a bad answer).
    """
    upper = text.upper()
    verdict = "REJECT"
    if "VERDICT:" in upper:
        after = upper.split("VERDICT:", 1)[1].lstrip()
        verdict = "ACCEPT" if after.startswith("ACCEPT") else "REJECT"
    elif "ACCEPT" in upper and "REJECT" not in upper:
        verdict = "ACCEPT"
    return verdict, text.strip()


class Level2BSolverCritic(BaseArchitecture):
    """Cyclic solver <-> critic graph.

    Non-interactive: START -> solver -> critic -> {solver | END}
    Interactive:     START -> solver [-> tools -> solver]* -> critic -> {solver | END}

    The solver proposes a solution; the critic reviews it against the task rules
    and either accepts (END) or rejects with feedback (back to solver). The cycle
    is bounded by config.max_critic_iterations.
    """

    def build_graph(self, domain: BaseDomain, config: ExperimentConfig) -> CompiledStateGraph:
        self.domain = domain
        self.config = config
        self.llm = create_llm(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            enable_thinking=config.enable_thinking,
        )
        self.tools = domain.get_tools()

        graph = StateGraph(AgentState)
        graph.add_node("critic", self._critic_node)

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            graph.add_node("solver", self._solver_agent_node)
            graph.add_node("tools", ToolNode(self.tools))

            graph.add_edge(START, "solver")
            graph.add_conditional_edges("solver", self._solver_should_act, {
                "tools": "tools",
                "critic": "critic",
            })
            graph.add_edge("tools", "solver")
        else:
            graph.add_node("solver", self._solver_simple_node)
            graph.add_edge(START, "solver")
            graph.add_edge("solver", "critic")

        graph.add_conditional_edges("critic", self._route_after_critic, {
            "solver": "solver",
            "end": END,
        })

        return graph.compile()

    # --- Solver (non-interactive) ---------------------------------------

    def _solver_simple_node(self, state: AgentState) -> dict:
        solver_instruction = (
            "You are a solver. Produce the best possible solution to the task. "
            "Return only the final answer in the required format — do not restate "
            "the task or explain your reasoning in the answer."
        )
        system_text = self._original_system(state)
        combined_system = "\n\n".join(p for p in [system_text, solver_instruction] if p)

        human_parts = [self._original_task(state)]
        feedback = state.get("critic_feedback", "")
        if feedback:
            human_parts.append(
                "A reviewer REJECTED your previous attempt with this feedback. "
                "Address every point and produce a corrected solution:\n" + feedback
            )

        messages = [
            SystemMessage(content=combined_system),
            HumanMessage(content="\n\n".join(human_parts)),
        ]
        response = self.llm.invoke(messages)
        return {
            "messages": [AIMessage(content=f"[SOLUTION]\n{response.content}")],
            "final_answer": response.content,
        }

    # --- Solver (interactive, tool-calling loop) ------------------------

    def _solver_agent_node(self, state: AgentState) -> dict:
        response = self.llm_with_tools.invoke(state["messages"])
        update: dict = {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }
        if response.content:
            update["final_answer"] = response.content
        return update

    def _solver_should_act(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        # Bound the inner tool loop; hand off to the critic when exhausted.
        if state.get("iteration", 0) >= state.get("max_iterations", 10) * 3:
            return "critic"
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "critic"

    # --- Critic ---------------------------------------------------------

    def _critic_node(self, state: AgentState) -> dict:
        task = Task.model_validate(state["task"])
        candidate = state.get("final_answer", "")
        critic_prompt = self.domain.format_critic_prompt(task, candidate)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=critic_prompt),
        ]
        response = self.llm.invoke(messages)
        verdict, feedback = _parse_verdict(response.content)

        return {
            "critic_verdict": verdict,
            "critic_feedback": feedback,
            "critic_iterations": state.get("critic_iterations", 0) + 1,
            "iteration": 0,
            "messages": [HumanMessage(content=f"[CRITIC VERDICT: {verdict}]\n{feedback}")],
        }

    def _route_after_critic(self, state: AgentState) -> str:
        if state.get("critic_verdict") == "ACCEPT":
            return "end"
        if state.get("critic_iterations", 0) >= self.config.max_critic_iterations:
            return "end"
        return "solver"

    # --- Helpers --------------------------------------------------------

    def _original_system(self, state: AgentState) -> str:
        parts = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, SystemMessage) and msg.content
        ]
        return "\n\n".join(parts)

    def _original_task(self, state: AgentState) -> str:
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                return msg.content
        return str(state.get("task", {}))
