from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
try:
    from langgraph.prebuilt import ToolNode
except ImportError:
    from langgraph.prebuilt.tool_node import ToolNode

from src.architectures.base import BaseArchitecture
from src.architectures.memory import EpisodicMemory, NoOpMemory
from src.config import ExperimentConfig
from src.domains.base import BaseDomain, Task
from src.llm import create_llm
from src.state import AgentState

PLANNER_SYSTEM_PROMPT = (
    "You are a strategic planner. Analyze the task and generate multiple "
    "distinct solution approaches. Each approach should be detailed and specific. "
    "Output your plan in a structured format ready for execution."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a meticulous critic. Your job is to evaluate proposed solutions "
    "against the task's rules. You do not solve the task yourself — you judge "
    "correctness and score each solution. Be strict: only high scores for solutions "
    "that fully satisfy every rule."
)

EXECUTOR_SYSTEM_PROMPT = (
    "You are a precise executor. Follow the plan provided and solve the task. "
    "Return only the final answer in the required format. "
    "Do not restate the plan or the task."
)


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse a critic response into (verdict, feedback).

    verdict is 'ACCEPT' or 'REJECT' (defaults to REJECT when ambiguous).
    """
    upper = text.upper()
    verdict = "REJECT"
    if "VERDICT:" in upper:
        after = upper.split("VERDICT:", 1)[1].lstrip()
        verdict = "ACCEPT" if after.startswith("ACCEPT") else "REJECT"
    elif "ACCEPT" in upper and "REJECT" not in upper:
        verdict = "ACCEPT"
    return verdict, text.strip()


def _extract_score(text: str) -> float:
    """Extract a numeric score from critic feedback. Default 0.0 if not found."""
    match = re.search(r"SCORE[:\s]+([0-9.]+)", text.upper())
    if match:
        try:
            score = float(match.group(1))
            return min(1.0, max(0.0, score))
        except ValueError:
            pass
    return 0.0


class Level3Adaptive(BaseArchitecture):
    """Tree-of-Thought with episodic memory.

    Graph: planner (N branches) → critic (scores branches) → executor (commits best)
    → loop or finalize. Reuses domain's existing format_* hooks.
    """

    def build_graph(
        self, domain: BaseDomain, config: ExperimentConfig, memory: EpisodicMemory | None = None
    ) -> CompiledStateGraph:
        self.domain = domain
        self.config = config
        # Explicit None check: an empty RecentSuccessMemory has len()==0 and is
        # falsy under `or`, which would silently swap in a NoOpMemory at the start
        # of every run and never accumulate episodes.
        self.memory = memory if memory is not None else NoOpMemory()
        self.llm = create_llm(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            thinking_token_budget=config.thinking_token_budget,
            request_timeout=config.request_timeout,
        )
        self.tools = domain.get_tools()

        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("critic", self._critic_node)

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            graph.add_node("executor", self._executor_agent_node)
            graph.add_node("tools", ToolNode(self.tools))
            graph.add_edge(START, "planner")
            graph.add_edge("planner", "critic")
            graph.add_edge("critic", "executor")
            graph.add_conditional_edges("executor", self._should_use_tool, {
                "tools": "tools",
                "route": "route",
            })
            graph.add_edge("tools", "executor")
        else:
            graph.add_node("executor", self._executor_simple_node)
            graph.add_edge(START, "planner")
            graph.add_edge("planner", "critic")
            graph.add_edge("critic", "executor")

        graph.add_node("route", self._route_node)
        graph.add_node("finalize", self._finalize_node)

        if self.tools:
            graph.add_conditional_edges("route", self._route_after_executor, {
                "loop": "planner",
                "finalize": "finalize",
            })
        else:
            graph.add_edge("executor", "route")
            graph.add_conditional_edges("route", self._route_after_executor, {
                "loop": "planner",
                "finalize": "finalize",
            })

        graph.add_edge("finalize", END)

        return graph.compile()

    def _planner_node(self, state: AgentState) -> dict:
        task = Task.model_validate(state["task"])

        # The episode shape stores `domain` as the lowercased domain class name; mirror
        # that here so retrieval can filter by the same key without changing the Protocol.
        domain_key = self.domain.__class__.__name__.lower()
        retrieve_task = {**state["task"], "domain": domain_key}
        retrieved = self.memory.retrieve(retrieve_task, k=2)
        retrieved_episodes = retrieved if retrieved else []

        memory_context = ""
        sentinels: list[str] = []
        if retrieved_episodes:
            summary_lines = []
            for idx, ep in enumerate(retrieved_episodes, 1):
                summary = ep.get("strategy_summary", "")
                if not summary:
                    continue
                tag = f"[MEM-{idx}]"
                sentinels.append(tag)
                summary_lines.append(f"- {tag} {summary}")
            if summary_lines:
                memory_context = (
                    "Similar solved problems (learned strategies). "
                    "If you reuse one, keep its [MEM-N] tag in your plan:\n"
                    + "\n".join(summary_lines)
                    + "\n\n"
                )

        system_text = self.domain.format_system_prompt(task)
        task_text = self.domain.format_task_prompt(task)

        combined_system = "\n\n".join(p for p in [system_text, PLANNER_SYSTEM_PROMPT] if p)
        combined_task = memory_context + task_text

        messages = [
            SystemMessage(content=combined_system),
            HumanMessage(content=combined_task),
        ]

        branches = []
        for i in range(self.config.num_branches):
            response = self.llm.invoke(messages)
            branches.append({"content": response.content, "score": 0.0})

        reuse_hits = sum(
            1 for b in branches if any(tag in b["content"] for tag in sentinels)
        )

        metadata = state.get("metadata", {})
        metadata["branch_count"] = self.config.num_branches
        metadata["mem_retrievals"] = len(retrieved_episodes)
        metadata["mem_reuse_hits"] = reuse_hits

        return {
            "branches": branches,
            "retrieved_episodes": retrieved_episodes,
            "metadata": metadata,
            "messages": [AIMessage(content=f"[BRANCHES GENERATED ({len(branches)})]")],
        }

    def _critic_node(self, state: AgentState) -> dict:
        task = Task.model_validate(state["task"])
        branches = state.get("branches", [])

        if not branches:
            return {
                "critic_iterations": state.get("critic_iterations", 0) + 1,
                "selected_branch": "",
                "critic_verdict": "REJECT",
                "messages": [HumanMessage(content="[NO BRANCHES TO CRITIQUE]")],
            }

        candidates_text = "\n\n".join(
            f"Candidate {i + 1}:\n{b['content']}" for i, b in enumerate(branches)
        )

        critic_prompt = (
            f"{self.domain.format_critic_prompt(task, candidates_text)}\n\n"
            "For EACH candidate, provide a SCORE (0.0-1.0) and brief justification. "
            "Use format:\nCandidate N: SCORE: X.X\nJustification: ..."
        )

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=critic_prompt),
        ]

        response = self.llm.invoke(messages)
        feedback = response.content

        for i, branch in enumerate(branches):
            score_pattern = f"Candidate {i + 1}[^0-9]*([0-9.]+)"
            match = re.search(score_pattern, feedback, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    branch["score"] = min(1.0, max(0.0, score))
                except ValueError:
                    branch["score"] = 0.0
            else:
                branch["score"] = 0.0

        best_idx = max(range(len(branches)), key=lambda i: branches[i]["score"])
        selected = branches[best_idx]["content"]

        best_score = branches[best_idx]["score"]
        verdict = "ACCEPT" if best_score >= 0.7 else "REJECT"

        return {
            "branches": branches,
            "selected_branch": selected,
            "critic_iterations": state.get("critic_iterations", 0) + 1,
            "critic_verdict": verdict,
            "messages": [HumanMessage(content=f"[CRITIC: {verdict} (best score: {best_score:.1f})]")],
        }

    def _executor_agent_node(self, state: AgentState) -> dict:
        selected = state.get("selected_branch", "")
        messages = state.get("messages", [])

        prompt = (
            f"Here is the selected plan:\n{selected}\n\n"
            "Now execute this plan step by step using the available tools."
        )
        executor_msg = HumanMessage(content=prompt)

        response = self.llm_with_tools.invoke(messages + [executor_msg])
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

    def _executor_simple_node(self, state: AgentState) -> dict:
        task = Task.model_validate(state["task"])
        selected = state.get("selected_branch", "")
        system_text = self.domain.format_system_prompt(task)

        combined_system = "\n\n".join(
            p for p in [system_text, EXECUTOR_SYSTEM_PROMPT, f"Selected plan:\n{selected}"] if p
        )

        task_text = self.domain.format_task_prompt(task)

        messages = [
            SystemMessage(content=combined_system),
            HumanMessage(content=task_text),
        ]

        response = self.llm.invoke(messages)
        return {"messages": [response], "final_answer": response.content}

    def _should_use_tool(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if state.get("iteration", 0) >= state.get("max_iterations", 10) * 3:
            return "route"
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "route"

    def _route_node(self, state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            return {"final_answer": last_msg.content}
        return {}

    def _route_after_executor(self, state: AgentState) -> str:
        verdict = state.get("critic_verdict", "REJECT")
        iterations = state.get("critic_iterations", 0)
        max_iterations = self.config.max_critic_iterations

        if verdict == "ACCEPT" or iterations >= max_iterations:
            return "finalize"
        return "loop"

    def _finalize_node(self, state: AgentState) -> dict:
        task = Task.model_validate(state["task"])
        final_answer = state.get("final_answer", "")
        selected_branch = state.get("selected_branch", "")
        critic_verdict = state.get("critic_verdict", "REJECT")

        puzzle_sig = f"{task.difficulty}_{task.task_id}"
        if "puzzle_id" in task.metadata:
            puzzle_sig = task.metadata["puzzle_id"]

        episode = {
            "domain": self.domain.__class__.__name__.lower(),
            "difficulty": task.difficulty,
            "puzzle_signature": puzzle_sig,
            "strategy_summary": selected_branch[:200],
            "outcome": critic_verdict,
            "score": 1.0 if critic_verdict == "ACCEPT" else 0.0,
        }

        self.memory.write(episode)

        # Carry counters forward from the planner — _planner_node already set them based
        # on the most recent ToT pass; don't zero them here.
        metadata = state.get("metadata", {})
        metadata.setdefault("branch_count", self.config.num_branches)
        metadata.setdefault("mem_retrievals", len(state.get("retrieved_episodes", [])))
        metadata.setdefault("mem_reuse_hits", 0)

        return {"metadata": metadata, "messages": [AIMessage(content="[FINALIZED]")]}
