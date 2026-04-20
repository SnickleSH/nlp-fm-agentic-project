from __future__ import annotations

import json
import re

from src.domains.base import BaseDomain, EvaluationResult, Task
from src.domains.gridworld.engine import GridWorld, generate_grid
from src.domains.gridworld.tools import create_gridworld_tools


class GridworldDomain(BaseDomain):
    def __init__(self) -> None:
        self._current_grid: GridWorld | None = None

    def generate_task(self, difficulty: str, task_id: int) -> Task:
        grid = generate_grid(difficulty, seed=task_id)
        self._current_grid = grid

        description = (
            f"Navigate a {grid.width}x{grid.height} grid from "
            f"({grid.agent_pos[0]}, {grid.agent_pos[1]}) to the goal at "
            f"({grid.goal_pos[0]}, {grid.goal_pos[1]})."
        )
        if grid.fog_of_war:
            description += f" Fog of war is enabled (view radius: {grid.view_radius})."

        rules = [
            "Use move_up, move_down, move_left, move_right to navigate.",
            "Walls (#) block movement. Boundaries also block movement.",
            "Reach the goal (G) to succeed.",
            f"You have a maximum of {grid.max_steps} steps.",
        ]
        if grid.fog_of_war:
            rules.append("Fog of war is active — you can only see nearby cells.")

        return Task(
            task_id=task_id,
            description=description,
            rules=rules,
            ground_truth={"goal_pos": list(grid.goal_pos)},
            difficulty=difficulty,
            metadata={
                "width": grid.width,
                "height": grid.height,
                "agent_start": list(grid.agent_pos),
                "goal_pos": list(grid.goal_pos),
                "num_walls": len(grid.walls),
                "fog_of_war": grid.fog_of_war,
            },
        )

    def format_system_prompt(self, task: Task) -> str:
        return (
            "You are a navigation agent in a 2D grid world. "
            "Your goal is to reach the target position by calling movement tools. "
            "Analyze the grid observation carefully before each move. "
            "Avoid walls (#) and grid boundaries. "
            "Plan an efficient path and execute it step by step."
        )

    def format_task_prompt(self, task: Task) -> str:
        if self._current_grid is None:
            raise RuntimeError("No grid generated — call generate_task first.")
        obs = self._current_grid.get_observation()
        rules_text = "\n".join(f"- {r}" for r in task.rules)
        return (
            f"{task.description}\n\n"
            f"Rules:\n{rules_text}\n\n"
            f"Current observation:\n{obs}\n\n"
            "Navigate to the goal using the movement tools. "
            "After reaching the goal, respond with DONE."
        )

    def evaluate(self, task: Task, answer: str) -> EvaluationResult:
        if self._current_grid is None:
            return EvaluationResult(success=False, score=0.0, details={"error": "no grid"})

        grid = self._current_grid
        reached_goal = grid.agent_pos == grid.goal_pos

        if reached_goal:
            score = 1.0
        else:
            # Partial credit based on distance reduction
            from src.domains.gridworld.engine import _shortest_path_length

            initial_dist = _shortest_path_length(
                grid.width, grid.height,
                tuple(task.metadata["agent_start"]),
                grid.goal_pos,
                grid.walls,
            )
            remaining_dist = _shortest_path_length(
                grid.width, grid.height,
                grid.agent_pos,
                grid.goal_pos,
                grid.walls,
            )
            if initial_dist and remaining_dist is not None:
                progress = (initial_dist - remaining_dist) / initial_dist
                score = max(0.0, progress * 0.5)  # Max 0.5 for partial progress
            else:
                score = 0.0

        return EvaluationResult(
            success=reached_goal,
            score=score,
            details={
                "reached_goal": reached_goal,
                "steps_taken": grid.steps_taken,
                "final_position": list(grid.agent_pos),
                "goal_position": list(grid.goal_pos),
            },
        )

    def get_tools(self) -> list:
        if self._current_grid is None:
            raise RuntimeError("No grid generated — call generate_task first.")
        return create_gridworld_tools(self._current_grid)
