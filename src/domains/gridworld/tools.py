from __future__ import annotations

from langchain_core.tools import tool

from src.domains.gridworld.engine import GridWorld


def create_gridworld_tools(grid: GridWorld) -> list:
    @tool
    def move_up() -> str:
        """Move the agent one step up on the grid."""
        obs, done, info = grid.step("up")
        return obs

    @tool
    def move_down() -> str:
        """Move the agent one step down on the grid."""
        obs, done, info = grid.step("down")
        return obs

    @tool
    def move_left() -> str:
        """Move the agent one step left on the grid."""
        obs, done, info = grid.step("left")
        return obs

    @tool
    def move_right() -> str:
        """Move the agent one step right on the grid."""
        obs, done, info = grid.step("right")
        return obs

    return [move_up, move_down, move_left, move_right]
