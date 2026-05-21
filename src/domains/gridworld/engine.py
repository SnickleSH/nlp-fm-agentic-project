from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GridWorld:
    width: int
    height: int
    agent_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    walls: set[tuple[int, int]] = field(default_factory=set)
    fog_of_war: bool = False
    view_radius: int | None = None
    steps_taken: int = 0
    max_steps: int = 50
    done: bool = False

    def get_observation(self) -> str:
        lines: list[str] = []
        lines.append(f"Grid: {self.width}x{self.height}")
        lines.append(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        lines.append(f"Goal: ({self.goal_pos[0]}, {self.goal_pos[1]})")
        lines.append(f"Steps taken: {self.steps_taken}/{self.max_steps}")
        lines.append("")
        lines.append(self._render_grid())
        return "\n".join(lines)

    def _render_grid(self) -> str:
        rows: list[str] = []
        for y in range(self.height):
            row: list[str] = []
            for x in range(self.width):
                if self.fog_of_war and self.view_radius is not None:
                    dx = abs(x - self.agent_pos[0])
                    dy = abs(y - self.agent_pos[1])
                    if max(dx, dy) > self.view_radius:
                        row.append("?")
                        continue
                if (x, y) == self.agent_pos:
                    row.append("A")
                elif (x, y) == self.goal_pos:
                    row.append("G")
                elif (x, y) in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def step(self, action: str) -> tuple[str, bool, dict]:
        if self.done:
            return "Game is already over.", True, {"reason": "already_done"}

        action = action.strip().lower()
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(
            action, (0, 0)
        )
        if dx == 0 and dy == 0:
            return f"Invalid action: {action}. Use up/down/left/right.", False, {}

        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            self.steps_taken += 1
            obs = f"You bumped into the boundary.\n\n{self.get_observation()}"
        elif (nx, ny) in self.walls:
            self.steps_taken += 1
            obs = f"You bumped into a wall.\n\n{self.get_observation()}"
        else:
            self.agent_pos = (nx, ny)
            self.steps_taken += 1
            if self.agent_pos == self.goal_pos:
                self.done = True
                return f"You reached the goal in {self.steps_taken} steps! Congratulations!", True, {"reason": "goal_reached"}
            obs = f"Moved {action}.\n\n{self.get_observation()}"

        if self.steps_taken >= self.max_steps:
            self.done = True
            return obs + "\n\nMax steps reached. Game over.", True, {"reason": "max_steps"}

        return obs, False, {}


def generate_grid(
    difficulty: str, seed: int | None = None
) -> GridWorld:
    rng = random.Random(seed)

    if difficulty == "easy":
        w, h = 4, 4
        fog = False
        view_radius = None
        max_steps = 20
        num_walls = rng.randint(1, 3)
        min_path, max_path = 3, 5
    else:
        w, h = 8, 8
        fog = True
        view_radius = 1
        max_steps = 50
        num_walls = rng.randint(10, 18)
        min_path, max_path = 10, 15

    # Place agent at top-left area, goal at bottom-right area
    agent_pos = (rng.randint(0, w // 4), rng.randint(0, h // 4))
    goal_pos = (rng.randint(3 * w // 4, w - 1), rng.randint(3 * h // 4, h - 1))

    # Generate walls ensuring a path exists
    walls: set[tuple[int, int]] = set()
    all_cells = [
        (x, y) for x in range(w) for y in range(h)
        if (x, y) != agent_pos and (x, y) != goal_pos
    ]
    rng.shuffle(all_cells)

    for cell in all_cells:
        if len(walls) >= num_walls:
            break
        walls.add(cell)
        if not _path_exists(w, h, agent_pos, goal_pos, walls):
            walls.discard(cell)

    # Verify path length is in desired range
    path_len = _shortest_path_length(w, h, agent_pos, goal_pos, walls)
    if path_len is not None and not (min_path <= path_len <= max_path):
        # Regenerate with a different seed if path length is out of range
        return generate_grid(difficulty, seed=(seed or 0) + 1000)

    return GridWorld(
        width=w,
        height=h,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        walls=walls,
        fog_of_war=fog,
        view_radius=view_radius,
        max_steps=max_steps,
    )


def _path_exists(
    w: int, h: int,
    start: tuple[int, int], end: tuple[int, int],
    walls: set[tuple[int, int]],
) -> bool:
    return _shortest_path_length(w, h, start, end, walls) is not None


def _shortest_path_length(
    w: int, h: int,
    start: tuple[int, int], end: tuple[int, int],
    walls: set[tuple[int, int]],
) -> int | None:
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in walls and (nx, ny) not in visited:
                if (nx, ny) == end:
                    return dist + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return None
