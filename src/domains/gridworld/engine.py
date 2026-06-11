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


# Locked level presets. Decoupling grid size, fog, view radius, max_steps, wall
# count, and path-length range means each can be varied independently — the four
# 'difficulty' labels just bundle a specific point in that space. max_steps is
# set as a shrinking multiple of the optimal-path upper bound across the ladder
# (easy 3×, medium 2.5×, hard 2×, extra_hard 1.5×) so the step budget tightens
# at each level; extra_hard further tightens the LLM thinking budget in config.
_LEVEL_PRESETS: dict[str, dict] = {
    "easy": {
        "width": 4, "height": 4,
        "fog": False, "view_radius": None,
        "max_steps": 15,
        "num_walls_range": (1, 3),
        "path_len_range": (3, 5),
    },
    "medium": {
        "width": 6, "height": 6,
        "fog": False, "view_radius": None,
        "max_steps": 20,
        "num_walls_range": (5, 10),
        "path_len_range": (5, 8),
    },
    "hard": {
        "width": 6, "height": 6,
        "fog": True, "view_radius": 1,
        "max_steps": 16,
        "num_walls_range": (5, 10),
        "path_len_range": (5, 8),
    },
    "extra_hard": {
        "width": 6, "height": 6,
        "fog": True, "view_radius": 1,
        "max_steps": 12,
        "num_walls_range": (5, 10),
        "path_len_range": (5, 8),
    },
}


def generate_grid(
    width: int,
    height: int,
    fog: bool,
    view_radius: int | None,
    max_steps: int,
    num_walls_range: tuple[int, int],
    path_len_range: tuple[int, int],
    seed: int | None = None,
) -> GridWorld:
    rng = random.Random(seed)
    num_walls = rng.randint(*num_walls_range)
    min_path, max_path = path_len_range

    agent_pos = (rng.randint(0, width // 4), rng.randint(0, height // 4))
    goal_pos = (rng.randint(3 * width // 4, width - 1), rng.randint(3 * height // 4, height - 1))

    walls: set[tuple[int, int]] = set()
    all_cells = [
        (x, y) for x in range(width) for y in range(height)
        if (x, y) != agent_pos and (x, y) != goal_pos
    ]
    rng.shuffle(all_cells)

    for cell in all_cells:
        if len(walls) >= num_walls:
            break
        walls.add(cell)
        if not _path_exists(width, height, agent_pos, goal_pos, walls):
            walls.discard(cell)

    path_len = _shortest_path_length(width, height, agent_pos, goal_pos, walls)
    if path_len is not None and not (min_path <= path_len <= max_path):
        return generate_grid(
            width, height, fog, view_radius, max_steps,
            num_walls_range, path_len_range,
            seed=(seed or 0) + 1000,
        )

    return GridWorld(
        width=width,
        height=height,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        walls=walls,
        fog_of_war=fog,
        view_radius=view_radius,
        max_steps=max_steps,
    )


def generate_grid_for_level(level: str, seed: int | None = None) -> GridWorld:
    if level not in _LEVEL_PRESETS:
        raise ValueError(f"Unknown level {level!r}. Expected one of {list(_LEVEL_PRESETS)}.")
    return generate_grid(seed=seed, **_LEVEL_PRESETS[level])


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
