"""Export the figures referenced in docs/report_benedek.md.

Generates fig3..fig6 from src.analysis.plots and a custom gridworld-example
panel (fig2) and a 4-panel architecture diagram (fig1). All outputs land in
docs/figures/ as 150 dpi PNGs.

Run from repo root:
    poetry run python scripts/export_report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import add_failure_mode, load_all
from src.analysis import plots as P
from src.domains.gridworld.engine import generate_grid_for_level

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def fig1_architectures() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("L1 — Baseline (ReAct)",
         ["START", "agent", "tools", "END"],
         [("START", "agent", "fwd"), ("agent", "tools", "fwd"),
          ("tools", "agent", "back"), ("agent", "END", "fwd")]),
        ("L2A — Planner + Executor",
         ["START", "planner", "executor", "tools", "END"],
         [("START", "planner", "fwd"), ("planner", "executor", "fwd"),
          ("executor", "tools", "fwd"), ("tools", "executor", "back"),
          ("executor", "END", "fwd")]),
        ("L2B — Solver + Critic (cyclic)",
         ["START", "solver", "tools", "critic", "END"],
         [("START", "solver", "fwd"), ("solver", "tools", "fwd"),
          ("tools", "solver", "back"), ("solver", "critic", "fwd_skip"),
          ("critic", "solver", "back"), ("critic", "END", "fwd")]),
        ("L3 — Adaptive (ToT + Episodic Memory)",
         ["START", "planner\n(N branches\n+mem read)", "critic\n(score)",
          "executor", "tools", "mem\nwrite", "END"],
         [("START", "planner\n(N branches\n+mem read)", "fwd"),
          ("planner\n(N branches\n+mem read)", "critic\n(score)", "fwd"),
          ("critic\n(score)", "executor", "fwd"),
          ("executor", "tools", "fwd"), ("tools", "executor", "back"),
          ("executor", "planner\n(N branches\n+mem read)", "fwd_skip"),
          ("executor", "mem\nwrite", "fwd"), ("mem\nwrite", "END", "fwd")]),
    ]
    tool_like = {"tools", "mem\nwrite"}
    BOX_H = 0.95
    BOX_Y = 3.2
    for ax, (title, nodes, edges) in zip(axes.flat, panels):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=11, weight="bold", pad=6)
        n_nodes = len(nodes)
        bw = min(1.25, 9.0 / n_nodes - 0.25)
        x_positions = [0.6 + bw / 2 + i * ((9.4 - bw) / max(1, n_nodes - 1))
                       for i in range(n_nodes)]
        positions = {n: (x, BOX_Y) for n, x in zip(nodes, x_positions)}
        for n, (x, y) in positions.items():
            color = ("#cfd8dc" if n in {"START", "END"}
                    else "#ffe082" if n in tool_like
                    else "#bbdefb")
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - bw / 2, y - BOX_H / 2), bw, BOX_H,
                boxstyle="round,pad=0.02", facecolor=color, edgecolor="#555",
                zorder=2))
            ax.text(x, y, n, ha="center", va="center", fontsize=7, zorder=3)
        box_top = BOX_Y + BOX_H / 2
        box_bot = BOX_Y - BOX_H / 2
        back_lane = box_bot - 0.7  # y-coordinate of the loop lane below the boxes
        over_lane = box_top + 0.7  # y-coordinate of the loop lane above the boxes
        for src, dst, kind in edges:
            x0, _ = positions[src]
            x1, _ = positions[dst]
            if kind == "fwd":
                ax.annotate("", xy=(x1 - bw / 2, BOX_Y),
                            xytext=(x0 + bw / 2, BOX_Y),
                            arrowprops=dict(arrowstyle="->", color="#444",
                                            lw=1.2, zorder=1))
            elif kind == "back":
                # Route: source-bottom -> down to back_lane -> across -> up to dest-bottom
                ax.plot([x0, x0], [box_bot, back_lane],
                        color="#c62828", lw=1.3, zorder=1)
                ax.plot([x0, x1], [back_lane, back_lane],
                        color="#c62828", lw=1.3, zorder=1)
                ax.annotate("", xy=(x1, box_bot), xytext=(x1, back_lane),
                            arrowprops=dict(arrowstyle="->", color="#c62828",
                                            lw=1.3, zorder=1))
            elif kind == "fwd_skip":
                # Route: source-top -> up to over_lane -> across -> down to dest-top
                ax.plot([x0, x0], [box_top, over_lane],
                        color="#1565c0", lw=1.2, zorder=1)
                ax.plot([x0, x1], [over_lane, over_lane],
                        color="#1565c0", lw=1.2, zorder=1)
                ax.annotate("", xy=(x1, box_top), xytext=(x1, over_lane),
                            arrowprops=dict(arrowstyle="->", color="#1565c0",
                                            lw=1.2, zorder=1))
    fig.suptitle("Agent architectures — LangGraph topologies", fontsize=12)
    fig.tight_layout()
    save(fig, "fig1_architectures.png")


def fig2_gridworld_examples() -> None:
    levels = ["easy", "medium", "hard", "extra_hard"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, level in zip(axes, levels):
        g = generate_grid_for_level(level, seed=0)
        grid = np.ones((g.height, g.width, 3))
        for x in range(g.width):
            for y in range(g.height):
                if g.fog_of_war and g.view_radius is not None:
                    dx, dy = abs(x - g.agent_pos[0]), abs(y - g.agent_pos[1])
                    if max(dx, dy) > g.view_radius:
                        grid[y, x] = [0.55, 0.55, 0.60]  # fog: grey
                        continue
                if (x, y) == g.agent_pos:
                    grid[y, x] = [0.20, 0.55, 0.95]  # agent: blue
                elif (x, y) == g.goal_pos:
                    grid[y, x] = [0.30, 0.78, 0.35]  # goal: green
                elif (x, y) in g.walls:
                    grid[y, x] = [0.25, 0.25, 0.25]  # wall: dark
                else:
                    grid[y, x] = [0.97, 0.97, 0.97]  # empty: near-white
        ax.imshow(grid, origin="upper")
        for x in range(g.width):
            for y in range(g.height):
                label = ""
                if g.fog_of_war and max(abs(x - g.agent_pos[0]),
                                        abs(y - g.agent_pos[1])) > (g.view_radius or 0):
                    label = "?"
                elif (x, y) == g.agent_pos:
                    label = "A"
                elif (x, y) == g.goal_pos:
                    label = "G"
                elif (x, y) in g.walls:
                    label = "#"
                if label:
                    ax.text(x, y, label, ha="center", va="center",
                            fontsize=10, color="white" if label in ("#", "?") else "black",
                            weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{level}\n{g.width}×{g.height}, max_steps={g.max_steps}"
                     + (", fog" if g.fog_of_war else ""), fontsize=9)
    fig.suptitle("Gridworld example layouts (seed=0)", fontsize=11)
    legend_handles = [
        mpatches.Patch(color=[0.20, 0.55, 0.95], label="A = agent"),
        mpatches.Patch(color=[0.30, 0.78, 0.35], label="G = goal"),
        mpatches.Patch(color=[0.25, 0.25, 0.25], label="# = wall"),
        mpatches.Patch(color=[0.55, 0.55, 0.60], label="? = hidden (fog)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "fig2_gridworld_examples.png")


def fig3_to_6_from_plots(df) -> None:
    save(P.plot_capability(df, "gridworld"), "fig3_capability_gridworld.png")
    save(P.plot_efficiency_pair(df, "gridworld"), "fig4_efficiency_gridworld.png")
    save(P.plot_dispersion(df, "gridworld", "num_llm_calls"),
         "fig5_dispersion_gridworld.png")
    save(P.plot_failure_stack(df, "gridworld"), "fig6_failures_gridworld.png")


def main() -> None:
    df = add_failure_mode(load_all("results/logic_final.jsonl",
                                   "results/gridworld_final.jsonl"))
    fig1_architectures()
    fig2_gridworld_examples()
    fig3_to_6_from_plots(df)
    print("done")


if __name__ == "__main__":
    main()
