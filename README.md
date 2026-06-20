# Evaluating LangGraph-Based Architectures Across Reasoning and Interactive Domains

An empirical investigation comparing agentic architectures of increasing complexity across different task domains and difficulty levels. Built for the **ELTE NLP & Foundation Models** course (2-person team project).

The core idea: rather than optimizing a single solution, we conduct controlled experiments to understand *when and why* certain agentic designs succeed or fail — and at what computational cost.

## Research Questions

1. How does the topology of agentic communication (e.g., linear delegation vs. cyclical reflection) impact the trade-off between task success and computational overhead across fundamentally different environments?
2. At what threshold of task complexity does explicit planning become a bottleneck rather than an advantage when compared to reactive, reflection-based correction?
3. How effectively can advanced cognitive patterns (Tree-of-Thought and Episodic Memory) generalize their benefits across both static logic puzzles and dynamic, partially observable simulations?

## Hypotheses

- Increasing architectural complexity from a single-agent baseline to a Level 3 multi-agent system will result in a massive, predictable increase in total token usage and runtime.
- Linear architectures (Planner + Executor) will exhibit highly consistent step counts per run regardless of correctness, while cyclic architectures (Solver + Critic) will show high variance — especially on hard tasks where the system may enter resource-draining validation loops.
- A Planner + Executor will dominate in static, fully observable tasks (logic puzzles) but underperform in dynamic simulations where rigid plans break down quickly.
- Adding a Critic to a Planner + Executor setup (Level 3) will initially degrade performance in low-complexity settings due to over-correction, but will be the only architecture capable of reliably solving tasks with high ambiguity or noise.
- Episodic memory will provide efficiency gains in static reasoning tasks but may cause "negative transfer" in interactive simulations if the state space changes unexpectedly.

## Architectures

All multi-agent pipelines are implemented using [LangGraph](https://github.com/langchain-ai/langgraph), leveraging its support for cyclic graphs and complex state handoffs.

| Level | Architecture | Graph Topology | Status |
|-------|-------------|----------------|--------|
| **Level 1** | Single-Agent Baseline | `START -> agent [-> tools -> agent]* -> END` | Implemented |
| **Level 2A** | Planner + Executor | `START -> planner -> executor [-> tools -> executor]* -> END` | Implemented |
| **Level 2B** | Solver + Critic | `START -> solver -> critic -> [solver ↔ critic]* -> END` | Implemented |
| **Level 3** | Adaptive (ToT + Episodic Memory) | `START -> planner -> critic -> executor [-> tools -> executor]* -> END` | Implemented |

### Level 1 — Single-Agent Baseline
A single LLM call with no explicit planning or persistent memory. For interactive domains (gridworld), uses a standard ReAct tool-calling loop.

### Level 2A — Planner + Executor
A two-node linear pipeline. The **planner** generates a step-by-step strategy, and the **executor** carries it out sequentially. For gridworld, the executor runs a tool-calling loop to execute the plan with move actions.

### Level 2B — Solver + Critic
A cyclic graph where the **solver** proposes a solution and the **critic** evaluates it against the task rules. If rejected, the state routes back to the solver with feedback for a retry, up to `max_critic_iterations` (default 3). The critic is an LLM-based self-review and never sees the ground truth. For interactive domains the solver runs a tool-calling loop before each critique (`START -> solver [-> tools -> solver]* -> critic -> {solver | END}`); for non-interactive domains it is a single solve call per cycle.

### Level 3 — Adaptive System
Extends Level 2B with **Tree-of-Thought (ToT)** branching and **Episodic Memory**. The **planner** generates `num_branches` (default 3) candidate solution approaches stored in `state["branches"]`, optionally primed with strategies retrieved from the episodic memory bank. The **critic** scores each branch and commits the best to `state["selected_branch"]`. The **executor** carries out the selected branch (tool-calling loop for interactive domains, single call for logic). On exit the run is appended to the memory bank. Role prompts (planner / critic / executor) are domain-agnostic defaults; domain owners can override via subclassing. The memory bank (`RecentSuccessMemory`) is **not** frozen: a fresh bank is created per `(domain, difficulty)` condition and written live after every run, so later runs in the same condition can be primed by earlier ones — runs within a condition are therefore not fully independent. Retrieval returns the top episodes for the matching domain/difficulty, accepted ones first. L3 diagnostics (`branch_count`, `mem_retrievals`, `mem_reuse_hits`) are written to `state["metadata"]` and persisted in `RunResult.state_metadata`.

## Domains

| Domain | Type | Status |
|--------|------|--------|
| **Gridworld** | Interactive environment (text-based simulation) | Implemented |
| **Logic Puzzles** | Reasoning tasks (constraint satisfaction) | Implemented |

### Gridworld
A custom text-based grid simulation where the agent navigates from a start position to a goal, avoiding walls. The agent interacts via LangChain tool calls (`move_up`, `move_down`, `move_left`, `move_right`), receiving text observations after each step.

### Logic Puzzles
Structured constraint satisfaction problems from the [`arg-tech/MysteryZebra`](https://huggingface.co/datasets/arg-tech/MysteryZebra) dataset. Agents must deduce a complete grid assignment (attributes × positions) from a set of natural-language clues. Non-interactive — the LLM reasons and outputs a JSON answer directly; no tool calls. Scored by cell-accuracy (fraction of correct attribute-position assignments), with `success=True` only for a perfect grid.

## Difficulty Settings

Logic puzzles use a 4-level difficulty ladder that varies two axes — grid size and `thinking_token_budget` — while holding the puzzle grade constant at **level3**. Gridworld varies grid size, observability (fog), and budget.


### Logic puzzle dataset

Puzzles come from [`arg-tech/MysteryZebra`](https://huggingface.co/datasets/arg-tech/MysteryZebra).

**Chosen grade: `level3`** — determined by a pilot that ran L1 at unlimited vs. budget=1500 across grades level2–level4:

| grade  | unlimited avg | 1500 avg | Δ    |
|--------|---------------|----------|------|
| level2 | 1.00          | 0.46     | 0.54 |
| **level3** | **1.00**  | **0.72** | **0.28** |
| level4 | 1.00          | 0.10     | 0.90 |

Level3 is the discrimination zone: unlimited solves perfectly, budget=1500 clearly degrades, but without near-zero collapse (level4) that would prevent architecture-level differentiation. The same grade is used for both 3×3 and 5×5 so the internal puzzle grade is not a confound across difficulty levels.

**Pinned puzzle IDs** (selection is independent of HuggingFace ordering):

| Difficulty | IDs |
|------------|-----|
| easy (3×3) | `Pt2_3x3_level3-0` … `Pt2_3x3_level3-4` |
| medium / hard / extra_hard (5×5) | `Pt2_5x5_level3-0` … `Pt2_5x5_level3-4` |

## Project Structure

```
src/
├── config.py                        # ExperimentConfig (Pydantic), YAML loader
├── llm.py                           # create_llm() — ChatOpenAI for ELTE endpoint; UNLIMITED_MAX_TOKENS
├── state.py                         # AgentState TypedDict (shared by all architectures)
├── metrics.py                       # MetricsCallback (token/step/finish-reason tracking)
├── runner.py                        # run_single(), RunResult, dedup/resume helpers
├── architectures/
│   ├── __init__.py                  # get_architecture() factory
│   ├── base.py                      # BaseArchitecture ABC
│   ├── level1_baseline.py           # Single LLM call (ReAct loop for interactive domains)
│   ├── level2a_planner_executor.py  # Linear: planner -> executor
│   ├── level2b_solver_critic.py     # Cyclic: solver <-> critic up to max_critic_iterations
│   ├── level3_adaptive.py           # ToT + episodic memory: planner -> critic -> executor
│   └── memory.py                    # EpisodicMemory protocol, RecentSuccessMemory (live per-condition bank), NoOpMemory
├── domains/
│   ├── __init__.py                  # get_domain() factory
│   ├── base.py                      # BaseDomain ABC, Task/EvaluationResult models
│   ├── gridworld/
│   │   ├── domain.py                # GridworldDomain
│   │   ├── engine.py                # GridWorld simulation engine (4-level difficulty)
│   │   └── tools.py                 # LangChain tools: move_up/down/left/right
│   └── logic_puzzles/
│       ├── domain.py                # LogicPuzzlesDomain
│       └── engine.py                # Puzzle loader, PINNED_*_IDS, parser, scorer
└── preprocessing/
    └── gridworld.py                 # Gridworld results -> common analysis schema
configs/
├── experiments.yaml                 # Default gridworld smoke-test matrix
├── gridworld_final.yaml             # Full gridworld sweep
├── gridworld_pilot.yaml             # Gridworld pilot runs
├── gridworld_smoke.yaml             # Quick gridworld sanity check
├── logic_final_anchor.yaml          # Logic saturation anchors: easy (3×3) + medium (5×5), unlimited budget
├── logic_final_main.yaml            # Logic discrimination tiers: hard (5×5 @4k) + extra_hard (5×5 @1.5k)
├── logic_pilot.yaml                 # Logic pilot runs (grade/budget calibration)
└── legacy/
    └── full_sweep_logic.yaml        # Superseded early logic sweep
scripts/
├── run_all.py                       # Run experiment matrix from YAML; resume-safe
├── run_experiment.py                # CLI: run a single condition
├── analyze_results.py               # Aggregate results, generate plots
├── analyze_pilot.py                 # Pilot-result analysis
├── k0_pilot.py                      # Grade-selection pilot (level2–4 vs budget)
├── test_logic_pipeline.py           # Logic pipeline smoke test
└── test_memory.py                   # Episodic-memory backend tests
results/
├── logic_final.jsonl                # Full logic 4-tier matrix
├── gridworld_final.jsonl            # Full gridworld 4-tier matrix
├── logic_pilot.jsonl                # Logic pilot results
└── k0_pilot.jsonl                   # Grade-selection pilot results
docs/
├── task_description.md              # Course project brief
└── draft.md                         # Initial project draft
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Access to the ELTE network (or VPN) for the LLM endpoint

### Install

```bash
git clone <repo-url>
cd nlp-fm-agentic-project
poetry install
```

### Configure Environment

Create a `.env` file in the project root:

```env
ELTE_API_BASE=<endpoint-url>
ELTE_API_KEY=<your-api-key>
ELTE_MODEL_NAME=<model-name>
```

Refer to the [ELTE LLM endpoint tutorial](https://github.com/elte-nlp/elte-nlp-course/blob/main/practice_examples/LLM_inference_endpoint.md) for setup details.

## Usage

### Run a single experiment

```bash
poetry run python scripts/run_experiment.py --architecture level1 --domain gridworld --difficulty easy --num-runs 3
```

### Run the full experiment matrix

```bash
poetry run python scripts/run_all.py --config configs/gridworld_final.yaml --output results/gridworld_final.jsonl
```

Scoped by domain or architecture to avoid cross-partner interference:

```bash
# Logic puzzles track (Kristóf) — anchors (easy/medium) then discrimination tiers (hard/extra_hard); both append to the same file
poetry run python scripts/run_all.py --config configs/logic_final_anchor.yaml --num-tasks 3 --output results/logic_final.jsonl
poetry run python scripts/run_all.py --config configs/logic_final_main.yaml  --num-tasks 5 --output results/logic_final.jsonl

# Gridworld track (Benedek)
poetry run python scripts/run_all.py --config configs/gridworld_final.yaml --output results/gridworld_final.jsonl
```

Runs are resume-safe — already-completed `(architecture, domain, difficulty, task_id, run_id, budget)` tuples are skipped on restart.

### Analyze results

```bash
poetry run python scripts/analyze_results.py --logic results/logic_final.jsonl --gridworld results/gridworld_final.jsonl --figdir figures
```

Or use the interactive notebook:

```bash
poetry run jupyter notebook notebooks/analysis.ipynb
```

## Evaluation Metrics

**Task performance:**
- `success` — binary (True only for perfect grid / reached goal)
- `score` ∈ [0, 1] — cell-accuracy for logic; goal-or-progress for gridworld

**Efficiency:**
- `total_tokens` — prompt + completion tokens (cost proxy)
- `num_llm_calls` — total LLM invocations per run (architecture-agnostic overhead)
- `runtime_seconds`

**Failure diagnosis:**
- `parse_failure` — model output could not be parsed as a solution
- `any_call_truncated` — at least one call hit `finish_reason=length`
- `budget_saturated` — completion tokens approached the `max_tokens` ceiling
- `revision_count` — critic iterations used (L2B / L3 only)

**L3 diagnostics** (in `state_metadata`):
- `branch_count` — ToT branches generated by the planner
- `mem_retrievals` — episodes retrieved from the memory bank
- `mem_reuse_hits` — retrievals that influenced the final answer

## Experimental Matrix

4 architectures × 4 difficulty levels × 2 domains = 32 conditions. N=8 runs per condition (N=10 at extra_hard). Results written to `results/logic_final.jsonl` and `results/gridworld_final.jsonl`.

### Logic Puzzles

| Architecture | easy (3×3 ∞) | medium (5×5 ∞) | hard (5×5 @4k) | extra_hard (5×5 @1.5k) |
|---|---|---|---|---|
| L1 Baseline | — | — | — | — |
| L2A Planner+Executor | — | — | — | — |
| L2B Solver+Critic | — | — | — | — |
| L3 Adaptive (ToT+Mem) | — | — | — | — |

## Extending the Framework

### Adding a new domain

1. Create `src/domains/<domain_name>/domain.py` implementing `BaseDomain`
2. Implement `generate_task()`, `format_system_prompt()`, `format_task_prompt()`, `evaluate()`
3. Override `get_tools()` only for interactive domains
4. Optionally override `format_critic_prompt()` to give the Level 2B critic domain-specific context (e.g. live environment state). The base default critiques against `task.rules`, so new domains work with Level 2B out of the box.
5. Register in `src/domains/__init__.py`
6. Add conditions to `configs/experiments.yaml`

### Adding a new architecture

1. Create `src/architectures/<name>.py` implementing `BaseArchitecture`
2. Implement `build_graph()` returning a compiled LangGraph
3. Extend `AgentState` in `src/state.py` if new state fields are needed
4. Register in `src/architectures/__init__.py`
5. Add conditions to `configs/experiments.yaml`

## GenAI Tool Usage

| Phase | GenAI Tool Used | Validation Method |
|-------|----------------|-------------------|
| Code implementation | Claude Code | Every generated module was read, run, and checked against the unit/smoke tests and manual experiment runs before use. |
| Code & report formatting | Claude Code | Reviewed all reformatted code and prose line-by-line to confirm meaning and structure were preserved. |
| Architecture figures | Nano Banana 2 | Each generated diagram was manually checked against the implemented graph topology for accuracy. |

## Team & Contributions

| Member | Domain | Architecture ownership |
|--------|--------|------------------------|
| **Kristóf** | Logic Puzzles | L2B (Solver+Critic), L3 base graph + ToT |
| **Benedek** | Gridworld | L1, L2A, L3 episodic memory |

Shared: repo setup, `src/state.py`, `src/runner.py`, `src/config.py`, `src/llm.py`, `src/metrics.py`, analysis notebook.

## Collaboration & Ownership

### Shared files — coordinate before editing

`src/state.py`, `src/runner.py`, `src/architectures/base.py`, `src/architectures/__init__.py`, `src/domains/base.py`, `src/domains/__init__.py`, `src/config.py`, `src/llm.py`, `src/metrics.py`, `scripts/run_all.py`

### Running experiments independently

Use `--config` to point at domain-specific YAML files so neither partner accidentally triggers the other's conditions:

```bash
# Kristóf — logic puzzles
poetry run python scripts/run_all.py --config configs/logic_final_anchor.yaml --num-tasks 3 --output results/logic_final.jsonl
poetry run python scripts/run_all.py --config configs/logic_final_main.yaml  --num-tasks 5 --output results/logic_final.jsonl

# Benedek — gridworld
poetry run python scripts/run_all.py --config configs/gridworld_final.yaml --output results/gridworld_final.jsonl
```

Both files write to the same common schema, so the analysis notebook loads both and treats them identically.

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ELTE NLP Course](https://github.com/elte-nlp/elte-nlp-course)
- [Awesome Agentic Patterns](https://github.com/nibzard/awesome-agentic-patterns)
