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
| **Level 2B** | Solver + Critic | `START -> solver -> critic -> [solver ↔ critic]* -> END` | Planned |
| **Level 3** | Adaptive (ToT + Episodic Memory) | Multi-node with branching, scoring, and memory retrieval | Planned |

### Level 1 — Single-Agent Baseline
A single LLM call with no explicit planning or persistent memory. For interactive domains (gridworld), uses a standard ReAct tool-calling loop.

### Level 2A — Planner + Executor
A two-node linear pipeline. The **planner** generates a step-by-step strategy, and the **executor** carries it out sequentially. For gridworld, the executor runs a tool-calling loop to execute the plan with move actions.

### Level 2B — Solver + Critic *(planned)*
A cyclic graph where the **solver** proposes a solution and the **critic** evaluates it against task rules. If rejected, the state routes back to the solver with feedback for a retry, up to a configurable iteration limit.

### Level 3 — Adaptive System *(planned)*
Extends Level 2B with **Tree-of-Thought** branching (planner generates multiple continuations, critic scores them, executor acts on the best) and **Episodic Memory** (saves execution logs to a persistent memory bank for retrieval in future runs).

## Domains

| Domain | Type | Status |
|--------|------|--------|
| **Gridworld** | Interactive environment (text-based simulation) | Implemented |
| **Logic Puzzles** | Reasoning tasks (constraint satisfaction) | Planned |

### Gridworld
A custom text-based grid simulation where the agent navigates from a start position to a goal, avoiding walls. The agent interacts via LangChain tool calls (`move_up`, `move_down`, `move_left`, `move_right`), receiving text observations after each step.

### Logic Puzzles *(planned)*
Structured constraint satisfaction problems requiring agents to hold multiple rules in context and deduce answers without external feedback loops. Non-interactive — the LLM reasons and outputs an answer directly.

## Difficulty Settings

| Setting | Gridworld | Logic Puzzles |
|---------|-----------|---------------|
| **Easy** | 4x4 grid, full observability, few walls, goal 3–5 steps away | Short puzzles, explicit rules, low ambiguity |
| **Hard** | 8x8 grid, partial observability (view radius = 1), maze-like walls, goal 10–15 steps away | Complex puzzles, implicit rules, noisy/ambiguous inputs |

## Project Structure

```
src/
├── __init__.py
├── config.py                        # ExperimentConfig (Pydantic), YAML loader
├── llm.py                           # create_llm() -> ChatOpenAI for ELTE endpoint
├── state.py                         # LangGraph TypedDict state schemas
├── metrics.py                       # MetricsCallback (token/step tracking)
├── runner.py                        # run_single() orchestrator, RunResult model
├── architectures/
│   ├── __init__.py                  # get_architecture() factory
│   ├── base.py                      # BaseArchitecture ABC
│   ├── level1_baseline.py           # Single LLM call
│   └── level2a_planner_executor.py  # Linear planner -> executor
└── domains/
    ├── __init__.py                  # get_domain() factory
    ├── base.py                      # BaseDomain ABC, Task/EvaluationResult models
    └── gridworld/
        ├── __init__.py
        ├── domain.py                # GridworldDomain
        ├── engine.py                # GridWorld simulation engine
        └── tools.py                 # LangChain tools: move_up/down/left/right
configs/
└── experiments.yaml                 # Experiment matrix definition
scripts/
├── run_experiment.py                # CLI: run single condition
├── run_all.py                       # Run full experiment matrix
└── analyze_results.py               # Aggregate results, generate plots
notebooks/
└── analysis.ipynb                   # Interactive results exploration
results/
└── .gitkeep                         # Populated by runs (git-ignored)
docs/
├── task_description.html            # Course task description
└── draft.md                         # Project draft with research questions
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
python scripts/run_experiment.py --architecture level1 --domain gridworld --difficulty easy --num-runs 3
```

### Run the full experiment matrix

```bash
python scripts/run_all.py
```

This reads `configs/experiments.yaml` and runs all defined conditions. Results are saved to `results/results.jsonl`.

### Analyze results

```bash
python scripts/analyze_results.py
```

Or use the interactive notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Evaluation Metrics

**Task performance:**
- Task success rate
- Score (0.0–1.0, supports partial credit)

**Efficiency:**
- Token usage (prompt / completion / total)
- Runtime (seconds)
- Number of LLM calls
- Number of reasoning / tool steps

## Experimental Matrix

All architectures are evaluated across all domains at both difficulty levels, with 3–5 runs per condition to account for LLM variance.

| Architecture | Domain | Easy | Hard |
|-------------|--------|------|------|
| Level 1 (Baseline) | Gridworld | `<score>` | `<score>` |
| Level 2A (Planner + Executor) | Gridworld | `<score>` | `<score>` |
| Level 2B (Solver + Critic) | Gridworld | `<score>` | `<score>` |
| Level 3 (Adaptive) | Gridworld | `<score>` | `<score>` |
| Level 1 (Baseline) | Logic Puzzles | `<score>` | `<score>` |
| Level 2A (Planner + Executor) | Logic Puzzles | `<score>` | `<score>` |
| Level 2B (Solver + Critic) | Logic Puzzles | `<score>` | `<score>` |
| Level 3 (Adaptive) | Logic Puzzles | `<score>` | `<score>` |

## Extending the Framework

### Adding a new domain

1. Create `src/domains/<domain_name>/domain.py` implementing `BaseDomain`
2. Implement `generate_task()`, `format_system_prompt()`, `format_task_prompt()`, `evaluate()`
3. Override `get_tools()` only for interactive domains
4. Register in `src/domains/__init__.py`
5. Add conditions to `configs/experiments.yaml`

### Adding a new architecture

1. Create `src/architectures/<name>.py` implementing `BaseArchitecture`
2. Implement `build_graph()` returning a compiled LangGraph
3. Extend `AgentState` in `src/state.py` if new state fields are needed
4. Register in `src/architectures/__init__.py`
5. Add conditions to `configs/experiments.yaml`

## GenAI Tool Usage

| Phase | GenAI Tool Used | Validation Method |
|-------|----------------|-------------------|
| ... | ... | ... |

## Team & Contributions

| Member | Responsibilities |
|--------|-----------------|
| *Member 1* | ... |
| *Member 2* | ... |

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ELTE NLP Course](https://github.com/elte-nlp/elte-nlp-course)
- [Awesome Agentic Patterns](https://github.com/nibzard/awesome-agentic-patterns)
