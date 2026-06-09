## Project Draft: Evaluating LangGraph-Based Architectures Across Reasoning and Interactive Domains

### Main Research Questions
* How does the topology of agentic communication (e.g., linear delegation vs. cyclical reflection) impact the trade-off between task success and computational overhead across fundamentally different environments?
* At what threshold of task complexity does explicit planning become a bottleneck rather than an advantage when compared to reactive, reflection-based correction?
* How effectively can advanced cognitive patterns (Tree-of-Thought and Episodic Memory) generalize their benefits across both static logic puzzles and dynamic, partially observable simulations?

### Initial Hypotheses

* Increasing architectural complexity from a single-agent baseline to a Level 3 multi-agent system will result in a massive, predictable increase in total token usage and runtime. The communication overhead of passing states between nodes (e.g., ToT branch generation and critic evaluations) will make the Level 3 system the most resource-intensive architecture across all tasks.
* Linear architectures (like the `Planner + Executor`) will exhibit a highly consistent number of reasoning and tool steps per run, regardless of whether the final answer is correct. In contrast, cyclic architectures (like the `Solver + Critic`) will show high variance in runtime and token usage, particularly on "Hard" tasks where the system gets caught in repeated, resource-draining validation loops.
* While a `planner + executor` architecture will dominate in static, fully observable tasks (logic puzzles) by preventing early logical drift, it will underperform compared to a `solver + critic` in dynamic simulations where rigid plans break down quickly upon execution.
* Adding a `critic` to a `planner + executor` setup (Level 3) will initially degrade performance in low-complexity settings due to over-correction and hallucinated flaws, but will be the only architecture capable of reliably solving tasks with high ambiguity or noise.
* Episodic memory caching will provide exponential efficiency gains in static reasoning tasks, but may lead to "negative transfer" (agents stubbornly applying old solutions to new contexts) in interactive simulations if the state space changes unexpectedly.

### Planned Architectures
All multi-agent pipelines will be implemented using LangGraph, leveraging its ability to manage cyclic graphs and complex state handoffs.

* **Level 1 Baseline:**
  * A single LLM call to process the puzzle or simulation state.
  * No explicit planning module.
  * No persistent memory beyond the prompt context.
* **Level 2 System A (`Planner + Executor`):**
  * A two-node LangGraph setup.
  * The `planner` agent breaks down the problem and generates a step-by-step strategy.
  * The `executor` agent carries out the instructions sequentially without questioning the plan.
* **Level 2 System B (`Solver + Critic`):**
  * A cyclic LangGraph setup.
  * The `solver` agent proposes a solution or takes an action.
  * The `critic` agent evaluates the state against the rules and either approves the action or routes the state back to the solver with feedback for a retry.
* **Level 3 Adaptive System (`Planner + Critic + Executor` with ToT & Episodic Memory):**
  * A complex multi-node graph acting as an adaptive system.
  * **Tree-of-Thought (ToT):** The `planner` generates multiple branch continuations for the next step, the `critic` scores them, and the `executor` acts on the highest-scoring branch. https://www.agentic-patterns.com/patterns/tree-of-thought-reasoning/#:~:text=Explore%20a%20search%20tree%20of%20intermediate%20thoughts%20instead,promising%20paths%20until%20a%20stopping%20condition%20is%20met.
  * **Episodic Memory Retrieval & Action Caching:** The system saves execution logs (both successes and failures) to a persistent memory bank. During the planning phase, it retrieves past episodes to avoid repeating errors or to instantly reuse a proven sub-path.
  https://www.agentic-patterns.com/


### Domains
* **Domain 1: Reasoning tasks (Logic Puzzles)**
  * Structured constraint satisfaction problems and logic puzzles.
  * Requires agents to hold multiple rules in context and deduce answers without explicit external feedback loops.
* **Domain 2: Interactive environments (Simulations)**
  * Text-based simulations or gridworlds.
  * Requires agents to interact with tools, navigate sequential decision-making, and respond to changing environmental states.

### Difficulty Settings
* **Easy:**
  * Fully observable environments, explicit rules, and short planning horizons.
  * Example: 2-3 step logic puzzles or small 4x4 simulation grids.
* **Hard:**
  * Longer planning horizons and increased task complexity (larger grids, more entities).
  * Partial observability in simulations (agents can only "see" adjacent cells).
  * Noisy or ambiguous inputs in logic puzzles.

### Experimental Setup & Evaluation Plan
* **Implementation:** Shared codebase building LangGraph runners that can dynamically swap out tools, prompts, and evaluation metrics based on the domain.
* **Matrix:** All four architectures evaluated across 2 domains at 2 difficulty levels.
* **Runs:** Conduct 3-5 runs per condition to account for non-deterministic LLM variance.
* **Evaluation Metrics:**
  * Main metrics: Task success rate, structural accuracy, final analysis accuracy.
  * Efficiency metrics: Token usage, runtime, and the number of reasoning/tool steps.
* **Analysis:**
  * Compare single-agent baseline to multi-agent structures.
  * Analyze failure cases, specifically attributing them to architectural bottlenecks (e.g., getting stuck in infinite LangGraph `critic-solver` loops).
  * Evaluate if the self-optimizing features (ToT and memory) result in overfitting by testing the populated memory buffers on novel problem samples.