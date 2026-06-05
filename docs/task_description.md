# Projects

## General rules

**Deadline**: The project must be defended at your exam. The project material (repo, report) must be submitted two days before the exam by sending it to all of the lecturers via email.

**Topic Selection**: Here we present:

1. A default project topic, which you can flexibly adapt and customize.
2. A few alternative project ideas related to the research interests of the lecturers. If you would like to choose these make sure to **contact** the lecturer before starting the project to get approval and guidance.

You have to choose one of these topics for your project.

**Expected Effort**: The project is expected to require approximately 80 hours of work. This includes time for research, implementation, experimentation, analysis, and report writing. Working in pairs is allowed but the effort should be 80 hours per person, not 80 hours total.

**When working in pairs:**

* **Division of Work**: You may divide the tasks as you see fit, but the contribution of each member must be clearly documented and identifiable.
* **Scope and Content**: The project should be increased/doubled content compared to a single project either through technical depth (complexity of the solution) or breadth (the volume/number of features implemented).
* **Documentation and Repository**: Pairs may use a single, shared code repository.
    * However, each student must submit an **individual** report. This report should highlight your specific contributions while still providing context for the complementary parts of the project.
* **Individual Evaluation**: The grading process and the project defense (oral exam) are handled entirely separately for each student. Your individual performance and understanding of your specific tasks will be assessed independently. This ensures that if one partner is unable to complete the course, it does not negatively impact the other's progress or grade.

## Default Project -- Agentic Architecture Research

*This project is proposed in collaboration with the HUN-REN AI Core Team who specialize in researching the connection between Agentic AI and Science.*

Large Language Models (LLMs) are increasingly used not as standalone predictors, but as agents capable of planning, reasoning, interacting with tools, and collaborating with other agents. While model capability matters, recent research shows that agent architecture design often has a larger impact on performance than model size alone.

### Requirements

* **Objective:** Design, compare, and analyze multiple agent team architectures across tasks of varying difficulty, and derive evidence-based conclusions about when and why certain agentic designs succeed or fail.
* **Focus:** This assignment emphasizes research thinking, experimental methodology, and critical analysis.
* **Hand-in:** Submit your deliverables (code, report, and presentation) as outlined in the Deliverables section.

### Core Project Types / Goals

Rather than optimizing a single solution, you are conducting an empirical investigation to answer a research question. Examples of research questions include:

* When do multi-agent systems outperform single agents?
* Does planning improve performance under limited context?
* What are the costs and benefits of reflection loops?
* How does communication overhead affect outcomes?

**You must implement and evaluate:**

* At least **three distinct agent architectures** of increasing complexity (see below).
* Across at least **two different task domains**.
* Under at least **two difficulty levels**.

---

### Agent Capability Levels

Your experiments must include architectures covering increasing agentic complexity.

**Level 1 — Single-Agent Baseline**
* Single LLM call.
* No explicit planning module.
* No persistent memory beyond prompt context.

**Level 2 — Multi-Agent System**
* At least two specialized agents communicating via messages. You must measure at least 2 different multi-agent architectures. Examples include:
    * `planner + executor`
    * `solver + critic`
    * `researcher + verifier`

**Level 3 — Adaptive Multi-Agent System**
* Memory summarization.
* Adaptive strategies.
* Dynamic role assignment.
* Experience replay and self-learning agents.

See the following collection for inspiration of variable complexity Agentic AI architectures: [https://github.com/nibzard/awesome-agentic-patterns](https://github.com/nibzard/awesome-agentic-patterns)

---

### Workflow / Required Steps

**1. Choose Task Domains**
You are encouraged to explore existing environments, datasets, or simulations. Your experiments must include tasks from at least two categories (justify your choices):

* **Reasoning tasks:** logic puzzles, coding repair, structured QA.
* **Interactive environments:** games, simulations, gridworlds.
* **Information-seeking tasks:** retrieval, investigation, querying tools.
* **Planning or coordination problems:** scheduling, negotiation, strategy.

**2. Define Difficulty Levels**
You must evaluate performance at multiple difficulty levels and clearly explain how difficulty changes between settings. Define and justify difficulty using at least two of the following:

* longer planning horizon
* partial observability
* noisy or ambiguous inputs
* limited token budget
* restricted tool usage
* delayed feedback
* increased task complexity

**3. Design the Experimental Pipeline**
Your study must include controlled comparisons between architectures, quantitative evaluation metrics, multiple experimental runs where appropriate, and an analysis of efficiency (e.g., tokens, steps, or runtime). You must evaluate using an experimental matrix similar to:

| Architecture | Easy Task | Hard Task |
| :--- | :--- | :--- |
| Single-Agent Baseline | `<score>` | `<score>` |
| Multi-Agent System A | `<score>` | `<score>` |
| Multi-Agent System B | `<score>` | `<score>` |
| Adaptive Multi-Agent System(optional) | `<score>` | `<score>` |

*Hint: In case of complex tasks consider an LLM-as-a-judge metric, but make sure to ground it on data, or clear set of rules/environments, not just a subjective judgement/hallucination.*

**4. Required Analysis**
Your report must encompass several analytical steps:

* **Research Question & Hypotheses:** State expectations before experiments.
* **Architecture Design:** Include diagrams explaining agent components and communication.
* **Experimental Setup:** Detail models used, tools, benchmarks, and evaluation metrics.
* **Results:** Provide quantitative comparisons and visualizations.
* **Failure Analysis:** Discuss when agents fail and why.
* **Discussion & Conclusions:** Interpret findings and tradeoffs, and summarize what you learned about agent design.

---

### Deliverables

**1. Research Notebook / Code**
* runnable experiments
* clear documentation
* reproducibility encouraged but not required
* Upload to a repository and write an extensive README.

**2. Written Report (5-8 pages recommended)**
* Short research-paper style.
* Upload to the repository if possible.
* Use the following structure written in either Markdown or LaTeX:
    * Abstract
    * Introduction
    * Related Work
    * Methodology
    * Experiments
    * Results
    * Analysis
    * Conclusion

---

### Allowed Resources & Tools

* **Required Framework:** An existing Python-based Agentic Framework from one of these: LangChain, LangGraph, Microsoft Agent Framework.
* **Allowed:** local LLMs, API-based LLMs, open-source tools.
* *Note:* You must clearly document all tools and models used. You may use external libraries and models, but all architecture design, experiments, and analysis must be your own work. Properly cite external resources.

#### Use of GenAI Tools

Usage of GenAI tools for writing, code generation, analysis or literature review is allowed with the following restrictions:

* You should fully understand your code and be able to explain any part of it during defense, failure to do so will result in a fail.
* You should check the output of GenAI tools and take full responsibility for the content of your project.
* You should put the effort freed up by GenAI tools into deeper analysis, architectural exploration, and critical thinking.
* You should document the use of GenAI tools by providing a table in the README file with the phase, the tool used and how you validated the output. For example:

| Phase | GenAI Tool Used | Validation Method |
| :--- | :--- | :--- |
| Literature Review | Gemini DeepResearch | Manually checked the relevance and accuracy of the summaries against original papers. |
| Experimental Code Architecture Design | OpenCode | Used the coding agent in plan mode to generate a detailed architecture; manually reviewed the md file. |
| Implementation of the Multi-Agent System | Copilot | Used Copilot to initialize the codebase; code was manually reviewed and adjusted. |

---

### Evaluation Criteria & Timeline

This is not a demo-building exercise; performance alone is not the main grading criterion. Strong projects will test hypotheses rather than showcase systems, analyze failures honestly, compare alternatives rigorously, and reflect critically on agent complexity. We encourage everyone to record experiments with multiple hyperparameter settings, or even optimize for them. Negative results are valuable and encouraged.

The grading will be binary (pass/fail) and will be based on the criteria outlined above. Only complete projects that meet the requirements and demonstrate a thorough investigation of agentic architectures will receive a passing grade.

**Suggested Timeline (80h total effort)**
* 10h — literature exploration & planning
* 25h — implementation
* 25h — experimentation
* 10h — analysis
* 10h — report and documentation

### Example project draft

This is a set of notes for a possible project, you are encouraged to use it as a starting point, but make sure to adapt it and expand it to fit your interests and the requirements of the project.:

Draft title: Comparing agent architectures for theory-guided grid construction and analysis

* **Main research question:**
    * When do multi-agent architectures outperform a single-agent baseline in converting domain theories into simplified grid-based models?
    * Does separating grid construction from validation improve cell-level accuracy?
    * How well do the same agent architectures transfer across different grid domains?
* **Initial hypothesis:**
    * A single agent will be enough when the source description is short and the mapping rules are explicit.
    * Multi-agent systems will help more when the task requires separating interpretation, grid construction, and validation.
* **Planned architectures:**
    * Level 1 baseline:
        * one main LLM call to read the description, create the grid, and produce the analysis.
    * Level 2 system A:
        * schema-builder architecture: first agent extracts mapping rules, second agent fills the grid.
    * Level 2 system B:
        * builder-critic architecture: builder creates the grid, critic checks against the source theory.
    * Level 3:
        * Use Level 2 system B but add a third agent that manages memory of previous runs and iteratively improves output.
* **Domains:**
    * Domain 1: archaeological / cemetery grids
    * Domain 2: microscopic liquid-droplet grids
* **Difficulty settings:**
    * Easy: smaller grids (8x8), explicit mapping rules, low ambiguity.
    * Hard: larger grids (32x32), implicit rules, noisy descriptions.
* **Experimental setup notes:**
    * Use one Python-based agent framework (e.g., LangGraph).
    * Do 3 runs per condition because LLM outputs can vary.
* **Evaluation plan:**
    * main metrics: labeling accuracy, structural similarity, final analysis accuracy.
    * efficiency metrics: token usage, runtime, number of reasoning/tool steps.
* **Analysis to include:**
    * Compare the single-agent baseline to the multi-agent one.
    * Analyse the failure cases; attribute them to specific architectural choices.
    * Compare the self-optimizing agentic system to the non-adaptive one.