# Runnable Experiment Design

The course deliverable "Research Notebook / Code: runnable experiments, clear
documentation, reproducibility encouraged" is satisfied by **two artifacts that
share one analysis library**, so the interactive and batch paths can never
disagree:

```
notebooks/analysis.ipynb        ← interactive "runnable experiment" (primary)
scripts/analyze_results.py      ← headless batch twin (same functions, savefig)
        ▲
        └── both call ── src/analysis/{loader,failure_modes,aggregate,plots}
```

The existing `scripts/run_all.py` / `run_experiment.py` (already in the repo)
*produce* the results; these artifacts *consume and analyse* them. Together they
form the runnable experiment end to end: run → upload JSONL → analyse → figures.

---

## 1. Design goals

1. **Wired to the real schema.** The loader reads the exact field names emitted
   by `runner.RunResult.model_dump_json`, so the moment you upload the two JSONL
   files the whole pipeline runs on them unchanged — no adapter, no reshaping.
   The backbone has been executed end-to-end to confirm every table and figure
   renders.
2. **Both domains, one pipeline.** Even though you built logic and Benedek built
   gridworld, the notebook loads and analyses both identically.
3. **L3-agnostic.** Empty L3 cells render as `n=0, NaN`; when the L3 rows are
   run and uploaded, the same notebook fills them with no code change.
4. **Defensible.** Every transformation is a named function you can point to and
   explain; nothing is a one-off cell hack.

---

## 2. The notebook (`notebooks/analysis.ipynb`) — section by section

Built by `scripts/build_notebook.py` (re-run it to regenerate; thereafter edit
the `.ipynb` directly).

| § | Cell does | Maps to report section |
|---|-----------|------------------------|
| 0 | **load results** — set the two JSONL paths and load both domains | — |
| 1 | **matrix coverage** — which cells have runs (spot empty L3) | Experiments |
| 2 | **capability matrices** + bootstrap CIs + grouped-bar figures | Results |
| 3 | **efficiency** (tokens, calls; runtime caveated) + line figures | Results |
| 4 | **H1–H5**, each its own slice + verdict prompt | Analysis |
| 5 | **failure analysis** — rate tables, stacked bars, sampled answers | Failure Analysis |
| 6 | **(optional) live demo** — one real agent run, guarded by a flag | Experiments (repro) |
| 7 | **export figures** to `figures/` for embedding in the report | — |

Section 0 is the **only** place you edit when results change: set `LOGIC_PATH`
and `GRIDWORLD_PATH` to the uploaded files. If a domain's file is absent the
loader warns and returns an empty frame for it, so the rest still runs on
whatever is present. The `RUN_LIVE_DEMO` flag (default off) gates the only
endpoint-touching cell.

---

## 3. The batch CLI (`scripts/analyze_results.py`)

For reproducible, headless regeneration of every table and figure:

```bash
poetry run python scripts/analyze_results.py \
    --logic results/logic_final.jsonl \
    --gridworld results/results_gridworld.jsonl \
    --figdir figures
```

Prints coverage, capability, efficiency, dispersion and failure-rate tables for
both domains; writes `figures/*.png` and a machine-readable `figures/summary.json`.
This is what you run after the real sweep (and again after L3 lands). It calls
the same `src.analysis` functions as the notebook, so the two cannot diverge.

---

## 4. The live demo (defense aid, not a result)

On the ELTE network with `.env` populated, set `RUN_LIVE_DEMO = True` in the
notebook. It runs **one** condition (`level1`, `logic_puzzles`, `easy`, one run)
through the real `run_single`, printing success/score/tokens/calls. Purpose: show
the agent pipeline genuinely executes against the model. Keep it to one run — it
does not feed the reported matrix and you do not want it perturbing your JSONL.

---

## 5. Reproducibility statement (put a version of this in the README/report)

- **Determinism that is in your control is fixed:** pinned puzzle IDs (selection
  independent of HuggingFace ordering), seeded grid generation
  (`generate_grid(seed=task_id)`), seeded bootstrap in analysis, locked config
  (model, `reasoning_effort=medium`, temperature, `max_critic_iterations=2`,
  `num_branches=3`, `max_tokens` rule).
- **Determinism that is not:** LLM sampling at `temperature=0.7` and endpoint
  load. This is why every condition runs N≥8 and why runtime is excluded as a
  metric. Reproducibility is at the level of *distributions and conclusions*,
  not bit-identical outputs — state this honestly.
- **Resume-safe runs:** `run_all.py` skips completed
  `(architecture, domain, difficulty, task_id, run_id, budget, max_critic_iter)`
  tuples, so an interrupted sweep continues without duplication.

---

## 6. Order of operations once L3 is ready

1. Confirm Benedek's B3 memory bank is built, **frozen**, and committed as a
   read-only artifact (so measured runs stay independent — D5).
2. Uncomment the L3 rows in `configs/logic_final.yaml`; confirm the gridworld L3
   rows and the gridworld `extra_hard` budget (500) with Benedek.
3. Run the L3 sweep (budget compute: L3 ≈ 5 calls/run, ~25–35k tokens/run at
   generous budgets — the K3 pilot measured ~25–26k at budget=4000).
4. Upload the updated JSONL; re-run `scripts/analyze_results.py` (or just re-run
   the notebook). L3 cells, H4/H5 slices, and L3 diagnostics populate
   automatically — no code change.
5. Regenerate figures; drop them into the report.
