# Phase 5: Comparison Report & Deliverables - Context

**Gathered:** 2026-10-29
**Status:** Ready for planning
**Source:** Locked project decisions, Phase 4 results, and the user's 2026-10-29 instruction to keep working while the Groq re-run finishes in the background (YOLO defaults; no discuss-phase)

<domain>
## Phase Boundary

Turn the Phase 4 data into the review deliverables:
- the cross-model comparison report with charts (DOC-01, DOC-02)
- the complete viva notes (DOC-03)
- README and SDD updated for the multi-LLM and local-model system (DOC-04)
- a Review-3 slide deck (DOC-05)

All of these come from one frozen results directory, through one command.
</domain>

<decisions>
## Implementation Decisions

### Data and regeneration
- **D-01:** Everything must be regenerable, because the Groq re-run is still running. It continues until the 2026-10-29 15:30 UTC freeze; see Phase 4 D-09 and WINDOWS.md entry 2. One command, `python -m src.deliverables build`, rebuilds the frozen directory `results/phase5-final/` from the Phase 4 directories, plus the report, charts, slide deck and number tables. The Phase 4 directories are: the five local models, phase4-control, and the three phase4-groq-* models.
  - Build it now on the current snapshot.
  - Re-run it once after the freeze. That re-run is a follow-up, not a blocker for this phase.
  - The snapshot time and the per-model completeness are printed on every deliverable.
- **D-02:** The October Gemini and Gemma rows (results/final-20261011) were produced by an older engine. They appear only in a clearly labelled "Legacy rows (older engine, not compared)" table, with their n. They are never shown in the headline ranking or in the charts. The headline compares only current-engine rows: 5 local models, 3 Groq models (labelled partial with n until they are complete), and the scripted control as a reference line.
- **D-03:** One source of numbers. The README, SDD, slides and viva pages take their result numbers from a generated `numbers.json` / `summary.csv` in the frozen directory. A consistency check fails if any of these documents carries a number that differs from the generated data. Extend `run_checks` or add a deliverables check for this. Prefer generated snippets, such as a results table inserted between markers in README/SDD, over hand-typed numbers.

### Report and charts (DOC-01, DOC-02)
- **D-04:** Build on the existing `src/benchmark/report.py` and `jobs report`; do not add a parallel report.
  - Add a confounds table. Columns: parameter count, quantization (Ollama Q4_K_M default vs bf16/fp32 local vs hosted unknown), local or hosted, context window, decoding (greedy/temp 0), and training provenance (scratch / LoRA / pretrained / hosted).
  - Add a threats-to-validity section. It covers: partial Groq n, the older-engine legacy rows, the 0 sandbox/seccomp blocks (a lower bound, since the policy engine stops attacks first), format failures that depress ASR for weak models, a single benchmark, and greedy decoding.
- **D-05:** Static matplotlib PNG charts, colour-blind-safe and labelled with n:
  1. ASR with 95% Wilson CIs per model, unprotected vs protected.
  2. Capability-vs-vulnerability scatter: benign utility vs unprotected ASR.
  3. Utility-vs-ASR arrow plot, from unprotected to protected.

  Charts are written into the frozen directory and copied to `docs/images/phase5/` only if that path is not on the never-stage list. Otherwise reference them from `results/phase5-final/` and embed them in the deck.

### Docs (DOC-03, DOC-04)
- **D-06:** README.md and docs/SDD.md are updated in place. Edit the markdown only. The SDD PDF/DOCX are binaries and must not be staged; regenerate them with the tracked `scripts/export_sdd.py` only if that runs offline without new installs, and otherwise note it as a manual step.
  - The SDD gains: the multi-LLM provider layer, hf_local and scratch providers, the local training pipeline, the jobs/run_checks tooling, and a results section.
- **D-07:** Viva notes. Add `docs/viva/05-evaluation-statistics-and-examiner-qa.md` and update the README index. It must include:
  - Wilson intervals, paired arms, ASR vs conditional ASR, completeness gating, confounds, and threats to validity.
  - A consolidated "likely examiner questions" set across all phases, with at least 30 questions and answers.
  - A one-page "defend the result" summary.

  Also fill the Phase 4 page placeholder "What the code review found and what we fixed" (WINDOWS.md entry 3) once 04-REVIEW-FIX exists. If it does not exist yet, leave it to the Phase 4 review step and do not block on it.

### Slides (DOC-05)
- **D-08:** Generate `docs/OS_Level_Guardrails_Review_3.pptx` with python-pptx.
  - Base it on `docs/template.pptx` if that file opens; otherwise use a clean built-in layout.
  - The generator lives in tracked code (`src/deliverables/slides.py`). The .pptx itself is a binary output and is never staged.
  - About 12–16 slides: the reviewer remark; the architecture (reference monitor, policy engine, namespaces, seccomp); the locally made models (ScratchGPT from scratch, Qwen LoRA); the model line-up and confounds; the headline results with charts; where blocks happened by layer, with an honest zero; the control agent; limitations; and next steps (v2).
  - Speaker notes on each slide.
  - Check whether python-pptx is importable in the WSL venv or on Windows. If it is missing, record a checkpoint asking for a pip install; it is not pre-approved.

### Claude's Discretion
- File layout under `src/deliverables/`, chart styling, slide visual design (clean and editorial; no stock gradients), test granularity, and the exact marker syntax for generated README/SDD blocks.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-03-SUMMARY.md`, `04-02-SUMMARY.md`, `04-04-SUMMARY.md` — the Phase 4 numbers and directories
- `results/phase4-report/summary.md` and `summary.csv` — the current merged report (snapshot 2026-10-29T03:06Z)
- `src/benchmark/report.py`, `src/benchmark/jobs.py`, `src/benchmark/run_checks.py` — the report tooling to extend
- `docs/viva/01..04` — viva page style
- `README.md`, `docs/SDD.md`, `scripts/export_sdd.py`, `docs/template.pptx`
- `.planning/WINDOWS.md` — open follow-ups
</canonical_refs>

<specifics>
## Specific Ideas

Headline story for the deck, from Phase 4:
- With guardrails, every current-engine model succeeded on 0 attacks. Qwen LoRA went from 111/120 to 0/120, and ScratchGPT from 33/120 to 0/120.
- The conditional block rate was 200/200.
- Sandbox and seccomp recorded 0 blocks in 1151 protected-arm episodes, because the policy engine stops attacks first. This is stated honestly and backed by the Phase 1 residual-bypass evidence (rows cited by number only).
- The control agent attacked 120/120 without guardrails and succeeded 0/120 with them.
</specifics>

<deferred>
## Deferred Ideas

- Final Groq numbers after the freeze. Re-run `python -m src.deliverables build` (WINDOWS.md entry 2).
- Exporting the SDD PDF/DOCX if the exporter needs network access or new installs.
</deferred>

---

*Phase: 05-comparison-report-deliverables*
*Context gathered: 2026-10-29*
