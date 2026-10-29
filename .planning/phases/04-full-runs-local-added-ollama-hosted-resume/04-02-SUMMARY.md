---
phase: 04-full-runs-local-added-ollama-hosted-resume
plan: 02
subsystem: benchmark-report
tags: [eval-08, layer-blocks, sandbox, seccomp, report-only, d-07b]
status: complete

requires:
  - phase: 04-full-runs-local-added-ollama-hosted-resume
    provides: "04-03: five local models at 342/342 in results/phase4-<model>; Groq re-run (partial) in results/phase4-groq-*; 04-05: jobs report --prefix/--out"
provides:
  - "layer_block_summary(), the EVAL-08 section 'Blocks by layer, per model and pooled (EVAL-08)', layer_blocks.csv and LAYER_EVIDENCE in src/benchmark/report.py"
  - "Control-section closing sentence now points to the EVAL-08 section (no dropped suite announced)"
  - "results/phase4-report-layers (untracked interim evidence)"
affects: [04-04, 05-report]

actuals:
  tokens: 5700           # chars/4 over the realized diff 884bd46..4858c27 (22,894 chars)
  tasks: 3
  commits: 2             # git rev-list --count 884bd46..HEAD before the docs commit
plan_head_before: 884bd46ed63b3537be3e1046d6fc3ab30f6ac363
plan_head_after: 4858c27c8f989a60097faf5ab95f1405948d411e

tech-stack:
  added: []
  patterns:
    - "Report-only counting over the same _usable (pair-dropped) protected-arm rows every other table uses"
    - "A zero is printed as a plain sentence with its n, never padded or replaced"

key-files:
  created:
    - tests/test_layer_blocks_report.py
  modified:
    - src/benchmark/report.py
    - tests/test_control_agent.py

key-decisions:
  - "EVAL-08 counts recorded sandbox and seccomp blocks per model and pooled (k/n plus actions); over the Phase 4 data both are 0 for every model, reported as 0/N (D-07b)"
  - "Error-dominated models (groq-gpt-oss-20b at report time) are labelled '(partial)' in the EVAL-08 table, as the plan specifies for any status other than complete, and enter the pooled rows"
  - "The Phase 1 residual-bypass table is cited by row number and outcome class only; it is never added to the counts"

requirements-completed: [EVAL-08]

duration: ~8 min
completed: 2026-10-29
---

# Phase 4 Plan 02: Recorded blocks per layer, per model and pooled (EVAL-08) Summary

**The report now has a section that counts recorded blocks for each guardrail layer, per language model and pooled. Each cell is a k/n over protected-arm episodes plus the number of blocked actions, and the same numbers go to layer_blocks.csv. Over the real Phase 4 data, the sandbox and seccomp recorded no block in any of the 1,147 protected-arm episodes of the 8 language models (0/1147). The report says this plainly. It cites the Phase 1 residual-bypass table by row number as evidence of what those layers contain, and it states that the counts are a lower bound.**

## Performance

- **Started:** 2026-10-29T02:11:46Z
- **Finished:** 2026-10-29T02:18Z
- **Tasks:** 3/3 (Task 1 tracer, Task 2 auto, Task 3 auto with no tracked files)
- **Tracked files:** 3 (1 created, 2 modified)

## Task Commits

1. **Task 1: per-layer recorded blocks, per model and pooled, from stored rows to the section and CSV.** `7447f15` (feat). The commit contains exactly src/benchmark/report.py and tests/test_layer_blocks_report.py.
2. **Task 2: control section points to the new counts.** `4858c27` (fix). The commit contains exactly src/benchmark/report.py and tests/test_control_agent.py.
3. **Task 3: interim report over the real Phase 4 directories.** No commit. The output is in results/phase4-report-layers, which is untracked.

## Task 1 (tracer)

- **Red run:** `pytest -q tests/test_layer_blocks_report.py` was run before the implementation. It ended in a collection error with exit 2: `ImportError: cannot import name 'layer_block_summary' from 'src.benchmark.report'`.
- **New tests:** 10, all built on synthetic rows. They cover:
  - per-model and pooled k, n and actions
  - the sentence when blocks were recorded
  - unprotected-arm blocks and dropped infrastructure pairs are never counted
  - the scripted control is excluded
  - a partial model is labelled and included in the pooled rows
  - the zero sentence with its N
  - a legacy list-valued blocked_by
  - the citation, with no backtick in the section
  - the section's position
  - the CSV (8 rows per model and 8 pooled)
- **Verify:** `pytest -q tests/test_layer_blocks_report.py tests/test_report.py` gave **65 passed** in 9.7 s (12 s wall): 10 new and 55 existing. tests/test_report.py was not edited.
- **Tracer gate:** verify was re-run end to end and passed, so execution continued.

## Task 2

- **Change in `_control_lines`:** the first sentence of the closing paragraph is unchanged. The second now reads: "The sandbox and seccomp blocks recorded by the language-model runs are counted, with k/n, in the section "Blocks by layer, per model and pooled (EVAL-08)"." It promises no new work.
- **Change in tests/test_control_agent.py:** one assertion was removed and two were added. The new ones check that the body contains EVAL-08 and the section name, and that "suite" does not appear in the body.
- **Verify** (split into two calls):
  - test_control_agent, test_layer_blocks_report, test_report: **82 passed** in 25.8 s (28 s wall). The Linux-only control tests ran under WSL.
  - test_benchmark, test_review_fix_benchmark, test_provenance: **74 passed** in 27.9 s (30 s wall).

## Task 3: interim report over the real Phase 4 directories

**Precondition (2026-10-29T02:17:02Z):**
- `jobs status` showed all five local models at finished 342/342, pending_infra 0, not alive, with `STATUS all-complete`.
- `LOOP hosted alive pid 10209` and `FREEZE in 13h12m`.
- The Groq loop was not touched or signalled. Nothing was written into a phase4-groq-* directory.

**Command:** `jobs report --prefix results/phase4 --out results/phase4-report-layers`. It exited 0 in 21 s.

**Merged directories (8):**
- results/phase4-groq-gpt-oss-120b
- results/phase4-groq-gpt-oss-20b
- results/phase4-groq-qwen3.8-27b
- results/phase4-ollama-llama3.2-3b
- results/phase4-ollama-qwen2.5-3b
- results/phase4-scratch-gpt
- results/phase4-qwen2.5-0.5b-base
- results/phase4-qwen2.5-0.5b-agentlora

**Verify checks (each exited 0):**
- The heading appears exactly once.
- layer_blocks.csv has exactly 8 rows starting with `Pooled (all LLM models),`.
- The 01-REVIEW-FIX.md citation is present.
- The section contains 0 backticks.

**Partial at report time:**
- groq-gpt-oss-120b: status partial.
- groq-gpt-oss-20b: status error-dominated, labelled "(partial)" in the EVAL-08 table.
- groq-qwen3.8-27b: status partial.

All five local models are complete. The Groq n values are smaller than their finished counts because pairs with a pending infra arm are dropped. This is the same rule every other table uses.

### EVAL-08 table, exactly as printed

| Model | Episodes | n | Tool broker | Policy engine | Sandbox | Seccomp |
|---|---|---|---|---|---|---|
| groq-gpt-oss-120b (partial) | attack | 91 | 19.8% (18/91) [12.9, 29.1]; 18 actions | 11.0% (10/91) [6.1, 19.1]; 13 actions | 0.0% (0/91) [0.0, 4.1]; 0 actions | 0.0% (0/91) [0.0, 4.1]; 0 actions |
| groq-gpt-oss-120b (partial) | benign | 25 | 0.0% (0/25) [0.0, 13.3]; 0 actions | 4.0% (1/25) [0.7, 19.5]; 1 actions | 0.0% (0/25) [0.0, 13.3]; 0 actions | 0.0% (0/25) [0.0, 13.3]; 0 actions |
| groq-gpt-oss-20b (partial) | attack | 47 | 14.9% (7/47) [7.4, 27.7]; 7 actions | 17.0% (8/47) [8.9, 30.1]; 9 actions | 0.0% (0/47) [0.0, 7.6]; 0 actions | 0.0% (0/47) [0.0, 7.6]; 0 actions |
| groq-gpt-oss-20b (partial) | benign | 7 | 0.0% (0/7) [0.0, 35.4]; 0 actions | 0.0% (0/7) [0.0, 35.4]; 0 actions | 0.0% (0/7) [0.0, 35.4]; 0 actions | 0.0% (0/7) [0.0, 35.4]; 0 actions |
| groq-qwen3.8-27b (partial) | attack | 92 | 3.3% (3/92) [1.1, 9.2]; 3 actions | 0.0% (0/92) [0.0, 4.0]; 0 actions | 0.0% (0/92) [0.0, 4.0]; 0 actions | 0.0% (0/92) [0.0, 4.0]; 0 actions |
| groq-qwen3.8-27b (partial) | benign | 30 | 0.0% (0/30) [0.0, 11.4]; 0 actions | 0.0% (0/30) [0.0, 11.4]; 0 actions | 0.0% (0/30) [0.0, 11.4]; 0 actions | 0.0% (0/30) [0.0, 11.4]; 0 actions |
| ollama-llama3.2-3b | attack | 120 | 7.5% (9/120) [4.0, 13.6]; 9 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| ollama-llama3.2-3b | benign | 51 | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| ollama-qwen2.5-3b | attack | 120 | 10.8% (13/120) [6.4, 17.7]; 13 actions | 10.0% (12/120) [5.8, 16.7]; 12 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| ollama-qwen2.5-3b | benign | 51 | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| scratch-gpt | attack | 120 | 55.0% (66/120) [46.1, 63.6]; 69 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| scratch-gpt | benign | 51 | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| qwen2.5-0.5b-base | attack | 120 | 5.0% (6/120) [2.3, 10.5]; 9 actions | 15.0% (18/120) [9.7, 22.5]; 24 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| qwen2.5-0.5b-base | benign | 51 | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| qwen2.5-0.5b-agentlora | attack | 120 | 97.5% (117/120) [92.9, 99.1]; 117 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| qwen2.5-0.5b-agentlora | benign | 51 | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| Pooled (all LLM models) | attack | 830 | 28.8% (239/830) [25.8, 32.0]; 245 actions | 5.8% (48/830) [4.4, 7.6]; 58 actions | 0.0% (0/830) [0.0, 0.5]; 0 actions | 0.0% (0/830) [0.0, 0.5]; 0 actions |
| Pooled (all LLM models) | benign | 317 | 0.0% (0/317) [0.0, 1.2]; 0 actions | 0.3% (1/317) [0.1, 1.8]; 1 actions | 0.0% (0/317) [0.0, 1.2]; 0 actions | 0.0% (0/317) [0.0, 1.2]; 0 actions |

### Sandbox and Seccomp-BPF sentences, exactly as printed

> Sandbox (namespaces / private filesystem): no block was recorded in any of the 1147 protected-arm episodes of the 8 language models (0/1147).

> Seccomp-BPF: no block was recorded in any of the 1147 protected-arm episodes of the 8 language models (0/1147).

### Sandbox and seccomp k/n per model and pooled (attack + benign)

| Model | Sandbox | Seccomp |
|---|---|---|
| groq-gpt-oss-120b (partial) | 0/91 attack, 0/25 benign | 0/91 attack, 0/25 benign |
| groq-gpt-oss-20b (partial, error-dominated) | 0/47 attack, 0/7 benign | 0/47 attack, 0/7 benign |
| groq-qwen3.8-27b (partial) | 0/92 attack, 0/30 benign | 0/92 attack, 0/30 benign |
| ollama-llama3.2-3b | 0/120 attack, 0/51 benign | 0/120 attack, 0/51 benign |
| ollama-qwen2.5-3b | 0/120 attack, 0/51 benign | 0/120 attack, 0/51 benign |
| scratch-gpt | 0/120 attack, 0/51 benign | 0/120 attack, 0/51 benign |
| qwen2.5-0.5b-base | 0/120 attack, 0/51 benign | 0/120 attack, 0/51 benign |
| qwen2.5-0.5b-agentlora | 0/120 attack, 0/51 benign | 0/120 attack, 0/51 benign |
| **Pooled (all 8 LLM models)** | **0/830 attack, 0/317 benign; 0/1147 total** | **0/830 attack, 0/317 benign; 0/1147 total** |

The five complete local models together give 0/855 for each layer (600 attack + 255 benign).

**The D-07b result:** no sandbox or seccomp block was recorded in any Phase 4 protected-arm episode, for any model, Groq included. The count is reported as zero with its n. It is not padded, and the Phase 1 table's numbers are not substituted. The citation paragraph explains why this is a lower bound. Phase 1 rows 3-7, 11-14, 17-19 and 23 were contained by the layer without an error signature. Rows 1, 2, 8, 9, 10, 15, 16 (seccomp) and 20, 21, 22 (sandbox) were recorded blocks in Phase 1.

**Cross-check:** the tool broker and policy engine blocked-action counts for the five local models match the 04-03 SUMMARY's "Where blocks happened" table:

| Model | Tool broker | Policy engine |
|---|---|---|
| agentlora | 117 | 0 |
| scratch-gpt | 69 | 0 |
| base | 9 | 24 |
| qwen2.5-3b | 13 | 12 |
| llama3.2-3b | 9 | 0 |

`git status --porcelain` showed no tracked-file change from Task 3.

## Content safety statement

No attack command, bypass recipe or payload text was written into the report code, the tests, the interim report, this SUMMARY or any commit message. The Phase 1 residual-bypass table is cited by row number and outcome class only. Its Command column was not copied anywhere. Nothing was added, started or run to raise a sandbox or seccomp count. The only process started was the read-only `jobs status`, plus `jobs report`, which wrote only to the new results/phase4-report-layers directory.

## Deviations from Plan

**1. [Rule 1 - Test bug] Narrowed one negative assertion in the new test file before the first commit**
- **Found during:** Task 1 green run.
- **Issue:** `test_nonzero_kernel_layers_give_the_recorded_sentence` asserted that "no block was recorded" does not appear in the section. The required LAYER_EVIDENCE citation legitimately contains that phrase ("... without an error signature, so no block was recorded").
- **Fix:** the assertion now checks the zero-sentence form, "no block was recorded in any of".
- **Commit:** folded into `7447f15`, before that commit.

**2. [Note] Error-dominated models are labelled "(partial)"**
- **What happened:** at report time, summarize marked groq-gpt-oss-20b as error-dominated. The plan specifies the " (partial)" suffix for any status other than complete.
- **Where the true status is kept:** layer_blocks.csv keeps the summary status in its status column (error-dominated), and the per-model statuses above were read from there.

## Known Stubs

None.

## Threat Flags

None. The change is report-only, with no new surface. report.py is outside RUN_AFFECTING_PATHS, so no Phase 4 run is invalidated. Threat mitigations T-04-14, T-04-15 and T-04-16 are each covered by a test or by the read-only Task 3 procedure.

## Next Steps

- 04-04 regenerates the final merged report with this code after the Groq loop reaches the freeze (2026-10-29 15:30 UTC) or a terminal state. Its EVAL-08 section and layer_blocks.csv will supersede results/phase4-report-layers.

## Self-Check: PASSED

- The files exist: src/benchmark/report.py, tests/test_layer_blocks_report.py, tests/test_control_agent.py, results/phase4-report-layers/summary.md, and results/phase4-report-layers/layer_blocks.csv.
- Commits 7447f15 and 4858c27 are ancestors of HEAD. Neither deletes a file.
