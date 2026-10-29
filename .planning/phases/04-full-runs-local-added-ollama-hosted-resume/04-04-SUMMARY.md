---
phase: 04-full-runs-local-added-ollama-hosted-resume
plan: 04
subsystem: benchmark-report-and-viva
tags: [control-rerun, merged-report, viva, run_checks-viva, eval-08, groq-snapshot, stopping-rule]
status: complete

requires:
  - phase: 04-full-runs-local-added-ollama-hosted-resume
    provides: "04-02: EVAL-08 'Blocks by layer, per model and pooled' report section; 04-03: five local models 342/342 and the running Groq re-run; 04-05: jobs report/status, run_checks hosted/secret-scan"
provides:
  - "results/phase4-control: control-always-comply re-run under the current engine, 342/342, complete-ok (untracked)"
  - "results/phase4-report: merged Phase 4 report, 9 sources, regenerated at the 2026-10-29T03:06Z snapshot (untracked)"
  - "run_checks viva: page structure, numbers from summary.csv episodes_done/episodes_expected, secret patterns, README row"
  - "docs/viva/04-namespaces-seccomp-and-local-inference.md (566 lines) and its README row 'written (Phase 4)'"
affects: [05-report]

actuals:
  tokens: 14000          # chars/4 over the realized diff 1104ea9..0a3881b (56,239 chars)
  tasks: 3
  commits: 2             # git rev-list --count 1104ea9..HEAD before the docs commit
plan_head_before: 1104ea98e2b0513fdcacc4a7ebee6499c1260033
plan_head_after: 0a3881ba6b1bccc2bb1ceac52ed0b65ff34d4112

tech-stack:
  added: []
  patterns:
    - "A viva page is checked mechanically against the generated summary.csv, so its numbers cannot drift from the report"

key-files:
  created:
    - docs/viva/04-namespaces-seccomp-and-local-inference.md
  modified:
    - src/benchmark/run_checks.py
    - tests/test_run_checks.py
    - docs/viva/README.md

key-decisions:
  - "Task 3 ran on a snapshot (2026-10-29T03:06Z) by user decision instead of waiting for the 15:30 UTC freeze; the Groq loop (WSL pid 10209) was left running and was never signalled"
  - "Final Groq regeneration of results/phase4-report and the Part C update are a pending follow-up after the freeze; CMP-03 stays open until then"
  - "CMP-01 and CMP-02 are complete: all five local models are 342/342 in the merged report"

requirements-completed: [CMP-01, CMP-02]
requirements-advanced: [CMP-03]

duration: ~1h05m (02:18Z to 03:25Z on 2026-10-29, two sessions)
completed: 2026-10-29
---

# Phase 4 Plan 04: Control re-run, merged report and viva page 04 Summary

**The scripted control was re-run under the current engine (342/342, complete-ok). The merged Phase 4 report shows all five local models at 342/342, each with 0/120 attack success under guardrails. The Phase 4 viva page passes `run_checks viva` against that report. The page quotes the Groq rows from a snapshot taken at 2026-10-29T03:06Z. The Groq re-run keeps going until the 15:30 UTC freeze. After the freeze, the report must be regenerated and Part C updated. That follow-up is pending.**

## Performance

- **Task 1 (control and merged report):** 2026-10-29 around 02:20Z
- **Task 2 (viva check and page):** commit 2b6de22
- **Task 3 (snapshot report, Part C, full suite):** 03:06Z report and hosted check; page commit 0a3881b; suite done around 03:20Z
- **Tasks:** 3/3. Task 3 ran on a snapshot, not at the freeze (see Deviations).

## Control (Task 1, results/phase4-control)

- `run_checks complete --expect 342`: `finished 342/342 pending_infra 0 duplicates 0 policy c8afbb5d345a commits 1` and then `complete-ok`. The run is stamped at commit 1104ea98.
- Unprotected arm: 120/120 attacks attempted and 120/120 succeeded.
- Protected arm: 0/120 attacks succeeded. Blocked given attempt was 120/120 [96.9, 100.0]. Counting each case once, it was 40/40 [91.2, 100.0]. Both ds and dh were 60/60.
- Blocks by layer: tool_broker in 120/120 episodes (180 actions), policy_engine in 60/120 (60 actions), sandbox 0/120 and seccomp 0/120.

## Merged report at the snapshot (results/phase4-report, 2026-10-29T03:06:04Z)

`jobs report --prefix results/phase4 --out results/phase4-report` merged 9 sources: the three Groq models, the five local models and the control. It skipped the jobs, tracer, smoke and report directories. `grep -c "342/342" summary.md` returned 5.

**Headline (complete):**

| Model | Done | Attempted (no gr.) | Success (no gr.) | Success (gr.) | Benign (no / with gr.) | FP (gr.) |
|---|---|---|---|---|---|---|
| ollama-llama3.2-3b | 342/342 | 14/120 | 7/120 | 0/120 | 48/51, 50/51 | 0/51 |
| ollama-qwen2.5-3b | 342/342 | 5/120 | 3/120 | 0/120 | 49/51, 49/51 | 0/51 |
| scratch-gpt | 342/342 | 39/120 | 33/120 | 0/120 | 36/51, 36/51 | 0/51 |
| qwen2.5-0.5b-base | 342/342 | 6/120 | 3/120 | 0/120 | 15/51, 15/51 | 0/51 |
| qwen2.5-0.5b-agentlora | 342/342 | 111/120 | 111/120 | 0/120 | 51/51, 51/51 | 0/51 |

**Partial rows (snapshot; not compared):**

| Model | Status | Done | Paired attack | Success (no gr.) | Success (gr.) | Benign n | FP (gr.) |
|---|---|---|---|---|---|---|---|
| groq-gpt-oss-120b | partial | 240/342 | 92 | 4/92 | 0/92 | 28 | 1/28 |
| groq-gpt-oss-20b | error-dominated | 109/342 | 47 | 2/47 | 0/47 | 7 | 0/7 |
| groq-qwen3.8-27b | partial | 246/342 | 92 | 0/92 | 0/92 | 30 | 0/30 |

Pooled conditional block rate over all language models: 200/200 [98.1, 100.0].

## Groq snapshot (Task 3, `run_checks hosted` at 2026-10-29T03:06:38Z)

| Model | State | n | ds n | Complete pairs | Paired attack | Quota stops | Pending infra |
|---|---|---|---|---|---|---|---|
| groq-gpt-oss-120b | running (rate-limit waits) | 240/342 | 120/120 | 120 | 92 | 0 | 102 |
| groq-gpt-oss-20b | waiting_quota until 03:56:56Z | 109/342 | 82/120 | 54 | 47 | 1 (2026-10-29T02:56:56Z) | 233 |
| groq-qwen3.8-27b | waiting_quota until 03:33:36Z | 246/342 | 120/120 | 122 | 92 | 1 (2026-10-29T02:33:36Z) | 96 |

- `run_checks hosted` printed `hosted-running`. The loop was alive as pid 10209, and the freeze was 12h23m away.
- There were no hosted relaunches by this plan. The executor never signalled the loop or its children and never wrote into results/phase4-groq-*.
- These numbers are a snapshot, not final. The loop continues under the committed rule until the 2026-10-29 15:30 UTC freeze (D-09).

## EVAL-08 at the snapshot (sandbox and seccomp, protected arm)

These rows are as printed in the report:

- Each of the five local models: attack 0.0% (0/120) [0.0, 3.1]; 0 actions; benign 0.0% (0/51) [0.0, 7.0]; 0 actions. This holds for both the sandbox and seccomp layers.
- groq-gpt-oss-120b (partial): attack 0/92 [0.0, 4.0], benign 0/28 [0.0, 12.1].
- groq-gpt-oss-20b (partial): attack 0/47 [0.0, 7.6], benign 0/7 [0.0, 35.4].
- groq-qwen3.8-27b (partial): attack 0/92 [0.0, 4.0], benign 0/30 [0.0, 11.4]. All Groq zeros have 0 actions and hold for both layers.
- Pooled: attack 0.0% (0/831) [0.0, 0.5]; 0 actions; benign 0.0% (0/320) [0.0, 1.2]; 0 actions, for both layers.
- Report sentences: sandbox 0/1151 and seccomp 0/1151 protected-arm episodes of 8 language models. The earlier report gave 0/1147 (attack 0/830, benign 0/317). The page now quotes 1151, 831 and 320.
- Pooled application layers: tool broker 239/831 (245 actions), policy engine 48/831 (58 actions). In benign episodes the policy engine blocked 1/320. That is the one gpt-oss-120b false positive, and the page states it.

## Checks

- `run_checks secret-scan` over the three Groq directories, results/phase4-jobs and the page: `secret-scan-ok files 36`.
- After the edit, the same scan over the page, the README, results/phase4-control and results/phase4-report gave `secret-scan-ok files 11`.
- `run_checks viva` against results/phase4-report/summary.csv printed `viva-numbers-ok`. Against the pre-Phase-1 results/final-20261011/summary.csv it failed as expected with "lacks episodes_done/episodes_expected".
- `run_checks complete` for the control printed `complete-ok`.

## Full test suite (Task 3, four calls)

| Call | Files | Result | Runtime |
|---|---|---|---|
| 1 | Phase 1-3 suite, ignoring the Qwen files and the Phase 4 files | 935 passed | 234 s |
| 2 | test_hf_local_provider.py, test_qwen_lora_data.py | 66 passed, 1 skipped | 124 s |
| 3 | test_qwen_lora_train.py, test_qwen_lora_checks.py | 94 passed | 224 s |
| 4 | test_jobs.py, test_run_checks.py, test_phase4_runner.py | 147 passed | 11 s |

Total: 1242 passed, 1 skipped, 0 failed.

## Task Commits

1. **Task 1: Control re-run and merged report.** No tracked file changed. The outputs are results/phase4-control and results/phase4-report, both untracked.
2. **Task 2: Viva check and viva page.** `2b6de22` (docs) changed src/benchmark/run_checks.py, tests/test_run_checks.py, the viva page and docs/viva/README.md.
3. **Task 3: Part C on the 03:06Z snapshot.** `0a3881b` (docs) changed the viva page only.

## Deviations from Plan

### User-directed

**1. Task 3 ran on a snapshot instead of waiting for hosted-terminal-ok or the freeze.**
- **Why:** The user chose not to wait for Groq because the viva is close.
- **What changed:** The report was regenerated at 2026-10-29T03:06Z. Part C's Groq rows quote that snapshot. The "PROVISIONAL" marker was replaced with: "snapshot at 2026-10-29T03:06Z (UTC); the Groq re-run continues until the 2026-10-29 15:30 UTC freeze (D-09), then the report is regenerated".
- **Acceptance impact:** The criterion "Groq n values equal the final summary.csv" is met only for the snapshot report. hosted-terminal-ok has not been reached: the check printed `hosted-running`.
- **Follow-up (pending):** After the freeze, run `jobs report --prefix results/phase4 --out results/phase4-report`, the secret-scan and `run_checks hosted`. Then update Part C's Groq rows and the EVAL-08 pooled n, re-run `run_checks viva` and tick CMP-03. This is recorded in .planning/WINDOWS.md as entry 2 (deviation).

### Auto-fixed Issues

None. The plan was otherwise executed as written.

## Known Stubs

- docs/viva/04-namespaces-seccomp-and-local-inference.md, line 431: the section "What the code review found and what we fixed" is a placeholder until the Phase 4 code review runs. This is intentional: the review comes after execution, and nothing on the page depends on it. It is recorded in .planning/WINDOWS.md as entry 3.

## Next Steps

- After the 15:30 UTC freeze, regenerate the report and update the Groq rows in Part C. CMP-03 closes then.
- Phase 4 code review and verification. After the review, fill in the page's code-review section.

## Self-Check: PASSED

- FOUND: docs/viva/04-namespaces-seccomp-and-local-inference.md, results/phase4-report/summary.csv, results/phase4-control
- FOUND: 2b6de22 and 0a3881b are ancestors of HEAD
