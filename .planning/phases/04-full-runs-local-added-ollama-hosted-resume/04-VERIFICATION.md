---
phase: 04-full-runs-local-added-ollama-hosted-resume
verified: 2026-10-29T04:00:00Z
status: human_needed
score: 5/5 must-haves verified (criterion 3 verified as an interim partial, by user decision)
covered_files:
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-01-PLAN.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-01-SUMMARY.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-02-PLAN.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-02-SUMMARY.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-03-PLAN.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-03-SUMMARY.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-04-PLAN.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-04-SUMMARY.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-05-PLAN.md"
  - ".planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-05-SUMMARY.md"
  - "config/models.yaml"
  - "docs/viva/04-namespaces-seccomp-and-local-inference.md"
  - "docs/viva/README.md"
  - "run_benchmark.py"
  - "src/agent_harness/loop.py"
  - "src/benchmark/jobs.py"
  - "src/benchmark/outcomes.py"
  - "src/benchmark/report.py"
  - "src/benchmark/run_checks.py"
  - "src/benchmark/runner.py"
  - "tests/test_control_agent.py"
  - "tests/test_jobs.py"
  - "tests/test_layer_blocks_report.py"
  - "tests/test_phase4_runner.py"
  - "tests/test_provenance.py"
  - "tests/test_run_checks.py"
covered_digest: "v3:sha256:a65cd030adbff0db47fd1bed12c6c950ef99695ec96bdee8f4a2f1652a2f0eac"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "After the 2026-10-29 15:30 UTC freeze, regenerate the merged report (python -m src.benchmark.jobs report --prefix results/phase4 --out results/phase4-report) and update docs/viva/04 Part C with the final Groq n; then re-run run_checks viva"
    expected: "run_checks viva prints viva-numbers-ok against the regenerated summary.csv. Today it does NOT: against the current results/phase4-report/summary.csv it prints FAIL viva: the page does not quote groq-qwen3.8-27b with 254/342 on one line. The page quotes the 03:06Z snapshot (246/342); the report was regenerated at 03:37Z (254/342). Against a copy of the csv with qwen3.8 set back to 246 it prints viva-numbers-ok. Tracked as WINDOWS.md #2 (open)."
    why_human: "CMP-03 is partial by user decision (do not wait for the Groq loop). The final numbers do not exist yet, so the page cannot match the report until the freeze. This is the expected drift, not a defect, but the 'passes run_checks viva' claim is true only against the snapshot."
  - test: "Fill the viva section 'What the code review found and what we fixed' (docs/viva/04, line 431) from 04-REVIEW.md and 04-REVIEW-FIX.md"
    expected: "The text 'To be filled after the Phase 4 code review. The review has not run yet' is replaced; the review has run (1 critical, 5 warnings, 7 info; 5 fixed, WR-01 deferred). Tracked as WINDOWS.md #3 (open)."
    why_human: "Planned follow-up recorded in 04-04-SUMMARY; the page text is now factually stale. Not a roadmap criterion and not a debt marker, so a warning, not a gap."
  - test: "Post-freeze, apply or consciously close the deferred run-affecting review items WR-01, IN-04, IN-05 (WINDOWS.md #4)"
    expected: "Decision recorded. Evidence that WR-01 has no effect on the Phase 4 data today: a scan of every results/phase4-* raw.jsonl found 0 rows with stop_reason context_overflow and 0 llm_error rows mentioning overflow, so the format-valid-ASR bias it describes touched no Phase 4 episode."
    why_human: "Deferred by user decision until after the freeze because the Groq loop stamps those paths; a human must decide whether to fix for Phase 5 or accept."
  - test: "Read the review-fix items marked 'fixed: requires human verification' (CR-01 loader ranking in report.py, WR-04 scheduler state in jobs.py)"
    expected: "Logic matches 04-REVIEW-FIX.md. Automated evidence: 260 tests in test_jobs, test_run_checks, test_phase4_runner, test_layer_blocks_report, test_report, test_control_agent and test_provenance pass; a report regenerated from the live data with the current code reproduces the on-disk results/phase4-report counts exactly (240, 109, 254 Groq; 342 for the other six)."
    why_human: "The fixer asked for a human read of record-selection and scheduler state handling."
  - test: "unverified-prohibition, human review recommended: 04-01, 04-03, 04-05 'Never stop, signal or restart a process this project did not start'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS on the evidence available. I sent no signal; pid 10209 (jobs hosted) is still alive after about 9 h 10 min, state Ss; the Phase 4 review-fix report records WR-02 (re-check ownership before SIGKILL) as fixed. Process history cannot be proven from files alone."
    why_human: "Judgment-tier prohibition, plan status unverified / flagged. Never a silent pass."
  - test: "unverified-prohibition, human review recommended: 04-01 'No benchmark setting changes in Phase 4 beyond the three planned runner changes'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. git diff over every run-affecting path (agent_harness, llm, policy_engine, sandbox, config, benchmarks, runner.py, outcomes.py, scoring.py, run_benchmark.py) is empty between 3b60b93e (the LoRA row stamp), d4ee817a (Groq stamp), 09a6638a (local stamp), 3e4dae36 and HEAD. policy_sha256 c8afbb5d345a is the same on every row; case_set_sha256, 57 cases, 3 runs, 2 arms are the same in every manifest. The report's 'MIXED' commit label on the Groq rows is therefore cosmetic."
    why_human: "Judgment-tier prohibition about process order, plan status unverified / flagged."
  - test: "unverified-prohibition, human review recommended: 04-02 'No attack command, bypass recipe or payload text is written into the report ... the Phase 1 table is cited by row number and outcome class only'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. The EVAL-08 paragraph in results/phase4-report/summary.md cites rows 1, 2, 8, 9, 10, 15, 16 (seccomp), 20, 21, 22 (sandbox) and the contained rows by number; I compared each against the table in 01-REVIEW-FIX.md lines 227-249 and they match. No command text appears."
    why_human: "Judgment-tier prohibition about wording, plan status unverified / flagged."
  - test: "unverified-prohibition, human review recommended: 04-03 and 04-05 'No unavailable Groq model replaced, results/final-20261011 never resumed or written, the freeze is never moved'; 04-04 'partial hosted rows never ranked or extrapolated'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. The three Groq model ids are unchanged in every row's resolved_model; results/final-20261011 newest mtime is 2026-10-12; FREEZE_UTC in src/benchmark/jobs.py:108 is 2026-10-29T15:30:00+00:00 and equals stopping_rule.json; the headline table lists only the five complete local models and the Groq rows sit under 'Partial or error-dominated runs (not compared)'."
    why_human: "Judgment-tier prohibitions, plan status unverified / flagged."
  - test: "Read docs/viva/04-namespaces-seccomp-and-local-inference.md (Parts A, B, C, D) as a first-time reader"
    expected: "Terms defined before use, the Part C numbers read naturally with the snapshot label, 20+ Q&A answers speakable in under a minute."
    why_human: "Readability and teaching quality cannot be checked by grep. run_checks viva proves structure, secret scan and per-model count quotes only."
---

# Phase 4: Full Runs - Local, Added Ollama, Hosted Resume Verification Report

**Phase Goal:** Every model in the comparison has its complete, like-for-like benchmark data (or an honestly labelled partial), and the report counts, with k/n per model and pooled, the sandbox and seccomp blocks those runs recorded, stating any zero plainly and citing the Phase 1 residual-bypass table as the evidence of what those layers contain.
**Verified:** 2026-10-29T04:00Z (the Groq loop was still running; its numbers are a moving snapshot)
**Status:** human_needed (all automated truths verified; CMP-03 is partial by user decision, and the follow-ups below need a person or the freeze)
**Re-verification:** No, initial verification

The goal is achieved for everything that can be achieved before the freeze. Nothing below is taken from SUMMARY.md: I recomputed the per-model counts from the raw.jsonl files with my own script, re-ran the repository's own gate commands in WSL, regenerated the merged report into a scratch directory with the current code and compared it with the on-disk report, and ran the 260 phase tests. I sent no signal to any process (pid 10209 was alive at 03:57Z) and wrote nothing under `results/phase4-groq-*`.

## Goal Achievement

### Observable Truths (ROADMAP success criteria, the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | scratch-gpt, Qwen base and Qwen LoRA each complete 57 cases x 3 runs x 2 arms = 342 episodes, shown 342/342 in the completeness column | VERIFIED | My own count over `results/phase4-{scratch-gpt,qwen2.5-0.5b-base,qwen2.5-0.5b-agentlora}/raw.jsonl`: 342 finished, 57 cases, 3 runs each. `run_checks complete --expect 342` printed `complete-ok` for each: 0 pending infra rows, 0 duplicates, 1 commit, policy c8afbb5d345a. Resolved stamps: `scratch-gpt@97bdace56294`, `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf`, `...+qwen2.5-0.5b-agentlora@924ac021f0cb`. Report `summary.csv` episodes_done/episodes_expected = 342/342 for all three. |
| 2 | qwen2.5:3b completes through Ollama with the context raised (verified by `ollama ps` and logged `prompt_tokens`); llama3.2:3b is re-checked for truncation | VERIFIED | `ollama-qwen2.5-3b` and `ollama-llama3.2-3b` 342/342, resolved ids `qwen2.5-3b-ctx8192` and `llama3.2-3b-ctx8192` (Modelfile variants with num_ctx 8192). `run_checks ollama-evidence` printed `ollama-evidence-ok` for both (ollama ps start and end files present; measured server default 4096, server 0.40.3, `default-context-ok`). `run_checks truncation --num-ctx 8192 --compare 4096`: llama3.2 1,251 steps, max prompt_tokens 1042, 0 steps at or above 95 % of the window, **0/342 episodes with a step above the 4096 default (k=0/342)**, 0 length finishes; qwen2.5-3b 951 steps, max 1002, 0/342, 1 length finish (a token-cap stop, not truncation). Both models were re-run in fresh directories, not just re-checked. |
| 3 | Hosted Groq models re-run under the current engine, ds first, in fresh directories, under the written stopping rule (hard freeze 2026-10-29 15:30 UTC plus QuotaExhausted handling); whatever is incomplete is reported as partial with its n | VERIFIED as an interim partial (by user decision; the loop is still running) | Rule committed (d4ee817, 18:47Z) before the loop started (18:48:03Z); `stopping_rule.json` freeze equals `FREEZE_UTC` (jobs.py:108). Fresh directories `results/phase4-groq-*`; October rows in `results/final-20261011` untouched (newest mtime 2026-10-12). ds first held: my count of finished episodes at 04:00Z: gpt-oss-120b 240/342 (ds 120/120, dh 64, benign 56); gpt-oss-20b 113/342 (ds 86/120; was 109 in the report); qwen3.8-27b 254/342 (ds 120/120). The report labels them partial (120b, qwen3.8) and error-dominated (20b, many infra rows after an overnight outage) and keeps them out of the headline table. **n is not final.** The loop (pid 10209, elapsed about 9 h 10 min) was in `waiting_quota` for 120b and qwen3.8 and `running` for 20b. Per the user, this is not counted as a gap and does not block Phase 5; the final n arrives after the freeze. |
| 4 | The report's where-blocked-by-layer table counts sandbox and seccomp blocks in the Phase 4 full runs, per model and pooled, with k/n; cites the Phase 1 residual-bypass table by row number; a layer with no blocks is reported as zero with its n (D-07b: no new attack suite) | VERIFIED (honest zeros) | `results/phase4-report/summary.md` section "Blocks by layer, per model and pooled (EVAL-08)" exists, with a k/n [Wilson] cell per model, per attack/benign group, per layer, plus a pooled row; `layer_blocks.csv` carries the same data. Sandbox and seccomp are **0/834 attack and 0/321 benign pooled; 0 of 1155 protected-arm episodes of 8 language models** (text printed by the report), and 0 for every model. My own scan of the protected rows' `blocked_by` field found 0 sandbox and 0 seccomp blocks in every Phase 4 directory. The report cites rows 1, 2, 8, 9, 10, 15, 16 (seccomp), 20, 21, 22 (sandbox) and 3-7, 11-14, 17-19, 23 (contained, no error signature) of 01-REVIEW-FIX.md; I checked each against the table at lines 227-249 and all match, and states the counts are a lower bound. The zeros are not a missing layer: `build_executor` puts `SandboxExecutor` under the protected arm (runner.py:51-53) and the runs were under WSL; attacks were stopped earlier by the tool broker (245 blocked actions in 239 of 834 pooled attack episodes) or the policy engine (58 actions in 48 episodes). No attack text is in the report. Protected-arm control: 120/120 attempts blocked, 0 reach sandbox or seccomp, and the report says that section is not evidence for those layers. |
| 5 | Viva page(s) for namespaces, seccomp-bpf and local-inference confounds added to `docs/viva/` | VERIFIED (readability: human; check passes only against the snapshot) | `docs/viva/04-namespaces-seccomp-and-local-inference.md`, 566 lines, Parts A-D plus "Limits of this phase"; README row 04 marked "written (Phase 4)". `run_checks secret-scan` over the page and the report: `secret-scan-ok files 3`. `run_checks viva` against the **current** `results/phase4-report/summary.csv`: **FAIL** (page quotes groq-qwen3.8-27b as 246/342, the report now says 254/342). Against a copy with that one value set to the snapshot 246, it prints `viva-numbers-ok`; every other model line (including the five 342/342) matches. See human item 1 and the Gaps Summary. |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified). Criterion 3 counts as verified only in the sense the roadmap defines it: partial with its n, reported honestly; the final n is pending.

### The user's checklist, against the data

| Claim | Result | Evidence |
|-------|--------|----------|
| 5 local models at 342/342 | CONFIRMED | Own count and `complete-ok` for all five; csv rows 342/342. |
| Protected ASR 0/120 | CONFIRMED for all five local models, and also for all three partial Groq models (0/92, 0/48, 0/95 in my count; 0/47 in the report for 20b after pair dropping) | Own count of `succeeded` over protected dh+ds rows: 0 each. Unprotected for contrast: LoRA 111/120, scratch-gpt 33/120, llama3.2 7/120, qwen2.5-3b 3/120, Qwen base 3/120; control 120/120. |
| llama re-check k=0/342 | CONFIRMED | `COMPARE episodes with a step above 4096: 0/342`; max prompt_tokens 1042. |
| EVAL-08 section exists with honest zeros | CONFIRMED | Truth 4. |
| Viva page 04 exists and passes `run_checks viva` | PARTLY: page exists, but the check **fails today** against the live report (254 vs 246 for one Groq row). It passed against the 03:06Z snapshot. | Truth 5. |

### Plan-level checks (not reduced from the ROADMAP)

| Plan-level truth | Status | Evidence |
|-----------------|--------|----------|
| Like-for-like engine across models | VERIFIED | All run-affecting paths (`src/agent_harness`, `src/llm`, `src/policy_engine`, `src/sandbox`, `config`, `benchmarks`, `runner.py`, `scoring.py`, `outcomes.py`, `run_benchmark.py`) have an empty `git diff` between every stamped commit (3b60b93e, d4ee817a, 09a6638a, 3e4dae36) and HEAD. One policy hash and one case-set hash on every row. |
| Review fixes did not invalidate runs | VERIFIED | `git diff 3e4dae3..HEAD --name-only` touched only report.py, run_checks.py, jobs.py and three test files, none in `RUN_AFFECTING_PATHS`. |
| Report is current after the CR-01 loader fix | VERIFIED | Regenerated `jobs report --prefix results/phase4` into the scratchpad with the current code: all nine rows identical to the on-disk `summary.csv` (episodes_done and status). |
| Resume and infra-error handling (CR-01, WR-04: behavior-dependent state transitions) | VERIFIED | Named behavioural tests: the parametrized CR-01 test over four record sequences and the WR-04 scheduler tests pass inside the 260. Not left PRESENT_BEHAVIOR_UNVERIFIED. |
| Control rerun in a fresh directory | VERIFIED | `results/phase4-control`: 342/342, stamped commit 1104ea9; 120/120 unprotected success, 0/120 protected. |
| 22 control-token strings absent from benchmark text (Phase 3 IN-01) | VERIFIED | `run_checks control-tokens`: strings 22, texts 308, matches 0. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `results/phase4-{scratch-gpt,qwen2.5-0.5b-base,qwen2.5-0.5b-agentlora,ollama-qwen2.5-3b,ollama-llama3.2-3b}` | five complete local runs | VERIFIED | 342 rows each, bench.done, manifest, progress.log; Ollama dirs also hold ollama_ps start/end and long-prompt evidence |
| `results/phase4-groq-*` | three fresh hosted runs | VERIFIED (partial, live) | read only by me; hosted_state.json and progress.log present |
| `results/phase4-control` | model-independent control | VERIFIED | 342/342 |
| `results/phase4-report/{summary.md,summary.csv,layer_blocks.csv,charts}` | merged report with EVAL-08 table | VERIFIED | 9 sources; regenerated at 03:37Z |
| `docs/viva/04-namespaces-seccomp-and-local-inference.md`, `docs/viva/README.md` | viva page and index row | VERIFIED with warnings | 566 lines; stale code-review section; snapshot numbers |
| `src/benchmark/jobs.py`, `run_checks.py`, `report.py`, `runner.py`, `outcomes.py`, `run_benchmark.py`, `config/models.yaml` | tooling and fixes | VERIFIED | exercised above; 260 tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `jobs hosted` loop | `run_benchmark` children, `stopping_rule.json` | relaunch, freeze constant | WIRED | pid 10209 alive, 3 launches recorded per model, quota stops parsed (`waiting_quota`) |
| `jobs report` | `generate_report(sources=...)` | nine exact directory names, never tracer/smoke/jobs | WIRED | reproduced into the scratchpad |
| `report.py` layer table | `blocked_by` on raw rows | per-layer k/n with Wilson interval | WIRED | my own scan agrees (0 and 0) |
| `report.py` EVAL-08 text | `01-REVIEW-FIX.md` | row numbers | WIRED | rows checked against the table |
| `run_checks viva` | `summary.csv` and the page | per-model `<done>/<expected>` on one line | WIRED but stale | works; fails on the one drifted Groq row |
| protected arm | `SandboxExecutor` | `build_executor` | WIRED | runner.py:51-53 |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| `summary.md` / `summary.csv` | counts, k/n | `raw.jsonl` of nine directories | Yes; my independent recount matches (342s, 240/109/254, 0 blocks) | FLOWING |
| `layer_blocks.csv` | per layer k, n | `blocked_by` in protected rows | Yes; zeros are real zeros, not a missing source | FLOWING |
| viva Part C | numbers | copied from the 03:06Z report | Yes, but frozen at the snapshot | FLOWING (stale for Groq) |

### Behavioral Spot-Checks (my own runs, WSL, venv python)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| five local models complete | `run_checks complete --results results/phase4-<m> --model <m> --expect 342` | `complete-ok` x5 | PASS |
| no truncation, llama | `run_checks truncation ... --num-ctx 8192 --compare 4096` | k=0/342, max 1042 | PASS |
| no truncation, qwen2.5-3b | same | 0/342, max 1002 | PASS |
| Ollama window evidence | `run_checks ollama-evidence` | `ollama-evidence-ok` x2 | PASS |
| default window measured | `run_checks default-context` | `default-context-ok` (4096) | PASS |
| control tokens | `run_checks control-tokens` | `control-tokens-ok` | PASS |
| secret scan | `run_checks secret-scan --paths <page> <summary.md> <summary.csv>` | `secret-scan-ok files 3` | PASS |
| viva vs current report | `run_checks viva --page ... --report-csv results/phase4-report/summary.csv` | FAIL: groq-qwen3.8-27b 254/342 not on the page | FAIL (expected drift, see warning) |
| viva vs snapshot csv | same with qwen3.8 set to 246 in a scratchpad copy | `viva-numbers-ok` | PASS |
| phase test suites | pytest test_jobs, test_run_checks, test_phase4_runner, test_layer_blocks_report, test_report, test_control_agent, test_provenance | 260 passed | PASS |
| independent episode recount | scratchpad script over every `results/phase4-*/raw.jsonl` | local 342 each; protected ASR 0/120 each; 0 sandbox, 0 seccomp | PASS |

I did not run the full repository suite; the caller records it separately.

### Probe Execution

SKIPPED. No `probe-*.sh` script exists or is declared by the Phase 4 plans. The `run_checks` subcommands above are the phase's own gate commands and were run.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CMP-01 | 04-01, 04-03 | qwen2.5:3b full run via Ollama with the window raised; llama3.2:3b re-checked (phi3.5, gemma3 to v2) | SATISFIED | truth 2 |
| CMP-02 | 04-03 | scratch-gpt, Qwen base, Qwen LoRA complete 57 x 3 x 2 | SATISFIED | truth 1 |
| CMP-03 | 04-03, 04-05 | Groq models re-run fresh, ds first, under the written rule; incomplete reported as partial with n | PARTIAL by user decision (REQUIREMENTS.md leaves it `[ ]` Pending; this is correct). Interim n: 120b 240/342, 20b 113/342, qwen3.8 254/342. Not a blocker for Phase 5. | truth 3 |
| EVAL-08 | 04-02, 04-04 | per-model and pooled sandbox and seccomp block counts with k/n, cites Phase 1 table, zero stated with n (D-07b) | SATISFIED | truth 4 |

No orphaned requirements: REQUIREMENTS.md maps exactly CMP-01, CMP-02, CMP-03 and EVAL-08 to Phase 4, and all four appear in the plans.

### Anti-Patterns Found

No `TBD`, `FIXME` or `XXX` in any covered file or the viva page. The viva page contains the stale phrase "To be filled after the Phase 4 code review" (see below); it is not one of the gated markers.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `docs/viva/04-namespaces-seccomp-and-local-inference.md` | 431 | Placeholder section says the review "has not run yet"; it has (04-REVIEW.md, 04-REVIEW-FIX.md) | Warning | The page misleads a reader. WINDOWS.md #3, planned follow-up. |
| `docs/viva/04-namespaces-seccomp-and-local-inference.md` | 11, 289-304, 396-418 | Groq numbers quote the 03:06Z snapshot (246/342 for qwen3.8, 1151 protected episodes); the report now says 254/342, 1155 | Warning | The page is labelled as a snapshot throughout, so it is honest, but `run_checks viva` fails against the live report. WINDOWS.md #2. |
| `src/benchmark/outcomes.py` (WR-01, IN-04), `runner.py`, `run_benchmark.py` (IN-05) | n/a | Deferred review findings | Info | Run-affecting, deferred by user decision until after the freeze (WINDOWS.md #4). No Phase 4 row contains a context overflow, so no current result is affected. |
| `src/dashboard/app.py` | n/a | Uncommitted change in the working tree | Info | Outside Phase 4's scope and the review's file list; not touched by this verification. |
| `results/` | n/a | Entire directory untracked | Info | Evidence exists on disk, not in git; the user's working-tree decision, same as Phase 3. |
| Report provenance | n/a | Groq rows show `MIXED` code commits and `dirty tree yes` | Info | Cosmetic: run-affecting paths are identical across all stamped commits (empty diff). `code_dirty` is true on every row, as in Phase 3. |

### Human Verification Required

See the `human_verification` list in the frontmatter. In short: (1) after the freeze, regenerate the report and update viva Part C so `run_checks viva` passes again; (2) fill the viva code-review section; (3) decide the deferred WR-01 / IN-04 / IN-05; (4) read the CR-01 and WR-04 fixes; (5) five flagged judgment-tier prohibitions (non-authoritative verdict: all hold); (6) viva readability.

### Gaps Summary

No blocking gaps. All five ROADMAP success criteria and all four requirement IDs are met by the data on disk, and the numbers the caller asked me to check reproduce exactly: five local models at 342/342, protected ASR 0/120 on each, the llama3.2 re-check at k=0/342 with a maximum of 1042 prompt tokens against a 4096 default, and an EVAL-08 section that reports 0 of 1155 sandbox and 0 of 1155 seccomp blocks with n and cites Phase 1 rows by number.

Two things differ from the stated state, both warnings and both already in the Broken Windows ledger:

1. `run_checks viva` does not pass today. The report was regenerated at 03:37Z (after the review-fix CR-01 commit) and now shows groq-qwen3.8-27b at 254/342, while the page, written against the 03:06Z snapshot, quotes 246/342. The SUMMARY's "regenerated at the 03:06Z snapshot" is also stale for the same reason. The page labels every Groq number as a snapshot, so it is honest, and the check passes against the snapshot values. It will pass again once the page is updated from the post-freeze report.
2. The viva "What the code review found and what we fixed" section is still the placeholder, and says the review has not run.

CMP-03 stays partial with an interim n (240, 113 and 254 of 342) because the Groq loop runs to the 15:30 UTC freeze; per the user's decision that is reported, not counted as a gap, and does not block Phase 5. Status is `human_needed` because the frontmatter list contains the follow-ups above and five flagged judgment-tier prohibitions, none of which I found failing.

---

_Verified: 2026-10-29T04:00:00Z_
_Verifier: Claude (gsd-verifier)_
