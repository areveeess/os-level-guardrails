---
phase: 04-full-runs-local-added-ollama-hosted-resume
plan: 05
subsystem: benchmark-infra
tags: [queue, freeze, stopping-rule, groq, resumable-runs, context-overflow, ds-first, secret-scan]
status: complete

requires:
  - phase: 04-full-runs-local-added-ollama-hosted-resume
    provides: "04-01: jobs run/preflight/ollama_up/ollama_down, run_checks complete_status, measured default context 4096"
provides:
  - "jobs.FREEZE_UTC = 2026-10-29T15:30:00+00:00 (D-09); run(model, out_dir, smoke, jobs_dir, deadline); done code 7"
  - "python -m src.benchmark.jobs local | status | report | hosted-probe | hosted"
  - "python -m src.benchmark.run_checks hosted | secret-scan"
  - "run_benchmark.py --priority-kinds (benign, dh, ds) and runner.run_benchmark(..., priority_kinds=())"
  - "outcomes.classify_llm_error -> 'context_overflow' (finished, never retried)"
affects: [04-03, 04-04, 05-report]

actuals:
  tokens: 26500          # chars/4 over the realized diff 64f1cf8..2759af1 (106,034 chars)
  tasks: 3
  commits: 3             # git rev-list --count 64f1cf8..2759af1
plan_head_before: 64f1cf88a0d8e87dcc2a8ab7b2b4778db077fdda
plan_head_after: 2759af147503eb8da5f2e6b7a6528e5eb1504971

tech-stack:
  added: []
  patterns:
    - "One injectable wall clock (jobs._now) for every freeze, staleness and quota comparison"
    - "Signals only through the Popen handle the same process created (stop_own_child); every other process lookup is a read-only /proc scan"
    - "Single instance per prefix: /proc cmdline scan that excludes this process and its ancestors (timeout/bash wrappers)"
    - "Per-model hosted_state.json plus a write-once stopping_rule.json, so a restarted loop resumes its decisions"

key-files:
  created:
    - tests/test_phase4_runner.py
  modified:
    - src/benchmark/jobs.py
    - src/benchmark/run_checks.py
    - src/benchmark/runner.py
    - src/benchmark/outcomes.py
    - src/agent_harness/loop.py
    - run_benchmark.py
    - tests/test_jobs.py
    - tests/test_run_checks.py

key-decisions:
  - "Local queue order is qwen2.5-0.5b-agentlora, scratch-gpt, qwen2.5-0.5b-base, ollama-qwen2.5-3b, ollama-llama3.2-3b (shortest GPU runs first for the D-09 freeze)"
  - "A local model's context overflow is a finished format_failure (attacked if the attack was attempted), never re-queued; only infra is retried"
  - "--priority-kinds is order only: recorded in manifest extra, not compared on resume; the default queue is unchanged"
  - "Hosted loop applies only D-09: hard freeze plus QuotaExhausted (retry + 120 s, or 60 min if no retry is stated; a next launch at or after the freeze ends the model as stopped_by_quota); other exits relaunch after 10 min; 3 zero-progress launches = stalled"
  - "status 'alive' means a run_benchmark child is running on that directory; running queues and loops are shown on separate LOOP lines"

requirements-completed: []          # CMP-02/CMP-03 need the full runs (04-03); this plan only builds the tooling
requirements-advanced: [CMP-02, CMP-03]

duration: 24min
completed: 2026-10-28
---

# Phase 4 Plan 05: Freeze-aware queue, runner fixes and hosted loop Summary

**The Phase 4 run tooling is ready for 04-03. A serial local queue respects the D-09 freeze and was tested on real hardware. Status is read-only and warns when a run stops making progress. The runner can no longer be stopped by a non-text reply or by a deterministic context overflow, and it can run data-stealing episodes first. A Groq loop applies the freeze plus quota rule and signals only its own children. All of it was built and tested without sending a hosted request.**

## Performance

- **Duration:** about 24 min
- **Started:** 2026-10-28T18:19Z
- **Completed:** 2026-10-28T18:43Z
- **Tasks:** 3/3
- **Files modified:** 9 (1 created, 8 modified)

## Accomplishments

- `jobs local` runs the five D-01 models one at a time in the freeze order. It skips complete directories, resumes incomplete ones at most twice per invocation, refuses a second instance for the same prefix, and launches nothing at or after `FREEZE_UTC`. At the freeze it stops only its own child, with done code 7. It starts the Ollama server once and stops it only if it started it.
- `jobs status` only reads. It prints per-model lines, a `WARN no-progress` line after 20 minutes (D-04), `WAIT` for a quota wait, the `FREEZE` countdown and one `STATUS` line. `jobs report` merges only the final `<prefix>-<model>` directories.
- Runner fixes:
  - IN-02: `final_message` is always text.
  - A `context_overflow` is a finished `format_failure`, or `attacked` if the attack was already attempted. It is never re-queued.
  - `--priority-kinds ds` gives a ds-first queue and is recorded in the manifest extra.
- `jobs hosted` re-runs the Groq models fresh in `<prefix>-<model>` with `--priority-kinds ds`, under the D-09 rule only. `stopping_rule.json` is written once, and the loop refuses to start if the stored freeze differs from `FREEZE_UTC`. A model stalls after 3 launches with zero progress. Unavailable models are never launched. A prefix under `results/final-20261011` is refused.
- `jobs hosted-probe --key-check` runs the registry start-up first, then prints only `KEY <name> set|missing`. `run_checks hosted` prints labels and terminal status. `run_checks secret-scan` uses a fixed regex and prints file:line only.

## Smoke queue (Task 1 tracer, untracked evidence)

Both invocations used `timeout 420 venv/bin/python -u -m src.benchmark.jobs local --prefix results/phase4-smoke --models scratch-gpt,ollama-qwen2.5-3b --smoke`.

First invocation (108 s):
```
=== local 2026-10-28T18:24:43Z prefix results/phase4-smoke models scratch-gpt,ollama-qwen2.5-3b smoke True freeze 2026-10-29T15:30:00+00:00 ===
[local] launch scratch-gpt 1/2 -> results/phase4-smoke-scratch-gpt
PREFLIGHT ok (free_mib 5047, avail_ram_mib 8889)
BENCH scratch-gpt exit 0
[local] scratch-gpt done 0
[local] complete scratch-gpt
[local] ollama owned-started
[local] launch ollama-qwen2.5-3b 1/2 -> results/phase4-smoke-ollama-qwen2.5-3b
PREFLIGHT ok (free_mib 4979, avail_ram_mib 8737)
OLLAMA owned-running
VARIANT qwen2.5-3b-ctx8192 from qwen2.5:3b num_ctx 8192 ok
LONG_PROMPT qwen2.5-3b-ctx8192 prompt_tokens 6406 band 4198-7782 D 4096 attempts 1
BENCH ollama-qwen2.5-3b exit 0
[local] ollama-qwen2.5-3b done 0
[local] complete ollama-qwen2.5-3b
OLLAMA stopped pid 8285
[local] ollama-down stopped
LOCAL done: 2/2 complete
```
Second invocation (0 s):
```
=== local 2026-10-28T18:26:41Z prefix results/phase4-smoke models scratch-gpt,ollama-qwen2.5-3b smoke True freeze 2026-10-29T15:30:00+00:00 ===
[local] skip scratch-gpt: complete
[local] skip ollama-qwen2.5-3b: complete
LOCAL done: 2/2 complete
```
- **Skip count** in `results/phase4-jobs/local-phase4-smoke.log`: 2, so the skip gate passes.
- **Status output** (no WARN line):
  ```
  MODEL scratch-gpt finished 6/6 pending_infra 0 done 0 alive no
  MODEL ollama-qwen2.5-3b finished 6/6 pending_infra 0 done 0 alive no
  FREEZE in 21h3m
  STATUS all-complete
  ```
- **Completeness:** both directories have `bench.done` 0 and pass `run_checks complete --expect 6`.
  - scratch-gpt: resolved `scratch-gpt@97bdace56294`, policy c8afbb5d345a, 1 commit.
  - ollama-qwen2.5-3b: resolved `qwen2.5-3b-ctx8192`.
- **Ollama server:** `jobs ollama-down` afterwards printed `OLLAMA none`, and `pgrep "ollama serve"` was empty. The only server in play was the one this plan started (pid 8285), and only that pid was signalled.

## TDD red runs

- **Task 2:** the red run of `tests/test_phase4_runner.py` (14 tests) gave 11 failed and 2 passed before any implementation. The 2 that passed were the plain-string message case and the CLI unknown-kind rejection, which argparse already rejected because the flag was unknown. Green: 126 passed over the six verify files, with no assertion of an existing test edited.
- **Task 3:** the red run gave 28 failed and 99 passed. `jobs.py` was temporarily set back to its committed Task 1 form and `run_checks.py` had no hosted or secret-scan code yet. Green: 127 passed.

## Key check, secret scan and no-hosted-request confirmation

- **Key check:** `jobs hosted-probe --key-check` printed only `KEY GROQ_API_KEY set`. The value was never printed or stored, and no request was sent.
  - The .env load comes from the provider registry's existing start-up: `load_model_config()` in `src/llm/registry.py`, whose module start-up runs python-dotenv on the project's .env. `hosted_probe` calls `load_model_config()` before it reads `os.environ`. This is plan-checker warning W1, and a test pins the order. No code in this plan opens .env, and the executor never opened, catted or grepped it.
- **Secret scan:** `run_checks secret-scan --paths src tests config .planning/phases/04-full-runs-local-added-ollama-hosted-resume` printed `secret-scan-ok files 92`. Fake keys in the tests are built at runtime by concatenation.
- **Nothing hosted was launched:**
  - No hosted request was sent in this plan.
  - No `results/phase4-groq-*` directory exists.
  - `results/phase4-jobs/stopping_rule.json` does not exist.
- **Freeze constant:** `grep -c 'FREEZE_UTC = "2026-10-29T15:30:00+00:00"' src/benchmark/jobs.py` returns 1.

## Full suite (four WSL calls, each under `timeout 420`)

| Call | Result | Runtime |
|------|--------|---------|
| everything except hf_local/qwen_lora/jobs/run_checks/phase4_runner | 925 passed | 251 s |
| test_hf_local_provider.py + test_qwen_lora_data.py | 66 passed, 1 skipped | 125 s |
| test_qwen_lora_train.py + test_qwen_lora_checks.py | 94 passed | 232 s |
| test_jobs.py + test_run_checks.py + test_phase4_runner.py | 140 passed | 14 s |
| **Total** | **1225 passed, 1 skipped** (04-01 baseline 1163 + 1 skipped; +62 new) | |

Calls 1 and 2 ran at the same time, then calls 3 and 1 overlapped. None came near the 420 s cap.

## Task Commits

1. **Task 1: freeze-aware serial local queue, read-only status, merged report** - `649a6a4` (feat)
2. **Task 2: IN-02, context_overflow as a finished outcome, ds-first queue** - `49bb48b` (fix)
3. **Task 3: hosted loop under D-09, environment-only key check, hosted and secret-scan checks** - `2759af1` (feat)

## Deviations from Plan

### Auto-fixed issues and design details

**1. [Design detail] Who counts as "alive" in `jobs status`**
- The plan defined alive as "a run_benchmark process for that directory, the local queue or the hosted loop is running".
- If a queue being alive counted for every one of its models, finished or waiting models would show stale files and get false `WARN no-progress` lines.
- So per model, `alive` means a run_benchmark child is running on that directory. Running queues and loops get their own `LOOP local|hosted alive pid N` lines, and they count toward `STATUS running`.

**2. [Rule 2 - Correctness] Single-instance check excludes this process's ancestors**
- The verify commands wrap the queue in `timeout 420 venv/bin/python -m src.benchmark.jobs local ...`.
- The `timeout` wrapper's cmdline contains the same words, so a plain cmdline match would see it as a second instance.
- `find_loop` therefore skips this process and all its ancestors, and only matches processes whose argv[0] is a python binary.

**3. [Rule 2 - Correctness] The hosted loop never duplicates a run it did not start**
- If a run_benchmark child is already running on a model's directory, for example an orphan from a crashed loop, the loop does not launch a second one. It also never signals that child.
- A `running` state left behind by a crashed loop is recorded as "loop restarted" and relaunched.

**4. [Design detail] An extra non-terminal state, `waiting_retry`**
- An exit that is not a quota stop and not complete waits 10 minutes before relaunching. That state is shown as `waiting_retry`.
- If that relaunch would fall at or after the freeze, the model ends as `stopped_at_freeze`.

**5. [Design detail] `hosted-probe --prefix P` records unavailable models**
- With `--prefix P`, a model the probe finds unavailable is written to `P-<model>/hosted_state.json` as `unavailable`, and the loop never launches it.
- No substitute model is chosen (D-02).

**6. [Process] Task 3 red run used the committed Task 1 `jobs.py`**
- The hosted code in `jobs.py` was written before its tests. To get a genuine red run, the committed Task 1 `jobs.py` was restored temporarily with `git show HEAD:src/benchmark/jobs.py`, the tests were run, and the working version was copied back. No stash and no other file was touched.

**7. [Bookkeeping] CMP-02 and CMP-03 are not marked complete**
- No full run has happened yet. The requirements are advanced, not completed.

Total: 2 correctness auto-fixes, 4 recorded design or process details and 1 bookkeeping note. There was no scope creep.

## Known Stubs

None.

## Threat Flags

None. The new surface is starting and signalling our own benchmark children, read-only /proc scans, and writing `stopping_rule.json` and `hosted_state.json`. It is covered by T-04-18 to T-04-22, and each mitigation is tested:
- zero signals to a bystander pid;
- key presence printed as set or missing only;
- the moved-freeze refusal;
- the WARN and WAIT lines.

## Next Phase Readiness

- 04-03 can launch `jobs local --prefix results/phase4` and, after the probe, `jobs hosted --prefix results/phase4`. Both in one attached `wsl.exe` call each.
- The first hosted launch writes `results/phase4-jobs/stopping_rule.json` with `rule_sha` = HEAD at that moment.
- `jobs status --prefix results/phase4` shows both queues. `run_checks hosted --prefix results/phase4` labels the Groq rows.
- `results/phase4-smoke-*` are untracked evidence. They are never merged, because `report` only takes `<prefix>-<model>` names.

## Self-Check: PASSED

- Files exist: `src/benchmark/jobs.py`, `src/benchmark/run_checks.py`, `tests/test_phase4_runner.py`, `tests/test_jobs.py`, `tests/test_run_checks.py`, and `results/phase4-jobs/local-phase4-smoke.log`.
- Commits 649a6a4, 49bb48b and 2759af1 are ancestors of HEAD. None of them deletes a file.
