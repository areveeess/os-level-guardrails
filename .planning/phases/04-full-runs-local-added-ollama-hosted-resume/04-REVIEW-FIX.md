---
phase: 04-full-runs-local-added-ollama-hosted-resume
fixed_at: 2026-10-29T03:52:00Z
review_path: .planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 5
skipped: 1
status: partial
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-10-29
**Source review:** .planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (CR-01 and WR-01 to WR-05). The Info findings IN-01 to IN-07 were out of scope, except two small
  parts: the `client_env` half of IN-07, and the first bullet of IN-01, which the WR-05 test needed. Both are described below.
- Fixed: 5 (CR-01, WR-02, WR-03, WR-04, WR-05)
- Skipped: 1 (WR-01, deferred until after the 15:30Z freeze because it is run-affecting)

**The freeze constraint.** The Groq hosted loop runs until the D-09 freeze at 2026-10-29 15:30 UTC. It relaunches
`run_benchmark.py` children, and each child stamps the code commit of every path in `RUN_AFFECTING_PATHS`
(`src/benchmark/run_checks.py`). These fixes therefore touch only paths outside that list:

- `src/benchmark/report.py`
- `src/benchmark/run_checks.py`
- `src/benchmark/jobs.py`
- `tests/test_jobs.py`, `tests/test_run_checks.py`, `tests/test_phase4_runner.py`

`git diff 3e4dae3..HEAD --name-only` lists exactly these six files. `runner.py`, `outcomes.py`, `run_benchmark.py`,
`src/llm`, `config` and every other run-affecting path are unchanged. A row written after these commits therefore has
the same run-affecting code as one written before them.

The running loop was never touched:
- No signal was sent to WSL pid 10209 or to any of its children.
- Nothing was written under `results/phase4-groq-*`.
- The loop keeps its old `jobs.py` in memory. The `jobs.py` changes take effect at the next `jobs` invocation.

**Verification environment:** `workflow.use_worktrees` is `false` in `.planning/config.json`. Every edit, test run and
commit therefore happened in the main checkout, on branch `docs/sdd`, with no worktree, temp branch or recovery
sentinel. Tests ran under WSL Ubuntu (`venv/bin/python -m pytest`) in split calls. Each new test was run first against
the old code and failed there, apart from the guard cases noted below. The key-check ordering test was also run against
a mutant that reads the key first, and it failed there.

**Test counts** (baseline at 3e4dae3, then after the last fix):

| File | Before | After |
|------|--------|-------|
| test_jobs.py | 77 | 88 |
| test_run_checks.py | 57 | 60 |
| test_phase4_runner.py | 13 | 17 |
| test_layer_blocks_report.py | 10 | 10 |
| test_report.py | 55 | 55 |
| test_control_agent.py | 17 | 17 |
| **Total** | **229** | **247** |

That is 18 new tests, and nothing failed. The other tests that read the loader (`test_benchmark.py`,
`test_provenance.py`, `test_review_fix_benchmark.py`, `test_runner_queue.py`) also pass, 89 passed in all.

**Merged report regenerated** after CR-01. The command reads the `results/phase4-<model>` directories and writes only
`results/phase4-report`:

```
python -m src.benchmark.jobs report --prefix results/phase4 --out results/phase4-report
```

- **Local and control rows.** The five local models and `control-always-comply` stay at `complete 342/342`, with 0 infra
  errors and 0 dropped pairs.
- **Effect of CR-01 on today's data.** A read-only comparison of the old and new loader rules over every
  `results/phase4-*/raw.jsonl` changes no episode's outcome (0 keys changed in each of the 9 directories). No stored key
  is in the infra-then-terminal state yet. The fix guards future resumes.
- **Groq rows.** `groq-qwen3.8-27b` moved from 246 to 254 between the old and new report. Those are new rows from the
  live loop, not a rule change. The Groq rows are still a mid-run snapshot (WINDOWS.md #2 stays open).

## Fixed Issues

### CR-01: An episode with an infra error followed by a context overflow or refusal is neither retried nor counted

**Status:** fixed: requires human verification (record-selection logic)
**Files modified:** `src/benchmark/report.py`, `tests/test_phase4_runner.py`
**Commit:** 94666b6
**Applied fix:**
- **New ranking in the loader.** `load_rows_counted` now ranks the records of a key with `_record_rank`:
  - 0: an error-free record
  - 1: a terminal error record (`context_overflow` or `provider_refusal`)
  - 2: an infra error (`classify_llm_error`)

  The first record of the best rank wins.
- **Same rule as the runner.** `runner._scan_raw` treats any non-infra record as done. The report, `summarize`, `_usable`
  and `run_checks.complete_status` all read through `load_rows`, so they now count such a key as finished.
- **Only change in behaviour:** an infra record followed by a terminal record now yields the terminal record. An
  error-free record still beats any error record, and between two records of the same rank the first one still wins.
- **Not changed:** `runner.py` (run-affecting). `run_checks.py` needed no code change, because `complete_status` reads
  `load_rows` and its duplicate count already used the runner's non-infra rule.

The test is parametrized over four sequences:
- infra then overflow
- infra then refusal
- infra, overflow, infra
- overflow then error-free

Each case checks `_scan_raw` (the key is done), `load_rows` (the terminal outcome), `complete_status` (finished 2,
pending_infra 0) and `summarize` (episodes_done 2, infra_errors 0, complete). The first three failed on the old loader
with `llm_error`. The fourth is a guard and passed both before and after.

### WR-02: `stop_owned` sends SIGKILL without re-checking ownership

**Status:** fixed
**Files modified:** `src/benchmark/jobs.py`, `tests/test_jobs.py`
**Commit:** f6a241c
**Applied fix:**
- **Re-check before SIGKILL.** SIGKILL is sent only when `_pid_alive(pid) and owned_pid(jobs_dir, pidfile_name) == pid`.
  `owned_pid` checks the pidfile, the `ollama serve` command line and the ownership marker in `/proc/<pid>/environ`.
- **Pidfile kept until after the check,** because `owned_pid` reads it.
- **New test** `test_ollama_down_from_another_process_never_kills_a_reused_pid`. The setup:
  - `_CHILDREN` is empty, as in a separate `jobs ollama-down` process.
  - The fake SIGTERM ends the server and hands its pid to another project's process.

  The old code sent `(77, SIGKILL)` to that process. The new code sends only the SIGTERM.
- **Unchanged:** the existing escalation test, where the pid stays our marked server, still sees SIGTERM then SIGKILL.
- **Not adopted:** the pidfd variant. A small window remains between the re-check and `kill`, microseconds instead of 30 s.

### WR-03: `jobs.run` checks the freeze only once, before long setup steps

**Status:** fixed
**Files modified:** `src/benchmark/jobs.py`, `tests/test_jobs.py`
**Commit:** dcb7b47
**Applied fix:**
- **Checks in `run`.** `run` checks the deadline in three places:
  - at entry (unchanged)
  - after `preflight_with_rechecks`, which can wait 6 × 300 s
  - after the Ollama setup (`ollama_up`, variants, `capture_start_evidence`)

  Each check prints the FREEZE line, appends `FREEZE not launched (D-09, reached during ...)` to `progress.log`, writes
  done code 7 and returns. On the Ollama path the existing `finally` block still unloads the variant and stops a server
  this run started.
- **Check before spawning.** `run_child_until` now checks the clock before `_spawn` and returns 7 without spawning when
  the deadline has passed.

Three tests:
- **The freeze passes during the preflight.** The GPU is busy at the first reading, 300 s pass, then the reading is ok.
- **The freeze passes during the Ollama setup.** The setup takes 30 minutes. The test asserts that no child starts, no
  end evidence is captured, and only our server gets SIGTERM.
- **`run_child_until` with a past deadline.**

All three spawned a child on the old code.

### WR-04: The schedulers' "complete" gate fails on provenance problems

**Status:** fixed: requires human verification (scheduler state handling)
**Files modified:** `src/benchmark/run_checks.py`, `src/benchmark/jobs.py`, `tests/test_run_checks.py`, `tests/test_jobs.py`
**Commit:** 6b23da2
**Applied fix:**
- **`complete_status`** now also returns `provenance`: the problems a relaunch cannot fix (duplicates, model not in
  config, code_commit stamps, `policy_sha256` drift, scoring, provider or config_model, a second `resolved_model`,
  run-affecting commit drift). These stay in `problems` too, so `run_checks complete` stays strict.
- **New gate `run_checks.progress_complete(status)`:** `expected` is known, `finished >= expected` and `pending_infra == 0`.
- **`jobs.is_complete`** uses that gate. A 342/342 directory with drift is no longer relaunched until it is `stalled`.
- **PROVENANCE FAIL lines.**
  - `local_queue` prints them on skip and on completion.
  - `status` prints them next to the MODEL line. Its `all-complete` word is about progress only.
  - The hosted loop prints them, and ends such a model in the new terminal state `complete_with_problems`
    (`HOSTED_COMPLETE`, part of `HOSTED_TERMINAL`).
- **`run_checks hosted`** labels that state `complete (provenance problems: see run_checks complete)`, counts it as
  terminal, and reports it as a FAIL that names the `run_checks complete` command.
- **Effect on the running loop:** none. It never writes the new state, because it runs the old code.

Seven tests:
- `progress_complete`, two tests: the gate itself, and its split from provenance on a policy and commit drift.
- `is_complete`
- The local queue skips a finished model that has drift.
- `status` prints the PROVENANCE line and still says `all-complete`.
- The hosted loop ends a drifted model `complete_with_problems` after one launch and does not relaunch it after a restart.
- `run_checks hosted` on `complete_with_problems`.

All seven failed on the old code.

### WR-05: No test covers these paths, and one key-check test cannot fail

**Status:** fixed
**Files modified:** `tests/test_jobs.py`, `src/benchmark/jobs.py` (first bullet of IN-01, see below)
**Commit:** 9d577a0 (the CR-01, WR-02 and WR-03 tests landed with their fixes: 94666b6, f6a241c, dcb7b47)
**Applied fix:**
- **Key-check ordering test.** `test_key_check_reads_the_environment_after_the_registry_start_up` replaces the old test.
  - **What it records:** it patches `jobs.os.environ` with a mapping that records each `get`, and asserts
    `order == ["registry", "env:GROQ_API_KEY"]` and the `KEY GROQ_API_KEY set` line.
  - **It can fail:** a mutant that reads the key before `load_model_config` failed it (`'env:GROQ_API_KEY' != 'registry'`).
    The mutant was reverted with `git checkout`.
  - **Docstring:** the `.env` load happens when `src.llm.registry` is imported, not inside `load_model_config`.
- **Exit-just-before-freeze test**
  (`test_child_exiting_in_the_last_poll_interval_before_the_freeze_is_finished_not_stopped`). The setup:
  - The freeze is 15 s away and the child exits at 12 s.
  - The loop polls at 10 s and next at 20 s.
  - Two cases: the child completes, or the child hits its quota.

  The old loop recorded both as `stopped_at_freeze` with the reason "stopped at the freeze". To pass, the hosted loop's
  freeze branch now polls each child first and sends an exited child through `finish()`, which yields `complete`,
  `stopped_by_quota` or another state. Only a child that is still running goes to `stop_own_child`. This is the first
  bullet of IN-01. Its other two bullets (serial SIGTERM, logging inherited children) are not done.
- **Tests for CR-01, WR-02 and WR-03:** listed under those findings.

### IN-07 (`client_env` half only): The Ollama CLI got the full environment

**Status:** fixed
**Files modified:** `src/benchmark/jobs.py`, `tests/test_jobs.py`
**Commit:** bcb28e9
**Applied fix:**
- **Minimal environment.** `client_env(port)` now returns PATH, HOME and LANG (`MINIMAL_ENV_KEYS`, as for the server)
  plus `OLLAMA_HOST`. API keys that `load_dotenv` placed in this process no longer reach any `ollama` CLI call.
- **New test.** It asserts the key set of `client_env` and of every `ollama` call that `ollama_variants` makes. It
  failed on the old code: the full WSL environment was passed.
- **Not done:** the duplicated constants (the other half of IN-07).

## Skipped Issues

### WR-01: A context overflow after valid steps is classed `format_failure`, which biases the format-valid ASR upward

**File:** `src/benchmark/outcomes.py:14-18, 88-91`
**Reason:** Deferred until after the 15:30Z freeze (run-affecting). `outcomes.py` is in `RUN_AFFECTING_PATHS`. A
change would make Groq rows written after it incomparable with those written before it in the same directory. Recorded
in `.planning/WINDOWS.md` (#4).
**Original issue:** Rule 2 (`attacked`) comes before rule 2b (`context_overflow` → `format_failure`). A mid-episode
overflow after the model ignored the injection therefore leaves the `asr_valid_*` denominator, while an overflow after
an attempted attack stays in it. The more often a model overflows, the higher its format-valid ASR looks.

## Deferred Info findings (run-affecting)

Recorded with WR-01 in `.planning/WINDOWS.md` (#4). Deferred until after the 15:30Z freeze (run-affecting):

- **IN-04.** The overflow marker is matched as a substring in any provider's error text (`src/benchmark/outcomes.py`).
- **IN-05.** Two problems in `run_benchmark.py` and `src/benchmark/runner.py`:
  - `priority_kinds` is stored in `plan["extra"]` and overwritten on resume.
  - The `_work_queue` docstring is stale.

## Info findings not addressed

These were out of scope and are untouched, apart from the two parts noted above:

- **IN-01:** only its first bullet was done, under WR-05. Serial SIGTERM at the freeze and logging inherited children
  are not done.
- **IN-02:** the stated retry time is rounded to 0.1 h. Its fix would also touch `runner.py`, which is run-affecting.
- **IN-03:** the probe's loose error classes, the sticky `unavailable` state, and an auth block that is not persisted.
- **IN-06:** the layer-table labels ("Episodes" column, "(partial)" on error-dominated).
- **IN-07:** only the `client_env` half was fixed. The duplicated constants and helpers remain.

---

_Fixed: 2026-10-29T03:52:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
