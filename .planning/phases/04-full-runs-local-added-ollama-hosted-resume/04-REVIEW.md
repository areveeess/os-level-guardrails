---
phase: 04-full-runs-local-added-ollama-hosted-resume
reviewed: 2026-10-29T03:30:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - config/models.yaml
  - run_benchmark.py
  - src/agent_harness/loop.py
  - src/benchmark/jobs.py
  - src/benchmark/outcomes.py
  - src/benchmark/report.py
  - src/benchmark/run_checks.py
  - src/benchmark/runner.py
  - tests/test_control_agent.py
  - tests/test_jobs.py
  - tests/test_layer_blocks_report.py
  - tests/test_phase4_runner.py
  - tests/test_provenance.py
  - tests/test_run_checks.py
findings:
  critical: 1
  warning: 5
  info: 7
  total: 13
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-10-29
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

**Scope.** I reviewed the source and tests changed in `0b4bc40..HEAD` under `src`, `tests`, `config` and `run_benchmark.py`, as the review request set out. I left out two files:

- `src/dashboard/templates/index.html`, which belongs to a separate quick task.
- `src/dashboard/app.py`, which holds uncommitted work in progress.

**Code read alongside.** I also read the code these files call into: `report.load_rows_counted`, `_usable`, `summarize`, `scoring.score_episode`, `llm/registry.py`, `llm/hf_local.py`, `llm/scratch.py` and the Phase 1 residual-bypass table in `01-REVIEW-FIX.md`.

**What I ran.** I ran no benchmark, sent no signal and did not read `.env`. The one probe I ran was a pure-Python check of `_scan_raw` and `load_rows` on a two-line temporary `raw.jsonl`. It confirms CR-01.

**What holds up:**

- **Ownership.** Signals go only to a pidfile pid that carries the environment marker and has an `ollama serve` command line, or to a `Popen` handle this process created. All `/proc` scans (`find_loop`, `find_bench_child`, `status`) only read.
- **Secrets.** The key check prints only `set` or `missing`. It relies on the registry's import-time `load_dotenv` and never opens `.env`.
- **Layer table and evidence text.** The EVAL-08 layer table agrees with the summary's attack-arm `blocks_*` counts. The row numbers that `LAYER_EVIDENCE` cites match the Phase 1 table (rows 1, 2, 8, 9, 10, 15 and 16 seccomp; rows 20 to 22 sandbox; rows 3-7, 11-14, 17-19 and 23 contained without a signature).
- **ds-first queue.** The queue is a correct stable partition, and the D-09 freeze constant cannot be moved.

**What is wrong:**

- **Completeness and outcomes (main issue).** The runner, the report loader and the new completeness check disagree about which record of an episode counts. One realistic sequence (an infrastructure error, then a context overflow on the retry) leaves an episode that is never retried and never counted (CR-01).
- **Classification bias.** The new context-overflow class skews the format-valid attack success rate (WR-01).
- **Ownership gap.** One path can send SIGKILL to a pid without checking ownership again (WR-02).
- **Freeze gap.** One path can launch a child after the freeze (WR-03).
- **Scheduler gate.** The loops treat provenance drift as unfinished work (WR-04).
- **Tests.** No test covers any of these four paths (WR-05).

## Critical Issues

### CR-01: An episode that hits an infrastructure error and is then retried into a context overflow (or a provider refusal) is neither retried nor counted, so the model can never reach 342/342

**File:** `src/benchmark/report.py:82`, `src/benchmark/runner.py:196-197`, `src/benchmark/run_checks.py:123-126`, `src/benchmark/jobs.py:992-997`

**Issue:** Three components each pick a record for a key, and they use different rules.

**The runner (`_scan_raw`)** marks a key done as soon as any record of it is non-infra:
```python
if classify_llm_error(r.get("llm_error")) != "infra":
    done.add(key)
```

**The loader (`load_rows_counted`)** replaces a stored record only with one that has no `llm_error` at all:
```python
if key not in best or (best[key].get("llm_error") and not row.get("llm_error")):
```

A context-overflow row and a provider-refusal row both carry a non-empty `llm_error`. Take the sequence: first an infra row (for example a CUDA `OutOfMemoryError`), then a `context_overflow` row on the bounded re-queue or on a resume. The two components then disagree:

- The runner treats the key as done and never runs it again.
- `load_rows` keeps the first (infra) row, so `classify_outcome` returns `llm_error`.

**Consequences:**

- `complete_status` counts the key under `pending_infra`, not `finished`.
- `is_complete` is never true.
- The local queue spends both launches. The hosted loop relaunches until the model is marked `stalled`.
- `run_checks complete` fails for good.
- The report drops the pair as an infrastructure error.

I reproduced this with the two-line `raw.jsonl` probe. `_scan_raw` returned the key as done, and `load_rows` returned the OOM row with outcome `llm_error`.

**Why this is realistic on this hardware:** the episode is deterministic at temperature 0. An OOM at step k on the 6 GB GPU happens on a long prompt. The re-run reaches step k again, may then fit, and overflows at step k+1. That is exactly the episode this sequence breaks. The same hole already existed for infra followed by `provider_refusal`. Phase 4 adds a second, more likely entry into it, and the new `complete_status` makes the loops depend on it.

**Fix:** Make the loader use the runner's rule: any non-infra record beats an infra record. Then add a test for the infra-then-overflow sequence that checks `_scan_raw`, `load_rows`, `complete_status` and `summarize` together.
```python
from src.benchmark.outcomes import classify_llm_error

def _is_infra(row):
    return classify_llm_error(row.get("llm_error")) == "infra"

if key not in best or (_is_infra(best[key]) and not _is_infra(row)):
    best[key] = row
```

## Warnings

### WR-01: A context overflow after valid steps is classed `format_failure`, which biases the format-valid attack success rate upward

**File:** `src/benchmark/outcomes.py:14-18, 88-91`

**Issue:** Rule 2 (`attacked`) comes before rule 2b (`context_overflow` becomes `format_failure`). The two cases are treated differently:

- **Attempted the attack, then overflowed:** stays `attacked`, so it counts in both the numerator and the denominator of `asr_valid_*`.
- **Ignored the injection for a step or two, then overflowed:** becomes `format_failure` and leaves the denominator. `FORMAT_VALID` is `("attacked", "ignored_injection", "refused")`.

So the more often a model overflows mid-episode, the higher its format-valid ASR looks. The scratch model has a small character window and runs five steps, so this is the model most exposed.

The module docstring defines `format_failure` as "the model never produced a usable action". A mid-episode overflow did produce usable actions, so it does not fit that definition. The two arms can also overflow at different steps, because BLOCKED observations differ in length, so the arms are not like-for-like.

**Fix:** Take one of these approaches:

- **Separate step-0 from mid-episode overflow.** Class an overflow at step 0 (no parsed action) as `format_failure`. Class a mid-episode overflow by its actions: `ignored_injection` or `refused` under rules 4 to 6, or a new `truncated` outcome reported next to the others.
- **Report the overflow count instead.** Keep the rule, but report the overflow count per arm next to `asr_valid_*` and state the direction of the bias in the report and the viva page.

### WR-02: `stop_owned` sends SIGKILL to a pid after a 30-second wait without checking ownership again

**File:** `src/benchmark/jobs.py:360-373`

**Issue:** Ownership is checked once, before SIGTERM. After that, `_pid_alive(pid)` decides whether to send SIGKILL. Whether this is safe depends on which process stops the server.

- **Same process that started the server.** The pid is in `_CHILDREN`. Our un-reaped `Popen` holds it, so it cannot be reused. This path is safe.
- **A different process.** This is the normal CLI flow: `jobs ollama-up` starts the server with `start_new_session=True` and exits, and later `jobs ollama-down` or a different `jobs run` stops it. The pid is not in `_CHILDREN`, so `_pid_alive` reads `/proc/<pid>/stat`.

In the second case, the server exits on SIGTERM and init reaps it. If the pid is reused by another project's process within the wait, `_pid_alive` sees a live process and SIGKILL goes to a process this project does not own. Reuse within 30 s is unlikely with WSL's large `pid_max`, but the requirement is "only ever signal processes it owns", and no test covers the cross-process case (`test_ollama_down_escalates_to_sigkill_only_for_the_owned_pid` relies on the world fake keeping the pid identity).

**Fix:** Check ownership again immediately before every signal after the first, and stop when the check fails:
```python
if _pid_alive(pid) and owned_pid(jobs_dir, pidfile_name) == pid:
    _send_signal(pid, signal.SIGKILL)
```
Keep the pidfile in place until after this check, because `owned_pid` reads it. Better still, open a pidfd (`os.pidfd_open`, Linux 5.3+) at the ownership check and send both signals with `signal.pidfd_send_signal`. A pidfd cannot point at a reused pid.

### WR-03: `jobs.run` checks the freeze only once, before setup steps that can take more than 30 minutes, so a child or an Ollama server can be launched after the freeze

**File:** `src/benchmark/jobs.py:849-877, 811-823`

**Issue:** The deadline is checked once, at line 849. The steps after it can take a long time:

- `preflight_with_rechecks` waits up to 6 × 300 s.
- The Ollama setup runs `ollama_up` (60 s), `ollama create` (timeout 600 s), `_load_model` (600 s) and up to three long-prompt requests of 900 s each.

After that, `run_child_until` calls `_spawn` before its first deadline check. A `local` launch that starts at about 15:00 UTC can therefore start an Ollama server and a `run_benchmark` child after 15:30. The child is killed at once, but before that it may already have appended an invocation entry to `manifest.json`.

D-09 and the written stopping rule say "nothing is launched at or after the freeze". The tests cover only a deadline that has already passed on entry (`test_run_after_the_deadline_launches_nothing`).

**Fix:** Check the deadline again after the preflight, after the Ollama setup, and at the top of `run_child_until` before `_spawn`. Each check returns `DONE_FREEZE` and writes the freeze line. Add one test that advances the fake clock past the freeze inside the preflight's `_sleep`.

### WR-04: The loops' "complete" gate also fails on provenance problems, so a fully finished model is relaunched and ends labelled `stalled` or partial

**File:** `src/benchmark/jobs.py:992-997, 1383, 1396-1408`, `src/benchmark/run_checks.py:146-178`, `src/benchmark/run_checks.py:481`

**Issue:** `is_complete` requires `not st["problems"]`. `complete_status` adds a problem in these cases:

- Any row's `policy_sha256` differs from the policy file as it is *now*.
- Run-affecting files differ between two of the rows' commits.
- The rows have more than one `resolved_model`.

So a commit to `src/llm` or `config/` during the 20-hour hosted run, or any later edit of the policy YAML, makes a 342/342 directory "incomplete" for the schedulers:

- **Hosted loop.** It relaunches the model every 10 minutes. Each child does nothing except rewrite the manifest and the report. After 3 zero-progress launches the model is recorded as `stalled`.
- **Local queue.** It spends both launches.
- **`check_hosted`.** It then prints `partial n=342`.

The real problem, provenance drift, is never shown as the reason. The check is right to fail; using it as the scheduling gate is the problem.

**Fix:** Gate scheduling on progress only: `finished >= expected and pending_infra == 0`. Report provenance problems separately, as a `PROVENANCE` FAIL line in `status` and a distinct hosted state such as `complete_with_problems`. Leave `run_checks complete` strict.

### WR-05: No test covers the paths above, and one key-check test cannot fail

**File:** `tests/test_phase4_runner.py:101-128`, `tests/test_jobs.py:217-235, 767-803, 1168-1178`

**Issue:** These cases have no test:

- **CR-01:** an infra row followed by an overflow or refusal row. The overflow tests only use a provider that always overflows.
- **WR-02:** `ollama-down` from a process that did not start the server, where `_CHILDREN` is empty and the pid is replaced during the wait.
- **WR-03:** the freeze passing during the preflight or the Ollama setup.
- **IN-01:** a hosted child that exits in the last poll interval before the freeze.

`test_key_check_runs_the_registry_start_up_first` only asserts that `order == ["registry"]`. It cannot fail on ordering, because nothing else is recorded. It also does not test what the docstring claims: the `.env` load happens when `src.llm.registry` is imported, not inside `load_model_config`.

**Fix:** Add the four tests. For the key check, record the `os.environ.get` call as well: patch `jobs.os.environ` with a mapping that appends to `order`. Then assert `order == ["registry", "env:GROQ_API_KEY"]`, or remove the test and state in the docstring that the import performs the start-up.

## Info

### IN-01: Edge cases in the hosted loop's freeze handling

**File:** `src/benchmark/jobs.py:1413-1427, 1440-1441`

**Issue:** Three edge cases:

- **Child exits just before the freeze.** If a child exits within the last poll interval (10 s) before the freeze, it is never passed through `finish()`. Its launch is recorded as "stopped at the freeze", and its state becomes `stopped_at_freeze` even when it had just completed or stopped on quota.
- **SIGTERM is sent serially.** `stop_own_child` waits up to 60 s for each child in turn.
- **Inherited children run past the freeze.** A run that the loop finds already on a directory (`find_bench_child`, for example a child left by a killed earlier loop) is correctly never signalled. But it keeps writing rows after the freeze, and nothing records that.

**Fix:**

- Poll each child and call `finish()` before applying the freeze.
- Send SIGTERM to all children first, then wait.
- Log a line for every inherited child that is alive at the freeze, so the post-freeze rows can be identified.

### IN-02: The stated retry time is rounded to 0.1 h, more than the 120 s margin

**File:** `src/benchmark/runner.py:230`, `src/benchmark/jobs.py:1217, 1386-1388`

**Issue:** `QuotaExhausted(f"API asks to retry in {wait / 3600:.1f}h")` rounds the wait to ±180 s. `QUOTA_MARGIN_S` is 120 s, so:

- A relaunch can come up to about a minute before the reset.
- Near the freeze, rounding can push the `next_launch >= freeze` decision either way.

**Fix:** Put the wait in the message in seconds (`retry in {wait:.0f}s`) and parse that.

### IN-03: The probe's error classes are loose, and an auth block is not persisted

**File:** `src/benchmark/jobs.py:1218-1221, 1504-1513`

**Issue:** Four problems:

- **Exception path.** It classes as `auth` any exception whose type name or message contains "key", so a config `KeyError` or a `TypeError` about an unexpected keyword argument blocks every hosted model as an auth failure.
- **Over-broad pattern.** `_UNAVAILABLE_RE` matches the generic "not found".
- **Sticky state.** An `unavailable` record is terminal in `hosted_state.json`, and no command clears it.
- **Auth block not persisted.** Under the stopping rule, an auth failure blocks every hosted model, but only by the operator's decision. `hosted_loop` does not read the probe's result.

**Fix:**

- Class exceptions by type: the registry's missing-key error is `auth`, anything else is `error`.
- Narrow the unavailable pattern to `404`, `model_not_found` and `decommissioned`.
- Record an auth block in the jobs directory, and have the loop refuse to start while that record exists.

### IN-04: The overflow marker is matched as a substring in any provider's error text

**File:** `src/benchmark/outcomes.py:53, 65-66`

**Issue:** `classify_llm_error` returns `context_overflow` for any error that contains the token, including hosted and Ollama errors whose text comes from the server. Such an episode would become a finished `format_failure` that is never retried.

**Fix:** Pass the step's `finish_reason == "context_overflow"` (both local providers set it) or the row's provider into the classifier, and stop matching free text.

### IN-05: `priority_kinds` provenance is overwritten, and the queue docstring is now wrong

**File:** `run_benchmark.py:86-88`, `src/benchmark/runner.py:243-250, 271-279`

**Issue:** Two problems:

- **Overwritten on resume.** `priority_kinds` is stored in `plan["extra"]`, which each invocation rewrites. A resume without `--priority-kinds` erases the record that the earlier invocations ran ds-first. The per-invocation entries do not carry it.
- **Stale docstring.** The `_work_queue` docstring still says "a run stopped by a quota still holds both attack kinds". With `("ds",)` that is false.

**Fix:**

- Store `priority_kinds` in each invocation entry and keep it out of `extra`.
- Update the docstring.

### IN-06: Labels in the EVAL-08 layer table

**File:** `src/benchmark/report.py:335-340`

**Issue:** Two label problems:

- The column headed "Episodes" holds the group name (attack or benign).
- An `error-dominated` model is also labelled "(partial)", but the legend says partial means the model has not finished its run.

**Fix:**

- Rename the column to "Group".
- Print the summary status itself, or add `error-dominated` to the legend.

### IN-07: Constants and helpers are duplicated across `jobs.py` and `run_checks.py`, and the Ollama CLI gets the full environment

**File:** `src/benchmark/jobs.py:94, 128-131, 1200, 305-307`, `src/benchmark/run_checks.py:203-206, 324-337, 448`

**Issue:** Two problems:

- **Duplicates that can drift.** `DEFAULT_JOBS_DIR`, `DEFAULT_CONTEXT_FILE`, `EVIDENCE_SOURCES`, `PROBE_PORTS` and `HOSTED_EXPECTED` are defined in both modules, as are the pairs `_read_json`/`_read_json_file` and `_read_file`/`_read_text`. A copy that drifts would make the checker and the launcher disagree.
- **Full environment for the CLI.** `client_env` passes the whole parent environment, including the API keys that `load_dotenv` placed there, to every `ollama` CLI call. The server is given a minimal environment for exactly this reason.

**Fix:**

- Define the shared values once (for example in `run_checks`, imported by `jobs`, which already imports it lazily).
- Build `client_env` from `MINIMAL_ENV_KEYS` plus `OLLAMA_HOST`.

---

_Reviewed: 2026-10-29T03:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
