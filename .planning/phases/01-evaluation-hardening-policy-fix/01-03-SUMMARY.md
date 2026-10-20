---
phase: 01-evaluation-hardening-policy-fix
plan: 03
subsystem: benchmark-runner-provenance
status: complete
tags: [provenance, manifest, finish-reason, requeue, interleaved-queue, resume, report]

requires:
  - phase: 01-01
    provides: classify_llm_error, drop_infra_pairs, report _cell/_usable/summarize(expected), DEFAULT_EXPECTED_EPISODES
  - phase: 01-02
    provides: PolicyEngine.policy_path / policy_sha256
provides:
  - "LLMResponse.finish_reason / resolved_model; Step.finish_reason / prompt_tokens / completion_tokens / resolved_model"
  - "Stamped raw.jsonl rows: code_commit, code_dirty, policy_sha256, provider, config_model, resolved_model, finish_reasons, prompt_tokens, completion_tokens"
  - "OpenAI-compatible provider reports finish_reason and server-resolved model id; recovered generations tagged recovered_from_error"
  - "runner._work_queue (run-major, seeded shuffle, arms adjacent), bounded infra re-queue, refusal-aware done set, manifest.json"
  - "report '## Provenance and data quality' section and manifest-driven completeness denominator"
affects: [01-04-control-agent-and-viva-notes, phase-2, phase-3-hf-local-provider, phase-4-full-runs, phase-5-report]

actuals:
  tokens: 10200
  tasks: 3
  commits: 3
plan_head_before: f250701bfd23e3ed695ccbed4cef86c0e698bab0
plan_head_after: 1de7488afe86dffa33130f990d7b22700c18df17
commits: 3

tech-stack:
  added: []
  patterns:
    - "Every long-lived evidence record is self-describing: code commit, policy hash and model ids are stamped at write time"
    - "Report precedence for the completeness denominator: explicit argument, then the source directory's manifest, then the default"
    - "Infra errors re-run from scratch; the best-record rule in load_rows replaces the error, so pairs are compared like-for-like"
    - "Subprocess for git: fixed argv, no shell, cwd=PROJECT_ROOT, 10 s timeout, failures map to 'unknown' / None"

key-files:
  created:
    - tests/test_provenance.py
    - tests/test_runner_queue.py
  modified:
    - src/llm/base.py
    - src/llm/openai_compat.py
    - src/agent_harness/loop.py
    - src/benchmark/runner.py
    - src/benchmark/report.py
    - run_benchmark.py
    - tests/test_llm_layer.py

key-decisions:
  - "Per-model manifest invocation count in the report: an invocation counts for a model when its models list includes it (entries without a models list count for all), so a shared directory does not overstate one model"
  - "When a model's rows come from several merged directories with different manifests, the largest expected_episodes_per_model wins (stricter completeness)"
  - "Queue shuffle seed is seed * 1_000_003 + run, an integer, so the order is identical across processes and Python versions"
  - "Re-queue log lines use '[requeue] <model> <case> run<k> <arm> <status>' and do not advance the [n/total] counter"
  - "Manifest is written atomically (tmp file + os.replace) at the start of run_benchmark so an interrupted run still has one"

patterns-established:
  - "Provenance table cells always pass through _cell; MIXED prefixes the value list when stamped rows disagree"

requirements-completed: [EVAL-05, EVAL-06]

coverage:
  - id: D1
    description: "Every raw.jsonl record carries code_commit, code_dirty, policy_sha256, provider, config_model, resolved_model, finish_reasons, prompt_tokens, completion_tokens"
    requirement: "EVAL-06"
    verification:
      - kind: unit
        ref: "tests/test_provenance.py#test_row_carries_every_stamp"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_provider_without_the_new_fields_does_not_crash"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_git_helpers_never_use_a_shell_and_fall_back"
        status: pass
    human_judgment: false
  - id: D2
    description: "OpenAI-compatible provider reports finish_reason and resolved model id; recovered output is tagged recovered_from_error; absent fields stay None"
    requirement: "EVAL-06"
    verification:
      - kind: unit
        ref: "tests/test_llm_layer.py#test_provider_reports_finish_reason_and_resolved_model"
        status: pass
      - kind: unit
        ref: "tests/test_llm_layer.py#test_recovered_generation_is_tagged_as_recovered_from_error"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_openai_compatible_provider_stamps_rows_end_to_end"
        status: pass
    human_judgment: false
  - id: D3
    description: "Report provenance section shows commits, policy hashes, resolved ids, length / content-filter stops, max prompt_tokens, MIXED markers, 'unstamped' for old records"
    requirement: "EVAL-06"
    verification:
      - kind: unit
        ref: "tests/test_provenance.py#test_report_shows_provenance"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_report_marks_records_without_stamps_as_unstamped"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_provenance_marks_mixed_commits_and_policy_hashes"
        status: pass
      - kind: unit
        ref: "tests/test_provenance.py#test_provenance_counts_length_and_content_filter_stops_and_max_prompt_tokens"
        status: pass
    human_judgment: false
  - id: D4
    description: "Queue is run-major with seeded shuffle and adjacent arms; a quota-truncated run holds both attack kinds"
    requirement: "EVAL-05"
    verification:
      - kind: unit
        ref: "tests/test_runner_queue.py#test_work_queue_is_run_major_paired_and_deterministic"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_work_queue_prefix_holds_both_attack_kinds"
        status: pass
    human_judgment: false
  - id: D5
    description: "Infra errors are re-queued once per invocation and on every resume; refusals are recorded once and never retried; no re-queue after a quota stop; re-queued pair compared like-for-like"
    requirement: "EVAL-05"
    verification:
      - kind: unit
        ref: "tests/test_runner_queue.py#test_infra_error_is_requeued_and_rerun_in_the_same_invocation"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_requeue_is_bounded_by_requeue_passes"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_requeued_protected_arm_is_compared_like_for_like"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_provider_refusal_is_recorded_once_and_never_retried"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_no_requeue_pass_after_quota_exhaustion"
        status: pass
    human_judgment: false
  - id: D6
    description: "manifest.json holds the plan and one entry per invocation, contains no secrets, and drives the report's completeness column (explicit argument overrides it)"
    requirement: "EVAL-06"
    verification:
      - kind: unit
        ref: "tests/test_runner_queue.py#test_manifest_records_plan_and_one_invocation_per_call"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_secrets_never_reach_raw_jsonl_or_manifest"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_report_completeness_follows_the_manifest"
        status: pass
      - kind: unit
        ref: "tests/test_runner_queue.py#test_explicit_expected_overrides_the_manifest"
        status: pass
    human_judgment: false
  - id: D7
    description: "A live hosted or Ollama run produces stamped rows with a real server-reported finish_reason and resolved model id"
    requirement: "EVAL-06"
    verification: []
    human_judgment: true
    rationale: "Provider behavior against a real server was exercised only through fakes (no hosted API calls, no live Ollama run in this plan); first real run is a Phase 4 concern"

duration: ~15 min
completed: 2026-10-20
---

# Phase 1 Plan 03: Stamped records, paired queue, infra re-queue and manifest Summary

**Every benchmark row now carries the code commit, policy hash, provider and resolved model id plus per-step finish_reason and prompt_tokens; the runner queues runs run-major with both arms adjacent, re-runs infrastructure-errored episodes (one bounded pass, and on every resume) while keeping provider refusals as outcomes, and writes a secret-free manifest.json that the report uses for completeness.**

## Performance

- **Duration:** about 15 min (851 s between the start timestamp and the last verification run)
- **Completed:** 2026-10-20
- **Tasks:** 3 (tracer, auto, auto)
- **Files modified:** 9 (2 created, 7 modified)
- **Suite:** 235 passed in WSL (baseline after 01-02 was 204; +31 new tests); `tests/test_benchmark.py` untouched

## Accomplishments

- Tracer: one provider reply travels `LLMResponse` -> `Step` -> `raw.jsonl` row -> "Provenance and data quality" section, committed before any expansion.
- The report provenance table: commit(s), dirty tree, policy hash(es), resolved model id(s), config model, length stops, content-filter stops, max prompt_tokens, invocations; `MIXED` leads the commit and policy cells when a model's stamped rows disagree; records written before this plan read "unstamped (recorded before EVAL-06)".
- Regenerating `results/phase1-report` from the October data (`--report-only ... --merge results/final-20261011/*/`) still works and prints "unstamped (recorded before EVAL-06)" for all six models.
- `_work_queue` replaces the case-major `itertools.product` loop; `itertools` is no longer imported.
- Bounded re-queue (`requeue_passes=1`): skipped after `QuotaExhausted`, never re-runs the good sibling arm (the existing best-record rule plus `drop_infra_pairs` make the pair like-for-like once the errored arm succeeds).

## Task Commits

1. **Task 1 (tracer): stamp one record end to end** - `ca3d308`
2. **Task 2: hosted finish_reason and model id; truncation / filter / MIXED flags** - `8c44775`
3. **Task 3: paired interleaved queue, infra re-queue, manifest** - `1de7488`

Tracer gate: auto mode was not active and the tracer `<verify>` is automated-only, so it was re-run end to end (provenance, agent-loop, benchmark and report tests plus the October `--report-only` regeneration: all pass) and expansion continued. The red state was seen for each task before implementation (7 failed in Task 1, 6 failed in Task 2, collection error in Task 3).

## Row stamp fields (as implemented, written by `run_benchmark`)

| Field | Type | Meaning |
|-------|------|---------|
| `code_commit` | str | 40-hex `git rev-parse HEAD`, or `"unknown"` if git is unavailable or output is not 40 hex |
| `code_dirty` | bool or null | tracked files differ from HEAD (`git status --porcelain --untracked-files=no`); null if git failed |
| `policy_sha256` | str | `PolicyEngine.policy_sha256` (SHA-256 of the policy YAML bytes) |
| `provider` | str | config provider type (`openai_compat`, `scratch`, ...) or `"unknown"` if the model is not in `config/models.yaml` |
| `config_model` | str or null | `cfg["model"]` |
| `resolved_model` | str or null | first non-null step `resolved_model` |
| `finish_reasons` | list[str or null] | one per step (`stop`, `length`, `content_filter`, `recovered_from_error`, ...) |
| `prompt_tokens` | list[int or null] | one per step |
| `completion_tokens` | list[int or null] | one per step |

These sit alongside the existing score fields. `code_commit`, `code_dirty` and `policy_sha256` are identical on every row of one invocation. A row is "stamped" in the report when it has a non-null `code_commit`.

## manifest.json schema (as implemented)

```json
{
  "schema": 1,
  "expected_episodes_per_model": 342,
  "cases": {"benign": 17, "dh": 20, "ds": 20},
  "runs": 3,
  "arms": ["unprotected", "protected"],
  "max_steps": 5,
  "models": ["..."],
  "extra": {"seed": 0, "dh": 20, "ds": 20, "benign": 17},
  "invocations": [
    {"started_at": "2026-10-20T18:30:00", "code_commit": "<40 hex | unknown>", "code_dirty": true,
     "policy_sha256": "<64 hex>", "models": ["..."]}
  ]
}
```

`expected_episodes_per_model = len(cases) * runs * len(arms)`; each call overwrites the plan fields, unions `models` (first-seen order) and appends one `invocations` entry. A corrupt file is rewritten. The write is atomic. `run_benchmark.py` passes `extra = {"seed", "dh", "ds", "benign"}` and `queue_seed=args.seed`.

New signature: `run_benchmark(models, cases, out_dir, runs=3, arms=ARMS, max_steps=5, sleep_s=0.0, log=print, queue_seed=0, requeue_passes=1, manifest_extra=None) -> str`. Report side: `generate_report(out_dir, charts=True, sources=None, expected=None)` with precedence explicit `expected`, then each source directory's manifest, then `DEFAULT_EXPECTED_EPISODES`; `summarize(rows, expected=None)` now also accepts a `{model: int}` dict; `provenance_summary(rows, invocations=None)` and `read_manifest(src)` are new public helpers in `report.py`.

## Files Created/Modified

- `src/llm/base.py` - `finish_reason`, `resolved_model` on `LLMResponse`
- `src/llm/openai_compat.py` - reads `resp.model` and `choices[0].finish_reason`; `recovered_from_error` tag
- `src/agent_harness/loop.py` - `Step` fields copied from the response
- `src/benchmark/runner.py` - git helpers, stamps, `_work_queue`, re-queue, refusal-aware done set, `_write_manifest`
- `src/benchmark/report.py` - provenance section, MIXED/length/filter/max-prompt columns, manifest-driven expected, Invocations column
- `run_benchmark.py` - passes `queue_seed` and `manifest_extra`; docstring note
- `tests/test_provenance.py`, `tests/test_runner_queue.py` (new), `tests/test_llm_layer.py` (FakeClient `model` / `finish_reason` options; 3 new tests)

## Decisions Made

See `key-decisions` above. One behavior to know: a `provider` of `"unknown"` appears for any model name not present in `config/models.yaml` (the tests' `"m"` model, for instance); real runs always use configured names.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Tool-induced escape mangling while writing test and patch files**
- **Found during:** Tasks 2 and 3
- **Issue:** Backslash sequences in shell heredocs were collapsed (`"\\n"` became a real newline inside a string literal), producing a SyntaxError at test collection in `tests/test_provenance.py`, and a heredoc containing a Python patch script failed to parse in bash.
- **Fix:** Repaired the test strings with the Edit tool (`"\n"`, `"\\|"`), and wrote the runner and report patch scripts with the Write tool into the scratchpad before running them.
- **Files modified:** tests/test_provenance.py, src/benchmark/runner.py (via scratchpad splice script)
- **Commit:** 8c44775, 1de7488

**Total deviations:** 1 auto-fixed (blocking tooling issue). **Impact:** none on scope. Additions beyond the plan's lists: `read_manifest` / `provenance_summary` helpers are public; extra tests (`test_requeue_is_bounded_by_requeue_passes`, `test_corrupt_manifest_is_rewritten`, `test_missing_manifest_falls_back_to_the_default`, `test_merged_sources_use_each_directorys_own_manifest`, `test_provenance_escapes_model_supplied_ids`, `test_provenance_single_commit_is_not_mixed`, `test_work_queue_seed_changes_the_order_and_runs_shuffle_independently`).

## Issues Encountered

- `code_dirty` will usually be true under WSL (line-ending differences make git report tracked files as modified), which is the honest value.
- Existing pre-plan result directories have no `manifest.json`, so their completeness still falls back to 342 unless `--expected` is passed.
- `run_benchmark.py` always passes an explicit `expected` to `generate_report`, so the manifest only drives completeness for `--report-only` / merged reports; the two values are equal for a normal run.

## Known Stubs

None.

## Threat Flags

None. The git subprocess, manifest and provider-supplied ids are the surfaces in the plan's threat model (T-01-11 to T-01-15): all mitigated as planned (stamps and MIXED markers, fixed manifest field list with a fake-key test, `_cell` escaping with a test, one bounded re-queue pass skipped after a quota stop, fixed-argv git with timeout).

## Next Phase Readiness

Ready for 01-04. Plans 01-04 and Phases 2-5 can read the stamp fields and manifest schema listed above; Phase 4 truncation checks can use `prompt_tokens` and `finish_reasons`.

## Self-Check: PASSED

- Created files present: tests/test_provenance.py, tests/test_runner_queue.py
- Commits reachable from HEAD: ca3d308, 8c44775, 1de7488
- Full WSL suite: 235 passed; all acceptance greps pass (`def _work_queue` 1, `itertools.product` 0, `[requeue]` 1, `classify_llm_error` 4, `expected_episodes_per_model` in report.py 1, `manifest_extra` in run_benchmark.py 1, `shell=True` 0); `git diff --stat tests/test_benchmark.py` empty
