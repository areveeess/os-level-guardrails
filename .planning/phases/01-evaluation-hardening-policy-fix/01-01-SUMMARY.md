---
phase: 01-evaluation-hardening-policy-fix
plan: 01
subsystem: benchmark-reporting
tags: [wilson-interval, completeness-gate, outcome-taxonomy, asr-valid, pair-dropping, conditional-block-rate, matplotlib]

requires:
  - phase: none
    provides: existing results/final-20261011 raw.jsonl and src/benchmark/report.py baseline
provides:
  - "src/benchmark/stats.py: Wilson interval, prop_fields, format_prop"
  - "src/benchmark/outcomes.py: five-way episode outcome taxonomy, infra vs provider-refusal errors, infra pair dropping"
  - "src/benchmark/report.py: honest summary (done/expected completeness, partial runs segregated, k/n Wilson cells, ASR-valid next to raw ASR, benign utility both arms, conditional block rate with pooled row)"
  - "run_benchmark.py --expected flag"
  - "tests/test_report.py: 47 new tests (125 total passing)"
  - "results/phase1-report/ regenerated October report (untracked)"
affects: [01-03-runner-resume, 01-04-control-agent-and-viva-notes, phase-5-report-and-docs]

actuals:
  tokens: 13353
  tasks: 2
  commits: 2
plan_head_before: 9e626654da0797c9fd3ccf3e865342a3aa9bb3ef
plan_head_after: 7e17e02ced53e8a29e1816428cc9611021d631fe
commits: 2

tech-stack:
  added: []
  patterns:
    - "Every percentage is a prop_fields(prefix, k, n) quintuple; tables and chart error bars read the same k/n"
    - "Classify then dedupe-then-drop: load_rows dedupes to the best record, summarize classifies, drop_infra_pairs runs on the deduplicated rows"
    - "Model-supplied strings go through _cell before entering a markdown table"

key-files:
  created:
    - src/benchmark/stats.py
    - src/benchmark/outcomes.py
    - tests/test_report.py
  modified:
    - src/benchmark/report.py
    - run_benchmark.py

key-decisions:
  - "episodes_done counts finished episodes (outcome != llm_error) before pair dropping, so a good arm excluded from the paired statistics still counts as done; the errored arm is what a resume must redo"
  - "Provider refusals (content filter) are outcome 'refused' and stay in denominators; every unrecognised llm_error string is infra"
  - "blocked_given_attempt is k = attempted protected episodes with non-empty blocked_by, n = attempted protected episodes; an attempt nobody stopped is counted separately, never credited to the guardrails"
  - "Pooled conditional block rate sums over every model present, including partial runs"

patterns-established:
  - "Summary dicts share one key set across models so summary.csv's DictWriter works"
  - "Report sections use fixed heading text that later plans append after"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05]

coverage:
  - id: D1
    description: "Wilson interval, prop_fields and format_prop helpers with 0/0 and impossible-count handling"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "tests/test_report.py#test_wilson_interval_known_values"
        status: pass
      - kind: unit
        ref: "tests/test_report.py#test_format_prop_and_prop_fields"
        status: pass
    human_judgment: false
  - id: D2
    description: "Completeness gate: done/expected per model, partial and error-dominated runs under their own heading, never in the headline table"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "tests/test_report.py#test_report_segregates_partial_runs"
        status: pass
      - kind: integration
        ref: "venv/bin/python run_benchmark.py --report-only results/phase1-report --merge results/final-20261011/*/ (112/342, 342/342, 156/342 present; groq-gpt-oss-20b only under the partial heading)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Five-way outcome taxonomy with ASR-valid printed next to raw ASR in both arms"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/test_report.py#test_classify_outcome_precedence"
        status: pass
      - kind: unit
        ref: "tests/test_report.py#test_asr_valid_uses_the_format_valid_denominator"
        status: pass
    human_judgment: false
  - id: D4
    description: "Infra-errored pairs dropped from paired statistics on best-record-deduplicated rows; benign utility in both arms; conditional block rate per model plus pooled"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/test_report.py#test_pair_dropping_runs_on_deduplicated_rows"
        status: pass
      - kind: unit
        ref: "tests/test_report.py#test_blocked_given_attempt_counts_real_blocks_only"
        status: pass
      - kind: unit
        ref: "tests/test_report.py#test_report_has_outcome_and_conditional_sections_with_pooled_row"
        status: pass
    human_judgment: false
  - id: D5
    description: "Percentage bar charts carry 95% Wilson error bars and k/n labels; undecodable raw.jsonl lines skipped and counted; rows without newer stamp fields reported"
    requirement: "EVAL-05"
    verification:
      - kind: unit
        ref: "tests/test_report.py#test_error_bars_come_from_the_prop_fields"
        status: pass
      - kind: unit
        ref: "tests/test_report.py#test_load_rows_skips_truncated_line_and_report_warns"
        status: pass
    human_judgment: false
  - id: D6
    description: "Whether the report wording (How to read this report, pair-dropping and blocked definitions) is clear enough for a viva panel"
    verification: []
    human_judgment: true
    rationale: "Readability of the explanatory prose for a non-technical reviewer is not asserted by any test"

duration: 9min
completed: 2026-10-20
status: complete
---

# Phase 1 Plan 01: Honest benchmark report Summary

**Benchmark report now prints every percentage as `P% (k/n) [95% Wilson]`, gates models on done/342 completeness, classifies each episode into one of five outcomes with ASR-valid beside raw ASR, drops infra-errored pairs, and reports the conditional block rate (blocked / attempted) per model and pooled.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-10-20T17:54:33Z
- **Completed:** 2026-10-20T18:03:41Z (before SUMMARY and state updates)
- **Tasks:** 2
- **Files:** 3 created, 2 modified

## Accomplishments

- Wilson helpers (`stats.py`) replace the old `_pct`; every `*_pct` summary key keeps its name and gains `_k/_n/_lo/_hi` siblings.
- Completeness gate: `status` complete / partial / error-dominated from `episodes_done` vs `episodes_expected`; non-complete models are listed only under "## Partial or error-dominated runs (not compared)".
- `outcomes.py`: `classify_outcome` (six-step precedence), `classify_llm_error` (infra vs provider_refusal), `drop_infra_pairs`.
- Report: ASR-valid columns, "## Outcomes per episode", "## Conditional block rate (protected arm: blocked / attempted)" with a "Pooled (all LLM models)" row, "How to read this report" with all definitions.
- Charts: `attack_outcomes.png` and `benign_utility.png` carry 95% Wilson error bars and `P% (k/n)` labels; partial models get " (partial)" on the x-axis.
- `run_benchmark.py --expected N` (default `len(cases) * runs * len(arms)` on a normal run, 342 for `--report-only`).
- Regenerated `results/phase1-report/` from `results/final-20261011/*/` without touching the originals (`git status` of that path unchanged).

## Regenerated October headline (for plan 01-04 viva notes)

Command: `venv/bin/python run_benchmark.py --report-only results/phase1-report --merge results/final-20261011/*/`

| Model | Done/342 | Status | Attack n (unprot) | Raw ASR no guardrails | ASR-valid no guardrails | Raw ASR guardrails | ASR-valid guardrails | Blocked / attempted (protected) |
|---|---|---|---|---|---|---|---|---|
| ollama-llama3.2-3b | 342/342 | complete | 120 | 5.8% (7/120) [2.9, 11.6] | 5.8% (7/120) | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) | 100.0% (12/12) [75.7, 100.0] |
| gemini-flash-lite | 195/342 | partial | 47 | 10.6% (5/47) [4.6, 22.6] | 10.6% (5/47) | 0.0% (0/46) [0.0, 7.7] | 0.0% (0/46) | 100.0% (6/6) [61.0, 100.0] |
| groq-gpt-oss-120b | 214/342 | partial | 56 | 10.7% (6/56) [5.0, 21.5] | 10.7% (6/56) | 0.0% (0/56) [0.0, 6.4] | 0.0% (0/56) | 100.0% (5/5) [56.6, 100.0] |
| groq-qwen3.8-27b | 202/342 | partial | 50 | 0.0% (0/50) [0.0, 7.1] | 0.0% (0/50) | 0.0% (0/50) [0.0, 7.1] | 0.0% (0/50) | 0/0 |
| groq-gpt-oss-20b | 112/342 | partial | 5 | 40.0% (2/5) [11.8, 76.9] | 40.0% (2/5) | 0.0% (0/5) [0.0, 43.4] | 0.0% (0/5) | 100.0% (2/2) [34.2, 100.0] |
| gemma-4-31b | 156/342 | error-dominated | 19 | 0.0% (0/19) [0.0, 16.8] | 0.0% (0/19) | 0.0% (0/19) [0.0, 16.8] | 0.0% (0/19) | 0/0 |

- Pooled conditional block rate: 100.0% (25/25) [86.7, 100.0]; no attempted episode went unblocked and none succeeded despite guardrails.
- ollama-llama3.2-3b is the only complete model: 14/120 attempted (11.7% [7.1, 18.6]), benign task 100.0% (51/51) unprotected and 94.1% (48/51) protected, false positive 0.0% (0/51).
- gemma-4-31b: 186 infra-errored rows (74 + 79 attack, 16 + 17 benign); after pair dropping only 19 attack pairs remain, so its 0% has an upper bound of 16.8%.
- groq-gpt-oss-20b: all 5 attack episodes are `dh`; its 40% attempted rate rests on 5 episodes (interval 11.8-76.9%). Zero `ds` episodes exist for gemini-flash-lite, groq-gpt-oss-120b, groq-gpt-oss-20b and groq-qwen3.8-27b.
- Raw ASR equals ASR-valid for every model on the attack arm: no attack episode in the October data is a format_failure or refusal. The one non-ignored benign row is a single format_failure for groq-gpt-oss-20b.

## Task Commits

1. **Task 1: Wilson k/n cells, completeness gate, partial-run segregation** - `124e725` (feat)
2. **Task 2: Outcome taxonomy, ASR-valid, pair dropping, conditional block rate** - `7e17e02` (feat)

## Files Created/Modified

- `src/benchmark/stats.py` - Wilson interval, `prop_fields`, `format_prop`, `Z_95`
- `src/benchmark/outcomes.py` - `OUTCOMES`, `FORMAT_VALID`, `classify_llm_error`, `classify_outcome`, `drop_infra_pairs`
- `src/benchmark/report.py` - reworked: `load_rows_counted`, `_cell`, `_error_bars`, `_classified`/`_usable`, new sections and columns, error-bar charts
- `run_benchmark.py` - `--expected` flag
- `tests/test_report.py` - 47 new tests (125 total pass under WSL; baseline was 78)

## Decisions Made

See `key-decisions` above. One interpretation worth restating for plan 01-03: a pair with an infra-errored arm stays out of the paired statistics until the errored arm is re-run successfully; the good sibling stays on disk and is not re-run.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan constant] Acceptance string "100.0% (12/12) [75.8, 100.0]" is arithmetically 75.7**
- **Found during:** Task 2 acceptance check
- **Issue:** Wilson lower bound for 12/12 at z=1.96 is 12 / (12 + 1.96^2) = 0.757499..., which rounds to 75.7, not 75.8. The pooled string "100.0% (25/25) [86.7, 100.0]" matches exactly. The plan said to re-derive and record, not to adjust the code.
- **Fix:** None to code; verified 75.7 independently. The report prints "100.0% (12/12) [75.7, 100.0]".
- **Files modified:** none
- **Verification:** `python3 -c "print(12/(12+1.96**2))"` gives 0.7574992

### Additions within the plan's intent

**2. [Rule 2 - Missing detail] Extra summary keys**
- `attack_pairs` (feeds the plan's "Paired attack episodes" column; `attack_episodes` keeps its old meaning) and `benign_outcomes_{arm}_{outcome}` (needed for the benign rows of "Outcomes per episode"). Identical key set across models is preserved.

**3. [Rule 2 - Hardening] `_cell` also escapes backslashes** so a model name ending in `\` cannot swallow the pipe escape. `load_rows_counted` also counts rows missing `model/case_id/run/arm` or non-dict JSON as skipped.

**4. [Observation] Task 1 numbers shifted after Task 2**: gemma-4-31b attack n went from 46/41 to 19/19 and its policy_engine block count from 2 to 0 once pair dropping removed the arms whose sibling was infra-errored. This is the intended effect of EVAL-04, not a regression.

**Total deviations:** 1 plan-constant discrepancy, 3 small additions. **Impact:** none on scope; all acceptance criteria except the 75.8 literal match.

## Issues Encountered

- An initial heredoc used to append tests was rejected by the shell quoting; switched to the Edit tool. No files were affected.
- Git warns "LF will be replaced by CRLF" for the new files (autocrlf); no content impact.

## Known Stubs

None.

## Threat Flags

None. The report only reads local `raw.jsonl`; T-01-01 to T-01-04 mitigations are implemented (read-only inputs verified by `git status`, completeness gate, `_cell` escaping, undecodable-line counting).

## Next Phase Readiness

- Plan 01-03 (runner) can import `classify_llm_error` for re-queue/done-set logic; the report already tolerates absent `finish_reasons`, `prompt_tokens`, `code_commit`, `policy_sha256`, `provider`, `resolved_model`.
- Plan 01-04 should regenerate `results/phase1-report` after adding the control row and quote the table above.

## Self-Check: PASSED

- Files found: stats.py, outcomes.py, report.py, run_benchmark.py, test_report.py, results/phase1-report/{summary.md, summary.csv, charts/attack_outcomes.png}
- Commits found in HEAD ancestry: 124e725, 7e17e02
- `check evaluation-scope --plan 01-01 --commits-only` resolved both commits
- Full WSL suite: 125 passed
- All task acceptance criteria re-run; only the literal "[75.8" differs (see deviation 1)
