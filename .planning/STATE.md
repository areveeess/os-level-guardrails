---
gsd_state_version: "1.0"
current_phase: 1
current_phase_name: Evaluation Hardening & Policy Fix
status: executing
stopped_at: Completed 01-03-PLAN.md
last_updated: "2026-10-20T18:40:17.733Z"
last_activity: 2026-10-20
last_activity_desc: Phase 1 execution started
state_head: 1de7488afe86dffa33130f990d7b22700c18df17
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 4
  completed_plans: 3
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-20)

**Core value:** The guardrails must be shown to hold independently of which LLM drives the agent, including a weak, unaligned, locally made model, backed by a fair, reproducible comparison table the student can defend in a viva.
**Current focus:** Phase 1 — Evaluation Hardening & Policy Fix

## Current Position

Phase: 1 (Evaluation Hardening & Policy Fix) — EXECUTING
Plan: 4 of 4
Status: Ready to execute
Last activity: 2026-10-20 — Phase 1 execution started

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: - min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 9 min | 2 tasks | 5 files |
| Phase 01 P02 | 50 min | 3 tasks | 6 files |
| Phase 01 P03 | 15 min | 3 tasks | 9 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5 phases (research-proposed structure kept; each is a distinct delivery boundary and the two local-model phases are separate serial GPU jobs)
- [Roadmap]: EVAL-08 (layer-targeted attack suite) placed in Phase 4 as CPU-side work alongside background GPU runs; EVAL-07 control agent stays in Phase 1
- [Roadmap]: DOC-03 (viva notes) traced to Phase 5, but every phase's success criteria require its viva page (definition of done)
- [Roadmap]: Policy fix (SEC-01) precedes all final protected-arm runs; policy-file hash is stamped on every result record
- [Phase 01]: [01-01] episodes_done counts finished episodes (outcome != llm_error) before pair dropping; provider content-filter refusals are outcome 'refused' and stay in denominators; blocked_given_attempt = attempted protected episodes with non-empty blocked_by
- [Phase 01]: Own shell lexer instead of shlex.split in policy_engine/shell_split.py; unparseable input, dynamic command/redirect names, heredocs and ANSI-C quoting fail closed
- [Phase 01]: Dashboard defaults to 127.0.0.1:5000 with debug off; DASHBOARD_HOST/PORT/DEBUG env vars opt in
- [Phase 01]: 01-03: report completeness denominator precedence is explicit expected, then source manifest.json, then 342; merged dirs use the largest manifest value per model
- [Phase 01]: 01-03: infra errors get one bounded re-queue pass per invocation (skipped after quota stop) and are retried on every resume; provider refusals are outcomes, never retried

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3] Python 3.14 compatibility of transformers/peft/accelerate is unverified; run the 15-minute smoke test (install, load Qwen, one LoRA step) early. Fallback: `uv` Python 3.12 venv with the cu128 torch build
- [Phase 2] Training-data compliance mix (50/25/25) determines weak-model behaviour under attack; decide and disclose in the report; same dataset for both local models
- [Phase 4] Hosted Groq quota may not allow completing all three models within the week; stopping rule must be written before resuming
- [Phase 4] Existing llama3.2:3b and gemma-4-31b rows may be affected by context truncation or API timeouts; re-score from `raw.jsonl`
- [Phase 4] GPU is serial: stop Ollama during training; scratch training, then LoRA training, then benchmark runs
- [Docs] REQUIREMENTS.md originally stated 26 v1 requirements; the actual count is 27 (corrected at roadmap creation)

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-10-20T18:40:17.641Z
Stopped at: Completed 01-03-PLAN.md
Resume file: None
