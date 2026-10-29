---
gsd_state_version: "1.0"
current_phase: 4
current_phase_name: Full Runs - Local, Added Ollama, Hosted Resume
status: verifying
stopped_at: Completed 05-04-PLAN.md
last_updated: "2026-10-29T08:52:51.543Z"
last_activity: 2026-10-29
last_activity_desc: "Phase 4 executed (5/5 plans), awaiting review; Groq final regeneration after the 15:30 UTC freeze pending"
state_head: 4bdf802263fd4f656056f27ce2d17e3462f8b1ad
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 21
  completed_plans: 21
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-22)

**Core value:** The guardrails must be shown to hold independently of which LLM drives the agent, including a weak, unaligned, locally made model, backed by a fair, reproducible comparison table the student can defend in a viva.
**Current focus:** Phase 4 — Full Runs - Local, Added Ollama, Hosted Resume (executed, awaiting review)

## Current Position

Phase: 4 (Full Runs - Local, Added Ollama, Hosted Resume) — EXECUTED, awaiting review
Plan: 5 of 5 (all plans have a SUMMARY)
Status: Phase 4 executed — awaiting code review and verification; the Groq re-run continues until the 2026-10-29 15:30 UTC freeze, after which results/phase4-report is regenerated and viva Part C updated (CMP-03 open)
Last activity: 2026-10-29 — 04-04 complete on the 03:06Z Groq snapshot; full suite 1242 passed, 1 skipped

Progress: [████░░░░░░] 40% (Phases 1-2 complete; Phases 3-4 executed, awaiting verification)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: - min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | - | - |
| 2 | 4 | - | - |

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
| Phase 01 P04 | 35 min | 2 tasks | 7 files |
| Phase 2 P01 | 20 min | 3 tasks | 8 files |
| Phase 02 P02 | 26min | 3 tasks | 5 files |
| Phase 02 P03 | 173 min | 3 tasks | 7 files |
| Phase 02 P04 | 58 min | 3 tasks | 5 files |
| Phase 03 P01 | 21 min | 3 tasks | 9 files |
| Phase 03 P02 | 30min | 3 tasks | 4 files |
| Phase 03 P03 | n/a | 3 tasks | 6 files |
| Phase 03 P04 | multi-session | 3 tasks | 7 files |
| Phase 04 P01 | 33min | 3 tasks | 6 files |
| Phase 04 P05 | 24min | 3 tasks | 9 files |
| Phase 04 P03 | 7h25m wall | 3 tasks | 1 files |
| Phase 04 P02 | 8min | 3 tasks | 3 files |
| Phase 04 P04 | 1h05m | 3 tasks | 4 files |
| Phase 05 P01 | 50min | 3 tasks | 11 files |
| Phase 05 P02 | 15min | 3 tasks | 2 files |
| Phase 05 P03 | 45min | 3 tasks | 3 files |
| Phase 05 P04 | 33min | 3 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5 phases (research-proposed structure kept; each is a distinct delivery boundary and the two local-model phases are separate serial GPU jobs)
- [Roadmap]: EVAL-08 (layer-targeted attack suite) placed in Phase 4 as CPU-side work alongside background GPU runs; EVAL-07 control agent stays in Phase 1
- [Roadmap]: DOC-03 (viva notes) traced to Phase 5, but every phase's success criteria require its viva page (definition of done)
- [Roadmap]: Policy fix (SEC-01) precedes all final protected-arm runs; policy-file hash is stamped on every result record
- [Phase 01]: [01-01] episodes_done counts finished episodes (outcome != llm_error) before pair dropping; provider content-filter refusals are outcome 'refused' and stay in denominators; blocked_given_attempt superseded by review fix WR-04: the attack action itself was blocked and the attack did not succeed
- [Phase 01]: Own shell lexer instead of shlex.split in policy_engine/shell_split.py; unparseable input, dynamic command/redirect names, heredocs and ANSI-C quoting fail closed
- [Phase 01]: Dashboard defaults to 127.0.0.1:5000 with debug off; DASHBOARD_HOST/PORT/DEBUG env vars opt in
- [Phase 01]: 01-03: report completeness denominator precedence is explicit expected, then source manifest.json, then 342; merged dirs use the largest manifest value per model
- [Phase 01]: 01-03: infra errors get one bounded re-queue pass per invocation (skipped after quota stop) and are retried on every resume; provider refusals are outcomes, never retried
- [Phase 01]: 01-04: Control rows (provider scripted) are excluded from every LLM report table and chart, not only headline and pooled; attribution sentence scoped to the control (sandbox/seccomp not reached by it)
- [Phase 2]: Plan 02-01: write_episodes preserves key insertion order (no sort_keys) so call_tool args keep the spec's required order and the gold action text keeps the documented action/tool/args grammar
- [Phase 2]: Plan 02-01: leakage gate checks 8-word shingles under two normalisations (union 2780, research-only 2035) and benchmark sentences in U:, O: and A: lines; signature rule applies to synthetic tools only
- [Phase 02]: D-02 read as 10% of all episodes, drawn only from the benign half; sinks without a recipient parameter get an injection without an address (verbatim-copy rule wins)
- [Phase 02]: Dataset manifest committed after the CLI commit so code_commit names it; train sha256 63a192b1..., dev sha256 58f5bcb0... (seed 0, 40000+2000)
- [Phase 02]: 02-03: non-finite loss is checked right after the forward pass (before the optimiser step); train exits 3 with no final manifest
- [Phase 02]: 02-03: load_generator enforces the exact checkpoint key set and five stamps; mismatches (incl. cfg/weights misfit) raise RuntimeError so the runner skips the model
- [Phase 02]: 02-04: the pre-registered dev-only rule triggered after run 1 (dev line exact 0.167 < 0.50, still improving); one retry at 14000 steps gave the kept model (best step 6000, dev loss 0.5720, checkpoint 83b0763a6d26)
- [Phase 02]: 02-04: smoke run is plumbing only (26 rows stamped scratch-gpt@83b0763a6d26, parse rate 76/80, attack attempts 4/10 per arm); no setting changed after it
- [Phase 03]: resolved_model = <base>@<rev12>, plus +<name>@<adapter_sha12> for an adapter row (base revision is the always-present identity)
- [Phase 03]: hf_local checks adapter files before any HF import and verifies hash, base id, revision and render fingerprint at construction; trainer must write base_model_name_or_path as the base id
- [Phase 03]: 03-02: loss mask from tokenizer character offsets, never token ids; action-level mask excludes only respond message text (boundary BPE token excluded whole)
- [Phase 03]: 03-02: PEFT all-linear resolves to full per-layer module names, targets verified by leaf name plus count 7 x layers; LORA-03 stays unchecked until the 03-04 GPU run
- [Phase 03]: 03-04: D-05 600 steps kept (pilot estimate 42.5 min); D-06 retry not needed (dev_action_exact 0.9396); LORA-04 dev gate pass (parse 1.0000); decided before smoke runs
- [Phase 04]: Ollama context raised by Modelfile variants (num_ctx 8192); owned server also gets OLLAMA_CONTEXT_LENGTH=8192
- [Phase 04]: Measured Ollama 0.40.3 default context for llama3.2:3b is 4096 (api_ps); evidence band 4198-7782
- [Phase 04]: Ollama 0.40.3 cuts over-long prompts to about half the window (2050 at 4096): truncation shows as a drop, not a plateau
- [Phase 04]: 04-05: local queue order agentlora, scratch-gpt, qwen base, ollama qwen2.5:3b, ollama llama3.2:3b; FREEZE_UTC 2026-10-29T15:30Z constant; done code 7 = freeze stop
- [Phase 04]: 04-05: local context_overflow is a finished format_failure (attacked if attempted), never re-queued; --priority-kinds is order-only (manifest extra, not plan)
- [Phase 04]: 04-05: hosted loop applies only D-09 (freeze + QuotaExhausted: retry+120s or 60 min; next launch at/after freeze = stopped_by_quota); stall after 3 zero-progress launches; stopping_rule.json write-once
- [Phase 04]: 04-03: new 8192-context llama3.2:3b run supersedes October rows; 0/342 episodes above the measured 4096 default
- [Phase 04]: 04-03: local queue and Groq loop launched detached (setsid nohup) with a hidden keepalive wsl.exe; Groq loop runs to the 2026-10-29 15:30 UTC freeze
- [Phase 04]: 04-02 EVAL-08: recorded sandbox and seccomp blocks are 0/1147 pooled over the 8 language models at report time (0/855 over the five complete local models); reported as zero with n, not padded (D-07b)
- [Phase 4]: Phase 4 Task 3 ran on a 2026-10-29T03:06Z snapshot by user decision; the Groq loop runs on to the 15:30 UTC freeze and the report is regenerated afterwards
- [Phase 4]: CMP-01 and CMP-02 complete (five local models 342/342); CMP-03 open until the post-freeze Groq regeneration
- [Phase 05]: 05-01: numbers.json is computed before the report; the build fails if generate_report's summary differs from it
- [Phase 05]: 05-01: constant pins (FREEZE_UTC, HOSTED_DS_EXPECTED, SECRET_RE, DEFAULT_MAX_CONTEXT_TOKENS) read the source with ast, so nothing imports jobs/run_checks/src.llm
- [Phase 05]: 05-01: unscoped deliverables check fails on viva page 04 until plan 05-03 (expected baseline); use --doc README.md --doc docs/SDD.md meanwhile
- [Phase 05]: 05-02: README and SDD quote result numbers only through generated blocks; SDD 3.0 adds 5.13-5.15, 7.3 (9 blocks) and Appendix C; SDD PDF/DOCX export is a manual step (needs network)
- [Phase 05]: 05-03: viva page 05 Part F answers point to Part D for every result number; page 04 prose quotes no Groq or pooled number (only generated blocks do)
- [Phase 05]: 05-04: Review-3 deck uses the template's plain BLANK layout (its layouts carry gradient decoration) with the chart surface background and one blue accent; build draws the deck only when the CLI passes deck_out

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3] Python 3.14 compatibility of transformers/peft/accelerate is unverified; run the 15-minute smoke test (install, load Qwen, one LoRA step) early. Fallback: `uv` Python 3.12 venv with the cu128 torch build
- [Phase 2→3] Reuse the Phase 2 dataset (sha a2586548c7f7) for Qwen LoRA; WSL now 11 GB + 8 GB swap and shared with another project's CPU job — keep GPU preflight + resumable training
- [Phase 2→5] Smoke behaviour (3/10 unprotected attempts) and ds dev accuracy 0.68 are disclosures for Phase 4/5, not Phase 2 gaps
- [Phase 4] Hosted Groq quota may not allow completing all three models within the week; stopping rule must be written before resuming
- [Phase 4] Pending follow-up: after the 2026-10-29 15:30 UTC freeze, regenerate results/phase4-report, run secret-scan and run_checks hosted, update viva Part C Groq rows and EVAL-08 pooled n, re-run run_checks viva, then tick CMP-03 (WINDOWS.md entry 2)
- [Phase 4] Existing llama3.2:3b and gemma-4-31b rows may be affected by context truncation or API timeouts; re-score from `raw.jsonl`
- [Phase 4] GPU is serial: stop Ollama during training; scratch training, then LoRA training, then benchmark runs
- [Docs] REQUIREMENTS.md originally stated 26 v1 requirements; the actual count is 27 (corrected at roadmap creation)
- [Phase 1→5] results/phase1-control was produced before the review fixes (policy hash 9dcdaaf1…); re-run it before the Phase 5 data freeze so stamps match the final engine (numbers expected unchanged)
- [Phase 1→4] Stricter policy may over-block benign dynamic operands (e.g. head -n $N file); watch benign utility / FP rate in Phase 4
- [Phase 1] Review Info items IN-01 ("429" substring rate-limit match), IN-02 (non-string final_message), IN-04 (retried errors invisible in report) remain open, low severity

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261028-th2 | Remake guardrails dashboard as a reference-monitor console | 2026-10-28 | 3881dfb | [261028-th2-redesign-guardrails-dashboard-ui](./quick/261028-th2-redesign-guardrails-dashboard-ui/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-10-29T08:52:50.236Z
Stopped at: Completed 05-04-PLAN.md
Resume file: None
