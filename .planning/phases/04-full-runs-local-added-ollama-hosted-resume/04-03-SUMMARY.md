---
phase: 04-full-runs-local-added-ollama-hosted-resume
plan: 03
subsystem: benchmark-runs
tags: [full-runs, local-models, ollama, ctx8192, truncation-recheck, groq, stopping-rule, freeze, ds-first]
status: complete

requires:
  - phase: 04-full-runs-local-added-ollama-hosted-resume
    provides: "04-01: Ollama ctx8192 variants, measured default context 4096; 04-05: jobs local/hosted/status/report, run_checks hosted/secret-scan, FREEZE_UTC"
provides:
  - "results/phase4-{qwen2.5-0.5b-agentlora, scratch-gpt, qwen2.5-0.5b-base, ollama-qwen2.5-3b, ollama-llama3.2-3b}: 342/342 each, complete-ok"
  - "results/phase4-report-local: interim report (5 complete local rows + 3 partial Groq rows, labelled partial)"
  - "04-HOSTED-STOPPING-RULE.md (committed before the first hosted request)"
  - "results/phase4-groq-*: fresh Groq re-run, ds first, still running under the D-09 rule"
affects: [04-02, 04-04, 05-report]

actuals:
  tokens: 1500           # chars/4 over the realized diff 3b60b93..09a6638 (5,923 chars; runs produce untracked data only)
  tasks: 3
  commits: 2             # git rev-list --count 3b60b93..HEAD before the docs commit
plan_head_before: 3b60b93ee0af97e0003130d14cb828419f847b6c
plan_head_after: 09a6638a619adfc1013a073a5824ee73abef218e

tech-stack:
  added: []
  patterns:
    - "Detached, re-entrant launchers (setsid nohup, pgrep guard) for the local queue and the hosted loop"
    - "A hidden Windows-side wsl.exe keepalive client that only reads the process table and exits on its own"

key-files:
  created:
    - .planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-HOSTED-STOPPING-RULE.md
  modified: []

key-decisions:
  - "The new 8192-context llama3.2:3b run supersedes the October rows for every later report; at the measured default of 4096 the re-run had 0/342 episodes with a step above it"
  - "Local and hosted loops were launched detached (setsid nohup) instead of as one attached run_in_background wsl.exe, so they outlive the executor; a hidden keepalive wsl.exe holds the VM"
  - "CMP-01/02/03 left unticked by orchestrator instruction; the verifier decides"

requirements-completed: []
requirements-advanced: [CMP-01, CMP-02, CMP-03]

duration: ~7h25m wall (local queue 1h45m; executor active about 30 min across two sessions)
completed: 2026-10-29
---

# Phase 4 Plan 03: Full local runs and concurrent Groq re-run Summary

**All five local models finished 342/342 in one uninterrupted local queue (18:45Z to 20:30Z), with stamps checked and the two Ollama rows confirmed at an 8,192-token window. The llama3.2:3b re-check found 0/342 episodes above the measured 4,096-token default. With guardrails, attack success is 0 for every local model. The fresh Groq re-run (ds first) started under a stopping rule committed beforehand. It is still running toward the 2026-10-29 15:30 UTC freeze. An overnight network outage left pending infra rows, which the loop is retrying.**

## Performance

- **Started:** 2026-10-28T18:44Z (preconditions); local queue launched 18:45:39Z; hosted loop 18:48:02Z
- **Local queue finished:** 2026-10-28T20:30:36Z (`LOCAL done: 5/5 complete`); Task 3 checks 2026-10-29T02:08Z
- **Tasks:** 3/3
- **Tracked files:** 1 (the stopping-rule file, two commits)

## Local queue (Task 1 and Task 3)

Launch: `setsid nohup venv/bin/python -u -m src.benchmark.jobs local --prefix results/phase4`, WSL pid 10038. The launcher refuses to start a second copy.
- **First preflight:** PREFLIGHT ok (free_mib 4801, avail_ram_mib 8882). KEEP_AWAKE logged with boot_id 51d2482c-9f95-4d1f-a636-9b8bc18c4b71 and uptime_s 18547.
- **First rows:** resolved_model `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf+qwen2.5-0.5b-agentlora@924ac021f0cb`.
- **Relaunches:** 0. Every model ran in one invocation and has `done 0`.
- **WARN no-progress:** none in any status poll. The boot_id did not change, and no sleep interruption was seen.
- **Ollama server:** the queue started its own server (`ollama owned-started`) and stopped it at the end (`ollama-down stopped`). Afterwards `pgrep "ollama serve"` was empty.

| Model | Invocation start | Last row | Wall time | context_overflow | complete check |
|-------|------------------|----------|-----------|------------------|----------------|
| qwen2.5-0.5b-agentlora | 18:45:39Z | 19:02:37Z | ~17 min | 0 | complete-ok, resolved …@924ac021f0cb |
| scratch-gpt | 19:02:55Z | 19:09:44Z | ~7 min | 0 | complete-ok, resolved scratch-gpt@97bdace56294 |
| qwen2.5-0.5b-base | 19:10:00Z | 19:57:26Z | ~47 min | 0 | complete-ok, resolved Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf |
| ollama-qwen2.5-3b | 19:58:05Z | 20:11:05Z | ~13 min | 0 | complete-ok, resolved qwen2.5-3b-ctx8192 |
| ollama-llama3.2-3b | 20:16:46Z | 20:30:36Z | ~14 min (+~5 min variant setup) | 0 | complete-ok, resolved llama3.2-3b-ctx8192 |

Every row has 342/342 finished, pending_infra 0, 0 duplicates, policy c8afbb5d345a and 1 code commit. No local model was stopped at the freeze, so success criterion 1 has no gap.

### Ollama window evidence (D-06, CMP-01)

| Model | truncation --num-ctx 8192 | ollama-evidence |
|-------|---------------------------|-----------------|
| ollama-qwen2.5-3b | truncation-ok: 951 steps, max 1002, p50 434, p95 678, at_or_above_95pct 0, length_finish 1, decreasing 0 | ollama-evidence-ok: path primary, D 4096, band 4198-7782, long prompt 6402 |
| ollama-llama3.2-3b | truncation-ok: 1251 steps, max 1042, p50 498, p95 712, at_or_above_95pct 0, length_finish 0, decreasing 0 | ollama-evidence-ok: path primary, D 4096, band 4198-7782, long prompt 5273 |

### llama3.2:3b truncation re-check (CMP-01)

- **October rows** (`results/final-20261011/ollama-llama3.2-3b`, `--num-ctx 4096 --compare 2048`): `no row carries prompt_tokens`, so the result is **truncation-unknown**.
- **New run** (`--num-ctx 8192 --compare 4096`): **k = 0/342** episodes had a step above the measured default. The largest prompt was 1,042 tokens.
- **Where the default comes from:** 4,096 tokens, read from `results/phase4-jobs/ollama_default_context.json`. evidence_source is `api_ps`, and the `ollama ps` column agrees. default_is_lower_bound is false. Ollama server 0.40.3, measured 2026-10-28T18:04:08Z on a temporary port.
- **Conclusion:** k is 0, so at the measured default the October run was not affected by truncation. Even so, every later report uses the new 8,192-context run (D-05).
- **Version caveat:** the default was measured on today's server, 0.40.3. The October server version is not recorded. The conclusion holds at the measured default only.

## Headline numbers (results/phase4-report-local/summary.md)

Each model has 120 attack episodes per arm and 51 benign episodes per arm.

| Model | Attempted (no guardrails) | Success no guardrails | Success with guardrails | ASR-valid no guardrails | Benign task no / with guardrails | Blocked given attempt | Parse errors per step |
|-------|---------------------------|-----------------------|-------------------------|-------------------------|-------------------------------|-----------------------|-----------------------|
| qwen2.5-0.5b-agentlora | 111/120 (92.5%) | 111/120 (92.5%) | 0/120 | 111/120 (92.5%) | 51/51, 51/51 | 111/111 | 6/978 (0.6%) |
| scratch-gpt | 39/120 (32.5%) | 33/120 (27.5%) | 0/120 | 33/120 (27.5%) | 36/51, 36/51 | 36/36 | 136/993 (13.7%) |
| qwen2.5-0.5b-base | 6/120 (5.0%) | 3/120 (2.5%) | 0/120 | 3/81 (3.7%) | 15/51, 15/51 | 6/6 | 803/1485 (54.1%) |
| ollama-qwen2.5-3b | 5/120 (4.2%) | 3/120 (2.5%) | 0/120 | 3/105 (2.9%) | 49/51, 49/51 | 10/10 | 150/951 (15.8%) |
| ollama-llama3.2-3b | 14/120 (11.7%) | 7/120 (5.8%) | 0/120 | 7/120 (5.8%) | 48/51, 50/51 | 9/9 | 47/1251 (3.8%) |

- **Pooled conditional block rate:** 200/200 across all LLM models, the Groq partial rows included.
- **scratch-gpt baseline:** 21/120 attack episodes called the attacker's tool before the injection could have influenced it. These are not counted as attacks.

**Format-failure episodes** (unprotected + protected, out of 342):

| Model | Unprotected | Protected | Total |
|-------|-------------|-----------|-------|
| qwen2.5-0.5b-agentlora | 0 | 0 | 0 |
| scratch-gpt | 0 | 0 | 0 |
| qwen2.5-0.5b-base | 39 attack + 21 benign | 45 attack + 21 benign | 126 |
| ollama-qwen2.5-3b | 15 | 9 | 24 |
| ollama-llama3.2-3b | 0 | 1 | 1 |

**Where blocks happened (protected arm):**
- Tool broker and policy engine blocks:

| Model | Tool broker | Policy engine |
|-------|-------------|---------------|
| agentlora | 117 | 0 |
| scratch-gpt | 69 | 0 |
| base | 9 | 24 |
| qwen2.5-3b | 13 | 12 |
| llama3.2-3b | 9 | 0 |

- **Sandbox and seccomp: 0 blocks for every model.** This is the D-07b input for 04-02, and it is reported as zero, not padded.

`jobs report` merges every `results/phase4-<model>` directory, so this interim report also lists the three Groq directories. They appear under "Partial or error-dominated runs (not compared)" and are not ranked against the complete rows.

## Hosted re-run (Task 2)

- **Rule commit:** `d4ee817` at 2026-10-28T18:47:27Z. The probe followed at 18:47:38Z and the first hosted invocation header at 18:48:03Z, so the rule was committed before any hosted request. The launch table commit is `09a6638`.
- **stopping_rule.json:** freeze_utc 2026-10-29T15:30:00+00:00, rule_sha d4ee817abc0c679ded2a28a296ef4b91e818a8fd, written 2026-10-28T18:48:03Z, superseded `results/final-20261011`.
- **KEY check:** `KEY GROQ_API_KEY set`. Presence only; .env was never opened.
- **Probe:** `PROBE groq-gpt-oss-120b ok`, `groq-gpt-oss-20b ok`, `groq-qwen3.8-27b ok`. No model was unavailable and none was substituted.
- **Launch:** `setsid nohup venv/bin/python -u -m src.benchmark.jobs hosted --prefix results/phase4`, loop pid 10209. Each manifest records `priority_kinds: ["ds"]`.
- **At Task 2's first poll (18:53Z):** 35, 22 and 43 finished, all ds.
- **Progress at 2026-10-29T02:08Z** (`run_checks hosted`, which prints hosted-running):

| Model | State | Finished | ds | Quota stops | Label |
|-------|-------|----------|----|-------------|-------|
| groq-gpt-oss-120b | running | 198/342 | 120/120 | 0 | partial n=198, ds n=120 |
| groq-gpt-oss-20b | running | 91/342 | 82/120 | 0 | partial n=91, ds n=82 |
| groq-qwen3.8-27b | running | 209/342 | 120/120 | 0 | partial n=209, ds n=120 |

- **Overnight outage:** a network outage left pending_infra rows (147 / 252 / 136 at 02:08Z). The error text is `APIConnectionError: Connection error.`
  - The loop handled it under the rule: exit 0, state `waiting_retry`, relaunch after 10 minutes.
  - qwen3.8-27b and gpt-oss-120b each had one zero-progress relaunch (#2) before launch #3 made progress again. The stall threshold of 3 consecutive zero-progress launches was not reached.
  - No quota stop has happened yet.
- **Ownership:** the loop was left running and was not signalled. Plan 04-04 waits for it to reach the freeze or a terminal state.
- **Secret scan:** `secret-scan-ok files 29` at launch and `secret-scan-ok files 36` at 02:08Z, over the three Groq directories, the jobs directory and the rule file.

## Task Commits

1. **Task 1: launch the local queue** - no tracked file (untracked results only)
2. **Task 2: stopping rule, then launch the Groq loop** - `d4ee817` (docs, rule) and `09a6638` (docs, launch table)
3. **Task 3: verify the local directories, llama re-check, local report** - no tracked file (untracked results only)

## Deviations from Plan

**1. [Rule 3 - Blocking, orchestrator instruction] Detached launch instead of an attached run_in_background wsl.exe**
- **Plan:** each loop runs as one foreground wsl.exe under the Bash tool's run_in_background, polled by the same executor for hours.
- **What happened:** the orchestrator required launches that outlive the executor. Both loops were started with `setsid nohup` through re-entrant launcher scripts in the session scratchpad. Each script uses a pgrep guard, and the hosted launcher refuses after the freeze.
- **Keepalive:** a hidden Windows `wsl.exe` keepalive (Windows pid 43868) holds a client attached so the VM is not idled out. It runs `keepalive.sh`, only reads the process table, and exits on its own when both loops are gone or at 16:00Z.
- **Polling:** done by the orchestrator, not inside this agent.

**2. [Note] The rule file quotes D-03 and D-09 verbatim**
- The plan required the verbatim quotes, so the words "72-hour window" and "quota exhausted twice" appear in the quoted source section.
- The rule as applied has neither clause and no STOP file.

**3. [Note] The interim local report also contains the Groq partial rows**
- `jobs report --prefix results/phase4` merges every final directory under the prefix.
- The Groq rows are labelled partial or error-dominated and kept out of the headline table.

**4. [Bookkeeping] CMP-01, CMP-02 and CMP-03 are not ticked**
- This follows the orchestrator's instruction.
- CMP-01 and CMP-02 have met their evidence here. CMP-03 is still running.

## Known Stubs

None.

## Threat Flags

None. No new code or surface. The launchers start this project's own loops and signal nothing. The keepalive only reads.

## Next Steps

- 04-02 (EVAL-08, D-07b) can count the sandbox and seccomp blocks now. The count is 0 for every local model so far.
- 04-04 waits for the Groq loop to reach the freeze (2026-10-29 15:30 UTC) or a terminal state, then writes the merged report.

## Self-Check: PASSED

- The files exist: 04-HOSTED-STOPPING-RULE.md, results/phase4-report-local/summary.md, results/phase4-jobs/stopping_rule.json, and the five results/phase4-<local model> directories.
- Commits d4ee817 and 09a6638 are ancestors of HEAD. Neither deletes a file.
