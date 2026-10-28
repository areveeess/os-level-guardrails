# Phase 4: Hosted Stopping Rule (Groq re-run)

**Written:** 2026-10-28, before the first hosted request of plan 04-03 other than the probe
**Applies to:** groq-gpt-oss-120b, groq-gpt-oss-20b, groq-qwen3.8-27b
**Implemented by:** `python -m src.benchmark.jobs hosted --prefix results/phase4` (`RULE_TEXT`, `FREEZE_UTC` in `src/benchmark/jobs.py`)

This file is the pre-registered rule for when the hosted re-run stops. It is committed before the hosted loop sends its first request. After that, the rule is not changed.

## Source decisions (verbatim from 04-CONTEXT.md)

- **D-03:** Written stopping rule for hosted runs: a model stops when its daily quota is exhausted twice (two QuotaExhausted stops on separate days/resets) or when the Phase 4 window ends; whatever remains is reported as partial with its n. Groq calls use the user's existing free-tier key; never print or store key values.
- **D-08:** Groq models are re-run fresh under the current engine, data-stealing (ds) episodes first, into fresh result dirs. The October partial rows in results/final-20261011 are superseded and kept as history only: they predate the current scoring and stamps and have no prompt_tokens. CMP-03 is read as "re-run under the current engine, ds first, under the stopping rule; incomplete models reported as partial with their n".
- **D-09:** Hard data freeze at **2026-10-29 15:30 UTC (21:00 IST)**, 24 hours from this decision. This replaces the 72-hour window and the "quota exhausted twice" clause of D-03. At the freeze every run stops, and whatever is unfinished is reported as partial with its n. Because GPU-bound local runs and network-bound Groq runs do not compete for the GPU, the Groq loop may start as soon as 04-01's tooling is ready and run concurrently with the local queue. This amends D-04's local-then-hosted ordering. A Groq model also stops early at its first QuotaExhausted that falls after the last daily reset before the freeze.

## The rule as applied

D-09 replaces D-03's time window and its quota clause. Only the two parts below decide when a model stops.

**(a) Hard freeze at 2026-10-29 15:30 UTC (21:00 IST).**
- Nothing is launched at or after the freeze.
- At the freeze, the hosted loop sends SIGTERM only to the run_benchmark children it started itself. If one is still alive 60 s later, the loop sends SIGKILL to the same pid.
- After the freeze the loop launches nothing more. Every model that is not complete is recorded as `stopped_at_freeze`.

**(b) QuotaExhausted handling.** A child invocation that ends with the runner's quota stop line is handled as follows:
- **Stated retry time:** the next launch is the stop time plus the stated retry time plus 120 s.
  - If that time is before the freeze, the loop waits and relaunches then.
  - If that time falls at or after the freeze, this is the model's first quota stop after the last reset before the freeze, so the model stops. It is recorded as `stopped_by_quota`.
- **No stated retry time:** the model is retried after 60 minutes, up to the freeze. A retry that would fall at or after the freeze ends the model as `stopped_by_quota`.

## Scope and data handling

- **Models:** groq-gpt-oss-120b, groq-gpt-oss-20b and groq-qwen3.8-27b only (D-02). Gemini and Gemma are not run; their existing partial rows stay as they are.
- **Fresh re-run (D-08):** each model is re-run under the current engine into a new directory `results/phase4-groq-<model>`, with the Phase 1 arguments (`--runs 3`, 57 cases, seed 0, max 5 steps) and `--priority-kinds ds`, so data-stealing episodes run first.
- **October rows superseded (D-08):** the October partial rows in `results/final-20261011` are superseded and kept as history only. They are never resumed, written or merged, because they predate the current scoring and stamps and carry no prompt_tokens. The hosted loop refuses any prefix under that directory.

## Failure handling outside the rule

- **Unavailable model:** a model the probe classes as unavailable is recorded and not launched.
- **No substitutes:** no other model replaces an unavailable one without a user decision.
- **Other exits:** an exit that is neither complete nor a quota stop is relaunched after 10 minutes, never at or after the freeze.
- **Stalled model:** 3 consecutive launches with zero new finished episodes and no quota stop mark the model as `stalled`.
- **Auth failure:** an auth failure in the probe blocks every hosted model.

## Reporting

- Whatever is unfinished at the freeze, or when a model stops by quota or stalls, is reported as **partial with its n**, its ds n and its paired count.
- Partial rows are never presented as complete or ranked against complete rows.

## Key handling

- The Groq key is never printed, logged or stored.
- Key presence is checked only by `jobs hosted-probe --key-check`, which prints `set` or `missing`.
- `run_checks secret-scan` checks the hosted directories, the jobs directory and this file for the fixed pattern `gsk_[A-Za-z0-9]{20,}`. It loads no key.

## Launch table

| Field | Value |
|-------|-------|
| Launch UTC | 2026-10-28T18:48:03Z (`written_utc` in stopping_rule.json; hosted loop pid 10209, children 10212, 10213, 10214) |
| freeze_utc (from results/phase4-jobs/stopping_rule.json) | 2026-10-29T15:30:00+00:00 |
| Rule commit sha | d4ee817abc0c679ded2a28a296ef4b91e818a8fd (`rule_sha` in stopping_rule.json) |
| Probe before launch | KEY GROQ_API_KEY set; PROBE groq-gpt-oss-120b ok, groq-gpt-oss-20b ok, groq-qwen3.8-27b ok |
