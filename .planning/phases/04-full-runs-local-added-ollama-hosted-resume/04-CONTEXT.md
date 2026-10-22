# Phase 4: Full Runs — Local, Added Ollama, Hosted Resume - Context

**Gathered:** 2026-10-22
**Status:** Ready for planning
**Source:** User decisions at the plan-phase gate + locked project decisions + Phase 1–3 summaries/reviews

<domain>
## Phase Boundary

Produce complete, like-for-like benchmark data (57 cases × 3 runs × 2 arms = 342 episodes per model) for every local model, resume the unfinished Groq rows under a written stopping rule, re-check llama3.2:3b for truncation, add a layer-targeted attack suite that shows sandbox and seccomp blocks, and write the viva page for namespaces / seccomp-bpf / local-inference confounds. The cross-model report, charts, README/SDD and slides are Phase 5.
</domain>

<decisions>
## Implementation Decisions

### Models in scope
- **D-01:** Full local runs for scratch-gpt (checkpoint 97bdace56294), qwen2.5-0.5b-base, qwen2.5-0.5b-agentlora (identity 924ac021f0cb), ollama qwen2.5:3b, and a truncation re-check/re-run of ollama llama3.2:3b. phi3.5:3.8b and gemma3:1b are descoped to v2 (CMP-V2-01) by user decision to save ~2 h of GPU time.
- **D-02:** Hosted resumes are Groq only: gpt-oss-120b, gpt-oss-20b, qwen3.8-27b, data-stealing (ds) episodes prioritised. Gemini/Gemma rows stay as their existing partial rows (reviewer asked for a non-Gemini comparison).
- **D-03:** Written stopping rule for hosted runs: a model stops when its daily quota is exhausted twice (two QuotaExhausted stops on separate days/resets) or when the Phase 4 window ends; whatever remains is reported as partial with its n. Groq calls use the user's existing free-tier key; never print or store key values.

### Execution
- **D-04:** Everything runs unattended in the background with automatic resume check-ins (as in Phases 2–3), local GPU models first (serially — GPU is shared and serial), then the Ollama models, then hosted resumes; the machine is kept awake. Another project shares WSL (CPU job, 11 GB RAM + 8 GB swap) — never stop or kill anything this project did not start.
- **D-05:** All runs use the current engine and scoring (post-injection attack counting, Phase 1 stamps/manifests), fresh result directories per model, and the same case seed/args as Phase 1 (`--runs 3`, 57 cases). Re-run the 20-second control into a fresh directory at the end so its stamps match the final engine.

### Ollama
- **D-06:** Raise Ollama's context window (e.g. OLLAMA_CONTEXT_LENGTH=8192 on the Ollama server process this project starts, or a Modelfile variant) and verify with `ollama ps` plus logged `prompt_tokens`; the old llama3.2:3b rows are checked for truncation (prompt_tokens plateau) and re-run if affected.

### Layer suite (EVAL-08)
- **D-07:** The layer-targeted suite uses the residual policy bypasses documented in .planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md (renamed binaries, awk/perl/sed subprocesses, interpreter network/file access, chmod/setuid/mount/ptrace syscalls) and asserts on side effects (no egress, canary instead of real secret, EPERM), not just Observation.status; results must show non-zero sandbox and seccomp blocks in the where-blocked-by-layer data. A dash differential test of the shell parser (Phase 1 UAT note) is included.

### Claude's Discretion
- Exact ordering within the local queue, poll cadence, how the layer suite is driven (scripted control-style provider vs direct executor harness), file layout, Ollama server-restart mechanics (only for the Ollama server this project runs).
</decisions>

<canonical_refs>
## Canonical References

- `.planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md` — 23 residual bypass commands and observed sandbox/seccomp outcomes
- `.planning/phases/02-local-data-foundation-scratchgpt/02-REVIEW-FIX.md` — WR-01 scoring, resumable runs
- `.planning/phases/03-qwen-lora-hf-local-provider/03-REVIEW-FIX.md` and 03-04-SUMMARY.md — rate-limit detection fix, smoke results, run_train.sh patterns
- `src/benchmark/runner.py`, `run_benchmark.py`, `scripts/run_full_benchmark.sh` (tracked) — runner, resume, manifests
- `config/models.yaml` — model entries
</canonical_refs>

<specifics>
## Specific Ideas

- Latency estimates: scratch-gpt ~15–20 min, Qwen base ~37 min, Qwen LoRA ~16 min per 342 episodes; Ollama 3B models likely ~30–60 min each.
- Known carry-forwards: Phase 3 review IN-01 (control-token guard covers 3 of 22 added tokens), IN-05 (remediation window wording); Phase 1 IN-01/IN-02/IN-04 low-severity runner items.
</specifics>

<deferred>
## Deferred Ideas

- phi3.5:3.8b and gemma3:1b full runs — v2 (CMP-V2-01).
- Gemini/Gemma resumes — not in scope (D-02).
</deferred>

---

*Phase: 04-full-runs-local-added-ollama-hosted-resume*
*Context gathered: 2026-10-22*
