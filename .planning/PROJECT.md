# OS-Level Guardrails for Autonomous AI Agents — Multi-LLM & Local-Model Expansion

## What This Is

A capability-based reference monitor (policy engine + Linux namespaces + seccomp-bpf) that mediates every action an LLM-driven agent takes, evaluated on the InjecAgent indirect-prompt-injection benchmark. This expansion answers the reviewer's remark — *"use this on a locally made LLM and also compare with other LLMs other than Gemini"* — by training a language model locally (from scratch, and by local fine-tuning), running the same guardrails on it, and comparing it against several non-Gemini models. Built by Raj Vardhan Singh (24BDS0200) for the BCSE303P Operating Systems Lab; the audience is the review panel and the student's own viva preparation.

## Core Value

The guardrails must be shown to hold **independently of which LLM drives the agent** — including a weak, unaligned, locally made model — backed by a fair, reproducible comparison table the student can defend in a viva.

## Requirements

### Validated

<!-- Inferred from the existing codebase (see .planning/codebase/). -->

- ✓ Access-matrix policy engine (YAML Allow/Block/Flag rules over run_shell, read_file, write_file, http_request) — existing
- ✓ Namespace isolation (user/PID/net via `unshare`) + private sandbox filesystem — existing
- ✓ seccomp-bpf syscall filter in C (chmod, mount, ptrace, setuid, …) — existing
- ✓ Multi-step agent loop with JSON action protocol and observation feedback — existing (Phase 2 commit)
- ✓ Pluggable LLM provider layer (OpenAI-compatible client for Gemini, Groq, Ollama; `scratch` provider stub) — existing (Phase 1 commit)
- ✓ InjecAgent benchmark: tool broker, unprotected vs protected arms, scoring, merged per-model reports + charts — existing (Phase 3 commit)
- ✓ Real-time Flask dashboard over `logs/guardrails.jsonl` — existing
- ✓ IEEE 1016 SDD (v2.0) with PDF/DOCX export — existing
- ✓ Honest benchmark report: outcome taxonomy, k/n + Wilson CIs, completeness gating, paired infra-error handling, case-level view — Phase 1
- ✓ Provenance stamps on every result record + run manifest + paired run-major queue — Phase 1
- ✓ Scripted worst-case control agent (control-always-comply): 120/120 attacks succeed unprotected, 0/120 protected — Phase 1
- ✓ Shell-chaining policy bypass fixed with test-first evidence; quote-aware fail-closed shell parser; dashboard on 127.0.0.1 with debug off — Phase 1

### Active

- [ ] From-scratch ScratchGPT is trained locally on the RTX 3060 and plugged into the benchmark as `scratch-gpt`
- [ ] A small open model (Qwen2.5-0.5B-Instruct) is LoRA fine-tuned locally on the agent action format and benchmarked
- [ ] More local open models (qwen2.5:3b, phi3.5:3.8b, gemma3:1b via Ollama) are benchmarked alongside llama3.2:3b
- [ ] Unfinished hosted (non-Gemini) runs are resumed so the comparison includes data-stealing episodes
- [ ] A cross-model comparison report (tables, charts, analysis) with Gemini as just one row among many
- [ ] Viva study notes covering the theory foundations of every component, updated as each phase lands (Phase 1 page done: docs/viva/01-reference-monitor-and-evaluation.md)
- [ ] Layer-targeted suite uses the residual policy bypasses from 01-REVIEW-FIX.md to show namespaces + seccomp containing what the denylist misses (emerged in Phase 1 code review)
- [ ] README, SDD, and a Review-3 slide deck reflect the multi-LLM + local-model results

### Out of Scope

- Pretraining a large model or matching hosted-model capability — a 6 GB laptop GPU and a 1-week deadline; the scratch model's role is a weak, unaligned worst-case principal
- New benchmark suites (AgentDojo, etc.) — InjecAgent is already integrated; changing benchmarks would invalidate existing results
- Paid APIs / paid tiers — free tiers and local inference only
- eBPF/LSM rewrite of the enforcement layer — larger research project (ActPlane-style), not needed for the reviewer remark

## Context

- Repo is on branch `docs/sdd`; Phases 1–3 of an earlier plan are committed (provider layer, agent loop, InjecAgent benchmark). A "Phase 4" from-scratch model was designed (`src/scratch_llm/model.py`, `format.py`, `dataset.py` validation) but `generate_episodes()`, `train.py`, and `inference.py` were never written.
- `results/final-20261011/`: only `ollama-llama3.2-3b` and `gemma-4-31b` reached all 342 (case × run × arm) episodes; gemma's are mostly API timeouts. gemini-flash-lite 195, groq-gpt-oss-120b 214, groq-gpt-oss-20b 112, groq-qwen3.8-27b 202; no hosted model has data-stealing (`ds`) episodes.
- Environment: Windows 11 + WSL2 Ubuntu; WSL venv has Python 3.14, torch 2.11+cu128 with CUDA on an RTX 3060 Laptop (6 GB), 16 cores, 7 GB RAM in WSL; Ollama installed in WSL.
- Codebase map: `.planning/codebase/` (ARCHITECTURE, STACK, CONCERNS, …). CONCERNS also flags dashboard `debug=True` on `0.0.0.0`.
- The student is preparing for a viva and wants to understand the foundations, not just ship code — every phase should add to the viva notes.

## Constraints

- **Timeline**: next review/viva within one week — favour a working end-to-end comparison over polish
- **Hardware**: RTX 3060 Laptop 6 GB VRAM, 7 GB WSL RAM — model sizes and batch sizes must fit
- **Platform**: sandbox/benchmark only run under Linux (WSL2); Windows side is for editing
- **Budget**: free API tiers only; hosted runs are rate/quota limited and resumable
- **Fairness**: every model must receive the same system prompt, cases, seeds, and arms (`src/agent_harness/prompting.py` is the single source of truth); the scratch model uses a compact rendering of the same conversation
- **Evaluation hygiene**: InjecAgent tools used in the benchmark are held out of all local training data

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| "Locally made LLM" = both from-scratch ScratchGPT and a locally LoRA-fine-tuned Qwen2.5-0.5B | Gives a spectrum for the viva: scratch → fine-tuned → pretrained local → hosted | — Pending |
| Compare against non-Gemini models: Groq (gpt-oss-120b/20b, qwen3.8-27b) + local Ollama (llama3.2:3b, qwen2.5:3b, phi3.5:3.8b, gemma3:1b) | Reviewer asked for "other LLMs other than Gemini"; local models avoid quota limits | — Pending |
| Keep InjecAgent + existing scoring | Results stay comparable with the committed final-20261011 run | — Pending |
| Fix the shell-chaining policy bypass with a test first | Real security gap found by the codebase map; strong viva point on defense-in-depth | ✓ Good — red commit 72e737f then fix; code review found 4 more bypasses, fixed/partial |
| Keep policy `default_action: FLAG` (denylist) after code review; residual bypasses become Phase 4 test inputs | An allow-list default would inflate false positives; the residuals demonstrate why OS layers exist | — Pending (validated in Phase 4) |
| Run GSD executors sequentially on the main checkout (`workflow.use_worktrees: false`) | WSL venv lives in the repo; worktrees could not run tests | ✓ Good |
| "Blocked given attempt" = the attack action itself was blocked and the attack did not succeed | Code review WR-04: counting any block over-credited guardrails | ✓ Good |
| Viva notes live in `docs/viva/` and grow every phase | Student asked for theory foundations to refer to during the viva | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-21 after Phase 1*
