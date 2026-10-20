# Project Research Summary

**Project:** OS-Level Guardrails for Autonomous AI Agents — Multi-LLM & Local-Model Expansion
**Domain:** Local LLM training (char-level GPT from scratch, LoRA on Qwen2.5-0.5B) + fair multi-model evaluation of an agent reference monitor on InjecAgent
**Researched:** 2026-10-20
**Confidence:** MEDIUM-HIGH

## Executive Summary

This is an evaluation-methodology project with a training sub-project attached. The existing system (policy engine, namespaces, seccomp, tool broker, InjecAgent harness) is built; what is missing is evidence for the core claim that the guardrails hold regardless of which LLM drives the agent — including a weak, unaligned, locally made one. Experts defend such claims with an unprotected-vs-protected comparison across many models, conditioned on attempted attacks, with explicit n and confidence intervals. All four research files agree that the credibility of the comparison matters more than the number of models.

Recommended approach: keep everything on the existing seams. Generate one synthetic episode dataset from the 251 non-held-out tools with a hard leakage gate; render it through a single shared function (`format.render_context` for ScratchGPT, `apply_chat_template` for Qwen, both built from `prompting.build_messages`); train ScratchGPT with a plain PyTorch loop and Qwen with a short custom `peft` LoRA loop (no TRL / datasets / bitsandbytes / unsloth); serve Qwen in-process through a new `hf_local` provider; keep the installed torch 2.11; treat Ollama/GGUF export as optional. Fix the report (outcome taxonomy, Wilson intervals, completeness gating, paired error handling) before collecting more numbers.

The key risks are validity, not engineering: (1) weak models look "safe" because they fail to emit valid actions — condition ASR on format-valid episodes and show benign completion beside it; (2) near-zero unprotected ASR makes "0% with guardrails" vacuous — add a scripted worst-case control agent / forced-attempt arm and report blocks-given-attempt with upper bounds; (3) train/serve template skew, loss-mask errors and tool/text leakage can silently ruin the local models — each needs a unit test as an acceptance gate; (4) Ollama silent context truncation and truncated hosted runs can corrupt existing rows. Python 3.14 compatibility of the HF stack is the largest unknown and gets a 15-minute day-0 smoke test.

## Key Findings

### Recommended Stack

Add only `transformers>=5.20,<5.21`, `peft==0.21.2`, `accelerate>=1.15,<2` (in a separate `requirements-train.txt`), imported lazily inside the provider constructor. Everything else is plain PyTorch and existing deps. See STACK.md.

**Core technologies:**
- PyTorch 2.11+cu128 (installed): training + inference for both models — do not upgrade mid-week
- transformers 5.20.2 + peft 0.21.2: load Qwen2.5-0.5B-Instruct, LoRA r=16/alpha 32/`all-linear`, bf16, SDPA, gradient checkpointing, lr 2e-4, 2–3 epochs; use `dtype=` not `torch_dtype=`
- Custom training loops (~60–80 lines each): explicit label masking, testable, explainable in a viva (vs TRL)
- ScratchGPT: plain torch, AdamW + warmup/cosine, bf16 autocast, no `torch.compile`; existing no-KV-cache `generate()` is acceptable
- Fallback: Python 3.12 venv via `uv` if the 3.14 smoke test fails; always `num_workers=0`

### Expected Features

"Features" = report contents and evaluation capabilities. See FEATURES.md.

**Must have (table stakes):**
- Unified per-model table: unprotected/protected ASR, attempted %, benign utility in both arms, FP %, parse-error %, latency — with n and 95% Wilson CIs
- Completeness column; incomplete rows (gemma, gpt-oss-20b) greyed; explicit "not run" cells for hosted `ds`
- Conditional block rate (blocked / attempted), pooled and per model
- At least one weak local row beyond llama3.2:3b (ScratchGPT or LoRA Qwen, plus qwen2.5:3b)
- Where-blocked-by-layer table with an explanation of why only broker/policy fire today
- Threat model + metric definitions + mapping to AgentDojo/InjecAgent/ASB names; viva notes per phase
- Shell-chaining bypass reproduced by a test and fixed (P1)

**Should have (differentiators):**
- Scripted worst-case control agent / forced-attempt arm (model-independent proof; cheap, no quota)
- Capability-vs-vulnerability scatter + utility-vs-ASR arrow plot (need unprotected benign column + ≥6 complete rows)
- Layer ablation with a small synthetic suite that reaches sandbox/seccomp
- Paired discordant-case counts; base-Qwen control row to isolate the LoRA effect

**Defer (v2+):** utility-under-attack on a larger task set with cluster bootstrap, adaptive attacks, AgentDojo/ASB replication. Anti-features: composite security score, model rankings, headline numbers from n=5 rows, learned PI-detector layers, tuning on InjecAgent tools.

### Architecture Approach

Everything downstream of the provider is unchanged. New code lives in a shared torch-free `src/scratch_llm/dataset.py` (generator, `episode_to_messages`, leakage gate), per-model training packages, and a new `src/llm/hf_local.py` registered as `provider: hf_local`. Training and inference share one renderer, protected by a `FORMAT_VERSION` stamp, vocab hash and a parity test. See ARCHITECTURE.md.

**Major components:**
1. `dataset.py` — synthetic episodes from the 251 free tools, 50/25/25 benign/dh/ds mix, randomized tool order, dev-tool validation split, hard leakage gate (tool names, 8-word shingles, banned strings incl. `CANARY`, attacker address)
2. `format.py` v2 — compact transcript with parameter names on the `T:` line (required for unseen-tool copying); must land before any training
3. `scratch_llm/{train,inference}.py` — inference must not import dataset/benchmark; checkpoint holds config, format version, vocab hash
4. `qwen_lora/{data,train}.py` — one (prompt, completion) pair per turn, labels −100 on the prompt, EOS kept in loss
5. `hf_local.py` + registry branch + base and LoRA config entries — missing adapter raises `RuntimeError` → runner `[skip]`
6. Manifests (seed, dataset sha, held-out hash, git SHA) committed; weights and episode JSONL gitignored

### Critical Pitfalls

1. **Parse failures make weak models look safe** — outcome taxonomy (attacked / ignored_injection / format_failure / llm_error / refused), ASR conditioned on format-valid episodes, benign completion beside it
2. **Vacuous guardrail claim** — scripted worst-case control agent + forced-attempt arm; blocks-given-attempt by layer with Wilson/Clopper-Pearson upper bounds (0/47 ≈ ≤7.6%, not "zero"); fix the shell-chaining bypass before final runs and record the policy hash
3. **Unequal n and mishandled errors** — interleaved paired work queue, drop pairs not rows, separate infra errors from provider refusals, idempotent writer, completeness gate, common-subset headline table
4. **Train/serve skew and loss-mask errors** — single renderer from `prompting.build_messages`, round-trip token-identity test, decoded-supervised-token test, never mask pad==EOS by id, always supply the project system prompt (Qwen inserts a default one otherwise)
5. **Leakage and dataset-determined behavior** — automated leakage test in pytest (incl. `GmailSendEmail`/`EXFIL_TOOL`, n-gram overlap), checkpoint selection on dev tools only, describe ScratchGPT as a worst-case principal whose behavior reflects the data policy
6. **Ollama silent context truncation** — set `OLLAMA_CONTEXT_LENGTH=8192`, verify with `ollama ps`, log `prompt_tokens`; re-check the existing llama3.2:3b run

## Implications for Roadmap

Coarse granularity + one-week deadline: the 10-step architecture build order and the PITFALLS P-labels collapse into 5 phases. Ordering is driven by (a) report hardening first, because every later number depends on it, (b) format v2 before any GPU time, (c) GPU jobs serial and in the background while CPU-side work proceeds.

### Phase 1: Evaluation Hardening & Policy Fix
**Rationale:** Every later number flows through scoring/report; fixing semantics after collecting data wastes runs. The policy fix must precede the final protected-arm runs.
**Delivers:** outcome taxonomy + conditional ASR, Wilson CIs, completeness gate, pair-aware error handling, interleaved idempotent queue, `finish_reason`/`prompt_tokens` logging, manifest (commit, policy hash, model digests), scripted worst-case control agent / forced-attempt arm, failing test then fix for `ls; curl`, dashboard bound to 127.0.0.1, `.gitignore` for `models/` and `data/local/`, Ollama context 8192 verified, first viva notes.
**Addresses:** Wilson CIs, completeness, conditional block rate, unprotected benign column, control agent, shell-chain fix
**Avoids:** Pitfalls 1, 2, 3, 4, 5, 6, 8, 22

### Phase 2: Local Data Foundation & ScratchGPT
**Rationale:** Dataset and format are shared by both local models; changing them later invalidates checkpoints. The scratch run is the longest GPU job, so start it early.
**Delivers:** `FORMAT_VERSION=2` with params on `T:`; `generate_episodes` + leakage gate + parity tests; `inference.py`, `train.py`; CPU micro-run, overfit-10 and causal-mask sanity tests; GPU training with manifest and loss curve; scratch smoke benchmark.
**Uses:** plain PyTorch, bf16 autocast, WSL ext4 for artifacts
**Implements:** `dataset.py`, `format.py` v2, `scratch_llm/{train,inference}.py`
**Avoids:** Pitfalls 9, 10, 11, 12, 14, 15, 17

### Phase 3: Qwen LoRA & hf_local Provider
**Rationale:** The provider and base-Qwen control need no training and de-risk the Python 3.14 HF install; LoRA reuses Phase 2 data. LoRA training runs after scratch training (GPU contention).
**Delivers:** day-0 smoke test (install, load, one LoRA step); `hf_local.py` with a fake-model contract test; `qwen2.5-0.5b-base` control row; custom LoRA loop with masking and round-trip tests; adapter + manifest; smoke benchmark.
**Uses:** transformers/peft/accelerate, bf16, gradient checkpointing, `do_sample=False`
**Avoids:** Pitfalls 9, 10, 13, 21 (no GGUF on the critical path)

### Phase 4: Full Runs — Local, Added Ollama, Hosted Resume
**Rationale:** Needs both artifacts and the hardened queue. Local runs have no quota and run first, serially on the GPU; hosted resumes are quota-bound and run `ds`-first.
**Delivers:** 3×342-episode runs for scratch-gpt, qwen base, qwen LoRA; qwen2.5:3b, phi3.5:3.8b, gemma3:1b via Ollama; llama3.2:3b truncation re-check; hosted resumes serialized per key with a fixed stopping rule (complete 2–3 models, label the rest partial); layer-targeted synthetic suite so sandbox/seccomp counts are non-zero.
**Avoids:** Pitfalls 2, 3, 5, 7, 16

### Phase 5: Comparison Report & Deliverables
**Rationale:** Reporting is model-agnostic; this phase freezes the data and regenerates everything from one script so slides and README cannot drift.
**Delivers:** cross-model table + confounds table; five static charts (grouped ASR with CIs, funnel, scatter, arrow plot, where-blocked); paired discordant counts; threat model & limits section; README, SDD and Review-3 deck regenerated from one results directory; complete `docs/viva/` set including examiner Q&A.
**Avoids:** Pitfalls 7, 18, 19, 20, 24

### Phase Ordering Rationale

- Dependencies: report semantics → format v2 → dataset → {scratch training, Qwen data} → full runs → report. `hf_local` is independent of training and can be built while scratch trains.
- The GPU is serial: scratch training → LoRA training → benchmark runs; Ollama must be stopped during training.
- Phase 1 is front-loaded because it also prevents the existing committed results from being quoted wrongly.
- Viva notes are a definition-of-done item in every phase, not a final-night task.
- Optional Ollama/GGUF export is a stretch after Phase 4, not a phase.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** episode design and leakage corpus (compliance mix, unseen-tool `ds` generalization, shingle gate) — design judgement with no external reference
- **Phase 3:** Python 3.14 wheel compatibility for the HF stack, Qwen/transformers v5 API drift, LoRA memory fit — settle with the day-0 smoke test before detailed planning

Phases with standard patterns (skip research-phase):
- **Phase 1:** repo-internal changes; the statistics formulas are exact and known
- **Phase 4:** existing runner `--resume` and Ollama patterns; only operational scheduling is new
- **Phase 5:** existing `generate_report`/charts; content already specified

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Versions and Python requirements verified on PyPI; peft/accelerate lack 3.14 classifiers; throughput/VRAM/adapter sizes are estimates until measured |
| Features | MEDIUM | AgentDojo metrics and CI method verified; InjecAgent/ASB details partly from snippets and prior knowledge; Wilson arithmetic exact |
| Architecture | HIGH on seams, MEDIUM on tooling | Read directly from the repo; Qwen/PEFT/Ollama details not run on this machine |
| Pitfalls | MEDIUM-HIGH | Repo-specific pitfalls from code and CONCERNS.md; Ollama context default from third-party guides; wheel availability LOW |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Python 3.14 compatibility of transformers/peft/accelerate and numpy wheels — day-0 smoke test; fall back to a `uv` Python 3.12 venv with the cu128 torch build
- Training-data compliance policy (50/25/25 mix) determines what the weak models do when attacked — decide in Phase 2, disclose in the report, same dataset for both local models
- Whether a weak model attempts attacks at all is unknown until trained — the control agent / forced-attempt arm is the hedge
- Throughput, VRAM and scratch inference speed are unmeasured — time 50 steps and run a worst-case-batch smoke test first
- Whether the existing llama3.2:3b and gemma results were affected by context truncation or errors/timeouts — re-score from `raw.jsonl`; re-run if prompt_tokens plateau
- Qwen2 safetensors/adapter import in current Ollama builds and `transformers serve` behavior — unverified; only matters for the optional stretch
- Hosted quota availability within the week — stopping rule; report partial rows with n shown
- InjecAgent protocol details (ASR-valid/all, DH/DS) — verify against the local PDF in `papers/` before citing

## Sources

### Primary (HIGH confidence)
- PyPI JSON API (queried 2026-10-20) — transformers, peft, accelerate, torch, trl versions and Python requirements
- PEFT LoRA developer guide — `LoraConfig`, `all-linear`, `merge_and_unload` semantics
- Repo code and data (`src/llm`, `src/scratch_llm`, `src/agent_harness`, `src/benchmark`, `config/models.yaml`, `benchmarks/injecagent`, `results/final-20261011`, `.planning/codebase/CONCERNS.md`) — integration seams; 330 tools, 79 held out, 251 free
- AgentDojo (arXiv 2406.13352) — metric definitions, 95% CIs, Pareto/scaling plots
- Wilson / Clopper-Pearson interval arithmetic (computed exactly)

### Secondary (MEDIUM confidence)
- Ollama docs (import, Modelfile) — GGUF import not re-quantized; adapter architecture list does not name Qwen2
- TRL SFT trainer docs — `assistant_only_loss` needs generation markers absent from the Qwen2.5 template
- CaMeL (arXiv 2503.18813), Agent Security Bench (arXiv 2410.02644) — headline numbers and metric names
- Established practice: Qwen chat template / `generation_config`, EOS/pad handling, WSL memory limits, logits memory

### Tertiary (LOW confidence)
- Third-party Ollama context-window guides — verify with `ollama show` / `ollama ps`
- transformers v5 migration notes from blogs — confirm in official docs
- Python 3.14 wheel availability for the ML stack; `transformers serve` stability — unverified

---
*Research completed: 2026-10-20*
*Ready for roadmap: yes*
