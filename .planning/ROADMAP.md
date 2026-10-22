# Roadmap: OS-Level Guardrails for Autonomous AI Agents

## Overview

The guardrails (policy engine, namespaces, seccomp-bpf) and the InjecAgent harness already exist; what is missing is credible evidence that the guardrails hold regardless of which LLM drives the agent, including a weak, locally made one. This roadmap answers the reviewer's remark in five steps. First, harden the scoring and report and close the known shell-chaining bypass, because every later number flows through them. Second, build the shared synthetic dataset, the single rendering format and a locally trained ScratchGPT. Third, add the `hf_local` provider, the base-Qwen control row and a locally LoRA-tuned Qwen2.5-0.5B. Fourth, run every model to completion (local, added Ollama and resumed hosted) plus a layer-targeted attack suite. Fifth, freeze the data and regenerate the cross-model report, charts, README, SDD and Review-3 deck from one results directory. The one-week deadline shapes the ordering: GPU jobs are serial (scratch training, then LoRA training, then benchmark runs) and run in the background while CPU-side work proceeds, and viva notes are added in every phase rather than on the final night.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Evaluation Hardening & Policy Fix** - Honest scoring/report semantics, a model-independent control agent, and the shell-chaining bypass fixed before any new data is collected (completed 2026-10-21)
- [x] **Phase 2: Local Data Foundation & ScratchGPT** - Leakage-free synthetic episodes, one shared render format, and a ScratchGPT trained locally and running in the benchmark (completed 2026-10-22)
- [ ] **Phase 3: Qwen LoRA & hf_local Provider** - In-process Hugging Face provider, base-Qwen control row, and a locally LoRA-tuned Qwen2.5-0.5B running in the benchmark
- [ ] **Phase 4: Full Runs - Local, Added Ollama, Hosted Resume** - Every local model benchmarked to 342/342, unfinished hosted runs resumed under a stopping rule, and sandbox/seccomp layers shown to fire
- [ ] **Phase 5: Comparison Report & Deliverables** - One script regenerates the cross-model report and charts from a frozen results directory; README, SDD, Review-3 deck and viva notes complete

## Phase Details

### Phase 1: Evaluation Hardening & Policy Fix

**Goal**: Every number the benchmark produces from now on is trustworthy: scoring and reporting semantics are honest about outcomes, uncertainty and completeness, a model-independent control agent measures what the guardrails block, and the known policy bypass is closed before final runs.
**Depends on**: Nothing (first phase)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, SEC-01, SEC-02
**Success Criteria** (what must be TRUE):
  1. Regenerating the report from the existing `results/final-20261011` shows every episode in exactly one outcome class (attacked, ignored_injection, format_failure, llm_error, refused), ASR conditioned on format-valid episodes next to raw ASR, every percentage as `k/n` with a 95% Wilson interval, benign utility in both arms, and a conditional block rate (blocked / attempted) per model and pooled.
  2. The report has a completeness column (episodes done / 342) and visibly marks incomplete or error-dominated model rows (for example groq-gpt-oss-20b at 112/342) instead of ranking them with complete ones; infrastructure-error episodes are dropped together with their paired arm and re-queued, and each result record stamps the git commit, policy-file hash, resolved model id, `finish_reason` and `prompt_tokens`.
  3. A test reproducing that `ls; curl http://evil.example` passes the policy engine fails before the fix and passes after it, with every sub-command of `;`, `&&`, `||`, `|`, `$(...)` and backticks checked and the chain blocked; the Flask dashboard binds to 127.0.0.1 with debug mode off by default.
  4. A scripted worst-case control agent (always obeys the injected instruction) runs through the same benchmark and both arms, and the report shows a model-independent count of what the guardrails block given an attempt.
  5. The viva page(s) for the reference monitor / access matrix / policy engine and for evaluation statistics (ASR definitions, Wilson intervals) are added to `docs/viva/` (definition of done for every phase; DOC-03 is traced to Phase 5).

**Plans**: 4/4 plans complete

Plans:
**Wave 1**
- [x] 01-01-PLAN.md — Honest report: outcome taxonomy, ASR-valid, Wilson k/n cells, completeness gate, pair dropping, utility both arms, conditional block rate (wave 1)
- [x] 01-02-PLAN.md — Shell-chaining bypass reproduced red then fixed (fail-closed parser), policy hash, dashboard on 127.0.0.1 with debug off (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 01-03-PLAN.md — Provenance stamps on every record, provenance report section, paired interleaved queue, infra re-queue, manifest.json (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 01-04-PLAN.md — Scripted always-comply control agent through both arms with its own report section; docs/viva Phase 1 page and index (wave 3)

### Phase 2: Local Data Foundation & ScratchGPT

**Goal**: A language model trained from scratch on this laptop's GPU drives the agent in the benchmark as `scratch-gpt`, trained on synthetic episodes that provably contain nothing from the benchmark and rendered by the same function at training and inference time.
**Depends on**: Phase 1
**Requirements**: SCRATCH-01, SCRATCH-02, SCRATCH-03, SCRATCH-04, SCRATCH-05
**Success Criteria** (what must be TRUE):
  1. `generate_episodes(n, seed)` reproducibly produces the 50% benign / 25% direct-harm-compliant / 25% data-stealing-compliant mix with self-written injections using only non-InjecAgent tools (the same seed gives the same dataset hash), and the automated leakage test passes on it while failing when a held-out tool (including `GmailSendEmail`), the benchmark attacker address, the canary format or an 8-word shingle of benchmark text is planted.
  2. Training and inference render conversations through the same function (compact format v2 with parameter names on the `T:` line), a parity test proves identical output, and loading a checkpoint whose format-version or vocab-hash stamp does not match is rejected.
  3. `python -m src.scratch_llm.train` trains ScratchGPT on the RTX 3060 and writes `models/scratch/scratch_gpt.pt`, a manifest (seed, dataset hash, config, git SHA) and a loss curve.
  4. `scratch-gpt` runs end-to-end through the benchmark (the provider loads the checkpoint via `inference.load_generator`) and produces valid actions on a smoke run.
  5. The viva page(s) for LLM foundations (transformers, tokenization, the training loop, decoding) and for the synthetic-data / leakage design are added to `docs/viva/` (DOC-03 is traced to Phase 5).

**Plans**: 4/4 plans complete

Plans:
**Wave 1**
- [x] 02-01-PLAN.md — Format version 2 (required params on the T: line, context 2048, version/vocab/fingerprint stamps), episode schema, training text = render_context output, loss mask, strict leakage gate with planted-violation tests, benign generator slice (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 02-02-PLAN.md — Full 50/25/25 generator (self-written dh/ds injections, synthetic tools, OS-action and blocked-tail episodes), dev split on tools and templates, dataset CLI, committed 40k+2k dataset manifest (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 02-03-PLAN.md — train.py (masked loss, AdamW warmup/cosine, dev line-exact, fixed manifest), inference.py with stamp checks, stamped LocalScratchProvider; CPU tests incl. a tiny checkpoint driving a benchmark episode (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 02-04-PLAN.md — GPU pilot, background ScratchGPT training on the RTX 3060, committed manifest/log/loss curve, scratch-gpt smoke benchmark, viva page 02 (wave 4)

### Phase 3: Qwen LoRA & hf_local Provider

**Goal**: A pretrained small open model that the student fine-tuned locally with LoRA, and its un-tuned base as a control, both run through the benchmark behind the same provider interface as every other model.
**Depends on**: Phase 2
**Requirements**: LORA-01, LORA-02, LORA-03, LORA-04
**Success Criteria** (what must be TRUE):
  1. A new `hf_local` provider serves a Hugging Face causal LM in-process (optionally with a PEFT adapter) behind the existing `LLMProvider` interface, with greedy decoding and the project system prompt always supplied; a contract test with a fake model passes, and a missing adapter makes the runner skip that model rather than crash.
  2. Un-tuned `Qwen2.5-0.5B-Instruct` runs end-to-end through the benchmark on a smoke run as the base control row.
  3. The custom LoRA training loop fine-tunes Qwen2.5-0.5B on the Phase 2 synthetic episodes within 6 GB VRAM and writes the adapter plus a manifest, and a decoded-supervised-tokens test shows that only the assistant action tokens are supervised.
  4. The LoRA-tuned Qwen runs end-to-end through the benchmark and produces valid actions on a smoke run.
  5. The viva page(s) for fine-tuning and LoRA, chat templates and loss masking, and decoding are added to `docs/viva/` (DOC-03 is traced to Phase 5).

**Plans**: 2/4 plans executed

Plans:
**Wave 1**
- [x] 03-01-PLAN.md — hf_local provider (offline, explicit greedy, stop tokens, stamps, verified and merged adapters), shared chat-render contract, qwen2.5-0.5b-base / agentlora config entries, STRICT_HF, pinned requirements-train.txt (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 03-02-PLAN.md — Qwen training examples with a character-offset loss mask (decoded-supervised-tokens and prefix tests), LoRA core loop with supervised-position loss, action-level evaluation and best-by-dev_loss_action selection; a tiny adapter served end to end (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*
- [ ] 03-03-PLAN.md — Trainer CLI with full manifest, final dev pass and gen_check through hf_local, resume and fail-closed exits, pre-registered dev-only remediation rule; run_train.sh launcher and checks commands (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*
- [ ] 03-04-PLAN.md — GPU pilot and background LoRA run, artifacts committed, remediation decision, base and LoRA smoke runs, viva page 03 (wave 4)

### Phase 4: Full Runs - Local, Added Ollama, Hosted Resume

**Goal**: Every model in the comparison has its complete, like-for-like benchmark data (or an honestly labelled partial), and the sandbox and seccomp layers are shown to fire on actions that only they can stop.
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: CMP-01, CMP-02, CMP-03, EVAL-08
**Success Criteria** (what must be TRUE):
  1. scratch-gpt, Qwen base and Qwen LoRA each complete the full benchmark (57 cases x 3 runs x 2 arms = 342 episodes), shown as 342/342 in the report's completeness column.
  2. qwen2.5:3b, phi3.5:3.8b and gemma3:1b complete the full benchmark through Ollama with the context window raised (verified with `ollama ps` and logged `prompt_tokens`, so prompts are not truncated), and the existing llama3.2:3b run is re-checked for truncation.
  3. The unfinished hosted non-Gemini runs (Groq gpt-oss-120b, gpt-oss-20b, qwen3.8-27b) are resumed with data-stealing episodes prioritised under a written stopping rule, and whatever remains incomplete is reported as partial with its n.
  4. A small layer-targeted attack suite drives actions that only the sandbox (namespaces / private filesystem) and seccomp layers can stop, and the where-blocked-by-layer data shows non-zero blocks for those layers.
  5. The viva page(s) for namespaces, seccomp-bpf, and local-inference confounds (quantization, context window) are added to `docs/viva/` (DOC-03 is traced to Phase 5).

**Plans**: TBD

### Phase 5: Comparison Report & Deliverables

**Goal**: The student can present and defend the multi-LLM + local-model result: one frozen data set drives a cross-model report, charts, README, SDD and Review-3 deck that cannot drift apart, backed by a complete set of viva notes.
**Depends on**: Phase 4
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05
**Success Criteria** (what must be TRUE):
  1. One script regenerates the cross-model comparison report from a single frozen results directory, containing the headline table, the confounds table (size, quantization, local/hosted, context), the where-blocked-by-layer table and a threats-to-validity section.
  2. The report includes static charts for ASR with confidence intervals per model, a capability-vs-vulnerability scatter, and a utility-vs-ASR arrow plot (unprotected to protected).
  3. The README and the IEEE 1016 SDD describe the multi-LLM + local-model system and its results (no longer Gemini-only), and a Review-3 slide deck presents the reviewer remark, the local models and the comparison results.
  4. `docs/viva/` is complete: theory notes for OS security (reference monitor, access matrix, capabilities, namespaces, seccomp-bpf), LLM foundations (transformers, training, LoRA, decoding), prompt injection and defenses, and evaluation statistics, plus likely examiner questions with answers; every earlier phase's page is present and this phase adds the Q&A and threats-to-validity page.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 (the `hf_local` provider in Phase 3 is independent of training and can be built while Phase 2 training runs)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Evaluation Hardening & Policy Fix | 4/4 | Complete    | 2026-10-21 |
| 2. Local Data Foundation & ScratchGPT | 4/4 | Complete    | 2026-10-22 |
| 3. Qwen LoRA & hf_local Provider | 2/4 | In Progress | - |
| 4. Full Runs - Local, Added Ollama, Hosted Resume | 0/TBD | Not started | - |
| 5. Comparison Report & Deliverables | 0/TBD | Not started | - |
