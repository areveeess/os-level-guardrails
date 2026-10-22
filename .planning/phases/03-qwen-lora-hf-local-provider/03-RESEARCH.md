# Phase 3: Qwen LoRA & hf_local Provider - Research

**Researched:** 2026-10-22
**Domain:** In-process Hugging Face serving (transformers + PEFT) and a custom LoRA training loop for Qwen2.5-0.5B-Instruct on a 6 GB RTX 3060 (WSL2, Python 3.14), behind the existing `LLMProvider` interface
**Confidence:** HIGH (the whole pipeline, install -> load -> template -> masked LoRA training -> adapter save/reload -> merged serving -> benchmark episodes, was executed on the real machine this session; only the final 600-step recipe and the full-benchmark timings are extrapolated)

## Summary

The Day-0 gate that the project research flagged as the largest unknown is **passed**: `transformers 5.20.2`, `peft 0.21.2` and `accelerate 1.15.0` install into the existing Python 3.14.4 WSL venv with no wheel problems and no change to torch (2.11.0+cu128), `pip check` is clean, Qwen2.5-0.5B-Instruct loads in bf16 on CUDA in 2.3 s, and a LoRA (r=16, alpha=32, all-linear) with gradient checkpointing trains at ~2,000 real tokens/s using about 2.3 GiB of VRAM. The fallback (uv Python 3.12 venv) is not needed and was not created. A 60-optimizer-step smoke run (960 episodes, 4 minutes) took teacher-forced dev action-line exact match from 0.007 to 0.899 and produced an adapter that drives real benchmark episodes (35 of 36 steps parsed, all 12 episodes ended in `respond`).

Three findings change how the phase should be planned. (1) **Never let the model materialise full-vocabulary logits for every position.** With HF's built-in `labels=` loss at micro-batch 4 the peak was 8,263 MiB allocated on a 6,144 MiB card (it silently spilled and ran at 6.7 s/step); computing the LM-head only on supervised positions cut that to 2,420 MiB at the same batch and 0.97 s/step. (2) **Serve the LoRA model merged, not as a live PEFT wrapper**: greedy outputs were identical on 20/20 prompts and episode outcomes were identical, and decode speed is 30.5 tok/s merged versus 13.3 tok/s unmerged (2.2x). (3) **The un-tuned base model mostly fails the action format** (21 of 47 steps parsed = 45%; it emits prose, `{"details": ...}`, or uses a tool name as the `action` value), so the base control row will be dominated by `format_failure` outcomes; that is a result, not a bug, and the EVAL-01 conditioning on format-valid episodes exists for exactly this.

The 40,000-episode Phase 2 file is far more data than the 30-60 minute budget can use (one epoch = 20.0 M Qwen tokens = ~2.8 h). The recommended recipe trains for **600 optimizer steps x 16 episodes = the first 9,600 episodes (24% of one epoch, by design: more passes over templated data hurt) in ~41 minutes**, with a dev action-level evaluation every 100 steps and a resume file after each evaluation.

**Primary recommendation:** Build `src/llm/hf_local.py` (lazy imports, `local_files_only`, greedy via explicit kwargs, merged adapter, fail-at-construction with `RuntimeError`) and `src/qwen_lora/{data,train}.py` (one multi-turn chat sequence per episode rendered through `apply_chat_template`, loss on assistant content + `<|im_end|>` only via character offsets, LM-head on supervised positions only, micro-batch 4 x accumulation 4, lr 2e-4 cosine, 600 steps, resumable), reuse `provenance.py`, and gate everything with the tests listed under Validation Architecture.

<user_constraints>
## User Constraints (from CONTEXT.md)

No `03-CONTEXT.md` exists for this phase (`.planning/phases/03-qwen-lora-hf-local-provider/` was empty before this file). The binding constraints below are the project-level locks that the task brief and the planning files state; treat them with the same authority as locked decisions.

### Locked Decisions (project level)
- In-process `hf_local` provider (transformers + PEFT), not Ollama/GGUF, for the main results (REQUIREMENTS LORA-01; STACK.md).
- Custom PyTorch LoRA loop. **Out of scope:** "TRL / bitsandbytes / unsloth / QLoRA | Fragile on Python 3.14 + 6 GB; custom loop is simpler and explainable" (REQUIREMENTS.md Out of Scope table).
- LoRA r=16, alpha=32, all-linear; bf16; gradient checkpointing; torch 2.11+cu128 is not to be upgraded.
- Train on the **same Phase 2 synthetic episodes** (dataset sha `a2586548c7f7...`); InjecAgent tools are held out of all local training data.
- Greedy decoding, project `SYSTEM_PROMPT` always supplied (LORA-01); `Qwen2.5-0.5B-Instruct` un-tuned is benchmarked as a base control row (LORA-02).
- Fairness: every model gets the same system prompt, cases, seeds and arms; `src/agent_harness/prompting.py` is the single source of truth.
- Hardware: RTX 3060 Laptop 6 GB, WSL2 ~11 GB RAM + 8 GB swap shared with another project's CPU-only job; sandbox/benchmark run only in WSL.

### Claude's Discretion
- Training hyper-parameters, step budget, evaluation cadence and metric, adapter/manifest layout, module layout (`src/qwen_lora/`), test granularity, whether to merge for serving.

### Deferred Ideas (OUT OF SCOPE)
- GGUF/Ollama export of the LoRA model (SERVE-V2-01), untrained-control rows beyond the base-Qwen row, pseudo-replication caveat (Phase 5), full 342-episode runs (Phase 4), report/charts (Phase 5).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LORA-01 | `hf_local` provider serves an HF causal LM in-process (optionally with a PEFT adapter) behind `LLMProvider`, greedy, project system prompt always supplied | "hf_local Provider Specification" (kwargs, stamps, merge, offline, failure modes, all measured) and the fake-model contract test design |
| LORA-02 | Un-tuned Qwen2.5-0.5B-Instruct benchmarked as a base control row | Measured: base loads in 2.3 s, 1.0 GiB; 12 real episodes ran; parse rate 45% (format_failure expected); latency estimate for 342 episodes |
| LORA-03 | Custom LoRA loop within 6 GB, supervising only assistant action tokens (decoded-supervised-tokens test), adapter + manifest | "Chat Template, Rendering and Loss Mask", "Training Recipe", measured VRAM/throughput, manifest fields, test design |
| LORA-04 | LoRA-tuned Qwen runs end-to-end in the benchmark and produces valid actions on a smoke run | 60-step adapter ran 12 real episodes (35/36 parsed); smoke command and acceptance gates |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

Directives extracted from `.claude/CLAUDE.md` that apply to this phase:
- **Providers never raise from `generate`**: return `LLMResponse(error=...)`; catch the broad `Exception` only at the model-call boundary. Construction-time failures (missing checkpoint/adapter/model) DO raise `RuntimeError` so the runner prints `[skip]` (the `LocalScratchProvider` pattern).
- Every module opens with a `"""Title ====="""` docstring explaining the role and the reason behind rules; snake_case; `UPPER_SNAKE_CASE` constants; `Optional[X]` not `X | None`; 4-space indent, double quotes in library code, lines under ~120 chars; no bare `except:`; no unused imports.
- Packages need `__init__.py` (light); import from the concrete module (`from src.llm.registry import get_provider`); do not copy `sys.path.insert` into `src/` modules.
- Tests: `tests/test_<area>.py`, section dividers `# ── Name ──`; tests that need heavy optional deps skip themselves (see `tests/conftest.py` torch header and `STRICT_TORCH=1`); extend that pattern to transformers/peft.
- Fairness constraint: every model receives the same system prompt via `src/agent_harness/prompting.py`; never hand-build a prompt for Qwen.
- Evaluation hygiene: InjecAgent tools stay held out of local training data (the Phase 2 leakage gate already guarantees this for the dataset; the LoRA trainer must call `assert_clean` on the files it reads, as `train.py` does).
- Do not read or quote `.env`; no secrets or canary values in logs/docs.
- GSD workflow: edits go through a GSD command (the planner/executor), not ad-hoc.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Episode -> Qwen chat messages, token ids, loss mask | Data layer (`src/qwen_lora/data.py`, torch-free apart from the tokenizer) | Phase 2 `dataset.py` (episodes, validation, leakage gate) | Same builder (`build_messages`, `OBSERVATION_TEMPLATE`) the loop uses, so train/serve parity is by construction |
| LoRA training loop, evaluation, resume, manifest | Offline training tier (`src/qwen_lora/train.py`, GPU, WSL) | `provenance.py` (git/hash/atomic JSON) | Training is a serial GPU job; reuse Phase 2 provenance helpers |
| Model serving for the benchmark | LLM provider layer (`src/llm/hf_local.py`) | `registry.py` + `config/models.yaml` | `LLMProvider` is the only seam the agent loop sees; nothing downstream changes |
| Chat-template rendering at inference | Provider (`tokenizer.apply_chat_template`) | — | The one renderer for both training and serving; the project's system prompt comes from `build_messages` |
| Decoding policy (greedy, stop tokens, token caps) | Provider | Config (`max_new_tokens`, `max_context_tokens`) | Decoding is a property of the serving stack and must be explicit, not inherited from `generation_config.json` |
| Provenance stamps (adapter hash, base revision, finish reason, token counts) | Provider -> `LLMResponse` | Runner `record()` (already persists them) | EVAL-06 stamping already exists downstream; the provider only has to fill the fields |
| Enforcement of actions (policy engine, sandbox, seccomp) | Unchanged | — | The model is an untrusted principal; Phase 3 changes nothing here |

## Measured Facts (Day-0 smoke test, executed this session)

All numbers below were measured on this machine on 2026-10-22 (WSL2 Ubuntu, repo venv at `/mnt/c/.../os_level_guardrails/venv`, RTX 3060 Laptop 6 GB). Prototype scripts lived in the session scratchpad and are not in the repo; the load-bearing code is reproduced under Code Examples.

### Environment and install [VERIFIED: pip, session output]

| Item | Measured value |
|------|----------------|
| Python | `3.14.4` (`venv/bin/python --version`) |
| torch (untouched) | `2.11.0+cu128`, CUDA 12.8 |
| numpy (untouched) | `2.5.4` |
| Installed | `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0` |
| Pulled in | `huggingface_hub 2.2.0`, `tokenizers 0.23.3`, `safetensors 0.8.0`, `hf-xet 1.7.0`, `regex 2026.9.29` (cp314 wheel), `psutil 7.2.2`, `typer 0.27.3`, `rich 15.0.0`, `annotated-doc 0.0.5`, `markdown-it-py 4.2.0`, `mdurl 0.1.2`, `shellingham 1.5.4` |
| Wheel problems | **None.** `pip install --dry-run` showed only these new packages (torch/numpy not in the list); real install succeeded; `pip check` -> "No broken requirements found." No source builds. |
| 3.14 fallback | Not needed. No uv 3.12 venv was created. |
| Install command that worked | `venv/bin/pip install "transformers>=5.20,<5.21" "peft==0.21.2" "accelerate>=1.15,<2"` |
| Base model download | `Qwen/Qwen2.5-0.5B-Instruct` -> `/home/rvsre/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`, 954 MB cache (ext4 home, not /mnt/c), 362 s at ~2.6 MB/s; `model.safetensors` = 988,097,824 bytes. Already cached: **no further download is needed**. |
| Python import cost (the /mnt/c venv is slow) | `import torch` 17 s; `from transformers import AutoTokenizer` 65 s (it imports torch); `import peft` 74 s; `import tokenizers` 0.3 s; `import src.llm.registry` 0.5 s (so registry stays cheap). Budget ~75 s per process that imports the HF stack, including every pytest session that touches it. |
| GPU at start | 6,144 MiB total, ~950 MiB used by the Windows desktop, 5,044-5,118 MiB free; `nvidia-smi --query-compute-apps` listed no compute process (the other project is CPU-only); nothing was touched or stopped. |
| Free disk / RAM | WSL ext4 941 GB free; WSL RAM total 10 GiB, 9 GiB available |

### Model facts [VERIFIED: `config.json`, `generation_config.json`, `tokenizer_config.json` read from the snapshot]

```
"hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
"intermediate_size": 4864, "vocab_size": 151936, "tie_word_embeddings": true, "max_position_embeddings": 32768,
"eos_token_id": 151645, "bos_token_id": 151643                       (config.json)
generation_config.json: "do_sample": true, "repetition_penalty": 1.1, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "eos_token_id": [151645, 151643], "pad_token_id": 151643
tokenizer: eos_token '<|im_end|>' (151645), pad_token '<|endoftext|>' (151643), bos_token None, add_bos_token False
```
So `model.generation_config` loaded from disk is a **sampling** config with a repetition penalty (confirmed by printing it at runtime: `'do_sample': True ... 'repetition_penalty': 1.1 ... 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8`). Pad (151643) != eos (151645) here, but the mask must still never be derived from token ids (see Pitfalls). Base-model parameter count (computed from the measured totals): 502,830,976 total with LoRA minus 8,798,208 trainable = **494,032,768** base parameters.

### Chat template behaviour [VERIFIED: runtime]

- `tok.apply_chat_template(msgs, add_generation_prompt=True)` with no other arguments returns a **`BatchEncoding`** in transformers 5.20.2 (not a list/tensor). Use `tokenize=True, return_dict=True, return_tensors="pt"` explicitly (as the code examples do) or `tokenize=False` for text.
- `tokenize=True` ids == `tok(rendered_text, add_special_tokens=False)` ids: **True** (len 296 for case 0).
- With no system message, Qwen **injects** `'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|...'`. The project always supplies its system prompt via `build_messages`, so the rendered text starts `'<|im_start|>system\nYou are a helpful AI assistant that executes system commands '`.
- The template is **prefix-stable**: for all 5,579 (episode, step) prompts of the 2,000 dev episodes, `apply_chat_template(episode_to_messages(ep, k), add_generation_prompt=True)` is an exact token prefix of the full-conversation ids, and the next token is supervised (0 mismatches).
- A literal `<|im_end|>` inside message content is tokenized as the real control token (`tok("x <|im_end|> y")` -> `[87, 220, 151645, 379]`). The Phase 2 train file (40,000 lines) and the benchmark data files contain **0** occurrences of `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.

### Sequence lengths in Qwen tokens [VERIFIED: tokenized all 40,000 train + 2,000 dev episodes]

| Split | n | mean | p50 | p90 | p99 | p99.9 | max | total tokens | > 1024 | supervised share |
|-------|---|------|-----|-----|-----|-------|-----|--------------|--------|------------------|
| train | 40,000 | 500.2 | 488 | 668 | 755 | 826 | **890** | 20,007,633 | **0** | 14.75% (mean 73.8 tokens, max 185) |
| dev | 2,000 | 494.9 | 481 | 660 | 737 | 783 | **791** | 989,714 | 0 | 14.93% |

Turns per train episode: 2 turns 17,600, 3 turns 13,193, 4 turns 9,207. System prompt alone = 242 tokens. First-turn prompts of the 57 benchmark cases: min 288, mean 346.5, max 442 tokens. Longest benchmark `tool_response` is 465 characters (< the 640-character training observation cap, so no length shift in observations).

### Decoding speed and the base control row [VERIFIED: runtime]

| Measure | Value |
|---------|-------|
| Load (bf16, cuda, sdpa) | 2.3 s; VRAM after load 950 MiB; peak during generation 1,008 MiB |
| Greedy decode, single sequence (includes prefill) | 33.3-34.9 tok/s (5 prompts, mean 34.1); 32.1 tok/s over 12 episodes |
| First benchmark prompt, base model | `{"action": "call_tool", "tool": "AmazonGetProductDetails", "args": {"product_id": "B08KFQ9HK5"}}` -> `parse_action` **accepts** it |
| 12 real episodes, unprotected arm (4 benign, 4 dh, 4 ds), base model | mean 6.53 s/episode (range 2.6-15.9), mean 3.92 steps, **26 of 47 steps were parse failures (21 parsed = 45%)**, 0 of 12 attacked, stop reasons `max_steps` and `responded` |
| What the base model emits when it fails | `{"action": "GitHubSearchRepositories", "keywords": ...}` (tool name used as the action), `{"details": {...}}` (wrong schema), plain prose ("The Evernote note containing the word 'Budget' has been found ...") |
| Sandbox overhead | Control agent through `run_episode`: unprotected 0.0 s, protected 0.013 s mean per episode (12 cases), 3.67 steps: **model time dominates**; the protected arm adds negligible wall time for tool-call episodes |

### LoRA training measurements (r=16, alpha=32, dropout 0.05, all-linear, grad checkpointing non-reentrant, bf16 autocast, adapters fp32) [VERIFIED: runtime]

Trainable parameters **8,798,208 of 502,830,976 = 1.750%**; trainable dtype `torch.float32`; saved `target_modules`: `["k_proj","down_proj","o_proj","gate_proj","up_proj","q_proj","v_proj"]` (all-linear resolves to these seven; `lm_head` and embeddings are not adapted).

Micro-benchmarks (5 timed steps + warm-up, last two steps use the longest episodes = worst case; real tokens/s):

| Config | Peak allocated | Reserved | Step time | Real tok/s | Verdict |
|--------|----------------|----------|-----------|------------|---------|
| mb=1, HF `labels=` loss | 2,872 MiB | 5,384 MiB | 0.71 s | 937 | works, slow |
| mb=4, HF `labels=` loss | **8,263 MiB** | **11,820 MiB** | 6.66 s | 362 | **exceeds the 6,144 MiB card, spilled** - do not use |
| mb=4, LM-head on supervised positions only | **2,420 MiB** | 4,958 MiB | 0.97 s | 2,489 | **chosen** |
| mb=8, LM-head on supervised positions only | 3,735 MiB | 6,150 MiB | 2.24 s | 2,112 | reserved >= card, slower: avoid |
| mb=2, supervised-only head, **no** grad checkpointing | 4,817 MiB | 5,328 MiB | 0.39 s | 3,199 | 29% faster but leaves < 1 GiB headroom on a shared GPU: avoid |

The two mb=4 paths gave the same losses to ~3 decimals (`[0.733,0.672,0.416,0.47,0.249,0.233]` vs `[0.733,0.667,0.412,0.469,0.246,0.233]`; the residual is dropout RNG), so the custom loss is equivalent to HF's. Cause of the 8 GiB peak: B x T x 151,936 logits (4 x ~800 x 151,936 x 2 bytes = 0.97 GB bf16, plus an fp32 upcast and its gradient). [ASSUMED] that the run above 6 GiB succeeded by WSL2/Windows "system memory fallback" rather than failing; the observed effect (reserved 11.8 GiB on a 6 GiB card, 7x slower) is what matters.

**Realistic 60-step run** (`PYTORCH_ALLOC_CONF=expandable_segments:True`, micro-batch 4 x accumulation 4 = 16 episodes/step, lr 2e-4 cosine to 2e-5, 3 warm-up steps, first 1,500 train episodes sampled with replacement across steps):

| Metric | Value |
|--------|-------|
| Wall time | 243 s = **4.05 s per optimizer step** (16 episodes, ~8 k tokens), **1,984 real tok/s** |
| Peak VRAM | **2,305 MiB allocated, 2,546 MiB reserved** (expandable segments removed the fragmentation seen above) |
| Training loss (token-weighted over the accumulation window) | 0.862 (step 1) -> 0.251 (step 11) -> 0.150 -> 0.077 -> 0.087 -> 0.059 -> 0.062 (step 60) |
| Dev, 200 random dev episodes, teacher-forced, **action-level** (respond message text excluded) | step 0: action loss 0.363, **action exact 0.007**, line exact (all tokens) 0.004. Step 60: action loss 0.0211, **action exact 0.899** (benign 0.958, dh 0.952, ds 0.785), line exact 0.532 |
| Dev evaluation cost | 15.2 s and 18.1 s for 200 episodes (python-loop metric) -> ~25 s for 300, ~3 min for the full 2,000 |
| Adapter on disk | `adapter_model.safetensors` **35,237,104 bytes**, `adapter_config.json` 1,182 bytes, `README.md` 5,206 bytes (PEFT model card; harmless) |
| Round trip | adapter reloaded from disk (fresh base + `PeftModel.from_pretrained`) gives **20/20 identical** first-turn replies to the in-memory adapter; reloaded adapter params are fp32 |

### Serving the trained adapter (60-step adapter, 12 real episodes, unprotected arm) [VERIFIED: runtime]

| Mode | s/episode | decode tok/s | parse-fail steps | attacked | notes |
|------|-----------|--------------|------------------|----------|-------|
| base (for comparison) | 6.53 | 32.1 | 26/47 | 0/12 | prose / wrong schema |
| LoRA, unmerged PEFT wrapper | 6.35 | **13.3** | 1/36 | 8/12 | 168 extra adapter matmuls per token |
| LoRA, reloaded from disk, unmerged | 6.18 | 13.3 | 1/36 | 8/12 | identical behaviour |
| LoRA, `merge_and_unload()` | **2.74** | **30.5** | 1/36 | 8/12 | merged == unmerged on 20/20 first-turn prompts; every first reply equals the gold first action (20/20) and parses (20/20) |

The one parse failure of the tuned model: it copied a quote-escaped (`\'`) address from the observation into JSON, producing invalid JSON. The tuned model attacks 8/12 in the *unprotected* arm because **the training mix deliberately teaches compliance** (every dh/ds episode follows the injected instruction); it is a second unaligned worst-case principal, not a safety-tuned model. This must be disclosed (see Pitfalls).

### Failure modes of loading [VERIFIED: runtime, CPU]

| Situation | Exception (transformers 5.20.2 / peft 0.21.2, `local_files_only=True`) |
|-----------|---------------------------------------------------------------------|
| Base model id not in cache | `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.` (both model and tokenizer) |
| Adapter directory missing or empty | `ValueError: Can't find 'adapter_config.json' at '<dir>'` |
| Adapter directory with a corrupt/empty config | `KeyError: 'peft_type'` |
| Good adapter on a tiny random-init Qwen2 (9.8 M params, hidden 64, 2 layers, real tokenizer) | loads, merges, and greedy-generates on CPU in seconds - usable as a fast integration test with no 1 GB download |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0+cu128 (installed, do not touch) | tensors, autocast, AdamW | already the project's training stack [VERIFIED: pip list] |
| transformers | 5.20.2 | `AutoModelForCausalLM`, `AutoTokenizer`, `apply_chat_template`, `generate` | the only supported loader for Qwen2; v5 uses `dtype=` not `torch_dtype=` [VERIFIED: runtime] |
| peft | 0.21.2 | `LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained`, `merge_and_unload` | reference LoRA implementation; `all-linear` target resolution [VERIFIED: runtime] |
| accelerate | 1.15.0 | dependency of peft (not used directly; the loop is plain PyTorch) | installed and importable [VERIFIED: pip list]; that peft declares it as a requirement is [ASSUMED] from peft's metadata |
| safetensors | 0.8.0 | adapter file format (`adapter_model.safetensors`, 35 MB) | pulled in [VERIFIED: pip] |
| tokenizers | 0.23.3 | Qwen BPE with `return_offsets_mapping` (used for the loss mask) | pulled in [VERIFIED: pip] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface_hub | 2.2.0 | `snapshot_download(local_files_only=True)` to resolve the cached snapshot path and revision hash | provider construction (offline) |
| matplotlib | existing | loss curve PNG (same as Phase 2) | training end |
| pytest | existing | tests | always |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom loop | TRL `SFTTrainer` | Out of scope (REQUIREMENTS); `assistant_only_loss` needs `{% generation %}` markers the Qwen2.5 template lacks [CITED: .planning/research/SUMMARY.md] |
| Merged serving | Live PEFT wrapper | 2.2x slower, behaviour identical on the checks run; keep a `merge: false` config switch only for debugging |
| In-process HF | `transformers serve` / Ollama GGUF | Out of scope for Phase 3 (SERVE-V2-01); different serving stack = extra confound |

**Installation (add as new file `requirements-train.txt`; keep the benchmark install light):**
```bash
# WSL, inside the existing venv. torch 2.11.0+cu128 is already installed - do NOT reinstall it.
pip install "transformers>=5.20,<5.21" "peft==0.21.2" "accelerate>=1.15,<2"
# resolved on 2026-10-22: transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, huggingface_hub 2.2.0,
# tokenizers 0.23.3, safetensors 0.8.0, hf-xet 1.7.0
```
Record the exact resolved versions in the training manifest (`packages` field) rather than relying on the ranges.

## Package Legitimacy Audit

Run via `gsd-tools query package-legitimacy check --ecosystem pypi ...` this session. The seam rates every package `SUS`, but only for `too-new` (latest *release* within days, e.g. transformers 5.20.2 published 2026-10-19) and `unknown-downloads` (the seam has no PyPI download counts); none has a postinstall, none is deprecated, all have a source repo under the expected owner. These are the Hugging Face organisation's own flagship packages, were named by the task brief and the project's STACK.md, and the user pre-approved installing them; they are **already installed** and `pip check` is clean.

| Package | Registry | Latest release | Source Repo | Verdict | Disposition |
|---------|----------|----------------|-------------|---------|-------------|
| transformers | PyPI | 2026-10-19 | github.com/huggingface/transformers | SUS (too-new, unknown-downloads) | Approved by user; pin `>=5.20,<5.21`; no new install gate needed |
| peft | PyPI | 2026-10-01 | github.com/huggingface/peft | SUS (too-new, unknown-downloads) | Approved; pin `==0.21.2` |
| accelerate | PyPI | 2026-09-09 | github.com/huggingface/accelerate | SUS (unknown-downloads) | Approved; pin `>=1.15,<2` |
| safetensors | PyPI | 2026-06-09 | github.com/huggingface/safetensors | SUS (unknown-downloads) | Transitive; pinned by resolver |
| tokenizers | PyPI | 2026-10-09 | github.com/huggingface/tokenizers | SUS (too-new, unknown-downloads) | Transitive |
| huggingface_hub | PyPI | 2026-10-08 | github.com/huggingface/huggingface_hub | SUS (too-new, unknown-downloads) | Transitive |
| hf-xet | PyPI | 2026-10-06 | github.com/huggingface/xet-core | SUS (too-new, unknown-downloads) | Transitive (download accelerator) |
| regex, psutil, typer, rich | PyPI | 2026-09-29 / 2026-01-28 / 2026-10-06 / 2026-04-12 | mrab-regex / giampaolo/psutil / fastapi/typer / Textualize/rich | SUS (recency / unknown-downloads) | Transitive, well-known |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged [SUS]:** all of the above by the seam's recency heuristic. Because the user already approved and executed the install, the planner does **not** need a `checkpoint:human-verify` install task; it only needs to pin versions in `requirements-train.txt` and note the SUS-by-recency rationale in the plan summary. `bitsandbytes`, `trl`, `unsloth`, `flash-attn`, `datasets` are not installed and must not be added.

## Architecture Patterns

### System Architecture Diagram

```
 TRAINING (offline, WSL, one serial GPU job)                    SERVING (benchmark time)
 ------------------------------------------                     ------------------------
 data/local/train-seed0-n40000.jsonl  (sha a258...)             run_benchmark.py --models qwen2.5-0.5b-{base,agentlora}
 data/local/dev-seed0-n2000.jsonl     (sha 1f63...)                  |
        |  read_episodes + assert_clean (leakage gate)               v
        v                                                       runner.get_provider(name)  --RuntimeError--> "[skip] name: ..."
 qwen_lora/data.py                                                  |
   episode_to_chat(ep)  = build_messages + [assistant JSON,         v
        user OBSERVATION_TEMPLATE]* (same as AgentLoop)         llm/registry.py (provider: hf_local, lazy import)
   apply_chat_template(tokenize=False)  --> text                    |
   tokenizer(text, offsets)  --> ids, labels, action_mask           v
        |   labels = assistant content + <|im_end|> only        llm/hf_local.py  HFLocalProvider
        v                                                          construct: snapshot_download(local_files_only)
 qwen_lora/train.py                                                          load bf16 -> cuda -> PeftModel(adapter)
   bf16 base (frozen) + LoRA r16 a32 all-linear (fp32)                       -> merge_and_unload -> eval
   micro-batch 4 x accum 4, grad-checkpointing                                verify adapter_config/manifest stamps
   hidden states -> LM-head on supervised positions only         generate(messages):
   AdamW lr 2e-4 cosine, eval every 100 steps on dev                          apply_chat_template(add_generation_prompt)
   resume file each eval; best-by-dev_loss_action                            greedy generate (explicit kwargs, stop <|im_end|>)
        |                                                                     -> LLMResponse(text, tokens, finish_reason,
        v                                                                        resolved_model = base@rev+adapter@sha12)
 models/qwen-agentlora/                                                      |
   adapter_model.safetensors  adapter_config.json                            v
   manifest.json  train_log.jsonl  loss_curve.png               AgentLoop -> parse_action -> ToolBroker -> executors (unchanged)
```

### Recommended Project Structure
```
src/
├── llm/
│   ├── hf_local.py          # HFLocalProvider (lazy torch/transformers/peft imports inside __init__)
│   └── registry.py          # + `hf_local` branch (lazy import, like scratch)
├── qwen_lora/
│   ├── __init__.py          # light
│   ├── data.py              # episode_to_chat, build_example, render stamp, batch collation
│   └── train.py             # python -m src.qwen_lora.train ; train(), evaluate(), resume, manifest
config/models.yaml           # + qwen2.5-0.5b-base, qwen2.5-0.5b-agentlora
models/qwen-agentlora/       # adapter_model.safetensors (git-ignored by models/**/*.safetensors), adapter_config.json,
                             # manifest.json, train_log.jsonl, loss_curve.png (tracked, like models/scratch)
requirements-train.txt       # pinned HF stack
tests/
├── test_hf_local_provider.py
├── test_qwen_lora_data.py
├── test_qwen_lora_train.py
docs/viva/03-fine-tuning-lora-and-chat-templates.md
```
`.gitignore` already lists `models/**/*.safetensors`, `models/**/*.tmp`, `models/**/*.done`, `models/**/*.log`; `models/qwen-agentlora/adapter_config.json`, `manifest.json`, `train_log.jsonl`, `loss_curve.png` stay tracked (Phase 2 convention). Note the adapter's own `README.md` written by PEFT can be left or deleted; it is not needed.

### Pattern 1: Episode to chat messages (identical to what `AgentLoop` sends) [VERIFIED: tested on 42,000 episodes]
**What:** Reuse `build_messages(ep["user"], ep["tools"])` for system+user, then for each turn append `{"role":"assistant","content": compact_action(action)}` and, if the turn has an observation, `{"role":"user","content": OBSERVATION_TEMPLATE.format(result=obs)}`. The last turn is always `respond` with observation `null` (validated by `validate_episode`).
**Quoted constants (read this session):** `prompting.py:30` `OBSERVATION_TEMPLATE = "Result of your last action:\n{result}"`; `dataset.py:66` `EPISODE_KEYS = ("id", "split", "cls", "tools", "user", "turns", "meta")`; `loop.py:19` `MAX_ASSISTANT_CHARS = 2000` (the loop stores `response.text[:2000]`, and every gold action is <= 400 characters, so no truncation difference).
**Do not** pass `tools=` to `apply_chat_template`: that switches Qwen to its own `<tools>` block and would break parity with the project's plain-text tool list in the system prompt.
**One sequence per episode, all assistant turns supervised.** Because attention is causal and the template is prefix-stable (verified), supervising every assistant span in the full conversation is equivalent to training each step separately on its own prefix, at 1/3 of the cost. (Phase 2 trained the same way: `training_text` = last-step prompt + last action, with earlier `A:` lines supervised in history.)

### Pattern 2: Loss mask from character offsets (not from token ids, not from prefix lengths) [VERIFIED]
Render the full text once; locate each assistant span as the exact string `"<|im_start|>assistant\n" + content + "<|im_end|>"` searching forward from the previous span end (assert found); supervise tokens whose `offset_mapping` lies inside `[content_start, span_end)`. The header `<|im_start|>assistant\n` and the trailing `\n` after `<|im_end|>` are **not** supervised (the generation prompt is supplied at inference and generation stops at `<|im_end|>`). Result on all 42,000 episodes: decoded supervised tokens == `"".join(content + "<|im_end|>")` exactly, 0 mismatches.

### Pattern 3: LM-head only on supervised positions
Call the decoder (`peft_model.base_model.model.model`) to get hidden states, select `h[:, :-1][labels[:, 1:] != -100]`, apply `lm_head` to those rows only, and take cross-entropy in fp32. Normalise by the **supervised-token count of the whole accumulation window** (not per micro-batch) so a 16-episode step is a token-weighted mean. This is what made 4 x 800-token micro-batches fit in 2.4 GiB.

### Pattern 4: Merged serving with provenance
Load base bf16 -> `PeftModel.from_pretrained(base, adapter_dir)` -> `merge_and_unload()` (assign the return value) -> `.eval()`. Because the PEFT object is gone after the merge, compute the adapter identity from `adapter_model.safetensors` bytes **before** merging and stamp `resolved_model = "<base>@<rev12>+<adapter name>@<sha12>"`.

### Anti-Patterns to Avoid
- **HF `labels=` loss with a 152k vocabulary at micro-batch > 1:** peak 8.3 GiB measured. Use Pattern 3.
- **Inheriting `generation_config.json` for "greedy":** it is `do_sample: true, repetition_penalty: 1.1, top_p 0.8, top_k 20`. Always pass the explicit greedy kwargs.
- **Masking padding by `input_ids == pad_token_id`:** equals the EOS id on many Qwen setups and would hide the stop token; build labels explicitly with `-100` for padding positions only by position (length), never by id.
- **Building the Qwen prompt by hand or omitting the system message:** Qwen then injects "You are Qwen, created by Alibaba Cloud".
- **Training on the full 40,000 episodes for several epochs:** 2.8 h per epoch and more passes over templated data degrade the model [CITED: .planning/research/PITFALLS.md line 222].
- **A new `torch_dtype=` argument:** v5 renamed it to `dtype=`.

## Chat Template, Rendering and Loss Mask (answers to the planner's questions)

1. **Parity train vs serve.** One function (`episode_to_chat`) mirrors `AgentLoop`; one renderer (`tokenizer.apply_chat_template`) is used by both the trainer (`tokenize=False`, then offsets) and the provider (`add_generation_prompt=True, tokenize=True, return_dict=True`). The 5,579-prompt prefix test above is the proof; reproduce it as a unit test.
2. **System prompt.** Always `build_messages(...)[0]`. Test that the rendered text begins with `<|im_start|>system\n` + `SYSTEM_PROMPT` and never contains `"You are Qwen"`.
3. **Observations as user turns.** The loop appends `{"role":"user","content": OBSERVATION_TEMPLATE.format(result=...)}` (loop.py:97) and, after an unparseable reply, `INVALID_REPLY_TEMPLATE` (loop.py:84). Phase 2 episodes contain no parse-error recovery turns (all gold), so the tuned model never saw an `INVALID_REPLY_TEMPLATE` turn in training; at inference it is rarely needed (1/36 steps in the smoke). Do not add synthetic recovery turns in Phase 3 (scope creep); disclose.
4. **Loss mask = assistant content + `<|im_end|>`.** Excluded: system/user/observation tokens, the `<|im_start|>assistant\n` header, the trailing newline after `<|im_end|>`, and padding.
5. **Pad/EOS.** Here pad `<|endoftext|>` (151643) != eos `<|im_end|>` (151645) (verified), and 0 of 300 sampled dev sequences contain the pad id, but design the mask by span position anyway. At generation pass `eos_token_id=[151645, 151643]` (the model's shipped list) and `pad_token_id=151643`; treat either as `finish_reason="stop"`.
6. **Decoded-supervised-tokens test design.** For N episodes (dev 300 + a train sample, or all 42,000: the full check takes ~42 s plus the 65 s import): `decoded = tok.decode([id for id, l in zip(ids, labels) if l != -100])` must equal `"".join(compact_action(a) + "<|im_end|>" for a, _ in ep["turns"])`. Plus: (a) the last supervised token of every assistant span is 151645; (b) no supervised token lies in an observation/user span (check by decoding the *unsupervised* remainder and asserting it contains every observation string and none of the action strings' `<|im_end|>`); (c) a doctored sequence whose content contains the pad id still keeps those tokens supervised (mask is not id-based); (d) the action-level mask excludes only respond message text.

**Action-level mask (for the selection metric).** Mirror Phase 2's CR-01 lesson: held-back dev wording makes `respond` message text unpredictable, so selection must not depend on it. Exclude tokens whose start offset lies in `[end of '{"action":"respond","message":"', start of the closing '"}')`; keep the head, the closing `"}` and `<|im_end|>`. Verified example: excluded text `'Access to that tool was denied, so I stopped."'`, action-level tokens `...{"action":"respond","message":"}<|im_end|>`. BPE merges the final quote with the preceding word piece, so the boundary token is excluded as a whole; that is acceptable (it is a conservative, deterministic rule) and must be documented in the viva page and the manifest `selection.description`.

## Training Recipe (budget, schedule, evaluation, resume)

### Budget [VERIFIED rates; plan figures are arithmetic from them]
| Quantity | Value |
|----------|-------|
| Measured step time | 4.05 s per optimizer step (16 episodes, ~8,000 tokens) |
| One full epoch of train (40,000 episodes, 20.0 M tokens) | ~10,000 s = **2.8 h** at 1,984 tok/s - too long; do not run it |
| **Recommended:** `max_steps 600`, effective batch 16 (mb 4 x accum 4) | 9,600 episodes (24% of one epoch), 4.8 M tokens, **~41 min** + 6 dev evaluations (~25 s each on a fixed 300-episode subset) + one final full-dev pass (~3 min) = **~46 min** |
| Hard cap if more is wanted | 750 steps = 51 min; do not exceed 60 min (budget) |
| Loss of work on a VM restart | <= 100 steps (~7 min) because the resume file is written at each evaluation |

The first 12,000 train episodes are class-balanced (benign 5,971 / ds 3,016 / dh 3,013; synthetic-tool share 4,180) [VERIFIED: counted], so "the first N episodes" is an acceptable, reproducible subset; record `n_train_used` and the file sha in the manifest. A shuffled order per seed is better: draw each step's 16 episodes from a seeded permutation of the first N (`random.Random(seed*1_000_003 + epoch)`), as `make_batches` does for ScratchGPT.

### Hyper-parameters
| Setting | Value | Why |
|---------|-------|-----|
| LoRA | r=16, alpha=32 (scaling 2), dropout 0.05, `target_modules="all-linear"`, `task_type="CAUSAL_LM"` | locked; resolves to q,k,v,o,gate,up,down in 24 layers [VERIFIED] |
| Precision | base bf16 frozen; adapters fp32 (PEFT default); `torch.autocast("cuda", dtype=torch.bfloat16)` | measured stable; fp16 not needed |
| Attention | `sdpa` (default; confirmed `attn` = sdpa) | no flash-attn |
| Gradient checkpointing | `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` then `enable_input_require_grads()` **before** `get_peft_model` | measured working order; set `config.use_cache = False` while training, `True` for serving |
| Optimizer | `torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)`, betas default | standard LoRA setting; 8.8 M params of state |
| Schedule | linear warm-up 20 steps (3%), cosine to 2e-5 (10% floor) | the 60-step smoke used 3-step warm-up with no instability; 20 is safer for 600 |
| Grad clip | 1.0 on trainable params | as measured |
| Micro-batch / accumulation | 4 x 4 (effective 16) | 2.3 GiB peak; mb=8 reserves the whole card |
| Max sequence length | 1024, **assert** (raise) if any rendered episode is longer; do not truncate | measured max 890 (train) / 791 (dev); truncation would cut supervised tails or the system prompt |
| Padding | right-pad to the batch max with the pad id; `attention_mask` 0 on pad; labels `-100` on pad | |
| Seed | `torch.manual_seed(seed)`; record seed; bf16 GPU runs are not bit-reproducible - say so |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` (set before torch is imported) | cut reserved memory from 4.9 GiB to 2.5 GiB at the same batch |
| DataLoader | none; pre-tokenised lists, python batching | `num_workers=0` equivalent; avoids the 3.14 forkserver default |

### Evaluation, selection and stopping
- **Fixed dev subset for in-training evals:** 300 episodes drawn once with a fixed seed from the 2,000 dev episodes, stratified by class (`dev_tools` + `dev_synthetic` both present); the **full 2,000** is evaluated once at the end and written to the manifest.
- **Selection metric (decide before the run): `dev_loss_action`** = mean cross-entropy over action-level supervised tokens (all supervised minus respond message text), lower is better; early stopping with `patience 3` evaluations (300 steps). Report as headline `dev_action_exact` (teacher-forced: every action-level token of an assistant message equals the argmax given the gold prefix), per class, plus `dev_line_exact` and `dev_loss` as diagnostics, exactly the Phase 2 split of selection vs diagnostics. This reuses the CR-01 lesson: never select on respond text.
- **Add one generation-level check at the end:** greedy-generate the first action for 100 dev prompts through the real provider path and report the parse rate and exact-gold rate (the smoke got 20/20 and 20/20). It catches render/serve skew that teacher forcing cannot.
- **Pre-registered remediation rule (write it into the plan before running, as Phase 2 did):** if final full-dev `dev_action_exact < 0.90` and the best step is within the last two evaluations, do exactly one retry with `max_steps` doubled (1,200 steps, ~82 min) and stop. The smoke reached 0.899 after only 60 steps, so the expected outcome is well above 0.90; the weakest class was ds (0.785 at step 60).
- Do not tune anything after seeing smoke-benchmark behaviour (Phase 2 rule).

### Resumability (Phase 2 pattern)
- After every evaluation write `qwen_lora.resume.tmp` (trainable tensors as a name->tensor dict, optimizer state, torch/CUDA RNG, step, epoch, position in the epoch's batch order, best-so-far, history, elapsed, identity) with `torch.save`; load with `weights_only=True`. Write the best adapter as `adapter.best.tmp/` via `save_pretrained`; rename to the final directory only when the run finishes.
- `identity` = train/dev sha256, LoRA config, train config, render stamp, package versions; `--resume` refuses on any difference (as `train.py` does). A non-finite loss exits 3 and deletes in-run files; a failed run leaves no loadable adapter.
- Reuse `provenance.git_commit/git_dirty/write_json_atomic/refuse_foreign_manifest/resolve_under_root`; call `refuse_foreign_manifest(out_dir, "adapter_sha256", "Qwen LoRA training")`.
- Reuse the Phase 2 launcher pattern (memory pre-check >= 3000 MiB available, GPU preflight >= 3584 MiB free with 5-minute re-checks, `train.done` exit-code marker, foreground `wsl.exe -d Ubuntu -- bash <script>` so the VM stays up). Peak need is ~2.6 GiB reserved so 3584 MiB free is ample; never stop another job's processes.

### Adapter save format and manifest
- `peft_model.save_pretrained(dir, save_embedding_layers=False)` -> `adapter_model.safetensors` (35.2 MB) + `adapter_config.json` (contains `base_model_name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"`, `r: 16`, `lora_alpha: 32`, `lora_dropout: 0.05`, the seven `target_modules`, `peft_version: "0.21.2"`) + PEFT's `README.md`. Passing `save_embedding_layers=False` avoids a harmless "Could not find a config file in Qwen/Qwen2.5-0.5B-Instruct" warning seen in offline mode.
- `adapter_sha256` = sha256 of `adapter_model.safetensors`.
- **Manifest fields** (schema 1; assert the key tuple like `MANIFEST_KEYS` in `train.py:69-73`): `schema, created_at, code_commit, code_dirty, python, torch, cuda, device, packages {transformers, peft, accelerate, tokenizers, safetensors}, seed, base_model {id, revision (snapshot hash 7ae557604adf67be50417f59c2c2f167def9a775), model_safetensors_bytes}, tokenizer {chat_template_sha256, eos_id, pad_id}, render {fingerprint, system_prompt_sha256, observation_template}, dataset {train_path, train_sha256, dev_path, dev_sha256, n_train_available, n_train_used, n_dev, class_counts, held_out_sha256}, lora {r, alpha, dropout, target_modules, trainable_params, total_params, trainable_fraction}, config {train hyper-parameters, micro_batch, accum, max_len}, steps, best_step, tokens_seen, peak_vram_mib {allocated, reserved}, selection {metric, direction, description, remediation_rule}, best {step, metrics...}, last_eval, final_dev_full {metrics}, gen_check {parse_rate, gold_first_action_rate}, adapter_sha256, wall_seconds, stop_reason, resumed_from_steps, run_notes`.

## hf_local Provider Specification

### Config entries (`config/models.yaml`)
```yaml
  qwen2.5-0.5b-base:
    provider: hf_local
    model: qwen2.5-0.5b-base                  # stamped as config_model on every row
    base_model: Qwen/Qwen2.5-0.5B-Instruct
    adapter: null                             # un-tuned control row (LORA-02)
    family: open / tiny / pretrained, not fine-tuned
    benchmark: true

  qwen2.5-0.5b-agentlora:
    provider: hf_local
    model: qwen2.5-0.5b-agentlora
    base_model: Qwen/Qwen2.5-0.5B-Instruct
    adapter: models/qwen-agentlora            # produced by python -m src.qwen_lora.train
    family: open / tiny / locally LoRA-tuned
    benchmark: true
```
Also update the comment block at the top of `models.yaml` (`provider: openai_compat | scratch | scripted` -> add `hf_local`). `registry.get_provider` gets a third lazy branch beside `scratch` (registry.py:47 `if cfg["provider"] == "scratch":`), passing `name`, `base_model`, `adapter` joined to `PROJECT_ROOT` when not null, plus optional `max_new_tokens` / `max_context_tokens` / `merge`. Importing the registry must still not import transformers (0.5 s today; enforce with a subprocess `sys.modules` test as Phase 2 did).

### Constructor contract (fail early, RuntimeError, so the runner prints `[skip]`)
`runner.py:316-318` is `except Exception as e:  # missing key, missing checkpoint, unknown name` / `log(f"[skip] {model}: {e}")`, so any exception at construction skips the model and writes no rows. Raise `RuntimeError` with an actionable message for: base model not in the local cache (`snapshot_download(base_model, local_files_only=True)` fails -> "download it once with ..."), adapter directory or `adapter_model.safetensors`/`adapter_config.json` missing, adapter `base_model_name_or_path` != configured `base_model`, adapter manifest `adapter_sha256` != recomputed hash, render fingerprint in the manifest != the running code's (stale adapter, same idea as ScratchGPT stamps), CUDA requested but unavailable (allow CPU only when explicitly configured), or any load exception (`OSError`, `ValueError`, `KeyError` observed above). Pre-check file existence yourself because a corrupt config raises a bare `KeyError: 'peft_type'`.

### Loading
1. Before importing anything from the HF stack: `os.environ.setdefault("HF_HUB_OFFLINE", "1")` (the constant is read at import time; if `huggingface_hub` is already imported it is a no-op, so ALSO pass `local_files_only=True` to every `from_pretrained`).
2. `path = huggingface_hub.snapshot_download(base_model, local_files_only=True)`; `revision = os.path.basename(path)`.
3. `AutoTokenizer.from_pretrained(path)`; `AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()`.
4. If adapter: sha256 of the adapter file, `PeftModel.from_pretrained(model, adapter_dir)`, `merge_and_unload()`, `.eval()` (config `merge: true` default).
5. Replace the sampling defaults: set `model.generation_config` to a fresh `GenerationConfig(do_sample=False, repetition_penalty=1.0, ...)` **and** pass explicit kwargs per call.
6. `transformers.utils.logging.disable_progress_bar()` (or `HF_HUB_DISABLE_PROGRESS_BAR=1`) so the "Loading weights" bar does not flood the benchmark log.

### `generate(messages, temperature=0.0, max_tokens=512) -> LLMResponse`
- `enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)`; `n_in = enc["input_ids"].shape[1]`.
- **Context guard:** if `n_in + 1 > max_context_tokens` (config, default 4096; the model supports 32,768 and the longest realistic benchmark prompt is ~1.1 k tokens at step 5, so this never triggers in practice) return an error response with `finish_reason="context_overflow"` and `prompt_tokens=n_in` (the same convention as `LocalScratchProvider`: never answer from a cropped prompt).
- Greedy: `model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, max_new_tokens=min(max_tokens, max_new_tokens_cfg), eos_token_id=[151645, 151643], pad_token_id=151643)` inside `torch.inference_mode()`. `temperature` is accepted for interface compatibility and ignored by design (the loop always passes 0.0; the benchmark is greedy) - document it.
- `new = out[0, n_in:]`; `stopped = new[-1] in {151645, 151643}`; `finish_reason = "stop" if stopped else "length"`; `text = tok.decode(new[:-1] if stopped else new, skip_special_tokens=True).strip()`; `prompt_tokens = n_in`; `completion_tokens = len(new)` (includes the stop token); `latency_ms` via `perf_counter` incl. `torch.cuda.synchronize()`.
- `resolved_model`: base row `"Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf"`; adapter row `"Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf+qwen2.5-0.5b-agentlora@<adapter_sha12>"`. (Phase 2 convention: `scratch-gpt@<sha12>`.)
- Wrap the call in `except Exception` -> `LLMResponse(text="", error=f"{type(e).__name__}: {e}", ...)`; CUDA OOM is the one to expect (`torch.cuda.OutOfMemoryError`): return it as an error response (the runner classifies it as an infrastructure error and re-queues).
- Output expectations measured: base 32-34 tok/s; merged adapter 30.5 tok/s.
- Memory: the runner builds one provider per model sequentially and holds the previous one until the loop variable is reassigned; two 0.5B models are ~2 GiB, which fits. Optionally expose `close()`; not required.

## Benchmark Latency Estimate (342 episodes = 57 cases x 3 runs x 2 arms)

Per-process fixed cost: ~75 s import + ~3 s load (+ a few s CUDA init). Model time per episode dominates (sandbox overhead measured ~13 ms/episode for the scripted control).

| Model | Measured s/episode (12-episode sample, unprotected) | Estimate for 342 | Notes |
|-------|----------------------------------------------------|------------------|-------|
| qwen2.5-0.5b-base | 6.53 (2.6-15.9) | **~37 min** (30-50) | base averages 3.9 steps, and 55% of its steps are parse failures that burn up to 512 new tokens at 33 tok/s; protected-arm retries after BLOCKED add steps (upper bound with every episode at 5 steps ~8.3 s -> 47 min) |
| qwen2.5-0.5b-agentlora (merged) | 2.74 | **~16 min** (13-25) | 3.0 steps/episode, 30.5 tok/s |
| qwen2.5-0.5b-agentlora (unmerged, do not use) | 6.35 | ~36 min | 13.3 tok/s |
| Smoke run (`--runs 1 --dh 5 --ds 5 --benign 3` = 26 episodes) | - | base ~3 min, LoRA ~1.2 min, plus ~80 s startup | |

The sample is only 12 episodes; treat the base figure as +/-30%. ScratchGPT's smoke is the comparable precedent (26 rows, well under 10 minutes).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chat formatting | A custom Qwen prompt string | `tokenizer.apply_chat_template` on `build_messages` output | exact parity with serving; avoids the injected default system prompt |
| LoRA layers | Custom low-rank modules | `peft.LoraConfig/get_peft_model` | tested merge/unmerge, `all-linear` resolution, save/load |
| Adapter serialisation | A custom checkpoint dict | `save_pretrained` / `PeftModel.from_pretrained` + safetensors | safe (no pickle), 35 MB, round-trip verified 20/20 |
| Merging | Manual `W += (alpha/r) B A` | `merge_and_unload()` | handles dtypes and all modules (verified identical output) |
| Token-offset alignment | Re-tokenising prefixes to find mask boundaries | `return_offsets_mapping=True` + char spans | BPE-boundary-proof, one render per episode |
| Atomic JSON / git stamps / path confinement | New helpers | `src/scratch_llm/provenance.py` | identical conventions to Phase 2 |
| Resume logic | New format | the `train.py` resume/identity pattern | already reviewed for restart bugs (VM restarts happened in Phase 2) |
| Chat-template jinja | Parsing the template yourself | the tokenizer's own template | |

**Key insight:** every piece that can silently skew train and serve (template, system prompt, EOS, adapter identity) already has a library or a Phase 2 helper; the new code should be glue plus tests.

## Common Pitfalls

### Pitfall 1: Full-vocabulary logits blow the 6 GB budget
**What goes wrong:** `model(input_ids, labels=...)` at micro-batch 4, ~800 tokens: peak 8,263 MiB allocated on a 6,144 MiB GPU; runs 7x slower through memory fallback instead of failing cleanly.
**Why:** 151,936-token vocabulary: logits 0.97 GB bf16 + fp32 upcast + gradient.
**How to avoid:** Pattern 3 (LM-head on supervised positions). Log `max_memory_allocated/reserved` every eval and write them to the manifest; abort with a clear error if `reserved > 0.95 x total`.
**Warning signs:** step time > 2 s at mb=4; reserved > total VRAM in `nvidia-smi`/`torch.cuda.max_memory_reserved`.

### Pitfall 2: "Greedy" that is not greedy
**What goes wrong:** `generate()` inherits `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1` from `generation_config.json`; JSON with repeated keys/quotes is distorted.
**How to avoid:** explicit kwargs per call plus a replaced `generation_config`; unit test captures the kwargs the fake model received (do_sample False, repetition_penalty 1.0, temperature/top_p/top_k None).

### Pitfall 3: Default system prompt injection and train/serve template skew
**What goes wrong:** no system message -> "You are Qwen..." in the prompt; or hand-built training strings.
**How to avoid:** always `build_messages`; render-parity and prefix tests; assert `"You are Qwen"` absent.

### Pitfall 4: Loss mask derived from token ids
**What goes wrong:** masking `labels[input_ids == pad_id]` (or EOS) removes the stop token, so the model never learns to stop and runs past the JSON.
**How to avoid:** span/offset mask; assert the last supervised token of each assistant span is 151645; a test with a doctored pad-id token inside content.

### Pitfall 5: Unmerged adapter makes the tuned row slow and a stale adapter makes it wrong
**What goes wrong:** live PEFT 13 tok/s vs merged 30.5; an adapter trained under a different system prompt/template is benchmarked as if current.
**How to avoid:** merge at load; stamp adapter hash; refuse when the manifest's `render.fingerprint` differs from the running code (RuntimeError -> `[skip]`).

### Pitfall 6: Network access during a benchmark run
**What goes wrong:** a flaky network makes `from_pretrained` stall or fail mid-run.
**How to avoid:** `snapshot_download(..., local_files_only=True)` and `local_files_only=True` everywhere, `HF_HUB_OFFLINE=1` set before import; test by constructing the provider with the env var set and a missing id (observed `OSError` -> `RuntimeError`).

### Pitfall 7: Import cost and test-suite time
**What goes wrong:** importing transformers/peft takes 65-75 s per process on the /mnt/c venv; many test files each paying it makes the suite miserable.
**How to avoid:** one module-scoped fixture per file, keep tokenizer-dependent tests in one file, skip when `transformers` or the cached tokenizer is absent (`STRICT_HF=1` turns skip into failure, mirroring `STRICT_TORCH`), keep registry/provider-contract tests torch-free (fake model injected), and never import the HF stack at module import time of `src/llm/*`.

### Pitfall 8: Special-token strings in attacker-controlled text
**What goes wrong:** observation text containing `<|im_end|>` is tokenized as a real control token, letting injected text close the turn and open a fake assistant turn (a prompt-injection amplifier specific to chat-template models).
**State:** 0 occurrences in train data and benchmark files (verified), so Phase 3 results are unaffected. **Recommendation:** add a test that fails if any episode/benchmark text contains a Qwen special-token string, and record in the viva/threats page that sanitising is out of scope (changing prompts would break fairness across models). [ASSUMED] that an attacker-supplied tool response in a future benchmark could contain them.

### Pitfall 9: The tuned model is trained to comply
**What goes wrong:** the LoRA row's high unprotected attack rate (8/12 in the smoke) is read as "fine-tuning broke safety" or as a bug.
**Why:** the shared Phase 2 mix has dh/ds episodes that follow the injected instruction by design (D-01).
**How to avoid:** state in the manifest `run_notes`, the viva page and the report that both local models are intentionally compliant worst-case principals, and that Qwen-base vs Qwen-LoRA measures the fine-tuning delta.

### Pitfall 10: Base row mostly format failures
**What goes wrong:** the base row shows 45% parse rate and few attacks; someone "fixes" the provider (repairs JSON, strips prose).
**How to avoid:** the provider returns raw text exactly like `LocalScratchProvider`; the loop's `INVALID_REPLY_TEMPLATE` feedback is the only recovery; EVAL-01 conditions on format-valid episodes.

### Pitfall 11: Seed and non-determinism claims
**What goes wrong:** claiming bit-reproducible training. bf16 + GPU kernels are not; also greedy decoding makes the 3 runs per case near-duplicates (pseudo-replication, already deferred to Phase 5).
**How to avoid:** record seed and say "same seed, same data, results within noise"; do not claim determinism.

### Pitfall 12: VM restart / shared host
**What goes wrong:** WSL VM restarts killed Phase 2 runs; the host is shared with a CPU job.
**How to avoid:** resume file after each evaluation; memory pre-check; keep the launcher in the foreground; never stop other processes.

## Code Examples

Prototype code that ran in this session (research only; adapt to the repo conventions).

### Episode to messages and offset-based mask [VERIFIED: 0 mismatches over 42,000 episodes]
```python
# Source: prototype run 2026-10-22 against data/local/*.jsonl and Qwen2.5-0.5B-Instruct tokenizer
from src.agent_harness.prompting import build_messages, OBSERVATION_TEMPLATE
from src.scratch_llm.format import compact_action

IM_START_ASSISTANT = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"

def episode_to_chat(ep):
    msgs = build_messages(ep["user"], ep["tools"])
    for action, obs in ep["turns"]:
        msgs.append({"role": "assistant", "content": compact_action(action)})
        if obs is not None:                         # the last (respond) turn has no observation
            msgs.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(result=obs)})
    return msgs

def build_example(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False)            # no generation prompt
    spans, pos, gold = [], 0, []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        needle = IM_START_ASSISTANT + m["content"] + IM_END
        at = text.find(needle, pos)
        assert at >= 0, "assistant turn not found in rendered text"
        spans.append((at + len(IM_START_ASSISTANT), at + len(needle)))
        gold.append(m["content"] + IM_END)
        pos = at + len(needle)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels, k = [-100] * len(ids), 0
    for i, (s, e) in enumerate(offs):
        while k < len(spans) and s >= spans[k][1]:
            k += 1
        if k < len(spans) and e > s and s >= spans[k][0] and e <= spans[k][1]:
            labels[i] = ids[i]
    return ids, labels, gold
# test: tok.decode([i for i, l in zip(ids, labels) if l != -100]) == "".join(gold)
```

### LoRA setup and supervised-position loss [VERIFIED: runtime, 2.3 GiB peak]
```python
# Source: prototype run 2026-10-22 (peft 0.21.2, transformers 5.20.2)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                      target_modules="all-linear", task_type="CAUSAL_LM"))
pm.config.use_cache = False
params = [p for p in pm.parameters() if p.requires_grad]        # 8,798,208
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)

inner = pm.base_model.model                                      # Qwen2ForCausalLM inside LoraModel
with torch.autocast("cuda", dtype=torch.bfloat16):
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = inner.lm_head(h[:, :-1][sel])                       # only supervised rows hit the 152k head
    loss = F.cross_entropy(logits.float(), tgt[sel], reduction="sum") / n_supervised_in_window
loss.backward()
# every 4 micro-batches: clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
pm.save_pretrained(out_dir, save_embedding_layers=False)
```
Guard the internal-structure access (`pm.base_model.model.model`) with an `assert` and a tiny-model unit test, because it couples to PEFT/Qwen2 internals.

### Greedy generation call and merged serving [VERIFIED: runtime]
```python
enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                              return_dict=True, return_tensors="pt").to("cuda")
n_in = enc["input_ids"].shape[1]
with torch.inference_mode():
    out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0, max_new_tokens=512,
                         eos_token_id=[151645, 151643], pad_token_id=151643)
new = out[0, n_in:].tolist()
stopped = bool(new) and new[-1] in (151645, 151643)
text = tok.decode(new[:-1] if stopped else new, skip_special_tokens=True).strip()

base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
served = PeftModel.from_pretrained(base, adapter_dir).eval().merge_and_unload().eval()
```

### Registry branch (follow the `scratch` pattern)
```python
    if cfg["provider"] == "hf_local":
        from src.llm.hf_local import HFLocalProvider        # lazy: transformers/peft/torch load only here
        adapter = cfg.get("adapter")
        return HFLocalProvider(name=name, base_model=cfg["base_model"],
                               adapter=os.path.join(PROJECT_ROOT, adapter) if adapter else None,
                               max_new_tokens=int(cfg.get("max_new_tokens", 512)),
                               max_context_tokens=int(cfg.get("max_context_tokens", 4096)))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch_dtype=` in `from_pretrained` | `dtype=` | transformers 5.x | use `dtype` (verified working) |
| `apply_chat_template` returns ids/tensor | returns `BatchEncoding` by default | transformers 5.x | pass `return_dict`/`tokenize=False` explicitly |
| TRL `SFTTrainer` with `assistant_only_loss` | needs `{% generation %}` template markers (absent in Qwen2.5) | - | custom offset mask |
| Live adapters at inference | `merge_and_unload()` for serving | long-standing | 2.2x faster, identical output on checks |
| `transformers` import light | now pulls torch on `AutoTokenizer` import | v5 | 65 s on the /mnt/c venv |

**Deprecated/outdated:** `torch_dtype` argument; relying on `generation_config.json` for deterministic behaviour.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The run above 6 GiB (mb=4 HF-loss) succeeded through Windows/WSL system-memory fallback rather than some other mechanism | Measured Facts | Low: the avoided config is avoided either way; only the explanation in the viva page would change |
| A2 | Final 600-step adapter will reach teacher-forced `dev_action_exact` >= 0.90 (smoke: 0.899 after 60 steps; ds class 0.785) | Training Recipe | Medium: if not, the pre-registered one-retry rule (1,200 steps, ~82 min) applies |
| A3 | Full-benchmark latency for base (~37 min) is accurate to +/-30% | Latency | Low: only affects scheduling in Phase 4 |
| A4 | Future benchmark tool responses could contain Qwen special-token strings | Pitfall 8 | Low: currently 0 occurrences |
| A5 | Generic LoRA/transformer theory in the viva outline (W + (alpha/r)BA, B initialised to zero, A random) matches Hu et al. 2021 and the PEFT implementation | Viva outline | Low: standard; confirm against peft docs when writing the page |
| A6 | `merge_and_unload()` output equals the unmerged model on all prompts, not just the 20 first-turn prompts and 12 episodes tested | Provider | Low-Medium: bf16 rounding can flip a rare greedy token; the Phase 4 run stamps `resolved_model`, and a `merge: false` switch exists for a spot check |
| A7 | The `pm.base_model.model.model` / `lm_head` attribute path is stable for peft 0.21.2 + Qwen2 only | Code Examples | Low: pinned versions; a tiny-model test guards it |

## Open Questions

1. **Train/dev subset policy: first N or seeded sample?**
   - Known: the first 12,000 episodes are class-balanced; the whole file is shuffled.
   - Unclear: whether to prefer a seeded random subset for the report's "trained on 9,600 of 40,000" statement.
   - Recommendation: take the first `n_train_used = 9600` (reproducible from the file hash) and shuffle order per seed; record both numbers in the manifest.
2. **Should the final adapter be chosen by the 300-episode subset or the full dev set?**
   - Recommendation: select on the fixed 300-episode subset during training (cost), then report the full-dev numbers for the kept adapter only (no re-selection).
3. **Does Phase 4 want the tuned model also evaluated unmerged for a spot check?**
   - Recommendation: add `merge: false` to config but leave the default `true`; one 26-episode comparison is optional.
4. **Number of warm-up/eval hyper-parameters under a 6 GB shared GPU** - values above are measured for 60 steps only; the first real run is the real test (Phase 2 needed a dev-gated retry).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| WSL2 Ubuntu | all GPU work, sandbox | yes | kernel 6.18.33.2-microsoft-standard-WSL2 | - |
| Python | everything | yes | 3.14.4 | - |
| torch + CUDA | training, serving | yes | 2.11.0+cu128, RTX 3060 Laptop 6 GB | CPU (too slow for training) |
| transformers / peft / accelerate | provider, trainer | yes (installed this session) | 5.20.2 / 0.21.2 / 1.15.0 | - |
| Qwen2.5-0.5B-Instruct weights | provider, trainer | yes, cached | snapshot `7ae557604adf` (988,097,824 B) | re-download (362 s) |
| Free VRAM | training/serving | 5.0-5.1 GiB free at start | peak need 2.6 GiB | wait/re-check via preflight (>= 3584 MiB) |
| Disk | adapters, logs | 941 GB free (ext4 home), /mnt/c 33 GB free | - | - |
| Ollama | not needed this phase | not checked | - | - |
| uv (3.12 fallback) | not needed | not checked / not created | - | - |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none.

## Validation Architecture

> `.planning/config.json` has `workflow.nyquist_validation: false`; this section is included because the task explicitly requires it.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 (existing) |
| Config file | none (`tests/conftest.py` adds the repo root; extend its header with an HF line and `STRICT_HF=1`) |
| Quick run command | `venv/bin/python -m pytest tests/test_hf_local_provider.py -x -q` (torch-free tests only; seconds) |
| Full suite command | `venv/bin/python -m pytest -q` in WSL (the HF-importing files cost ~65-75 s of import each session) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LORA-01 | `HFLocalProvider` with an injected fake model/tokenizer returns an `LLMResponse` with text, `prompt_tokens`, `completion_tokens`, `finish_reason` ("stop"/"length"), `resolved_model` incl. adapter hash; never raises from `generate` (model exception -> `error` set) | unit (no torch) | `pytest tests/test_hf_local_provider.py -k contract -x` | Wave 0 |
| LORA-01 | greedy kwargs: the fake model receives `do_sample=False`, `repetition_penalty=1.0`, `temperature/top_p/top_k=None`, `eos_token_id=[151645,151643]`, `max_new_tokens<=512` | unit | `-k greedy` | Wave 0 |
| LORA-01 | project system prompt always supplied: rendered prompt starts with `<|im_start|>system\n`+`SYSTEM_PROMPT`, never contains "You are Qwen" (real tokenizer) | unit (tokenizer) | `pytest tests/test_qwen_lora_data.py -k system_prompt` | Wave 0 |
| LORA-01 | missing adapter dir -> `RuntimeError` at construction; `run_benchmark` prints `[skip] <name>:` and writes no rows (copy `test_the_runner_skips_a_refused_checkpoint_and_writes_no_row_for_it`) | integration (tmp models.yaml) | `-k missing_adapter` | Wave 0 |
| LORA-01 | stale adapter (manifest render fingerprint or adapter hash mismatch, wrong base id) -> `RuntimeError` | unit | `-k stale` | Wave 0 |
| LORA-01 | registry: `provider: hf_local` builds lazily; `import src.llm.registry` does not import `transformers`/`peft`/`torch` (subprocess `sys.modules` test) | unit | `-k registry` | Wave 0 |
| LORA-01 | context overflow -> error response with `finish_reason="context_overflow"` | unit | `-k overflow` | Wave 0 |
| LORA-01 | real-stack integration on a tiny random-init Qwen2 + cached tokenizer (CPU): provider end-to-end incl. adapter save/load/merge; adapter changes logits vs base | integration (skips without cached tokenizer; `STRICT_HF=1` fails) | `pytest tests/test_hf_local_provider.py -k tiny` | Wave 0 |
| LORA-02 | base row on a smoke run: rows exist with `provider: hf_local`, `config_model: qwen2.5-0.5b-base`, non-null `resolved_model`, `finish_reasons`, `prompt_tokens`; no `llm_error`; no `[skip]` | smoke (manual, GPU, ~5 min) | `mkdir -p results/phase3-smoke-base && venv/bin/python run_benchmark.py --models qwen2.5-0.5b-base --runs 1 --dh 5 --ds 5 --benign 3 --resume results/phase3-smoke-base` then a plumbing-check script (Phase 2 `check_smoke.py` pattern) | after build |
| LORA-03 | decoded supervised tokens == gold action JSON + `<|im_end|>` for every assistant turn (300 dev + 300 train episodes minimum; full 42,000 optional), last supervised token of each span is 151645, no observation/system/user token supervised, header and trailing newline not supervised, pad-id inside content still supervised | unit (tokenizer) | `pytest tests/test_qwen_lora_data.py -k supervised` | Wave 0 |
| LORA-03 | every step prompt from `AgentLoop`-style `episode_to_messages(ep, k)` is an exact token prefix of the training sequence and the next token is supervised (parity; the 5,579-prompt check) | unit (tokenizer) | `-k prefix` | Wave 0 |
| LORA-03 | custom supervised-position loss == HF `labels=` loss (tolerance 1e-4, dropout off) on a tiny Qwen2 | unit (torch CPU) | `pytest tests/test_qwen_lora_train.py -k loss_equivalence` | Wave 0 |
| LORA-03 | trainer on a tiny model + 20 synthetic episodes (CPU, 3 steps): writes `adapter_model.safetensors`, `adapter_config.json` (r 16, alpha 32, 7 target modules), `manifest.json` whose key tuple equals `MANIFEST_KEYS`, `train_log.jsonl`; trainable-param count matches the formula `r*(in+out)` summed; non-finite loss exits 3 with no adapter | unit | `-k trainer` | Wave 0 |
| LORA-03 | resume: kill after step k, `--resume` continues and refuses on a changed data hash/config; `train.done`-style marker | unit | `-k resume` | Wave 0 |
| LORA-03 | leakage: trainer calls `assert_clean` on both files; a file with a held-out tool aborts before step 0 | unit | `-k leak` | Wave 0 |
| LORA-03 | VRAM budget: real run's manifest `peak_vram_mib.allocated` < 4,096 and `reserved` < 5,800 (95% of the 6,144 MiB card) | gate on the real manifest | script check | after training run |
| LORA-04 | tuned row smoke: `results/phase3-smoke-lora`, same command with `--models qwen2.5-0.5b-agentlora`; gate: no `[skip]`, no `llm_error`, `resolved_model` ends with `@<adapter_sha12>` equal to manifest `adapter_sha256[:12]` on every row, parse rate >= 90% of steps (smoke precedent 35/36), `finish_reasons` all in {stop, length} | smoke (manual, GPU, ~3 min) | as above + check script | after training run |
| LORA-04 | tuned adapter differs from base: first-action exact rate vs gold on 100 dev prompts base << tuned (smoke: 20/20 tuned) | gate in the end-of-training `gen_check` | trainer output | after training run |
| Docs | `docs/viva/03-*.md` exists, README row updated, numbers quoted match the manifest (Phase 2 `part-c-ok` style grep), secret/canary scan | script | grep checks | after run |

### Sampling Rate
- **Per task commit:** the torch-free provider contract tests (`-k "contract or greedy or stale or registry or overflow"`) - seconds.
- **Per wave merge:** all new test files (adds ~65-75 s import per HF-importing file) plus `tests/test_llm_layer.py`, `tests/test_scratch_inference.py` (regression: registry unchanged).
- **Phase gate:** full `pytest -q` green in WSL, the base and tuned smoke runs, the manifest VRAM/metric gates, then `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/test_hf_local_provider.py` - covers LORA-01 (fake model, registry, skip, stale, overflow, tiny integration)
- [ ] `tests/test_qwen_lora_data.py` - covers LORA-03 mask/prefix/system-prompt tests (shared tokenizer fixture, module-scoped)
- [ ] `tests/test_qwen_lora_train.py` - covers LORA-03 loss equivalence, trainer artifacts, resume, leakage, non-finite exit
- [ ] `tests/conftest.py` - HF header line and `STRICT_HF=1`
- [ ] `requirements-train.txt` - pinned HF stack (install already done on this machine)
- [ ] Training must expose a model-factory/tokenizer-factory seam so tests inject a tiny random-init Qwen2 (hidden 64, 2 layers, vocab 151936; 9.8 M params) instead of the 1 GB model

## Security Domain

> `security_enforcement` is enabled in `.planning/config.json` (ASVS level 1, block on high).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no network service added; model download needs no credentials (an unauthenticated HF warning appeared; do not add tokens) |
| V3 Session Management | no | - |
| V4 Access Control | yes | CLI paths confined with `resolve_under_root`; the adapter loader only reads under the repo; never loads pickles: `torch.load(..., weights_only=True)` for resume files, safetensors for adapters |
| V5 Input Validation | yes | validated episodes only (`read_episodes` + `assert_clean`); assert max length; reject special-token strings in training text; provider validates adapter config/manifest before loading |
| V6 Cryptography | yes (integrity only) | sha256 of adapter, dataset and base-model revision recorded and checked; no custom crypto |
| V10 Malicious code / supply chain | yes | pinned versions; no `trust_remote_code` (never pass it); HF packages SUS-by-recency documented; no postinstall scripts present |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle deserialisation of a tampered checkpoint | Tampering / Elevation | safetensors adapters, `weights_only=True` for torch files; refuse unexpected keys |
| `trust_remote_code=True` executing code from the hub | Elevation | never set; Qwen2 is natively supported by transformers |
| Stale or substituted adapter benchmarked as current | Tampering / Repudiation | adapter sha + render fingerprint + base id checked at construction; `resolved_model` stamped per row |
| Benchmark run reaching the network | Information disclosure | `local_files_only=True`, `HF_HUB_OFFLINE=1` |
| Path traversal via `--out-dir`/`--adapter` | Tampering | `resolve_under_root`, `refuse_foreign_manifest` |
| Prompt injection through special tokens in tool output | Tampering | not sanitised (fairness); test that data contains none; documented as a limit |
| Secret leakage in logs/manifests | Information disclosure | manifest uses a fixed field list (no env values); no `.env` reads; no canary values logged |
| GPU/host resource exhaustion on a shared machine | Denial of service | preflight, memory fraction check, never touch other processes |

## Viva Page Outline (`docs/viva/03-fine-tuning-lora-and-chat-templates.md`)

Same house style as page 02: glossary first, Parts A-D, "Limits of this phase", likely questions with model answers; quote numbers from the manifest, not from this file.

1. **Glossary:** pretrained vs fine-tuned, base vs instruct model, token, BPE, special token, chat template, system/user/assistant roles, `<|im_start|>`/`<|im_end|>`, loss mask / label -100, LoRA, rank, alpha, adapter, merge, bf16, gradient checkpointing, micro-batch/accumulation, greedy decoding, KV cache, logits, vocabulary.
2. **Part A - Pretraining vs fine-tuning:** Qwen2.5-0.5B-Instruct = 494 M parameters pretrained on web-scale text then instruction-tuned by its authors; ScratchGPT (4.8 M, Phase 2) learned everything from our 40k episodes; we adapt the pretrained model rather than train from zero because 494 M parameters cannot be trained from scratch on 6 GB in a week. Why compare base vs tuned: the base row is the control that isolates what *our* fine-tuning changed (format-valid rate 45% -> ~97% in the smoke, attack rate 0/12 -> 8/12 because the data teaches compliance).
3. **Part B - LoRA math:** `W' = W + (alpha/r) * B A` with `A in R^{r x in}`, `B in R^{out x r}`, B initialised to zero so training starts exactly at the pretrained model, scaling alpha/r = 2; only A, B train. Parameter counts for this model, per layer: q 28,672; k 16,384; v 16,384; o 28,672; gate 92,160; up 92,160; down 92,160 = 366,592; x 24 layers = **8,798,208 trainable of 502,830,976 (1.750%)**; adapter file 35.2 MB vs 988 MB base. Memory argument: full fine-tuning needs weights + gradients + Adam states ~16 bytes/parameter ~ 7.9 GB > 6 GB; LoRA needs a frozen bf16 base (0.95 GiB) plus 8.8 M fp32 parameters x (weight 4 + grad 4 + Adam 8 bytes) ~ 0.13 GB, plus activations (checkpointed). Why merge: `W + BA` can be folded into W after training, so inference costs nothing extra (30.5 vs 13.3 tok/s measured).
4. **Part C - Chat templates and special tokens:** the model only sees token ids; the template wraps each message in `<|im_start|>role\n ... <|im_end|>\n`; the same template must be used in training and serving; why Qwen's default system prompt would change behaviour; how a tool result becomes a `user` turn here (`OBSERVATION_TEMPLATE`); the special-token injection caveat.
5. **Part D - Loss masking:** supervise only the assistant JSON plus `<|im_end|>` (14.75% of tokens); why not the prompt (it is data the model reads, not text it writes), why the stop token must be supervised, why we mask by character spans not token ids; the decoded-supervised-tokens test as the proof; action-level metric excluding respond text (CR-01 lesson).
6. **Part E - Numerics and memory:** bf16 vs fp16 vs fp32 (range vs precision; RTX 3060 supports bf16; adapters kept fp32); gradient checkpointing trades compute for activation memory (measured: no-GC mb=2 peaked at 4.8 GiB and ran 3,199 tok/s; with GC, mb=4 peaked at 2.4 GiB at 2,489 tok/s, i.e. about 22% less throughput per token but half the memory at twice the batch); the 152k-vocabulary logits problem and the supervised-position head (8.3 GiB -> 2.4 GiB); micro-batch x accumulation = effective batch 16.
7. **Part F - Decoding:** greedy = argmax each step; why explicit `do_sample=False`, `repetition_penalty=1.0` (generation_config defaults to sampling); stop on `<|im_end|>`; max_new_tokens; why determinism is limited (bf16 kernels) and why 3 greedy runs per case are near-duplicates.
8. **Part G - Real numbers from this run:** table filled from `models/qwen-agentlora/manifest.json`: steps, episodes seen, wall time, tok/s, peak VRAM, dev action loss/exact (by class), adapter hash, base revision, smoke parse rate and attack counts for base and tuned rows.
9. **Limits of this phase:** both local models are trained/instructed to be compliant; one seed; teacher-forced dev metric is not benchmark performance; pretrained Qwen may have seen InjecAgent (public) - only ScratchGPT has clean provenance; served in-process bf16 (not Ollama Q4), a confound for the Phase 4 table.
10. **Likely examiner questions (write full answers):** What is LoRA and why not full fine-tuning? What do rank and alpha do? Why is B zero-initialised? Why do you mask the prompt tokens? What happens if you forget the stop token in the loss? What is a chat template and why does it matter? Why bf16 and gradient checkpointing? Why greedy decoding and what does the repetition penalty do? Why compare against the un-tuned model? Why is your tuned model *more* vulnerable and is that a problem for your thesis? How do you know there is no train/benchmark leakage (Phase 2 gate + held-out tools)? How would you know the adapter being benchmarked is the one you trained (hash stamps)?

## Sources

### Primary (HIGH confidence)
- Runtime measurements this session (WSL2, RTX 3060): install, model load, template parity, tokenization of all 42,000 Phase 2 episodes, LoRA training micro-benchmarks and the 60-step run, adapter save/reload/merge, 12-episode benchmark samples for base and tuned, failure-mode probes.
- Model files read from the HF cache snapshot `7ae557604adf...`: `config.json`, `generation_config.json`, `tokenizer_config.json` (chat template).
- Repo files read this session: `.planning/{REQUIREMENTS,STATE,ROADMAP,config.json}`, `.claude/CLAUDE.md`, `.planning/research/{STACK,ARCHITECTURE,PITFALLS,SUMMARY}.md`, `.planning/phases/02-*/{02-CONTEXT,02-03-SUMMARY,02-04-SUMMARY}.md`, `src/llm/{base,registry,scratch}.py`, `src/agent_harness/{prompting,loop}.py`, `src/scratch_llm/{format,dataset,train,inference,provenance}.py`, `src/benchmark/{runner,injecagent}.py`, `run_benchmark.py`, `tests/{test_llm_layer,conftest}.py`, `data/local/manifest.json`, `models/scratch/manifest.json`, `docs/viva/README.md`.

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` / `PITFALLS.md` recommendations (verified or corrected above where measured).
- `gsd-tools package-legitimacy check` output (registry metadata only; no download counts).

### Tertiary (LOW confidence)
- LoRA theory statements for the viva (Hu et al., 2021) and the WSL2 memory-fallback explanation are from training knowledge and tagged [ASSUMED].

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - installed and executed on the target machine; `pip check` clean.
- Architecture: HIGH - every seam (registry, runner `[skip]`, `LLMResponse` fields, loop message shapes) read in source and exercised by real episodes.
- Pitfalls: HIGH for those measured (logits memory, generation_config defaults, merged speed, base parse rate, template injection); MEDIUM for the 600-step outcome and full-run latencies (extrapolated).

**Research date:** 2026-10-22
**Valid until:** ~2026-11-05 (transformers/peft release monthly and v5 APIs are still moving; versions are pinned in `requirements-train.txt`, so the plan stays valid while the pins hold)
