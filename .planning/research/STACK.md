# Stack Research

**Domain:** Small-scale local LLM training (from-scratch char-level GPT), LoRA fine-tuning of Qwen2.5-0.5B-Instruct, and local inference serving inside an existing OpenAI-compatible agent benchmark
**Researched:** 2026-10-20
**Confidence:** MEDIUM-HIGH (package versions and Python requirements verified against PyPI today; Ollama/Qwen2 import behaviour is documented-but-unverified, and the throughput numbers are estimates)

## Bottom Line (read this first)

1. **Serve the fine-tuned Qwen in-process with HF transformers + peft. Do not go through GGUF/Ollama for the main results.** Merge conversion adds three fragile seams (llama.cpp convert script, Modelfile chat template, quantization drift) and gives nothing the benchmark needs. The repo already has the in-process pattern (`LocalScratchProvider`); add a sibling `LocalHFProvider` registered as a new `provider:` type in `src/llm/registry.py`.
2. **Train with a short custom PyTorch loop around `peft`, not TRL.** Skip `trl` and `datasets` entirely. They add the heaviest dependency chain (pyarrow, pandas) and the fastest-moving API, and the project's data is plain JSONL. A ~80-line loop with explicit label masking is also easier to explain in a viva.
3. **Keep the installed torch 2.11+cu128.** Every HF package below accepts `torch>=2.6`. Do not upgrade torch (latest on PyPI is 2.14.1) mid-week.
4. **Python 3.14 is probably fine for this set but is the single biggest unknown.** Gate the whole milestone on a 15-minute smoke test (install, load Qwen, one LoRA step). Escape hatch: a Python 3.12 venv via `uv`.
5. **No bitsandbytes / QLoRA.** A 0.5B model in bf16 is about 1 GB; quantizing it is pure added fragility.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| PyTorch | **2.11.x+cu128 (already installed; do not upgrade)** | Training and inference for both models | Already working with CUDA on the 3060. transformers 5.20.2 needs `torch>=2.6`, peft needs `>=1.13`, accelerate `>=2.0`. PyPI shows torch 2.14.1 supports 3.14, so the installed 2.11 build running on 3.14 is expected. | HIGH |
| transformers | **5.20.2** (pin `>=5.20,<5.21`) | Load Qwen2.5-0.5B-Instruct, tokenizer, `apply_chat_template`, generation | Current major (v5). Requires Python >=3.10, lists 3.14 classifier, `huggingface-hub>=1.31,<3`, `tokenizers>=0.23.1,<0.24`, `safetensors>=0.8.0`. Qwen2 is a first-class architecture. | HIGH (version) / MEDIUM (v5 API details) |
| peft | **0.21.2** | LoRA adapters (`LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained`, `merge_and_unload`) | The standard LoRA library. Requires Python >=3.10, `torch>=1.13`, `transformers` unpinned, `accelerate>=0.21`. `target_modules="all-linear"` removes the need to hand-name Qwen's projection layers. | HIGH |
| accelerate | **1.15.0** | Hard dependency of peft; device placement | Pulled in by peft. Do not write accelerate launch configs; single GPU needs none. | HIGH |
| Plain PyTorch (no framework) | n/a | ScratchGPT training loop | `model.py` is already pure torch. AdamW (fused), cosine LR with warmup, bf16 autocast, gradient clipping. No Lightning, no HF Trainer. | HIGH |
| Qwen/Qwen2.5-0.5B-Instruct | HF hub, Apache-2.0 | Base model for LoRA | About 494M params, tied embeddings, 24 layers. bf16 weights about 1 GB. Fits the 6 GB budget with large headroom. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface_hub | pulled by transformers (`>=1.31,<3`) | Download the model once (`snapshot_download`) | Always; download once to the WSL ext4 home cache, then set `HF_HUB_OFFLINE=1` for the benchmark runs so a flaky network cannot break a run. |
| tokenizers | 0.23.x (pulled in) | Qwen BPE tokenizer | Automatic. abi3-style wheels, 3.14 classifier present. |
| safetensors | 0.8.0 (pulled in) | Adapter and checkpoint format | Automatic. Adapter file is about 35 MB at r=16 on all linear layers (estimate). |
| numpy | whatever pip resolves (needs a 3.14-capable release, >=2.3) | Dependency of everything | Automatic. Verify a cp314 wheel installs (see smoke test). |
| pyyaml / openai / flask / matplotlib | existing | Already in `requirements.txt` | Unchanged. |
| gguf | 0.19.0 | Only if the optional GGUF stretch is attempted | Do not install by default. |
| llama.cpp (git checkout, not pip) | pin a release tag | Only for the optional GGUF stretch: `convert_hf_to_gguf.py` | Optional Ollama export. Not `llama-cpp-python`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `requirements-train.txt` (new, separate file) | Keeps `transformers/peft/accelerate` out of the base install | The benchmark and dashboard must keep installing without the training stack. Lazy-import `transformers`/`peft` inside the provider constructor, matching the existing `LocalScratchProvider` pattern. |
| `uv` | Fallback Python 3.12 venv | `uv venv --python 3.12 --system-site-packages` is NOT recommended (mixes torch builds). If 3.14 fails, create a clean 3.12 venv and `pip install torch --index-url https://download.pytorch.org/whl/cu128` (match the existing cu128 build). |
| `nvidia-smi` / `torch.cuda.max_memory_allocated()` | VRAM monitoring | Log peak VRAM at the end of each run. WSL2 shares the GPU with the Windows desktop, so budget about 5 GB, not 6. |
| pytest (existing) | Provider contract tests | Add a test that `LocalHFProvider.generate()` returns an `LLMResponse` and never raises, using a tiny random-init Qwen2 config so CI does not need the 1 GB download. |

## Concrete Training Recipes

### Qwen2.5-0.5B LoRA (the 6 GB budget)

| Setting | Value | Why |
|---------|-------|-----|
| Base weights dtype | `bfloat16` | RTX 3060 is Ampere and supports bf16. Do not use fp16: Qwen models can overflow in fp16 and need a GradScaler. |
| Attention implementation | `sdpa` (default) | No flash-attn build needed; PyTorch's fused SDPA is used. Avoid installing `flash-attn` (source compile, CUDA toolchain, 3.14 risk). |
| LoRA | `r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"`, `task_type="CAUSAL_LM"` | Docs recommend `all-linear` as the robust default. Roughly 9M trainable params (about 2% of the model). Do not target embeddings or `lm_head` (tied weights). |
| Adapter dtype | fp32 (peft default autocasts adapters to fp32) | Stable optimizer states, tiny memory cost. |
| Optimizer | `torch.optim.AdamW`, lr 2e-4, cosine, 3% warmup, weight decay 0 | Standard LoRA setting. Optimizer state covers only about 9M params. |
| Sequence length | cap 1024 tokens; micro-batch 4; grad accumulation 4 (effective 16) | With gradient checkpointing this should sit around 3 GB peak (estimate; measure it). |
| Gradient checkpointing | On (`model.gradient_checkpointing_enable()` plus `model.enable_input_require_grads()`) | Cheap insurance for the VRAM budget. Required call order with peft. |
| Loss masking | Labels = -100 everywhere except the assistant's action JSON turn(s) | Same idea as `ScratchGPT.forward(mask=...)`: learn only the agent's own action lines. Build via `apply_chat_template` on the prefix versus the full conversation and mask the prefix length. |
| Epochs | 2 to 3 over 5k to 20k transcripts | Rough estimate: 6M tokens per epoch at an assumed 8 to 12k tok/s is on the order of 10 to 15 minutes per epoch. LOW confidence on throughput, so time 50 steps first. |
| DataLoader | `num_workers=0` | Python 3.14 changed the default multiprocessing start method on Linux from fork to forkserver. Workers would need picklable datasets and a `__main__` guard. Data is pre-tokenized in memory, so workers buy nothing. |
| Eval hygiene | Hold out all InjecAgent tools from training data; log eval loss on a held-out split of synthetic tools | Required by PROJECT.md. |
| Output | Save adapter only (`model.save_pretrained("models/qwen-lora/")`) plus tokenizer copy | Small, git-ignorable, reload with `PeftModel.from_pretrained`. |

### ScratchGPT (5M params, char-level)

| Setting | Value | Why |
|---------|-------|-----|
| Framework | Plain torch | Model exists. Vocab 96 means no tokenizer library; the char map lives in `format.py`. |
| Precision | `torch.autocast("cuda", dtype=torch.bfloat16)` | 3060 supports it; 5M params would train fine even in fp32. |
| Optimizer | AdamW (betas 0.9/0.95, lr 1e-3 to 3e-3 for this size, wd 0.1, cosine, warmup), grad clip 1.0 | Standard for tiny GPTs. |
| `torch.compile` | **Do not use** | Unneeded at 5M params and the least certain feature on a brand-new Python version. |
| Inference | Existing `generate()` is fine | It recomputes the full prefix per token (no KV cache). At context 1536 and short JSON action outputs, this is acceptable on GPU. Only add a KV cache if a benchmark run measures it as too slow. |
| Checkpoint | `torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "chars": ...})` at `models/scratch/scratch_gpt.pt` | Matches the path in `config/models.yaml`. Use `torch.load(..., weights_only=True)` and keep the checkpoint to plain tensors and dicts so that works. |

## Serving to the Benchmark

### Recommendation: in-process HF provider (primary)

Add `provider: hf_lora` to `config/models.yaml` and `src/llm/registry.py`, implemented as `src/llm/hf_local.py`:

```yaml
qwen-lora:
  provider: hf_lora
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  adapter: models/qwen-lora
  family: open / tiny / locally fine-tuned
  benchmark: true
```

Implementation notes:
- Constructor: `AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")`, then `PeftModel.from_pretrained(...)`, then `merge_and_unload()` (assign the return value; it is not in-place) and `.eval()`. Merging removes adapter latency and the adapter is no longer needed at inference.
- `generate(messages, temperature, max_tokens)`: `tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`, `torch.inference_mode()`, greedy when `temperature == 0` (`do_sample=False`), `max_new_tokens=max_tokens`. Fill `prompt_tokens` and `completion_tokens` in `LLMResponse`. Wrap everything in try/except and return `LLMResponse(error=...)`, never raise (contract in `base.py`).
- Use the identical `apply_chat_template` call in training and serving. This is the only way to guarantee prompt-format parity, and it is the main thing that goes wrong with a GGUF/Ollama round trip.
- Fairness: messages come from `src/agent_harness/prompting.py` unchanged, so Qwen sees the same system prompt as every other model. The base model also should be benchmarked un-tuned (`adapter: null`) so the report can show the fine-tuning delta.
- VRAM hygiene: run the benchmark for one local GPU model at a time, and run `ollama stop <model>` (or set `OLLAMA_KEEP_ALIVE=0`) before training or HF serving so Ollama is not holding VRAM.

### Why not merge, convert to GGUF, and serve via Ollama

| Concern | Detail | Severity |
|---------|--------|----------|
| Ollama native safetensors import | Official docs list Llama, Mistral, Gemma, Phi3 for `FROM <safetensors dir>`, and safetensors `ADAPTER` support for Llama, Mistral, Gemma only. Qwen2 is not listed. One third-party mirror lists Qwen2, which conflicts with the official docs. Treat Qwen2 native import as unverified. | High |
| GGUF conversion | Works (llama.cpp supports Qwen2) but needs a llama.cpp checkout whose `convert_hf_to_gguf.py` matches the installed `gguf`/`transformers` versions. A new moving part, with 3.14 untested. | Medium |
| Chat template | `ollama create` from a bare GGUF needs an explicit `TEMPLATE` and `PARAMS` (stop tokens `<|im_end|>`). A mismatch silently degrades the model and invalidates the train/serve parity. | High |
| Quantization | Ollama does not quantize GGUF on import; you would run `llama-quantize` yourself. Quantization drift confounds the comparison. Use f16 or Q8_0 if you do export. | Medium |
| Benefit | The only gain is a uniform `openai_compat` code path and a nicer viva demo. The benchmark already supports non-OpenAI providers. | Low |

**Treat GGUF/Ollama as an optional stretch after the in-process results are in**, using: `save merged -> convert_hf_to_gguf.py --outtype f16 -> Modelfile (FROM x.gguf, explicit Qwen ChatML TEMPLATE, PARAMETER stop "<|im_end|>") -> ollama create`. Then check that Ollama output matches the HF output on 20 fixed prompts before trusting it.

### Alternative: `transformers serve` (v5)

Transformers v5 ships an OpenAI-compatible `transformers serve` command, which would let the model sit behind the existing `openai_compat` provider with no new provider class. I did not verify its behaviour with LoRA/merged local checkpoints or its stability, so treat it as LOW confidence. Use it only if the in-process provider causes trouble (for example, to isolate GPU memory in a separate process).

## Installation

```bash
# WSL Ubuntu, inside the existing venv (torch 2.11+cu128 already present — do NOT reinstall torch)
pip install "transformers>=5.20,<5.21" "peft==0.21.2" "accelerate>=1.15,<2"

# Pull the model once (ext4 home, not /mnt/c), then work offline
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-0.5B-Instruct')"
export HF_HUB_OFFLINE=1

# Day-0 smoke test (gate for the whole milestone)
python - <<'EOF'
import torch, transformers, peft
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).cuda()
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
x = tok.apply_chat_template([{"role":"user","content":"hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = m(**x, labels=x["input_ids"]); out.loss.backward()
print("loss", out.loss.item(), "peak GB", torch.cuda.max_memory_allocated()/1e9)
EOF
```

Notes: `requirements-train.txt` should list exactly the three pip lines above. Do NOT add `trl`, `datasets`, `bitsandbytes`, `unsloth`, or `llama-cpp-python`.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Custom loop + peft | TRL `SFTTrainer` (1.16.0, Python >=3.11, lists 3.14) | If the team wants built-in packing, chat-template handling, or `assistant_only_loss` and accepts the `datasets`/`pyarrow`/`pandas` chain. API churn between TRL releases is the cost. Reasonable once training works. |
| Custom loop + peft | HF `Trainer` | If you want checkpoint/resume and logging for free. Fine, but v5 renamed several arguments (for example `tokenizer=` to `processing_class=`), so budget time for migration reading. |
| In-process HF serving | Ollama via GGUF | If a live demo through the Ollama UI matters, or the model must run alongside others without sharing a Python process. |
| LoRA in bf16 | QLoRA (bitsandbytes 0.50.2, lists 3.14) | Only for models that do not fit (7B+). A 0.5B model has no need. |
| Plain PyTorch for ScratchGPT | Lightning / HF Trainer | Never for this project; the loop is about 60 lines and is itself viva material. |
| Keep torch 2.11 | Upgrade to 2.14.1 | Only if the smoke test finds a real bug. Upgrading risks breaking the already-working CUDA setup. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `unsloth` (2026.10.7, requires Python <3.15) | Installs/pins its own torch/triton/xformers stack, aggressive monkey patching, and tends to fight an existing torch install. Wrong trade-off with one week and a working CUDA env. | peft + custom loop |
| `bitsandbytes` / QLoRA | Unneeded at 0.5B; CUDA-library lookup issues in WSL are a classic time sink. | bf16 base + fp32 LoRA |
| `llama-cpp-python` (0.3.36) | Source build with CMake/CUDA flags; fragile on a brand-new Python. If GGUF is needed, use a llama.cpp binary/checkout directly. | llama.cpp release tools or skip GGUF |
| `flash-attn` | Source compile, long builds, mismatched CUDA toolchain risk. SDPA is enough at these sizes. | `attn_implementation="sdpa"` |
| fp16 training for Qwen | Overflow/NaN risk and needs loss scaling. | bf16 |
| `torch.compile` on either model | No payoff at this scale, and new-Python compile support is the least certain area. | Eager mode |
| `DataLoader(num_workers>0)` on Python 3.14 | Default start method is now forkserver on Linux; breaks closures and unguarded scripts. | `num_workers=0` with in-memory tensors |
| Storing the HF cache or checkpoints under `/mnt/c` | Very slow I/O across the WSL boundary. | WSL ext4 home directory |
| Copy-pasting v4-era tutorials verbatim | v5 renamed `torch_dtype` to `dtype` and `tokenizer=` to `processing_class=` in Trainer, and removed TF/Flax (secondary source; confirm against the official v5 docs). | Read the v5 docs for each call used |
| Training and serving prompts built by hand | The classic source of train/serve skew. | `tok.apply_chat_template` in both |

## Stack Patterns by Variant

**If the day-0 smoke test fails to install on Python 3.14 (most likely culprit: a compiled dependency such as numpy/pyarrow/psutil without a cp314 wheel):**
- Create a clean Python 3.12 venv with `uv`, install the cu128 torch build, then the three pip lines.
- Because the sandbox and benchmark code is plain Python and runs unchanged on 3.12 (the repo documents Python 3.10+).

**If peak VRAM exceeds about 5 GB:**
- Lower micro-batch to 2, cap sequence length at 768, keep gradient checkpointing on, drop to `r=8`.
- Because activations, not weights, dominate; the 1 GB base and tiny LoRA state are fixed.

**If WSL RAM (7 GB) is tight during model load:**
- Load with `dtype=torch.bfloat16` directly to GPU (`device_map="cuda"`), close the Windows-side browser/IDE load, and ensure Ollama has unloaded its models.
- Because fp32 materialisation of 0.5B params in CPU RAM is about 2 GB, which is fine alone but not alongside other loaded models.

**If the in-process provider is too slow for 342 episodes:**
- Batch is not needed; 0.5B on a 3060 should do a short JSON action in a few seconds or less (estimate). If not, cap `max_new_tokens` to the action schema size and keep greedy decoding.
- Because the benchmark is sequential per episode; only switch to Ollama/GGUF if measured latency is the blocker.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| transformers 5.20.2 | torch >=2.6 (installed 2.11 OK) | Python >=3.10; 3.14 classifier present. Needs huggingface-hub >=1.31,<3, tokenizers >=0.23.1,<0.24, safetensors >=0.8.0. |
| peft 0.21.2 | transformers 5.x, accelerate >=0.21 | PyPI classifiers stop at 3.13, but `requires_python>=3.10` and it is pure Python, so 3.14 should work. Verified only by the smoke test, so MEDIUM. |
| accelerate 1.15.0 | torch >=2.0 | PyPI classifiers list only 3.10; pure Python, expected fine on 3.14. |
| trl 1.16.0 (not recommended) | datasets >=4.7, pyarrow >=21, pandas >=2.1.1, transformers >=4.56.2 | Requires Python >=3.11. pandas 3.0.6 has cp314 manylinux wheels (verified). pyarrow and numpy cp314 wheel availability was NOT verified (PyPI JSON fetch failed). Another reason to skip it. |
| bitsandbytes 0.50.2 (not recommended) | Python >=3.10, 3.14 classifier | Not needed. |
| unsloth 2026.10.7 (not recommended) | Python >=3.9,<3.15 | Works on 3.14 on paper; torch pinning is the risk. |
| llama-cpp-python 0.3.36 (not recommended) | Python >=3.8 | Source build. |
| ScratchGPT checkpoint | torch >=2.6 `weights_only=True` default | Store only tensors and dicts. |

## Python 3.14 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| A transitive compiled dependency lacks a cp314 wheel and tries to build from source | Low-Medium | Install stalls | Smoke test first. Use `pip install --only-binary=:all:` to fail fast. Fallback: Python 3.12 venv via uv. |
| peft/accelerate not officially classified for 3.14 | Low | Runtime error in a corner path | Pure Python; smoke test covers `get_peft_model`, backward, `save_pretrained`, `PeftModel.from_pretrained`, `merge_and_unload`. |
| Multiprocessing default changed to forkserver | Certain if workers are used | DataLoader crash | `num_workers=0`. |
| `torch.compile` / Dynamo on 3.14 | Unknown | Crash | Do not use it. |
| v5 API drift versus tutorials and older answers | Medium | Time lost on renamed arguments | Use `dtype=`, `processing_class=`, and check each call against the v5 docs. |

## Sources

- PyPI JSON API, queried 2026-10-20: transformers 5.20.2 (requires_python >=3.10, 3.14 classifier; deps huggingface-hub >=1.31,<3, tokenizers >=0.23.1,<0.24, safetensors >=0.8.0, torch >=2.6 via extra) — HIGH
- PyPI: peft 0.21.2 (>=3.10; classifiers to 3.13; torch >=1.13, accelerate >=0.21) — HIGH
- PyPI: accelerate 1.15.0, trl 1.16.0 (>=3.11, 3.14 classifier), datasets 5.1.0, bitsandbytes 0.50.2, torch 2.14.1 (3.14 classifier), safetensors 0.8.0, tokenizers 0.23.3, unsloth 2026.10.7 (<3.15), llama-cpp-python 0.3.36, gguf 0.19.0, pandas 3.0.6 (cp314 manylinux wheel present) — HIGH
- https://huggingface.co/docs/peft/main/en/developer_guides/lora — `LoraConfig` (`r`, `lora_alpha`, `target_modules="all-linear"`, `use_rslora`), `merge_and_unload()` returns a new model (not in-place) — HIGH
- https://docs.ollama.com/import — safetensors directory import via `FROM`, GGUF import, Ollama does not quantize GGUF on import — HIGH (docs fetched); supported-architecture list taken from the Ollama import and Modelfile docs via web search (https://docs.ollama.com/modelfile) — MEDIUM; Qwen2 safetensors import in current Ollama builds is unverified
- Web search for transformers v5 migration (`torch_dtype` to `dtype`, `tokenizer` to `processing_class`, TF/Flax removal) — secondary blog sources only — LOW-MEDIUM; confirm in official v5 migration docs
- Python 3.14 multiprocessing default start method (forkserver on Linux) — from the CPython 3.14 release notes, recalled not re-fetched — MEDIUM-HIGH
- Training throughput, VRAM peak, and adapter size figures are my estimates, not measured; measure in the first 50 steps — LOW
- Repo files read: `.planning/PROJECT.md`, `.planning/codebase/STACK.md`, `src/scratch_llm/model.py`, `src/llm/{base,scratch,registry}.py`, `config/models.yaml`, `requirements.txt`

### Not verified (carry as open questions)

- cp314 wheels for pyarrow and numpy (PyPI JSON fetch failed); only matters if `datasets`/`trl` are added.
- Whether `transformers serve` works cleanly with a locally merged Qwen checkpoint.
- Whether current Ollama builds import Qwen2 safetensors natively (docs do not list it).
- The research-plan/classify-confidence/research-store seam commands were not run; confidence tags above are assigned manually from source type.

---
*Stack research for: local LLM training, LoRA fine-tuning and local serving for an agent-security benchmark*
*Researched: 2026-10-20*
