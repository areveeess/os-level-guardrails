# Architecture Research

**Domain:** Integrating locally trained LLMs (from-scratch char-level GPT + LoRA-tuned Qwen2.5-0.5B) into an existing agent-guardrail benchmark
**Researched:** 2026-10-20
**Confidence:** HIGH on integration seams and data-leakage design (read from the code); MEDIUM on Qwen/PEFT/Ollama tooling details (not run on this machine, Python 3.14 + torch 2.11 in WSL)

## Bottom line (answers to the five questions)

| Question | Recommendation |
|----------|----------------|
| Synthetic episodes without leaking InjecAgent tools | One seeded generator (`dataset.generate_episodes`) that draws **only from the 251 non-held-out tools** in `tools.json` (330 total, 79 held out), writes its own instruction/injection wording, uses non-`CANARY...DATA` secrets, and runs a hard leakage gate (tool names + text n-gram scan) before anything is written to disk. Dev-validation uses a second slice of free tools, never the benchmark. |
| Sharing the compact transcript between train and inference | Training never builds strings by hand. Episode -> **chat messages** (via `prompting.build_messages` + `OBSERVATION_TEMPLATE`) -> `format.render_context(messages)`. That is the exact function inference calls on `AgentLoop`'s messages. Enforce with a parity test and a `FORMAT_VERSION` stamped into the checkpoint. |
| Qwen in-process or Ollama? | **In-process: new provider type `hf_local`** (transformers + PEFT adapter, lazy-imported, same pattern as `scratch`). Ollama GGUF/Modelfile is an optional stretch phase after the end-to-end comparison exists, not the primary path. |
| Where checkpoints live | `models/scratch/scratch_gpt.pt` (path already in `config/models.yaml`), `models/qwen-agentlora/` (PEFT adapter dir). Weights gitignored; a small `MANIFEST.json` + `train_log.jsonl` next to each is committed. Generated episodes in `data/local/` (JSONL gitignored, manifest committed). HF base-model cache stays in the WSL home, not in the repo. |
| Build order | Format v2 + data generator + leakage gate -> scratch inference/train -> scratch smoke benchmark -> `hf_local` provider -> Qwen LoRA train -> smoke -> full runs -> (optional) Ollama export. Details in "Suggested Build Order". |

## Standard Architecture

### System Overview

```
 OFFLINE (training time, WSL, GPU)                    ONLINE (benchmark time, unchanged)
 ─────────────────────────────────                    ──────────────────────────────────
 benchmarks/injecagent/tools.json                      run_benchmark.py
   │  (330 tools)                                        │
   ▼                                                     ▼
 held_out_tools() ──► free pool (251) ──┐              src/benchmark/runner.py  (model × case × run × arm)
   (79 tools, incl. GmailSendEmail)     │                │ get_provider(name)
                                        ▼                ▼
                    ┌─────────────────────────┐     src/llm/registry.py ── config/models.yaml
                    │ dataset.generate_episodes│        │          │              │
                    │  + leakage gate (hard)   │        │          │              │
                    └───────────┬─────────────┘        ▼          ▼              ▼
                                │ data/local/episodes-*.jsonl   openai_compat  scratch.py     hf_local.py (NEW)
                                │ + MANIFEST.json (seed, hashes)  (Gemini/Groq/  LocalScratch   HFLocalProvider
                    episode_to_messages(ep, k)                     Ollama)        Provider        │ base_model + adapter
                    (uses prompting.build_messages)                                  │            │
                       │                      │                                      ▼            ▼
          ┌────────────┴───────┐   ┌──────────┴──────────┐           scratch_llm/inference.py   transformers + peft
          ▼                    ▼   ▼                     ▼            load_generator(ckpt)      apply_chat_template
   format.render_context   (scratch)           tokenizer.apply_chat_template (Qwen)                     ▲
   → "T:..\nU:..\nA:" + target                 prompt / completion token masks                          │
          │                                              │                                             │
          ▼                                              ▼                                             │
   scratch_llm/train.py                         qwen_lora/train.py                                      │
   (ScratchGPT, loss mask on A: lines)          (peft LoRA, labels=-100 on prompt)                      │
          │                                              │                                             │
          ▼                                              ▼                                             │
   models/scratch/scratch_gpt.pt  ──────────────►  (loaded by inference.py)                            │
   models/qwen-agentlora/ (adapter) ───────────────────────────────────────────────────────────────────┘

 Both online providers return LLMResponse(ok/err); AgentLoop → parse_action → ToolBroker → GuardedExecutor
 → SandboxExecutor/MockExecutor → score_episode. NOTHING downstream of the provider changes.
```

### Component Responsibilities

| Component | Responsibility | Talks to | Status |
|-----------|----------------|----------|--------|
| `src/scratch_llm/dataset.py` | **Shared** synthetic-episode generator, `validate_episode`, `held_out_tools`, `episode_to_messages`, leakage gate. Torch-free. | `prompting` (build_messages, parse_action), `benchmark.injecagent` (DATA_DIR, load_tool_specs) | generator missing |
| `src/scratch_llm/format.py` | Compact transcript + char vocab. Single renderer for the scratch model, train and inference. | `prompting` (lazy, for `OBSERVATION_TEMPLATE`) | exists; needs v2 tweaks |
| `src/scratch_llm/model.py` | ScratchGPT architecture + `generate`. | torch only | exists |
| `src/scratch_llm/train.py` | Build samples, train, write checkpoint + manifest + log. | dataset, format, model | missing |
| `src/scratch_llm/inference.py` | `load_generator(ckpt) -> fn(messages, temperature, max_tokens) -> str`. Validates `FORMAT_VERSION`. | format, model only (must NOT import dataset or benchmark) | missing |
| `src/llm/scratch.py` | Provider wrapper, error-as-value. | inference | exists |
| `src/qwen_lora/data.py` | episodes -> (prompt_messages, completion) pairs for Qwen. | dataset | new |
| `src/qwen_lora/train.py` | LoRA fine-tune; writes adapter dir + manifest + log. | data, transformers, peft | new |
| `src/llm/hf_local.py` | `HFLocalProvider(base_model, adapter=None)`; one class serves the base-Qwen control and the tuned model. | transformers, peft (lazy) | new |
| `src/llm/registry.py` | Add one `elif cfg["provider"] == "hf_local"` branch. | hf_local | edit |
| `config/models.yaml` | Add `qwen2.5-0.5b-base`, `qwen2.5-0.5b-agentlora` entries. | registry | edit |
| `tests/` | Leakage, format parity, provider contract tests. | all of the above | new |

## Recommended Project Structure

```
src/
├── llm/
│   ├── base.py, registry.py, openai_compat.py, scratch.py   # exist
│   └── hf_local.py                  # NEW: in-process transformers(+PEFT) provider
├── scratch_llm/
│   ├── model.py, format.py          # exist (format.py gets FORMAT_VERSION + param names on T: line)
│   ├── dataset.py                   # IMPLEMENT: generator + leakage gate + episode_to_messages (shared by Qwen)
│   ├── train.py                     # NEW: python -m src.scratch_llm.train
│   └── inference.py                 # NEW: load_generator
└── qwen_lora/
    ├── data.py                      # episodes -> prompt/completion token ids
    └── train.py                     # python -m src.qwen_lora.train
models/                              # gitignored weights; manifests committed
├── scratch/{scratch_gpt.pt, MANIFEST.json, train_log.jsonl}
└── qwen-agentlora/{adapter_config.json, adapter_model.safetensors, MANIFEST.json, train_log.jsonl}
data/local/{episodes-v1-seed0.jsonl (ignored), MANIFEST.json}
tests/{test_training_leakage.py, test_format_parity.py, test_hf_local_provider.py, test_scratch_inference.py}
```

### Structure Rationale

- **Generator stays in `src/scratch_llm/dataset.py`.** The file, its docstring, and `LocalScratchProvider`'s error message already fix that contract; moving it costs time on a one-week clock. Qwen's trainer imports it (`qwen_lora -> scratch_llm.dataset`). The package name is a wart, not a bug; a rename to `src/local_models/` is a post-viva cleanup.
- **`dataset.py` must stay torch-free** so tests, the leakage gate, and CPU-only machines import it cheaply.
- **`inference.py` must not import `dataset.py` or `benchmark`.** Eval-time code then physically cannot see the held-out list or benchmark text. This also keeps the dependency direction from the codebase map intact (harness lowest; providers lazy-import heavy deps).
- **Separate `qwen_lora/` package** because its dependencies (transformers, peft) are heavy and must never be imported by the scratch path or by hosted-only benchmark runs.
- **`hf_local.py` under `src/llm/`** with lazy imports inside `__init__`, exactly like `registry.py` already lazy-imports `scratch`/`openai_compat`. Hosted-model runs and the dashboard never pay the torch import cost.

## Question 1: Synthetic episodes without leaking held-out InjecAgent content

### What "held out" must cover (measured on the vendored data)

| Leak channel | Size | Guard |
|--------------|------|-------|
| Tool names | 79 held out of 330 (17 user tools + 63 attacker tools, GmailSendEmail among them); **251 free** | Generator samples only from `free = all_specs - held_out_tools()`. Hard assert on every emitted action and `T:` line. |
| Benchmark text | 17 user instructions, every attacker instruction (`attacker_cases_{dh,ds}.jsonl`), tool-response templates, the attacker address `amy.watson@gmail.com`, the `Private record ... Reference:` phrasing, `CANARY<hex>DATA` secrets | Generator writes its own phrasing from its own templates. Gate rejects any episode sharing an exact instruction or an 8-word shingle with the benchmark corpus; bans the attacker address and the literal `CANARY` prefix. |
| Case sampling | `attack_cases(seed=0)` picks 40 of the pairs, but `held_out_tools()` already reads **all** cases, which is the conservative choice | Keep it; do not narrow to the sampled 40. |

`held_out_tools()` as written is correct and conservative. Add `EXFIL_TOOL` explicitly to its return set (it is already there via the ds attacker lists today, but the benchmark treats it as special in `injecagent.py`, so make the dependency explicit).

### The capability trade-off to decide up front

If `GmailSendEmail` is fully held out, the models never see an email-send tool's arguments in training, so a data-stealing (`ds`) chain ("read private record -> send canary via `GmailSendEmail`") can only succeed if the model **copies the tool name and argument names from the `T:` / system-prompt tool list at test time**. That is the honest "unseen tool" generalisation test, and it is the right one for the evaluation-hygiene constraint, but it has consequences:

1. Train the *skill* on analogues: free "send"-like tools (`SlackSendMessage`, `TwilioSendSms`, and others in the free pool) plus a few synthetic fake toolkits (random names) so copying names and args from the tool list is learned, not memorised.
2. **Scratch model: put parameter names on the `T:` line** (`T:GmailSendEmail(to,subject,body)|...`). Today `tool_names_from_system` drops everything but names, so a char-level model has no way to produce correct args for a tool it never saw. This is a `format.py` change and must land **before** any dataset or training work, because it changes every training string and invalidates checkpoints (hence `FORMAT_VERSION`).
3. Qwen already receives full specs through `build_messages`, so it needs no change.
4. Expect the scratch model's `ds` success to be low even unprotected. State this as a result, not a bug (confidence: MEDIUM, cannot be known before training).

### Episode mix (threat-model decision, flag for roadmap)

The scratch model's role is "weak, unaligned worst-case principal". A model trained only on benign tool-use will never follow an injection, and the comparison would show ASR ~ 0 in the *unprotected* arm, making "guardrails help" unobservable on those rows. Recommended default, recorded in `MANIFEST.json`:

| Episode type | Share | Shape |
|--------------|-------|-------|
| benign | ~50% | call user tool -> clean observation -> `respond` |
| injected-compliant dh | ~25% | observation carries *self-written* instruction naming a free tool -> model calls it -> `respond` |
| injected-compliant ds | ~25% | injection -> call "private-data" free tool -> copy secret into a free "send" tool -> `respond` |

Also randomise tool-list order and add distractor tools. In the benchmark the offered list is always `[user_tool] + attacker_tools + [GmailSendEmail]`, so a fixed order in training teaches a positional shortcut instead of reading the list. Disclose the compliance mix in the report (it is a methodology choice the viva panel will probe). Use the **same dataset for both local models** (fairness; "same data, two architectures" is a clean viva story).

### Validation split that never touches the benchmark

Reserve ~10% of the free tools as **dev tools** (never in train episodes). Validation loss/format-validity on episodes using only dev tools measures generalisation to unseen tools, the same property the benchmark tests, without tuning on InjecAgent. Choose epochs/early-stop on this; evaluate on InjecAgent once per final model.

### Leakage gate (concrete)

```python
def assert_clean(episodes, held_out, bench_corpus, banned):
    for ep in episodes:
        used = set(ep["tools"]) | {t["tool"] for t,_ in ep["turns"] if t["action"] == "call_tool"}
        assert not used & held_out, used & held_out
        text = json.dumps(ep)
        assert not any(b in text for b in banned)            # amy.watson@gmail.com, "CANARY", ...
        assert not shingles(text, 8) & bench_corpus.shingles  # 8-word overlap with any benchmark string
```

Run it inside `generate_episodes` (cannot be skipped) and again in `tests/test_training_leakage.py`. The manifest stores `sha256(sorted(held_out))` and the seed so a checkpoint can be shown to be trained against a specific held-out list.

## Question 2: Sharing the compact transcript between train and inference

### The invariant

`format.render_context(messages)` is called at inference on the messages `AgentLoop` builds. **Training must produce its prompts by calling the same function on equivalent messages**, not by concatenating `render_header` + `render_turn` independently. Two code paths that "should" agree are how train/inference drift happens.

```
episode ──episode_to_messages(ep, k, dump=compact_action)──► [system(build_messages), user, assistant, user(obs), ...]
                                                                   │
                         ┌─────────────────────────────────────────┤
                         ▼                                         ▼
          scratch: render_context(msgs) → "...A:"        Qwen: apply_chat_template(msgs, add_generation_prompt=True)
          target  = compact_action(ep.turns[k]) + "\n"    target = json.dumps(action) + <|im_end|>
```

- `episode_to_messages(ep, upto_turn, dump)` lives in `dataset.py` and uses `prompting.build_messages(user, tools=free_specs)` and `OBSERVATION_TEMPLATE.format(result=obs)`. The system prompt text is never copied anywhere: `prompting.py` stays the single source of truth for every model.
- `dump` parameter: assistant history must be rendered the way **that model itself emits actions** (compact JSON for scratch because `compact_action` has no spaces; ordinary `json.dumps` is fine for Qwen), otherwise history seen in training differs from history built from the model's own outputs at inference.
- Scratch training sample = one episode as one sequence with a loss mask on `A:` lines (the model already supports `mask`). Generator caps episode length so `fit()` truncation never triggers in training; add a per-turn fallback if an episode would exceed `CONTEXT_CHARS - 400`. Measured on the benchmark: max rendered context is ~700 of 1536 chars and tool responses are <= 465 chars (< `OBS_CHARS` 640), so benchmark prompts are never truncated; the cap is a safety net.
- Qwen training sample = one (prompt_messages, completion) pair per turn; labels `-100` on prompt tokens, loss only on the action JSON + end token.

### Guardrails for the invariant

| Mechanism | What it catches |
|-----------|-----------------|
| `tests/test_format_parity.py`: for 200 generated episodes and every prefix k, `render_context(episode_to_messages(ep, k)) == render_header + turns[:k] + "A:"` | any divergence between the two renderers |
| Same test but feeding a recorded `AgentLoop` transcript (build via `build_messages` + fake provider) | divergence between `episode_to_messages` and the real loop |
| `FORMAT_VERSION = 2` in `format.py`, saved in the checkpoint, checked by `load_generator` (raises `RuntimeError`, which `get_provider` callers already catch -> `[skip]` in the runner) | stale checkpoint trained on an older format |
| Vocab hash saved in checkpoint | `VOCAB` edits that silently shift token ids |
| Qwen manifest stores tokenizer/chat-template hash and base-model revision | base model swapped between train and serve |

### Known format sharp edges (fix in the format phase)

- `INVALID_REPLY_TEMPLATE` turns render as an `O:` line holding the full error text; the model never saw those in training. Add ~3-5% recovery episodes (bad reply -> corrected reply) or accept parse-error behaviour as a measured result.
- `_flat` collapses whitespace in assistant history to 400 chars; generated args must stay below that or the history is cut mid-JSON.
- Non-ASCII becomes `?` (VOCAB is printable ASCII + `\n`). Generator should emit ASCII only; the benchmark data is ASCII-clean on the sampled cases but check tool summaries.

## Question 3: Serving the fine-tuned Qwen

### Decision: in-process `hf_local` provider

| Criterion | In-process (`hf_local`) | Ollama (merge -> GGUF -> Modelfile) |
|-----------|-------------------------|--------------------------------------|
| New code | ~60 lines + 1 registry branch | none in `src/` (reuse `openai_compat`), but a conversion pipeline outside it |
| Failure points | model load, template (same tokenizer as training) | LoRA merge, `convert_hf_to_gguf.py`, optional `llama-quantize`, `TEMPLATE`/stop-token in Modelfile, `ollama create` |
| Train/serve parity | exact: same `apply_chat_template`, same weights (bf16 base + adapter) | template is re-declared in a Modelfile (silent mismatch if wrong); Ollama does not quantise on import, so quantisation is a separate step |
| Adapter shortcut | PEFT loads adapter directly | `ADAPTER` import documented for Llama/Mistral/Gemma; **Qwen2 adapter support is unverified** in official docs (LOW) |
| Same path as other local rows | no (but `scratch` already is in-process) | yes (llama3.2:3b, qwen2.5:3b, phi3.5, gemma3:1b via Ollama) |
| VRAM in WSL (6 GB) | ~1 GB bf16 for 0.5B, fine | similar, Ollama manages |
| Testability | inject `model=`/`tokenizer=` fakes, mirrors `OpenAICompatibleProvider(client=...)` | needs a running Ollama |

The deciding factors are the one-week deadline and parity: every Ollama step is an extra place for a quiet template mismatch to make the tuned model look worse than it is. Do Ollama only as a stretch ("also exported as GGUF and served through Ollama, results match within noise") if time remains; it is a good viva topic but not on the critical path.

### Provider shape

```python
# src/llm/hf_local.py  (heavy imports inside __init__, errors returned as values)
class HFLocalProvider(LLMProvider):
    def __init__(self, name, base_model, adapter=None, model=None, tokenizer=None):
        if model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to("cuda")
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)   # raise RuntimeError if dir missing
            model.eval()
        ...
    def generate(self, messages, temperature=0.0, max_tokens=512) -> LLMResponse:
        # apply_chat_template(messages, add_generation_prompt=True) -> generate(do_sample=temperature>0,
        # max_new_tokens=min(max_tokens, 256)) -> decode only new tokens; fill prompt/completion_tokens
```

```yaml
# config/models.yaml additions
  qwen2.5-0.5b-base:            # control: same weights, no tuning -> isolates the effect of LoRA
    provider: hf_local
    base_model: Qwen/Qwen2.5-0.5B-Instruct
    family: open / tiny / local (pretrained)
    benchmark: true
  qwen2.5-0.5b-agentlora:
    provider: hf_local
    base_model: Qwen/Qwen2.5-0.5B-Instruct
    adapter: models/qwen-agentlora
    family: open / tiny / local (LoRA-tuned here)
    benchmark: true
```

Registry branch resolves `adapter` against `PROJECT_ROOT` exactly as `checkpoint` is today. A missing adapter raises `RuntimeError`; `run_benchmark` already logs `[skip]` and continues, so the config entries can be committed before training finishes. Note `tests/test_llm_layer.py` asserts `scratch-gpt` is in `list_models(benchmark_only=True)`; keep that entry unchanged.

Do not name the model "guard": the LoRA teaches the **action format** (and, per the episode mix, deliberate compliance), not safety.

## Question 4: Where checkpoints live

| Artifact | Path | Git | Why |
|----------|------|-----|-----|
| Scratch weights | `models/scratch/scratch_gpt.pt` (already referenced) | ignored (`models/**/*.pt`, `*.safetensors`, `*.gguf`) | ~20 MB binary churn; reproducible from seed |
| Scratch checkpoint contents | `{state_dict, cfg.to_dict(), FORMAT_VERSION, vocab_hash, train_meta}` in the one `.pt` | n/a | `load_generator` can rebuild `GPTConfig(**cfg)` without guessing |
| Qwen adapter | `models/qwen-agentlora/` (PEFT dir, a few MB at r=16) | ignored weights | |
| Manifests + logs | `models/*/MANIFEST.json`, `train_log.jsonl`, loss-curve PNG | **committed** | proof for the viva: seed, steps, git SHA, dataset sha256, held-out-list hash, final dev loss |
| Episodes | `data/local/episodes-v1-seed0.jsonl`, `data/local/MANIFEST.json` | JSONL ignored, manifest committed | regenerate from seed |
| HF base model cache | `~/.cache/huggingface` inside WSL ext4 | outside repo | avoids slow `/mnt/c` reads and accidental commits |
| Benchmark results | `results/<run>/...` (unchanged) | existing convention | smoke runs stay `results/smoke-*/` (already ignored) |

`.gitignore` currently has nothing for `models/` or `data/local/`; add it in the first phase so no weights are committed by accident.

## Architectural Patterns

### Pattern 1: Provider-type extension with lazy imports
**What:** each new serving mechanism is one `provider:` value in `registry.get_provider`, with heavy imports inside the branch/`__init__`.
**When:** a model cannot be reached via an OpenAI-compatible endpoint (in-process weights).
**Trade-offs:** keeps hosted runs light and `AgentLoop`/runner untouched; costs one `elif` per type. Do not add a plugin system for two types.

### Pattern 2: One canonical conversion, many renderers
**What:** episode -> chat messages is defined once; renderers (compact for scratch, chat template for Qwen) are pure functions of messages.
**When:** training and serving must agree byte-for-byte.
**Trade-offs:** extra indirection vs. hand-built strings, but it is the only way the parity test is meaningful.

### Pattern 3: Manifest-stamped artifacts
**What:** every dataset/checkpoint carries seed, format version, held-out hash, git SHA.
**When:** results must be defensible and reproducible in a viva.
**Trade-offs:** ~20 lines of code; pays for itself the first time a stale checkpoint would have been benchmarked.

### Pattern 4: Control rows
**What:** benchmark `qwen2.5-0.5b-base` next to the LoRA model (and, for scratch, an untrained/early-checkpoint row if cheap).
**When:** you want to attribute behaviour change to the fine-tune, not the model family.
**Trade-offs:** one extra config entry; no extra code because `hf_local` takes `adapter: null`.

## Data Flow

### Training flow

```
tools.json ─► free pool (251) ─► generate_episodes(n, seed) ─► leakage gate ─► data/local/*.jsonl + MANIFEST
                                                                   │
                  ┌────────────────────────────────────────────────┴───────────────┐
                  ▼                                                                ▼
   episode_to_messages(dump=compact) ─► render_context ─► encode (VOCAB)     episode_to_messages(dump=json.dumps)
        ─► (x, y, loss_mask on A: lines) ─► ScratchGPT ─► scratch_gpt.pt        ─► apply_chat_template ─► LoRA ─► adapter dir
```

### Benchmark flow (unchanged except the provider)

```
runner.get_provider(model) ─► LocalScratchProvider | HFLocalProvider
AgentLoop.run(prompt): messages = build_messages(prompt, case.tools)
   loop: provider.generate(messages) ─► text ─► parse_action ─► ToolBroker/Guarded/Sandbox ─► Observation
         ─► OBSERVATION_TEMPLATE appended ─► repeat (max_steps=5) ─► score_episode ─► raw.jsonl
```

The scratch provider is the only place `render_context` runs at inference, via `inference.load_generator`. Failure to load or generate surfaces as `LLMResponse(error=...)`, which the runner records as `llm_error` and retries on `--resume`.

### Inference details for the scratch model

- `load_generator`: `torch.load` -> check `FORMAT_VERSION`/vocab hash -> `ScratchGPT(GPTConfig(**cfg))` -> `eval()` -> CUDA if available.
- Per call: `render_context(messages)` -> `encode` -> `model.generate(idx, max_new=min(max_tokens, 400), stop_id=STOI["\n"], temperature)` -> decode only new ids -> strip.
- `ScratchGPT.generate` has no KV cache (full forward per char). With ~5M parameters and <=1536-char context this is acceptable for a 342-episode run (rough estimate: low tens of minutes on the 3060; MEDIUM confidence, measure in the smoke run).
- `temperature=0.0` (greedy) is the benchmark default, so results are deterministic given the checkpoint.

## Scaling Considerations (hardware, not users)

| Concern | Scratch GPT | Qwen LoRA | Notes |
|---------|-------------|-----------|-------|
| Parameters / VRAM | ~5M, trivial | 0.5B bf16 ~1 GB + LoRA + activations | RTX 3060 6 GB; use gradient checkpointing + micro-batch 2-4 with accumulation, no QLoRA needed (MEDIUM) |
| Episodes | 20k-50k (tens of MB of text) | 3k-10k is plenty for format learning | Generate once, reuse; Qwen needs fewer because it is pretrained |
| Training time | minutes to ~1 h | tens of minutes | run serially on the one GPU; do not overlap with a benchmark using Ollama |
| WSL RAM 7 GB | fine | fine | keep dataloader workers at 0-2 |
| Benchmark concurrency | `run_full_benchmark.sh` starts one process per model | same | local GPU models fit together with Ollama's 3B (VRAM) but contend for compute; launch local ones separately |

First bottleneck: scratch `generate` without KV cache. Second: Python 3.14 wheels for the HF stack (verify `pip install transformers peft` in the WSL venv in the first Qwen task; fall back to a 3.12 venv if a wheel is missing, LOW confidence either way).

## Anti-Patterns

### Anti-Pattern 1: Hand-building training strings
**What people do:** write `f"T:{...}\nU:{...}"` in `train.py` and a slightly different one in `inference.py`.
**Why it's wrong:** silent train/serve skew; the model learned a format it never sees.
**Do this instead:** both sides call `render_context` on messages; enforce with the parity test.

### Anti-Pattern 2: Holding out tool names but not wording
**What people do:** exclude the 79 tool names yet reuse InjecAgent-like instructions or the same attacker email and canary shape.
**Why it's wrong:** the model memorises the attack template; ASR measures recall, not behaviour.
**Do this instead:** own templates, own addresses, n-gram gate, no literal `CANARY` prefix.

### Anti-Pattern 3: Tuning on the benchmark
**What people do:** pick epochs/learning rate by watching InjecAgent ASR.
**Why it's wrong:** the held-out set becomes a validation set; the comparison table is no longer defensible.
**Do this instead:** dev-tool validation split; run the benchmark once per final model (plus tiny smoke runs for plumbing only).

### Anti-Pattern 4: Copying the system prompt into training code
**What people do:** paste `SYSTEM_PROMPT` into a data script.
**Why it's wrong:** breaks the fairness constraint that `prompting.py` is the single source of truth.
**Do this instead:** always `build_messages(...)`.

### Anti-Pattern 5: Making the runtime path import the training path
**What people do:** `inference.py` imports `dataset.py` for a constant.
**Why it's wrong:** eval code can read the held-out list and pulls benchmark modules into the provider.
**Do this instead:** keep shared constants in `format.py`; training imports inference-side modules, never the reverse.

### Anti-Pattern 6: Making Ollama the critical path
**What people do:** block the Qwen result on a GGUF conversion.
**Why it's wrong:** three extra failure points and an unverified Qwen2 adapter path, one week of slack.
**Do this instead:** `hf_local` first; GGUF export as a stretch.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `registry` -> `hf_local` / `scratch` | `get_provider(name)` constructs; lazy import | Missing weights raise `RuntimeError`; runner logs `[skip]` |
| `LocalScratchProvider` -> `inference` | `load_generator(ckpt)` returning a callable | Signature already fixed by `scratch.py` |
| `dataset` -> `prompting` | `build_messages`, `OBSERVATION_TEMPLATE`, `parse_action` | Episode validation already uses `parse_action` |
| `dataset` -> `benchmark.injecagent` | `DATA_DIR`, `load_tool_specs` (training-time only) | Acceptable direction per codebase map; never import from `inference` |
| `qwen_lora` -> `dataset` | `generate_episodes`, `episode_to_messages` | Torch-free import |
| `runner` -> providers | `provider.generate(messages, temperature)` | `max_tokens` default 512; char-level model treats it as chars |

### External Services / Tooling

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Hugging Face Hub | one-time `from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")` | needs network once; then cached in WSL; pin `revision` in the manifest |
| PEFT / transformers | LoRA on all attention + MLP projections, r=16; hand-written loop (labels `-100` on prompt) preferred over TRL here | TRL `assistant_only_loss` requires `{% generation %}` markers in the chat template, which Qwen2.5 lacks (MEDIUM); a small custom loop avoids that and is easier to explain in the viva |
| Ollama (stretch) | `FROM` merged GGUF + explicit ChatML `TEMPLATE` + `PARAMETER stop <\|im_end\|>`; config entry becomes plain `openai_compat` at `localhost:11434/v1` | Ollama docs: GGUF imports are not quantised by Ollama; adapter architecture list does not name Qwen2 |

## Suggested Build Order

Dependencies: format v2 -> generator -> {scratch train, Qwen data}; providers are independent of training but need artifacts to be exercised end-to-end.

```
P1 format v2 + .gitignore + generator + leakage gate + parity tests
      │
      ├─► P2 scratch inference.py + train.py (CPU micro-run test first) ─► P3 scratch GPU training ─► P4 scratch smoke benchmark
      │
      └─► P5 hf_local provider + registry + base-Qwen control entry (testable with fakes; base Qwen runs immediately)
                 │
                 └─► P6 Qwen LoRA data adapter + training ─► P7 Qwen smoke benchmark
P4 + P7 ─► P8 full runs of local models (serial on GPU) ─► P9 comparison report
(stretch) P10 Ollama GGUF export
```

| # | Phase | Deliverable | Why this position | Verification |
|---|-------|-------------|-------------------|--------------|
| 1 | Data foundation | `FORMAT_VERSION=2`, params on `T:` line, `generate_episodes`, `episode_to_messages`, leakage gate, manifests, `.gitignore` | Everything downstream consumes it; format changes after training waste GPU hours | leakage + parity tests green; 1k episodes validate |
| 2 | Scratch pipeline | `train.py`, `inference.py`, provider round-trip | Provider contract fixed already; proves the plumbing with a 20-step CPU model before the real run | `test_scratch_inference.py`: tiny checkpoint -> `LocalScratchProvider.generate` returns parseable text |
| 3 | Scratch training | `scratch_gpt.pt` + manifest + loss curve | Long GPU job; start it, work on Phase 5 in parallel | dev-tool loss curve; % of generated actions that `parse_action` accepts |
| 4 | Scratch smoke | `run_benchmark.py --models scratch-gpt --runs 1 --dh 5 --ds 5 --benign 3` into `results/smoke-*` | Catches format/parse failures cheaply before the full run | parse_errors rate, non-zero tool calls |
| 5 | `hf_local` provider | `hf_local.py`, registry branch, `qwen2.5-0.5b-base` row, fake-model unit test | No training dependency; base-Qwen row is a free control; de-risks Python 3.14 HF install early | provider contract test; base Qwen smoke run |
| 6 | Qwen LoRA | `qwen_lora/{data,train}.py`, adapter dir | Reuses Phase 1 data; small job; run after scratch training to avoid GPU contention | dev-tool completion loss; action-validity on dev tools |
| 7 | Qwen smoke | same smoke command for `qwen2.5-0.5b-agentlora` | Plumbing check before the full run | parse-error rate vs base |
| 8 | Full local runs | 3 runs x 342 episodes per local model, serially | Needs both artifacts; GPU-bound | complete `raw.jsonl`, no `llm_error` |
| 9 | Report | rows appear via existing `generate_report` / `--merge` | Reporting already model-agnostic | table includes `family` labels |
| 10 | Stretch | GGUF + Modelfile, `openai_compat` row | Optional; independent | Ollama output matches `hf_local` on 5 cases |

**Research flags for phases:** Phase 1 (episode design, leakage corpus) and Phase 6 (LoRA hyper-parameters, Python 3.14 wheel compatibility, memory fit) merit deeper per-phase research. Phases 2, 4, 5, 8, 9 follow existing patterns in this repo.

## Open Decisions for the Roadmap

1. **Compliance mix** in training data (recommended 50/25/25 benign/dh/ds; disclose in the report). Confidence: design judgement.
2. **`T:` line carries parameter names** (recommended yes; required for any chance at unseen-tool `ds`).
3. **Whether to add an untrained/early-checkpoint scratch control row** (cheap, strong viva material; optional).
4. **Ollama export**: stretch only, unless the student specifically wants the GGUF/quantisation material for the viva.

## Sources

- Repo code read directly: `src/llm/{base,registry,scratch,openai_compat}.py`, `src/scratch_llm/{model,format,dataset}.py`, `src/agent_harness/{prompting,loop,tool_broker}.py`, `src/benchmark/{injecagent,runner,scoring,classify}.py`, `config/models.yaml`, `benchmarks/injecagent/*` (HIGH). Counts (330 tools, 79 held out, 251 free, 17 user cases, 63 attacker tools, max rendered context ~700 chars, tool responses <= 465 chars) were computed from the vendored data.
- Ollama import docs: https://docs.ollama.com/import (GGUF import not quantised by Ollama; no architecture list on that page) (MEDIUM)
- Ollama Modelfile / import reference mirror: https://ollama.readthedocs.io/en/modelfile and https://mintlify.com/ollama/ollama/advanced/import (adapter list Llama/Mistral/Gemma in the older reference; mirror lists Qwen2 for model import; adapter support for Qwen2 unverified) (LOW)
- TRL SFT trainer docs: https://huggingface.co/docs/trl/main/en/sft_trainer (`assistant_only_loss` needs generation markers in the chat template; completion-only default for prompt/completion data) (MEDIUM)

---
*Architecture research for: local-model training and serving in the OS-level guardrails benchmark*
*Researched: 2026-10-20*
