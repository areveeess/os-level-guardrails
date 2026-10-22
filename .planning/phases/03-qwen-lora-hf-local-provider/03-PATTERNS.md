# Phase 3: Qwen LoRA & hf_local Provider - Pattern Map

**Mapped:** 2026-10-22
**Files analyzed:** 14 new/modified
**Analogs found:** 13 / 14 (all analog paths verified git-tracked via `git ls-files`; no mirror paths)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/llm/hf_local.py` | provider | request-response (in-process inference) | `src/llm/scratch.py` + `src/scratch_llm/inference.py` | exact (role + flow) |
| `src/llm/registry.py` (modify) | config/registry | request-response | itself, `scratch` branch lines 47-49 | exact |
| `config/models.yaml` (modify) | config | n/a | `scratch-gpt` entry lines 64-70 | exact |
| `src/qwen_lora/__init__.py` | package marker | n/a | `src/scratch_llm/__init__.py` (16 lines, light) | exact |
| `src/qwen_lora/data.py` | utility | transform (episode -> chat -> ids/labels) | `src/scratch_llm/dataset.py::episode_to_messages` (152-160) + `train.py::encode_episode`/`collate` (118-163) | role-match |
| `src/qwen_lora/train.py` | training script | batch / file-I/O (resumable job) | `src/scratch_llm/train.py` | exact (structure) |
| `requirements-train.txt` | config | n/a | `requirements.txt` torch comment block | partial |
| `models/qwen-agentlora/{adapter_config.json,manifest.json,train_log.jsonl,loss_curve.png}` | artifact | file-I/O | `models/scratch/{manifest.json,train_log.jsonl,loss_curve.png}` (tracked) | exact |
| `tests/test_hf_local_provider.py` | test | request-response, fake model | `tests/test_scratch_inference.py` | exact |
| `tests/test_qwen_lora_data.py` | test | transform | `tests/test_scratch_train.py` (loss-mask section) | role-match |
| `tests/test_qwen_lora_train.py` | test | batch | `tests/test_scratch_train.py` | exact |
| `tests/conftest.py` (modify, `STRICT_HF`) | test config | n/a | itself, lines 18-33 (`STRICT_TORCH`) | exact |
| `docs/viva/03-fine-tuning-lora-and-chat-templates.md` + `docs/viva/README.md` row | docs | n/a | `docs/viva/02-llm-foundations-and-synthetic-data.md` | exact |
| Launcher / GPU preflight script (WSL) | script | batch | `scripts/run_full_benchmark.sh` (weak; Phase 2 launcher is not tracked, only its `models/scratch/gpu_preflight.log` output) | partial / see No Analog |

## Pattern Assignments

### `src/llm/hf_local.py` (provider, request-response)

**Analog:** `src/llm/scratch.py` (provider shell) and `src/scratch_llm/inference.py` (loader, stamps, overflow, sha256).

**Module docstring + imports** (`src/llm/scratch.py` 1-17): title docstring explaining role AND why construction raises; only `os, time` and `from src.llm.base import LLMProvider, LLMResponse` at module top. All torch/transformers/peft imports go INSIDE `__init__` (research: registry import stays 0.5 s, HF import costs ~75 s).
```python
import os
import time

from src.llm.base import LLMProvider, LLMResponse
```

**Construction fails with RuntimeError so runner prints `[skip]`** (`scratch.py` 22-32):
```python
def __init__(self, name: str, checkpoint: str):
    self.name = name
    self.checkpoint = checkpoint
    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"Scratch model checkpoint not found: {checkpoint}. "
            f"Train it first: python -m src.scratch_llm.train"
        )
    # Lazy import: torch is only needed when this provider is built, not when the registry loads.
    from src.scratch_llm.inference import load_generator
    self._generate = load_generator(checkpoint)
```
Copy: pre-check files yourself (adapter dir, `adapter_config.json`, `adapter_model.safetensors`), wrap `OSError/ValueError/KeyError` from `from_pretrained` into `RuntimeError` with actionable text (`raise ... from e`, as `inference.py` 145-146). Set `os.environ.setdefault("HF_HUB_OFFLINE", "1")` before the first HF import and pass `local_files_only=True` everywhere.

**Stamps verified at load, RuntimeError naming field/expected/found** (`inference.py` 53-59):
```python
def check_stamps(ckpt: dict) -> None:
    expected = current_stamps()
    for field in STAMP_KEYS:
        if ckpt[field] != expected[field]:
            raise RuntimeError(f"checkpoint {field} is {ckpt[field]!r} but the running code has {expected[field]!r}; "
                               f"retrain it with python -m src.scratch_llm.train")
```
Apply to: adapter manifest `adapter_sha256` vs recomputed hash, render fingerprint, `base_model_name_or_path` vs configured `base_model`.

**sha256 helper** (`inference.py` 69-74, copy verbatim, compute BEFORE merge):
```python
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

**Context overflow = error response, never a cropped answer** (`inference.py` 77-80 and 113-117; `scratch.py` 38-44):
```python
class ContextOverflowError(ValueError):
    finish_reason = "context_overflow"
...
except Exception as e:
    overflow = getattr(e, "finish_reason", None) == "context_overflow"
    return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name,
                       error=f"{type(e).__name__}: {e}",
                       prompt_tokens=... if overflow else None,
                       finish_reason="context_overflow" if overflow else None)
```
For hf_local the guard is `n_in + 1 > max_context_tokens` -> return the same shape with `prompt_tokens=n_in`.

**Provenance-stamped response + `_elapsed_ms`** (`scratch.py` 45-58):
```python
return LLMResponse(
    text=text.strip(), latency_ms=_elapsed_ms(start), model=self.name,
    prompt_tokens=..., completion_tokens=..., finish_reason=..., 
    resolved_model=f"{self.name}@{sha[:12]}" if sha else None,
)

def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
```
hf_local differences (from RESEARCH, not in analog): `resolved_model = "<base>@<rev12>"` (+ `"+<name>@<adapter_sha12>"` when adapter); `finish_reason = "stop"` if last new token in `{151645, 151643}` else `"length"`; `completion_tokens` includes the stop token; call `torch.cuda.synchronize()` before stopping the timer; explicit greedy kwargs `do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=[151645,151643], pad_token_id=151643`; render with `apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")` (never `tools=`); `dtype=` not `torch_dtype=`; `merge_and_unload()` return value assigned; `temperature` accepted and ignored (documented). Wrap the model call in `except Exception` (CUDA OOM -> error response); CLAUDE.md: providers never raise from `generate`.

---

### `src/llm/registry.py` (modify) and `config/models.yaml` (modify)

**Analog:** same files. Add a third lazy branch after `scratch` (`registry.py` 47-49):
```python
if cfg["provider"] == "scratch":
    from src.llm.scratch import LocalScratchProvider
    return LocalScratchProvider(name=name, checkpoint=os.path.join(PROJECT_ROOT, cfg["checkpoint"]))
```
New branch: `if cfg["provider"] == "hf_local": from src.llm.hf_local import HFLocalProvider; return HFLocalProvider(name=name, base_model=cfg["base_model"], adapter=os.path.join(PROJECT_ROOT, cfg["adapter"]) if cfg.get("adapter") else None, max_new_tokens=cfg.get(...), max_context_tokens=cfg.get(...), merge=cfg.get("merge", True))`. Registry uses single-quoted paths at the top but the existing branches use double quotes; follow the branch style. Keep `raise ValueError(f"Unknown provider type ...")` last.

**models.yaml** entry shape (`models.yaml` 64-70) including the `model:` stamp comment and header comment (line 6 `provider: openai_compat | scratch | scripted` must gain `hf_local`):
```yaml
  scratch-gpt:
    provider: scratch
    model: scratch-gpt                       # stamped as config_model on every row
    # Produced by `python -m src.scratch_llm.train`; the .pt file is git-ignored.
    checkpoint: models/scratch/scratch_gpt.pt
    family: from-scratch / unaligned baseline
    benchmark: true
```
Add `qwen2.5-0.5b-base` (adapter: null) and `qwen2.5-0.5b-agentlora` (adapter: models/qwen-agentlora) per RESEARCH "Config entries". Place before `control-always-comply`.

---

### `src/qwen_lora/data.py` (utility, transform)

**Analog:** `src/scratch_llm/dataset.py::episode_to_messages` (152-160) and `train.py::encode_episode` (118-123).

**Single source of truth for prompts** (`dataset.py` 152-160):
```python
def episode_to_messages(ep: dict, upto: int) -> list:
    messages = build_messages(ep["user"], ep["tools"])
    for action, observation in ep["turns"][:upto]:
        messages.append({"role": "assistant", "content": compact_action(action)})
        messages.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(result=observation)})
    return messages
```
Reuse `build_messages`/`OBSERVATION_TEMPLATE` from `src/agent_harness/prompting.py` and `compact_action` from `src/scratch_llm/format.py:67` (`json.dumps(action, separators=(",", ":"), ensure_ascii=True)`). New `episode_to_chat(ep)` = same loop over ALL turns, appending the assistant action and, when the observation is not None, the user observation (last respond turn has `null` observation). Do not import `inference.py`-style isolation concerns: data.py is training-time and MAY import `dataset.py` (`read_episodes`, `assert_clean`, `file_sha256`, `held_out_sha256`).

**Encode returns tensors already shifted + class tag; collate right-pads with mask 0** (`train.py` 118-123, 154-163):
```python
def encode_episode(ep: dict) -> tuple:
    text = training_text(ep)
    ids, mask = encode(text), supervised_mask(text)
    return (torch.tensor(ids[:-1], ...), torch.tensor(ids[1:], ...), torch.tensor(mask[1:], ...), ep["cls"])

def collate(encoded: list, idxs: list, device) -> tuple:
    longest = max(len(encoded[i][0]) for i in idxs)
    x = torch.zeros(len(idxs), longest, dtype=torch.long)
    ...
    for row, i in enumerate(idxs):
        n = len(encoded[i][0])
        x[row, :n], y[row, :n], m[row, :n] = encoded[i][0].long(), ...
```
Qwen difference (RESEARCH Pattern 2): mask comes from character offsets (`return_offsets_mapping=True`), spans found as `"<|im_start|>assistant\n" + content + "<|im_end|>"` searching forward, assert found; never `input_ids == pad_id`; pad by position/length; `MAX_LEN = 1024` raise (no truncation). Provide an action-level mask (excludes respond message text between `{"action":"respond","message":"` and closing `"}`), mirroring `RESPOND_HEAD` in `train.py:175`.

---

### `src/qwen_lora/train.py` (training script, batch/file-I/O)

**Analog:** `src/scratch_llm/train.py` (copy the skeleton wholesale; swap model/loss internals).

**Module docstring style** (1-38): title with `====`, explain the decisions (D-xx), list outputs and the stdout progress-line grammar a background poller reads (`device`, `step`, `eval step`, `DONE`, `FAILED non_finite_loss`).

**Constants / manifest contract** (63-79):
```python
ALLOWED_ROOT = PROJECT_ROOT      # --train, --dev and --out-dir must stay inside this directory (tests point it at a tmp dir)
EXIT_NON_FINITE = 3
IN_RUN_CHECKPOINT = "scratch_gpt.best.tmp"
RESUME_FILE = "scratch_gpt.resume.tmp"
MANIFEST_KEYS = ("schema", "created_at", "code_commit", ...)   # + assert tuple(manifest) == MANIFEST_KEYS (line 603)
CLASSES = ("benign", "dh", "ds")
EVAL_KEYS = ("dev_loss", "dev_loss_action", ..., "dev_action_exact", ...)
```
Qwen: `IN_RUN_CHECKPOINT = "adapter.best.tmp"` (directory via `save_pretrained`), `RESUME_FILE = "qwen_lora.resume.tmp"`, manifest keys per RESEARCH "Adapter save format and manifest" (include `adapter_sha256` so `refuse_foreign_manifest(out_dir, "adapter_sha256", "Qwen LoRA training")` works).

**Config dataclass + schedule** (86-113): `@dataclass TrainConfig` with `to_dict()`; reuse `lr_at(step, total, peak, floor, warmup)` verbatim (import it from `src.scratch_llm.train` or copy; importing pulls torch only, which is fine; prefer import to keep one tested schedule). Note it hits `floor` at the last step and warms up linearly: matches D-05 (20 warmup, 2e-4 -> 2e-5).

**Autocast helper** (166-170): `torch.autocast("cuda", dtype=torch.bfloat16)` else `nullcontext()`.

**Selection constants + pre-registered remediation rule** (177-201): `SELECTION_METRIC = "dev_loss_action"`, `REMEDIATION_THRESHOLD`, `REMEDIATION_RULE` text, and `remediation_needed(manifest)` pure function. D-06 threshold is `dev_action_exact < 0.90`; rule is "doubled max_steps once, gated on dev metrics only".

**Evaluate** (204-312): `@torch.no_grad()`, `was_training` restore, returns dict with exactly `EVAL_KEYS`; action-level vs diagnostic metrics documented in the docstring. For Qwen, compute logits only on supervised rows (RESEARCH Pattern 3: hidden states from `peft_model.base_model.model.model`, `h[:, :-1][labels[:, 1:] != -100]`, `lm_head` on those rows, fp32 CE, normalise by window token count).

**Training loop, resume, non-finite guard, atomic writes** (413-612), key excerpts:
```python
train_eps, dev_eps = read_episodes(train_path), read_episodes(dev_path)
assert_clean(train_eps)       # a leaking file stops here, before the first step
assert_clean(dev_eps)
...
identity = {"train_sha256": file_sha256(train_path), "dev_sha256": file_sha256(dev_path),
            "model": model_cfg.to_dict(), "train": train_cfg.to_dict(), **current_stamps()}
if resume:
    state = torch.load(resume_path, map_location="cpu", weights_only=True)
    differs = [k for k in identity if state["identity"].get(k) != identity[k]]
    if differs:
        raise ValueError(f"the resume file is from a different run: {', '.join(differs)} differ")
...
if not bool(torch.isfinite(loss)):
    log(f"FAILED non_finite_loss step {step + 1}")
    _append_jsonl(log_path, {"step": step + 1, "event": "non_finite_loss"})
    raise NonFiniteLossError(...)
...
if result[SELECTION_METRIC] < best[SELECTION_METRIC]:       # never the respond-text loss (CR-01)
    best, stale_evals = {...}, 0
    _save_checkpoint(best_path, ...)
else:
    stale_evals += 1
...
except KeyboardInterrupt:
    raise                         # Ctrl-C leaves the resume file and the best weights: --resume continues
except BaseException:
    _remove(*leftovers)           # a failed run leaves no loadable checkpoint (review WR-05)
    raise
```
Also copy: `_remove`, `_save_resume` (tmp + `os.replace`, 353-357), `_append_jsonl`, `_rss_mb` (341-350), `_rel` (317-322), `_write_curve` with lazy matplotlib `Agg` (365-381, wrapped in try/except "a plotting problem must not cost a finished run its manifest", 567-570), `refuse_foreign_manifest` called before anything is deleted (438), stale-file cleanup on non-resume (457-459), resume-writes only when `step < max_steps` and no stop_reason (546), checkpoint renamed to final name only on success (566). Resume file must hold only trainable tensors (name->tensor dict) + optimizer + RNG + step/epoch/batch_pos/best/history/elapsed + identity; loaded with `weights_only=True`. Step batching: seeded permutation `random.Random(seed * 1_000_003 + epoch)` as in `make_batches` (138). Gradient accumulation (mb 4 x 4) is NEW (no analog); the optimizer step/zero-grad/clip order at 498-501 is the template (`torch.nn.utils.clip_grad_norm_(trainable, 1.0)`).

**Manifest assembly** (574-603): `schema`, `created_at` (`datetime.now(timezone.utc).isoformat(timespec="seconds")`), `code_commit/code_dirty` from `provenance`, `python/torch/cuda/device`, `dataset {train_path (via _rel), sha256s, counts, class_counts}`, `held_out_sha256()`, `selection {metric, direction, description, remediation_rule}`, `best` (the kept checkpoint's eval) vs `last_eval`, `wall_seconds`, `stop_reason`, `resumed_from_steps`, `run_notes`.

**CLI** (627-679): argparse with `prog="python -m src.qwen_lora.train"`, defaults from the dataclass, `resolve_under_root(..., ALLOWED_ROOT)` for `train/dev/out_dir` with `parser.error`, `os.path.isfile` check, `--resume`, repeatable `--run-note` via `_note` validator (one line ASCII <=200 chars), `NonFiniteLossError -> return EXIT_NON_FINITE`, `if __name__ == "__main__": sys.exit(main())`, `_flush_print` for log. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` via `os.environ.setdefault` BEFORE `import torch`.

---

### `src/scratch_llm/provenance.py` (reuse, do not modify)

Import from `src.scratch_llm.provenance`: `git_commit, git_dirty, refuse_foreign_manifest, resolve_under_root, write_json_atomic` (as `train.py` 60-61). Note `provenance.py` itself imports `src.benchmark.injecagent.PROJECT_ROOT`; in `hf_local.py` (serving), do NOT import `dataset.py` or `provenance.py` if you want to keep the isolation property that `inference.py` documents ("never imports dataset.py or anything from src/benchmark"), since evaluation-time code must not read the held-out list. Compute sha in hf_local with a local `_sha256_file`.

---

### `tests/test_hf_local_provider.py` (test, request-response with fake model)

**Analog:** `tests/test_scratch_inference.py`.

**File header + gating** (1-38): docstring stating CPU only / no network / what is slow; `torch = pytest.importorskip("torch")` plus `pytest.importorskip("transformers")` / `("peft")`; section dividers `# ── name ──`.
```python
@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
```

**Module-scoped tiny fixture** (49-58): build once with `tmp_path_factory`, return `SimpleNamespace`. For Qwen use the tiny random-init Qwen2 (hidden 64, 2 layers, real tokenizer from cache, `local_files_only=True`) + saved LoRA adapter (RESEARCH "Failure modes": works on CPU in seconds). Keep HF-touching tests few (65-75 s import).

**Provider stamps / end-to-end episode** (103-122):
```python
provider = LocalScratchProvider("scratch-gpt", tiny.path)
response = provider.generate(messages_for_a_notes_request())
assert response.ok
...
episode, score = run_episode(provider, case, "unprotected", policy, 3)
assert episode.steps[0].resolved_model.startswith("scratch-gpt@")
```
**Import-isolation subprocess test** (127-133): copy for "importing `src.llm.registry` does not import transformers/torch/peft" (`sys.modules` check, `cwd=PROJECT_ROOT`, `timeout=600`).

**Refusal tests** (147-175): `pytest.raises(RuntimeError, match="<field>")` for each tampered stamp (adapter sha, base mismatch, missing adapter dir, corrupt `adapter_config.json` -> KeyError wrapped); "missing checkpoint says how to produce it" analog (173-175).

**Runner [skip] integration** (178-202): `write_models_yaml` + `point_runner_at` monkeypatching `runner_module.get_provider` / `load_model_config`; assert `any(line.startswith("[skip] name:") ...)` and no rows in `raw.jsonl`. Add `provider: hf_local` variant of `write_models_yaml`.

**Error response from broken generator + stand-in replies** (318-345): swap `provider._generate`-like seam for a fake `generate`; assert `not response.ok and response.error.startswith("ValueError") and response.text == ""`. Greedy-kwargs capture test (RESEARCH Pitfall 2): fake model records kwargs -> assert `do_sample is False`, `repetition_penalty == 1.0`, `temperature/top_p/top_k is None`.

**models.yaml stamp test** (357-358): `assert registry.load_model_config()["models"]["qwen2.5-0.5b-base"]["model"] == "qwen2.5-0.5b-base"`.

---

### `tests/test_qwen_lora_data.py` / `tests/test_qwen_lora_train.py`

**Analog:** `tests/test_scratch_train.py` (header 1-72). Copy: docstring listing what is pinned ("cheapest first"), `pytest.importorskip`, `TINY` config constant, module-scoped `episodes`/`dev_episodes` via `generate_episodes(64, 0)` and `generate_episodes(8, 1, "dev_tools")`, the `files` fixture (`write_episodes` to `tmp_path`), `quiet(*_)`, `quick_config(**changes)`, `read_log(out_dir)`. Tests to mirror: lr schedule maths (76-84), batching covers every item once and follows seed (87+), loss-mask decode test (decoded supervised tokens == `"".join(compact_action(a) + "<|im_end|>")`), manifest has fixed field list (`tuple(manifest) == MANIFEST_KEYS`), failure paths (leaking file -> `LeakageError`, path outside repo -> `ValueError`, non-finite loss -> `NonFiniteLossError`, resume refuses on identity differences), `remediation_needed` on synthetic manifests. Tokenizer-dependent tests need the cached Qwen tokenizer; keep one module-scoped load.

---

### `tests/conftest.py` (modify: add `STRICT_HF`)

**Analog:** same file, lines 18-33:
```python
def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None

def pytest_report_header(config):
    if _torch_installed():
        return "torch: installed (the ScratchGPT training and inference tests run)"
    return "torch: NOT INSTALLED - tests/test_scratch_train.py and tests/test_scratch_inference.py will be SKIPPED"

def pytest_sessionstart(session):
    if os.environ.get("STRICT_TORCH") == "1" and not _torch_installed():
        raise pytest.UsageError("STRICT_TORCH=1 but torch is not installed: the ScratchGPT tests would be skipped")
```
Add `_hf_installed()` using `find_spec("transformers") and find_spec("peft")` (find_spec does not import: no 65 s cost); extend `pytest_report_header` to return a list of lines (pytest accepts `str | list[str]`) so the existing torch line is preserved; add `STRICT_HF == "1"` check raising `pytest.UsageError`. Keep the review-IN-04 comment style (why the header exists).

---

### `docs/viva/03-fine-tuning-lora-and-chat-templates.md` (+ README row update)

**Analog:** `docs/viva/02-llm-foundations-and-synthetic-data.md`. Structure: `# Viva notes N: ...` title; intro paragraph (what the phase built, why, honest framing: the tuned model is a deliberately compliant worst-case principal); "How to use this page"; a bulleted glossary ("- **Term**: definition", 1-30); `## Part A` concepts with `###` subsections tying each idea to a file (e.g. "### The loss mask (decision D-06)": what is masked, bullet reasons, then "A test decodes the supervised characters..." sentence, lines 210-224); `## Part B` design in this repo; `## Part C` real numbers (copy from `models/qwen-agentlora/manifest.json`); `## Limits of this phase` (honest list); `## Part D - Likely viva questions and answers` as `**Q1. ...?**` + prose answer (569+). Topics per CONTEXT: LoRA W+BA with 8,798,208 / 502,830,976 = 1.75%, chat templates and special tokens, char-offset loss masking, bf16, gradient checkpointing, greedy decoding, base-vs-tuned. `docs/viva/README.md` table row for 03 currently reads `| 03 Fine-tuning, LoRA, chat templates and decoding | ... | Phase 3 | planned (Phase 3) |`: replace with a link row and `written (Phase 3)` matching rows 01/02 format.

---

### `requirements-train.txt` (config)

**Analog:** `requirements.txt` trailing comment block (header comment style, torch is "installed separately"):
```
# From-scratch model (Phase 2: src/scratch_llm) — needs torch>=2.6; install the CUDA build separately
# (tested with torch 2.11.0+cu128):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
```
New file: same comment style; `transformers>=5.20,<5.21`, `peft==0.21.2`, `accelerate>=1.15,<2`; note torch 2.11.0+cu128 NOT to be reinstalled; do not add trl/bitsandbytes/unsloth/flash-attn/datasets. Mention SUS-by-recency rationale in plan summary, not here.

---

## Shared Patterns

### Providers: construction raises, `generate` never raises
**Source:** `src/llm/base.py` 33-41 and `src/llm/scratch.py` 22-54. **Apply to:** `hf_local.py`, its tests.
`LLMProvider.generate(messages, temperature=0.0, max_tokens=512) -> LLMResponse`; failures become `LLMResponse(error=...)`; `LLMResponse` fields: `text, latency_ms, model, prompt_tokens, completion_tokens, error, finish_reason, resolved_model`.

### Lazy heavy imports
**Source:** `scratch.py` line 30 comment; `registry.py` 38-52 (each provider imported inside its branch). **Apply to:** `registry.py` hf_local branch, `hf_local.py`, `src/qwen_lora/__init__.py` (keep light).

### Atomic writes, provenance, path confinement
**Source:** `src/scratch_llm/provenance.py` 50-91 (`write_json_atomic`, `refuse_foreign_manifest`, `resolve_under_root`), `train.py` `_save_resume` 353-357 (tmp + `os.replace`). **Apply to:** `qwen_lora/train.py`.

### Leakage gate before step 0
**Source:** `train.py` 413-417 (`read_episodes` -> `assert_clean`). **Apply to:** `qwen_lora/train.py` and `data.py` loading (D-02).

### Selection on action-level metric, not respond text
**Source:** `train.py` 177, 214-219, 531 (CR-01 comment). **Apply to:** `qwen_lora/train.py` selection, early stopping, adapter-best save.

### Prompt single source of truth
**Source:** `src/agent_harness/prompting.py` (`SYSTEM_PROMPT`, `OBSERVATION_TEMPLATE = "Result of your last action:\n{result}"`, `build_messages`, `parse_action`), `dataset.py::episode_to_messages` 152-160. **Apply to:** `data.py`, provider tests (render parity: rendered text starts `<|im_start|>system\n` + SYSTEM_PROMPT, never contains "You are Qwen"; prefix-stability test).

### Code style (CLAUDE.md)
Module docstring `"""Title ====="""` giving the reason behind rules; snake_case / `UPPER_SNAKE` constants; `Optional[X]`; double quotes in library code; lines < ~120 chars; no bare `except:`; no `sys.path.insert` in `src/`; test section dividers `# ── name ──`.

### Tracked artifacts convention
`models/scratch/` tracks `manifest.json`, `train_log.jsonl`, `loss_curve.png` (git ls-files confirms) and ignores weights (`models/**/*.safetensors` already in `.gitignore`). `models/qwen-agentlora/` follows suit: track `adapter_config.json`, `manifest.json`, `train_log.jsonl`, `loss_curve.png`; PEFT's `README.md` optional.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Gradient accumulation + LM-head-on-supervised-rows loss (inside `qwen_lora/train.py`) | training core | batch | ScratchGPT computes loss inside the model with one batch per optimizer step; use RESEARCH Pattern 3 and the 60-step prototype description |
| PEFT save/reload, `merge_and_unload`, `snapshot_download(local_files_only=True)` | provider/trainer | file-I/O | No HF/PEFT code exists in the repo; follow RESEARCH "Loading" steps 1-6 and "Failure modes of loading" |
| Character-offset loss mask for a BPE tokenizer (`data.py`) | utility | transform | Analog (`format.supervised_mask`) is per-character; logic differs, use RESEARCH Pattern 2 |
| WSL launcher with GPU preflight (>= 3584 MiB free, memory pre-check >= 3000 MiB, `train.done` marker, foreground `wsl.exe -d Ubuntu -- bash <script>`) | script | batch | Phase 2 launcher is not git-tracked (only `models/scratch/gpu_preflight.log` and `train.done` outputs exist); `scripts/run_full_benchmark.sh` is the nearest tracked shell-script style reference. Planner should treat as a small new script specified by RESEARCH "Resumability" and must never stop another project's processes |

## Metadata

**Analog search scope:** `src/llm/`, `src/scratch_llm/`, `src/agent_harness/prompting.py`, `config/`, `tests/`, `docs/viva/`, `scripts/`, `models/`, `requirements.txt`
**Files scanned/read:** 14 (03-CONTEXT, 03-RESEARCH lines 1-455 of 759, scratch.py, registry.py, base.py, provenance.py, inference.py, train.py, conftest.py, models.yaml, test_scratch_inference.py, test_scratch_train.py header, viva 02 outline/excerpts, prompting.py excerpts, dataset.py 150-175)
**Tracked-source gate:** all analog paths confirmed via `git ls-files` (src/llm, src/scratch_llm, config/models.yaml, tests/conftest.py, docs/viva, scripts, models/scratch/{loss_curve.png,manifest.json,train_log.jsonl}). `tests/test_scratch_*.py` were read from disk (present in repo working tree; the truncated ls-files head did not list them, planner should confirm they are tracked before citing).
**Not read:** RESEARCH.md lines 456-759 (Validation Architecture, Code Examples, remaining pitfalls); planner should consult them for the prototype code and test list.
**Pattern extraction date:** 2026-10-22
