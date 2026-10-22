"""
Hugging Face in-process provider
================================
Serves a Hugging Face causal language model inside the benchmark process behind the LLMProvider
interface, optionally with a PEFT LoRA adapter merged into the weights at load. It is how the
un-tuned Qwen2.5-0.5B-Instruct control row and the locally LoRA-tuned row reach the agent loop.

Decisions, each with the reason behind it:

  Explicit greedy decoding. The shipped generation_config.json of Qwen samples (temperature 0.7,
  top_p 0.8, top_k 20) with a repetition penalty of 1.1, so inheriting it would not be greedy and
  the penalty would distort JSON, which repeats quotes and keys. Every call passes do_sample=False,
  temperature/top_p/top_k=None and repetition_penalty=1.0, and the loaded model gets a fresh
  GenerationConfig. temperature is accepted for interface compatibility and ignored: the benchmark
  is greedy for every model.

  Offline. A benchmark must not depend on the network (a flaky connection would stall or fail a run
  half way). HF_HUB_OFFLINE is set before any Hugging Face import and every snapshot_download /
  from_pretrained call passes local_files_only=True. A model that is not in the local cache is a
  construction error, never a download. The remote-code flag is never passed: Qwen2 is native to the
  transformers library, so no code from a model repository is ever executed.

  Merged adapter. merge_and_unload() folds the low-rank update into the base weights: the greedy
  outputs are identical (measured on 20 prompts and 12 episodes) and decoding is about 2.3 times
  faster than the live PEFT wrapper (30.5 against 13.3 tokens/s, measured). merge=False keeps the
  wrapper for a debugging spot check.

  The reply is never repaired. The text is the raw decode, stripped of whitespace and nothing else.
  A base model that answers in prose or in the wrong schema has a measured property (a format
  failure), reported like for every other model; the agent loop's INVALID_REPLY_TEMPLATE is the only
  recovery. The prompt is rendered only by tokenizer.apply_chat_template over the messages the loop
  built with build_messages (the project system prompt, no tools argument), so every model sees the
  same words.

  Construction fails, generation does not. A missing or stale adapter, an uncached base model or an
  unavailable GPU raise RuntimeError naming the cause; the benchmark runner turns that into a
  [skip] line and writes no rows, instead of scoring a stale model as the trained one. generate()
  itself never raises: a model error (CUDA out of memory included) is returned as LLMResponse.error,
  and a prompt that does not fit the context window is an error response, never an answer from a
  cropped prompt.

Adapter directory contract (written by the LoRA trainer, src/qwen_lora/train.py, read here):
adapter_config.json (PEFT), adapter_model.safetensors and manifest.json holding adapter_sha256,
base_model {id, revision} and render {fingerprint}.

Heavy libraries (torch, transformers, peft, huggingface_hub) are imported inside the functions that
need them, so importing this module or the model registry stays cheap (about 0.5 s; the Hugging
Face stack costs about a minute on the Windows drive).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from src.llm.base import LLMProvider, LLMResponse

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
GREEDY_KWARGS = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0}
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_CONTEXT_TOKENS = 4096
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "manifest.json")


@dataclass
class LoadedModel:
    """What load_model returns: the tokenizer and model plus the identity stamps of what was loaded."""
    tokenizer: object
    model: object
    revision: str
    adapter_sha256: Optional[str]
    stop_ids: list
    pad_id: int
    device: str


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _offline_environment() -> None:
    """Must run before the first Hugging Face import: the hub reads these variables at import time."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def resolve_base_model(base_model: str) -> tuple:
    """(local path, revision) of a base model without touching the network.

    A directory resolves to its real path with revision "local". A Hub id resolves through the local
    cache; the revision is the snapshot directory name (the commit hash of the cached files).
    """
    if os.path.isdir(base_model):
        return os.path.realpath(base_model), "local"
    _offline_environment()
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(base_model, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"base model {base_model} is not in the local Hugging Face cache (offline mode, nothing is downloaded "
            f"during a benchmark); download it once with: venv/bin/python -c "
            f"\"from huggingface_hub import snapshot_download; snapshot_download('{base_model}')\""
        ) from e
    return path, os.path.basename(path)


RETRAIN = "retrain it with python -m src.qwen_lora.train"


def _require_adapter_files(adapter_dir: str) -> None:
    """The cheap part of the adapter check (no Hugging Face import): the directory and its three files exist."""
    if not os.path.isdir(adapter_dir):
        raise RuntimeError(f"adapter not found: {adapter_dir}. "
                           f"Train it first: python -m src.qwen_lora.train (see src/qwen_lora/run_train.sh)")
    for name in ADAPTER_FILES:
        if not os.path.isfile(os.path.join(adapter_dir, name)):
            raise RuntimeError(f"adapter file {name} is missing in {adapter_dir}. "
                               f"Train it first: python -m src.qwen_lora.train (see src/qwen_lora/run_train.sh)")


def _read_json_object(path: str, label: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"{label} is unreadable: {path} ({type(e).__name__}: {e}); {RETRAIN}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}; {RETRAIN}")
    return data


def _same_base_model(found, base_model: str) -> bool:
    """A model id must match exactly; a local directory matches by real path."""
    if os.path.isdir(base_model):
        return isinstance(found, str) and os.path.realpath(found) == os.path.realpath(base_model)
    return found == base_model


def verify_adapter(adapter_dir: str, base_model: str, revision: str, tokenizer) -> str:
    """Refuse a missing or stale adapter; return the sha256 of its weights.

    Checked, each mismatch a RuntimeError naming the field with the expected and the found value: the three
    files exist; adapter_config.json parses and its base_model_name_or_path is the configured base model;
    manifest.json holds adapter_sha256, base_model.id, base_model.revision and render.fingerprint, the base
    model id and the cached revision are the ones the adapter was trained on, the hash is that of
    adapter_model.safetensors as it is on disk now (computed before any merge), and the render fingerprint is the
    running code's. A stale adapter would be benchmarked as if it were the trained one.
    """
    _require_adapter_files(adapter_dir)
    config = _read_json_object(os.path.join(adapter_dir, "adapter_config.json"), "adapter_config.json")
    for key in ("peft_type", "base_model_name_or_path"):
        if key not in config:
            raise RuntimeError(f"adapter_config.json in {adapter_dir} has no {key}; {RETRAIN}")
    if not _same_base_model(config["base_model_name_or_path"], base_model):
        raise RuntimeError(f"adapter_config.json base_model_name_or_path is {config['base_model_name_or_path']!r} "
                           f"but the configured base model is {base_model!r}; {RETRAIN}")

    manifest = _read_json_object(os.path.join(adapter_dir, "manifest.json"), "manifest.json")

    def field(dotted: str):
        node = manifest
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise RuntimeError(f"manifest.json in {adapter_dir} has no {dotted}; {RETRAIN}")
            node = node[part]
        return node

    if not _same_base_model(field("base_model.id"), base_model):
        raise RuntimeError(f"manifest base_model.id is {field('base_model.id')!r} "
                           f"but the configured base model is {base_model!r}; {RETRAIN}")
    if field("base_model.revision") != revision:
        raise RuntimeError(f"manifest base_model.revision is {field('base_model.revision')!r} "
                           f"but the cached base model is at {revision!r}; {RETRAIN}")
    sha256 = sha256_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    if field("adapter_sha256") != sha256:
        raise RuntimeError(f"manifest adapter_sha256 is {field('adapter_sha256')!r} "
                           f"but adapter_model.safetensors hashes to {sha256!r}; {RETRAIN}")
    from src.qwen_lora.chat import render_fingerprint
    expected = render_fingerprint(tokenizer)
    if field("render.fingerprint") != expected:
        raise RuntimeError(f"manifest render.fingerprint is {field('render.fingerprint')!r} "
                           f"but the running code renders {expected!r}; {RETRAIN}")
    return sha256


def load_model(base_model: str, adapter: Optional[str] = None, merge: bool = True, device: str = "cuda") -> LoadedModel:
    """Load the tokenizer and the base model (bf16 on CUDA, fp32 on the CPU) and merge the adapter when one is given.

    Every failure is a RuntimeError naming the cause, so the benchmark runner skips the model.
    """
    _offline_environment()
    if adapter is not None:
        _require_adapter_files(adapter)        # fail before the minute-long Hugging Face imports
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; set device: cpu for this model in config/models.yaml "
                               "to run it on the CPU")
        path, revision = resolve_base_model(base_model)
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # The adapter is checked after the tokenizer (its render fingerprint needs it) and before the base weights load.
        adapter_sha256 = verify_adapter(adapter, base_model, revision, tokenizer) if adapter is not None else None
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True)
        model = model.to(device).eval()
        stop_ids = tokenizer.convert_tokens_to_ids(list(STOP_TOKENS))
        if any(token_id is None for token_id in stop_ids):
            raise RuntimeError(f"the tokenizer of {base_model} does not know the stop tokens {STOP_TOKENS}")
        pad_id = stop_ids[1]
        # Replace the sampling defaults of generation_config.json; generate() also passes them per call.
        model.generation_config = GenerationConfig(do_sample=False, repetition_penalty=1.0,
                                                   eos_token_id=stop_ids, pad_token_id=pad_id)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            if merge:
                model = model.merge_and_unload()
            model.eval()
        return LoadedModel(tokenizer=tokenizer, model=model, revision=revision, adapter_sha256=adapter_sha256,
                           stop_ids=stop_ids, pad_id=pad_id, device=device)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"cannot load {base_model} (adapter {adapter}): {type(e).__name__}: {e}") from e


class HFLocalProvider(LLMProvider):

    def __init__(self, name: str, base_model: str, adapter: Optional[str] = None,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
                 merge: bool = True, device: str = "cuda"):
        self.name = name
        self.base_model = base_model
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.merge = merge
        # Looked up on the module so a test can replace the loader with a fake.
        loaded = load_model(base_model, adapter=adapter, merge=merge, device=device)
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.revision = loaded.revision
        self.adapter_sha256 = loaded.adapter_sha256
        self.stop_ids = loaded.stop_ids
        self.pad_id = loaded.pad_id
        self.device = loaded.device
        self.resolved_model = f"{base_model}@{loaded.revision[:12]}"
        if loaded.adapter_sha256:
            self.resolved_model += f"+{name}@{loaded.adapter_sha256[:12]}"

    def generate(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        """One greedy reply. `temperature` is accepted and ignored: the benchmark is greedy for every model."""
        start = time.perf_counter()
        try:
            import torch
            enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors="pt")
            n_in = int(enc["input_ids"].shape[1])
            if n_in + 1 > self.max_context_tokens:
                # Never answer from a cropped prompt: the model would be scored on text it did not read.
                return LLMResponse(
                    text="", latency_ms=_elapsed_ms(start), model=self.name, prompt_tokens=n_in,
                    finish_reason="context_overflow", resolved_model=self.resolved_model,
                    # No bare numbers in the text (review WR-01): the count is in prompt_tokens, the limit is
                    # max_context_tokens in config/models.yaml.
                    error="ContextOverflowError: context_overflow: the prompt leaves no room in the configured "
                          "context window (max_context_tokens)")
            max_new = max(0, min(max_tokens, self.max_new_tokens, self.max_context_tokens - n_in))
            if max_new == 0:
                return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name, prompt_tokens=n_in,
                                   completion_tokens=0, finish_reason="length", resolved_model=self.resolved_model)
            enc = {key: value.to(self.device) for key, value in enc.items()}
            with torch.inference_mode():
                out = self.model.generate(**enc, **GREEDY_KWARGS, max_new_tokens=max_new,
                                          eos_token_id=self.stop_ids, pad_token_id=self.pad_id)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            new = out[0, n_in:].tolist()
            stopped = bool(new) and new[-1] in self.stop_ids
            text = self.tokenizer.decode(new[:-1] if stopped else new, skip_special_tokens=True).strip()
            return LLMResponse(
                text=text, latency_ms=_elapsed_ms(start), model=self.name, prompt_tokens=n_in,
                completion_tokens=len(new), finish_reason="stop" if stopped else "length",
                resolved_model=self.resolved_model)
        except Exception as e:
            # The one broad catch, at the model-call boundary: CUDA out of memory comes back as a value too.
            return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name,
                               error=f"{type(e).__name__}: {e}")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
