"""
Shared helpers for the Qwen tests (provider, and the LoRA data and trainer tests of later plans).

  - HF_HUB_OFFLINE is set before any Hugging Face import: a test must never reach the network, and the
    hub library reads this variable once, at import time.
  - require_hf() skips a test module when transformers or peft are missing; qwen_snapshot_or_skip()
    skips when Qwen/Qwen2.5-0.5B-Instruct is not in the local cache. STRICT_HF=1 turns the missing
    cache into a failure (a missing package already fails the session in conftest.py), so a green CI
    run cannot hide skipped Qwen tests.
  - build_tiny_qwen() makes a random-init Qwen2 (about 10 M parameters, hidden size 64, 2 layers) with
    the real cached tokenizer, and write_lora_adapter() saves a PEFT LoRA adapter for it with the same
    manifest the trainer writes. Together they exercise the real transformers and peft code paths on
    the CPU in seconds, without the 1 GB model.

transformers and peft are imported inside the functions, never at module level.
"""

import json
import os
import shutil

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def require_hf() -> None:
    """Skip the calling module when transformers or peft is not installed."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")


def qwen_snapshot_or_skip() -> str:
    """Path of the cached Qwen snapshot; skip (fail under STRICT_HF=1) when it is not in the local cache."""
    require_hf()
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(QWEN_ID, local_files_only=True)
    except Exception as e:
        message = f"{QWEN_ID} is not in the local Hugging Face cache ({type(e).__name__})"
        if os.environ.get("STRICT_HF") == "1":
            pytest.fail(message)
        pytest.skip(message)


def build_tiny_qwen(directory: str, seed: int = 0) -> str:
    """Save a random-init Qwen2 and the real cached tokenizer into `directory`; return the directory."""
    snapshot = qwen_snapshot_or_skip()
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                         eos_token_id=151645, bos_token_id=151643)
    Qwen2ForCausalLM(config).save_pretrained(directory)
    AutoTokenizer.from_pretrained(snapshot, local_files_only=True).save_pretrained(directory)
    return directory


def write_lora_adapter(base_dir: str, out_dir: str, seed: int = 0, perturb: bool = True) -> str:
    """Save a PEFT LoRA adapter (r 16, alpha 32, all-linear) for the model in `base_dir` into `out_dir`.

    A fresh LoRA starts with lora_B at zero, which changes nothing; with perturb the B matrices are
    filled from a seeded normal(0, 0.02) so the adapter really moves the logits. manifest.json carries
    what src.llm.hf_local.verify_adapter checks: the weights, config and identity hashes, the base model and the render
    fingerprint.
    """
    require_hf()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.llm.hf_local import adapter_config_sha256, adapter_identity_sha256, sha256_file
    from src.qwen_lora.chat import render_fingerprint

    base_dir = os.path.realpath(base_dir)
    model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True)
    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                                  task_type="CAUSAL_LM"))
    if perturb:
        generator = torch.Generator().manual_seed(seed)
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.data = (torch.randn(param.shape, generator=generator) * 0.02).to(param.dtype)
    peft_model.save_pretrained(out_dir, save_embedding_layers=False)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    weights_sha256 = sha256_file(os.path.join(out_dir, "adapter_model.safetensors"))
    config_sha256 = adapter_config_sha256(os.path.join(out_dir, "adapter_config.json"))
    manifest = {
        "adapter_sha256": weights_sha256,
        "adapter_config_sha256": config_sha256,
        "adapter_identity_sha256": adapter_identity_sha256(weights_sha256, config_sha256),
        "base_model": {"id": base_dir, "revision": "local"},
        "render": {"fingerprint": render_fingerprint(tokenizer)},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_dir


def copy_adapter(adapter_dir: str, destination: str) -> str:
    """A throwaway copy of an adapter directory, for tests that damage or edit it."""
    shutil.copytree(adapter_dir, destination)
    return destination
