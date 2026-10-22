import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.policy_engine.engine as policy_engine_module


@pytest.fixture(autouse=True)
def isolated_decision_log(tmp_path, monkeypatch):
    """Send policy decisions to a temp file so tests never touch logs/guardrails.jsonl."""
    monkeypatch.setattr(policy_engine_module, 'LOG_FILE', str(tmp_path / 'guardrails.jsonl'))


# torch is installed separately (CUDA build), so tests/test_scratch_train.py and tests/test_scratch_inference.py
# skip themselves without it. A green run must not hide that (review IN-04): the header says it, and
# STRICT_TORCH=1 turns a missing torch into a failure for CI.
def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


# The Qwen provider and LoRA tests need transformers and peft (requirements-train.txt) and skip themselves without
# them, so the header says it too, and STRICT_HF=1 turns their absence into a failure. find_spec does not import the
# packages, so the header costs nothing (importing transformers takes about a minute from the Windows drive).
def _hf_installed() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("transformers", "peft"))


def pytest_report_header(config):
    if _torch_installed():
        torch_line = "torch: installed (the ScratchGPT training and inference tests run)"
    else:
        torch_line = ("torch: NOT INSTALLED - tests/test_scratch_train.py and tests/test_scratch_inference.py "
                      "will be SKIPPED")
    if _hf_installed():
        hf_line = ("transformers/peft: installed (the Qwen provider and LoRA tests run; they need "
                   "Qwen/Qwen2.5-0.5B-Instruct in the local Hugging Face cache)")
    else:
        hf_line = ("transformers/peft: NOT INSTALLED - tests/test_hf_local_provider.py, tests/test_qwen_lora_data.py "
                   "and tests/test_qwen_lora_train.py will be SKIPPED (pip install -r requirements-train.txt)")
    return [torch_line, hf_line]


def pytest_sessionstart(session):
    if os.environ.get("STRICT_TORCH") == "1" and not _torch_installed():
        raise pytest.UsageError("STRICT_TORCH=1 but torch is not installed: the ScratchGPT tests would be skipped")
    if os.environ.get("STRICT_HF") == "1" and not _hf_installed():
        raise pytest.UsageError("STRICT_HF=1 but transformers or peft is not installed: the Qwen tests would be skipped")
