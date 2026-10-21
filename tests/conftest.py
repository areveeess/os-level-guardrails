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


def pytest_report_header(config):
    if _torch_installed():
        return "torch: installed (the ScratchGPT training and inference tests run)"
    return "torch: NOT INSTALLED - tests/test_scratch_train.py and tests/test_scratch_inference.py will be SKIPPED"


def pytest_sessionstart(session):
    if os.environ.get("STRICT_TORCH") == "1" and not _torch_installed():
        raise pytest.UsageError("STRICT_TORCH=1 but torch is not installed: the ScratchGPT tests would be skipped")
