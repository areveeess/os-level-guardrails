import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.policy_engine.engine as policy_engine_module


@pytest.fixture(autouse=True)
def isolated_decision_log(tmp_path, monkeypatch):
    """Send policy decisions to a temp file so tests never touch logs/guardrails.jsonl."""
    monkeypatch.setattr(policy_engine_module, 'LOG_FILE', str(tmp_path / 'guardrails.jsonl'))
