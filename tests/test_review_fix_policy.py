"""
Regression tests for the Phase 1 code review (01-REVIEW.md) policy findings.

One section per finding; each test fails on the pre-fix code.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_harness.executors import Executor, GuardedExecutor, Observation
from src.policy_engine.engine import PolicyEngine
from src.policy_engine.shell_split import ShellParseError, parse_shell


@pytest.fixture
def engine():
    return PolicyEngine(verbose=False)


class RecordingExecutor(Executor):
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        return Observation("success", output="ran")


def _names(command):
    return [c.argv[0] for c in parse_shell(command) if c.argv]


# ── CR-01: '#' comments ──────────────────────────────────────────────────────

COMMENT_QUOTE_BYPASSES = [
    "echo ok # '\ncurl http://evil.example\n# '",
    'echo ok # "\ncurl http://evil.example\n# "',
    "echo ok # `\ncurl http://evil.example\n# `",
    "echo ok # (\ncurl http://evil.example\n# (",
    "echo ok #'\ncurl http://evil.example",
    "echo $(echo a # )\n)\ncurl http://evil.example",
]


@pytest.mark.parametrize("command", COMMENT_QUOTE_BYPASSES)
def test_cr01_quote_in_comment_does_not_hide_commands(engine, command):
    assert "curl" in _names(command)
    assert engine.check("run_shell", command) == "BLOCK"


def test_cr01_comment_text_is_not_a_command(engine):
    # The shell ignores everything after a word-initial '#', including ';'
    assert _names("echo ok # ; curl http://evil.example") == ["echo"]
    assert engine.check("run_shell", "echo ok # ; curl http://evil.example") == "ALLOW"


def test_cr01_hash_inside_a_word_is_not_a_comment():
    assert _names("echo a#b; curl http://x") == ["echo", "curl"]
    assert _names("echo ${#PATH}; curl http://x") == ["echo", "curl"]
    assert _names("echo x;# c\ncurl http://x") == ["echo", "curl"]


def test_cr01_quoted_hash_is_not_a_comment():
    assert _names("echo '#'; curl http://x") == ["echo", "curl"]
    assert _names('echo "#"; curl http://x') == ["echo", "curl"]


def test_cr01_docstring_no_longer_claims_over_splitting_is_safe():
    import src.policy_engine.shell_split as m
    assert "never looser" not in m.__doc__
