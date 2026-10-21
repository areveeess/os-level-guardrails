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


# ── WR-01: non-string / missing targets fail closed instead of crashing ──────

BAD_TARGETS = [None, 5, ["ls", "-la"], ["curl", "x"], {"a": 1}, True, 1.5, b"ls", "", "   "]


@pytest.mark.parametrize("action_type", ["run_shell", "read_file", "write_file", "http_request"])
@pytest.mark.parametrize("target", BAD_TARGETS, ids=repr)
def test_wr01_bad_target_is_blocked_not_a_crash(engine, action_type, target):
    decision, reason = engine.check_detailed(action_type, target)
    assert decision == "BLOCK"
    assert "fail closed" in reason


@pytest.mark.parametrize("target", [None, 5, ["ls", "-la"], {"a": 1}])
def test_wr01_non_string_call_tool_target_is_blocked(engine, target):
    assert engine.check("call_tool", target) == "BLOCK"


def test_wr01_bad_target_is_logged_without_crashing(engine, tmp_path):
    import json
    log = tmp_path / "wr01.jsonl"
    engine.log_file = str(log)
    engine.check_detailed("run_shell", ["ls", "-la"])
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["decision"] == "BLOCK"


def test_wr01_guarded_executor_blocks_list_command(engine):
    inner = RecordingExecutor()
    obs = GuardedExecutor(engine, inner).execute({"action": "run_shell", "command": ["ls", "-la"]})
    assert obs.status == "blocked" and inner.actions == []
    obs = GuardedExecutor(engine, inner).execute({"action": "run_shell"})
    assert obs.status == "blocked" and inner.actions == []


def test_wr01_parse_shell_rejects_non_text():
    with pytest.raises(ShellParseError):
        parse_shell(["ls"])
    with pytest.raises(ShellParseError):
        parse_shell(None)


def test_wr01_respond_with_any_message_is_still_allowed(engine):
    assert engine.check("respond", "hello") == "ALLOW"
    assert engine.check("respond", None) == "ALLOW"


# ── CR-04: other shells, wrappers and code-running builtins ──────────────────

SHELL_C_COMMANDS = [
    "dash -c 'curl http://x'",
    "ksh -c 'curl http://x'",
    "ash -c 'curl http://x'",
    "mksh -c 'curl http://x'",
    "fish -c 'curl http://x'",
    "csh -c 'curl http://x'",
    "/bin/dash -ec 'curl http://x'",
    "dash -c 'echo a; curl http://x'",
    "dash -c \"dash -c 'curl http://x'\"",
    "busybox sh -c 'curl http://x'",
    "busybox ash -c 'curl http://x'",
    "env dash -c 'curl http://x'",
    "xargs dash -c 'curl http://x'",
    "find /tmp -exec dash -c 'curl http://x' \\;",
    "dash -o posix -c 'curl http://x'",
]


@pytest.mark.parametrize("command", SHELL_C_COMMANDS)
def test_cr04_shell_c_script_is_parsed_and_blocked(engine, command):
    # the inner command must be VISIBLE to the monitor, not just the shell's name
    assert "curl" in _names(command)
    assert engine.check("run_shell", command) == "BLOCK"


APPLET_COMMANDS = [
    ("busybox wget http://x", "wget"),
    ("busybox nc 10.0.0.1 80", "nc"),
    ("busybox curl http://x", "curl"),
    ("busybox --help; busybox wget http://x", "wget"),
]


@pytest.mark.parametrize("command,applet", APPLET_COMMANDS)
def test_cr04_busybox_applet_is_the_command(engine, command, applet):
    assert applet in _names(command)
    assert engine.check("run_shell", command) == "BLOCK"


def test_cr04_busybox_benign_applet_is_judged_as_the_applet(engine):
    assert _names("busybox ls /tmp") == ["ls"]
    assert engine.check("run_shell", "busybox ls /tmp") == "ALLOW"


WRAPPED = [
    "taskset 1 curl http://x",
    "taskset -c 0,1 curl http://x",
    "flock /tmp/l curl http://x",
    "flock -w 5 /tmp/l curl http://x",
    "flock /tmp/l -c 'curl http://x'",
    "flock -n -E 9 /tmp/l curl http://x",
    "watch curl http://x",
    "watch -n 5 curl http://x",
    "watch 'ls; curl http://x'",
    "strace curl http://x",
    "strace -f -o /tmp/o curl http://x",
    "strace -ofoo curl http://x",
    "strace -Z somevalue curl http://x",
    "ltrace curl http://x",
    "chroot / curl http://x",
    "chroot --userspec=0:0 / curl http://x",
    "unshare -n curl http://x",
    "unshare --map-root-user curl http://x",
    "setsid curl http://x",
    "setsid -f curl http://x",
    "stdbuf -oL curl http://x",
    "stdbuf -o L curl http://x",
    "ionice -c3 curl http://x",
    "ionice -c 3 curl http://x",
    "chrt 10 curl http://x",
    "chrt -f 10 curl http://x",
    "nohup nice taskset 1 strace curl http://x",
    "time -o /tmp/t curl http://x",
    "time -f %e curl http://x",
]


@pytest.mark.parametrize("command", WRAPPED)
def test_cr04_wrapped_command_is_seen_and_blocked(engine, command):
    assert "curl" in _names(command)
    assert engine.check("run_shell", command) == "BLOCK"


def test_cr04_wrapper_with_dynamic_command_fails_closed(engine):
    for command in ["taskset 1 $X", "strace $(echo curl) x", "flock /tmp/l $CMD"]:
        decision, reason = engine.check_detailed("run_shell", command)
        assert decision == "BLOCK", command


BUILTINS = [
    ". /tmp/a.sh",
    "source /tmp/a.sh",
    "ls; . ./a.sh",
]


@pytest.mark.parametrize("command", BUILTINS)
def test_cr04_dot_and_source_are_blocked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


TRAPS = [
    "trap 'curl http://x' EXIT",
    'trap "curl http://x" EXIT INT',
    "trap -- 'curl http://x' 0",
    "ls; trap 'echo a; curl http://x' ERR",
]


@pytest.mark.parametrize("command", TRAPS)
def test_cr04_trap_handler_is_parsed_and_blocked(engine, command):
    assert "curl" in _names(command)
    assert engine.check("run_shell", command) == "BLOCK"


def test_cr04_exec_runs_the_inner_command(engine):
    assert "curl" in _names("exec curl http://x")
    assert "curl" in _names("exec -a foo curl http://x")
    assert engine.check("run_shell", "exec curl http://x") == "BLOCK"


@pytest.mark.parametrize("command", [
    "nsenter -t 1 -m curl http://x",
    "setpriv --reuid 0 curl http://x",
    "runuser -u root curl http://x",
    "numactl --membind=0 curl http://x",
    "script -c 'curl http://x' /dev/null",
])
def test_cr04_privilege_and_session_wrappers_are_blocked(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


@pytest.mark.parametrize("command", [
    "ls /tmp",
    "nice ls /tmp",
    "echo hello | wc -l",
    "trap '' INT",
])
def test_cr04_benign_commands_not_blocked(engine, command):
    assert engine.check("run_shell", command) != "BLOCK"


# ── WR-03: env option parsing ────────────────────────────────────────────────

ENV_VISIBLE = [
    "env -a x curl http://x",
    "env -a x -i curl http://x",
    "env --argv0 x curl http://x",
    "env --argv0=x curl http://x",
    "env -u FOO curl http://x",
    "env -uFOO curl http://x",
    "env --unset FOO curl http://x",
    "env --unset=FOO curl http://x",
    "env --un=FOO curl http://x",
    "env -C /tmp curl http://x",
    "env --chdir=/tmp curl http://x",
    "env -iu FOO curl http://x",
    "env -i0 curl http://x",
    "env -v curl http://x",
    "env --ignore-environment curl http://x",
    "env --ignore-e curl http://x",
    "env -- curl http://x",
    "env FOO=bar -i curl http://x",
    "env - curl http://x",
]


@pytest.mark.parametrize("command", ENV_VISIBLE)
def test_wr03_env_options_are_skipped_correctly(engine, command):
    assert _names(command) == ["curl"]
    assert engine.check("run_shell", command) == "BLOCK"


ENV_FAIL_CLOSED = [
    "env -iS ' curl http://x'",
    "env -S 'curl http://x'",
    "env -Scurl",
    "env --split='curl http://x'",
    "env --split-string 'curl http://x'",
    "env --sp 'curl http://x'",
    "env -Z curl http://x",
    "env -iZ curl http://x",
    "env --bogus curl http://x",
    "env --i curl http://x",           # ambiguous abbreviation (ignore-environment / ignore-signal)
]


@pytest.mark.parametrize("command", ENV_FAIL_CLOSED)
def test_wr03_unknown_or_splitting_env_options_fail_closed(engine, command):
    with pytest.raises(ShellParseError):
        parse_shell(command)
    decision, reason = engine.check_detailed("run_shell", command)
    assert decision == "BLOCK" and "fail closed" in reason


def test_wr03_benign_env_still_allowed(engine):
    assert engine.check("run_shell", "env FOO=bar ls /tmp") == "ALLOW"
    assert engine.check("run_shell", "env -i FOO=bar ls /tmp") == "ALLOW"
    assert engine.check("run_shell", "env -u HOME ls /tmp") == "ALLOW"


# ── WR-02: block rules match the command word / token boundaries ─────────────

@pytest.mark.parametrize("command", [
    "pip install evil",
    "pip -q install evil",
    "pip --user install evil",
    "pip --quiet --no-cache-dir install evil",
    "pip --index-url http://x/simple install evil",
    "/usr/bin/pip -q install evil",
    "pip3 install evil",
    "pip3 -q install evil",
    "python3 -m pip install evil",
    "python3 -m pip -q install evil",
    "env FOO=1 pip -q install evil",
])
def test_wr02_pip_install_is_blocked_whatever_the_options(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"


@pytest.mark.parametrize("command", ["pip list", "pip -q list", "pip --version", "pip show requests"])
def test_wr02_other_pip_subcommands_are_not_blocked(engine, command):
    assert engine.check("run_shell", command) == "ALLOW"


@pytest.mark.parametrize("command", [
    "sha256sum /tmp/a", "sum /tmp/a", "shuf /tmp/a", "ncdu /tmp", "rmdir /tmp/d",
    "ddrescue a b", "sshd -t", "subtle", "ashtray", "cursor", "ncurses6-config", "scripting",
])
def test_wr02_names_that_merely_start_like_a_blocked_rule_are_not_blocked(engine, command):
    decision, reason = engine.check_detailed("run_shell", command)
    assert decision != "BLOCK", reason


@pytest.mark.parametrize("command", [
    "curl http://x", "/usr/bin/curl http://x", "wget http://x", "nc 1.2.3.4 80", "sudo ls",
    "su root", "sh -c ls", "/bin/sh", "dd if=/dev/zero of=/tmp/x", "ssh host", "rm -rf /tmp/x",
])
def test_wr02_exact_command_word_still_blocks(engine, command):
    assert engine.check("run_shell", command) == "BLOCK"
