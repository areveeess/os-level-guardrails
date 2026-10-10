"""
Tests for sandbox block detection.
Ensures only failures with an enforcement signature are counted as blocks,
so ordinary command failures don't inflate the defense metrics.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sandbox.sandbox import detect_enforcement


def test_success_is_not_a_block():
    assert detect_enforcement(0, "ok", "") is None


def test_seccomp_eperm_is_attributed_to_seccomp():
    assert detect_enforcement(1, "", "chmod: changing permissions of 'x': Operation not permitted") == "seccomp"


def test_sigsys_kill_is_attributed_to_seccomp():
    assert detect_enforcement(159, "", "") == "seccomp"


def test_network_namespace_is_attributed_to_sandbox():
    assert detect_enforcement(6, "", "curl: (6) Could not resolve host: evil.com") == "sandbox"


def test_permission_denied_is_attributed_to_sandbox():
    assert detect_enforcement(1, "", "cat: /etc/shadow: Permission denied") == "sandbox"


def test_ordinary_failures_are_not_blocks():
    assert detect_enforcement(1, "", "") is None  # grep with no match
    assert detect_enforcement(127, "", "sh: 1: lss: not found") is None  # typo
    assert detect_enforcement(2, "", "ls: cannot access 'nope': No such file or directory") is None
