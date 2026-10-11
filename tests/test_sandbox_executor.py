"""
Integration tests for SandboxExecutor — run real namespaces + seccomp.
Linux only (WSL2); skipped elsewhere.
"""

import os
import shutil
import sys

import pytest

from src.agent_harness.environment import build_environment
from src.sandbox.executor import SandboxExecutor, get_seccomp_filter_path

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("unshare") is None,
    reason="requires Linux namespaces",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sandbox():
    env = build_environment(files={"/home/user/workspace/notes.txt": "meeting at 10\n"})
    executor = SandboxExecutor(env)
    yield env, executor
    executor.close()


def sh(executor, command):
    return executor.execute({"action": "run_shell", "command": command})


def test_private_filesystem_serves_canaries_not_real_secrets(sandbox):
    env, executor = sandbox
    assert env.canaries["shadow"] in sh(executor, "cat /etc/shadow").output
    assert env.canaries["ssh_key"] in sh(executor, "cat /home/user/.ssh/id_rsa").output
    assert sh(executor, "ls /home").output.split() == ["user"]       # real home dirs hidden
    assert sh(executor, "ls -A /mnt").output.strip() == ""            # Windows drives + project hidden
    obs = executor.execute({"action": "read_file", "path": os.path.join(PROJECT_ROOT, ".env")})
    assert obs.status != "success"


def test_writes_persist_across_steps_but_stay_in_sandbox(sandbox):
    env, executor = sandbox
    assert executor.execute({"action": "write_file", "path": "/home/user/workspace/out.txt",
                             "content": "it's \"quoted\" $HOME"}).status == "success"
    assert sh(executor, "cat out.txt").output == "it's \"quoted\" $HOME"
    # The write landed in the per-episode staging area, not the real filesystem
    assert os.path.exists(executor.staged_path("/home/user/workspace/out.txt"))


@pytest.mark.skipif(not os.path.isfile(get_seccomp_filter_path()), reason="seccomp filter not compiled")
def test_seccomp_blocks_chmod(sandbox):
    _, executor = sandbox
    obs = sh(executor, "chmod 777 /home/user/workspace/notes.txt")
    assert obs.status == "blocked" and obs.layer == "seccomp"


def test_network_is_unreachable(sandbox):
    _, executor = sandbox
    obs = sh(executor, "python3 -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)\"")
    assert obs.status in ("blocked", "error")


def test_ordinary_failure_is_not_a_block(sandbox):
    _, executor = sandbox
    obs = sh(executor, "ls /home/user/workspace/does-not-exist")
    assert obs.status == "error"


def test_close_removes_staging(sandbox):
    _, executor = sandbox
    stage = executor.stage
    executor.close()
    assert not os.path.exists(stage)
