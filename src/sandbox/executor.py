"""
SandboxExecutor — runs agent actions inside the OS-level sandbox
=================================================================
Each action runs in fresh Linux namespaces with a *private filesystem view*:

    unshare --user --map-root-user --mount --pid --fork --net
      ├─ /home, /root, /tmp  → per-episode staging dirs (fake home + canaries)
      ├─ /etc/shadow         → canary copy
      ├─ /mnt                → empty tmpfs (hides Windows drives, project, .env)
      └─ seccomp_filter → sh -c "<command>"   (chmod, mount, ptrace, connect … denied)

The staging dirs live for a whole episode, so the agent's own writes persist
between steps but never touch the real system. Mount setup happens *before*
the seccomp filter is applied, because the filter forbids mount() itself.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile

from src.agent_harness.environment import Environment
from src.agent_harness.executors import Executor, Observation

ACTION_TIMEOUT_SECONDS = 30


def is_linux() -> bool:
    """Check if we're running on Linux (required for namespaces/seccomp)."""
    return sys.platform.startswith('linux')


def get_seccomp_filter_path() -> str:
    """Get the path to the compiled seccomp filter binary."""
    return os.path.join(os.path.dirname(__file__), 'seccomp_filter')


# Error text that indicates an isolation layer stopped the command.
# seccomp_filter.c returns EPERM for blocked syscalls ("Operation not permitted");
# the network namespace has no interfaces, so network calls fail to resolve/route.
SECCOMP_SIGNATURES = ("operation not permitted", "bad system call")
NAMESPACE_SIGNATURES = (
    "permission denied",
    "network is unreachable",
    "could not resolve host",
    "temporary failure in name resolution",
    "unable to resolve host",
)


def detect_enforcement(returncode: int, stdout: str, stderr: str):
    """Return the layer that blocked a failed command, or None for an ordinary failure."""
    if returncode == 0:
        return None
    # 159 = 128 + SIGSYS: the process was killed by a seccomp kill action
    if returncode in (159, -31):
        return "seccomp"
    text = f"{stdout}\n{stderr}".lower()
    if any(sig in text for sig in SECCOMP_SIGNATURES):
        return "seccomp"
    if any(sig in text for sig in NAMESPACE_SIGNATURES):
        return "sandbox"
    return None

# Runs as namespace-root inside the new namespaces. $SBX_* come from the environment
# so the agent's command never has to be quoted into this script.
SETUP_SCRIPT = r'''
set -e
mount --bind "$SBX_STAGE/etc_shadow" /etc/shadow
mount --bind "$SBX_STAGE/home" /home
mount --bind "$SBX_STAGE/root" /root
mount -t tmpfs -o size=16m tmpfs /mnt
mount --bind "$SBX_STAGE/tmp" /tmp
cd "$SBX_CWD" 2>/dev/null || cd /
set +e
if [ -x /root/.seccomp_filter ]; then
  exec /root/.seccomp_filter sh -c "$SBX_CMD"
else
  exec sh -c "$SBX_CMD"
fi
'''


class SandboxExecutor(Executor):

    def __init__(self, env: Environment, policy=None, use_seccomp: bool = True):
        """
        Args:
            env: Files/canaries to stage into the sandbox's private filesystem.
            policy: Optional PolicyEngine, used only to log kernel-level blocks
                    to the dashboard with their layer.
            use_seccomp: Apply the seccomp-BPF filter (requires `make` in src/sandbox).
        """
        if not is_linux():
            raise RuntimeError("SandboxExecutor requires Linux (namespaces + seccomp)")
        env.validate_stageable()
        self.env = env
        self.policy = policy
        self.stage = tempfile.mkdtemp(prefix="guardrails_sbx_")
        self._stage_environment(use_seccomp)

    def _stage_environment(self, use_seccomp: bool):
        for sub in ("home", "root", "tmp"):
            os.makedirs(os.path.join(self.stage, sub))
        os.makedirs(os.path.join(self.stage, "home", "user", "workspace"), exist_ok=True)
        open(os.path.join(self.stage, "etc_shadow"), "w").close()

        for path, content in self.env.files.items():
            if path == "/etc/shadow":
                dest = os.path.join(self.stage, "etc_shadow")
            else:  # /home/... or /tmp/...
                dest = os.path.join(self.stage, path.lstrip("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)

        seccomp = get_seccomp_filter_path()
        self.seccomp_active = use_seccomp and os.path.isfile(seccomp)
        if self.seccomp_active:
            shutil.copy2(seccomp, os.path.join(self.stage, "root", ".seccomp_filter"))

    def staged_path(self, sandbox_path: str) -> str:
        """Host path of a file as the sandbox sees it at `sandbox_path` (for inspection)."""
        if sandbox_path == "/etc/shadow":
            return os.path.join(self.stage, "etc_shadow")
        return os.path.join(self.stage, sandbox_path.lstrip("/"))

    def execute(self, action: dict) -> Observation:
        kind = action.get("action")
        extra_env = {}
        if kind == "respond":
            return Observation("success", output=action.get("message", ""))
        if kind == "run_shell":
            command = action.get("command", "")
        elif kind == "read_file":
            command = f"cat -- {shlex.quote(action.get('path', ''))}"
        elif kind == "write_file":
            path = action.get("path", "/dev/null")
            command = f'mkdir -p -- "$(dirname -- {shlex.quote(path)})" && printf "%s" "$SBX_CONTENT" > {shlex.quote(path)}'
            extra_env["SBX_CONTENT"] = action.get("content", "")
        elif kind == "http_request":
            command = f"curl -sS --max-time 10 -- {shlex.quote(action.get('url', ''))}"
        else:
            return Observation("error", reason=f"Unknown action type: {kind}")

        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/home/user",
            "LANG": "C.UTF-8",
            "SBX_STAGE": self.stage,
            "SBX_CWD": self.env.cwd,
            "SBX_CMD": command,
            **extra_env,
        }
        cmd = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--net",
               "sh", "-c", SETUP_SCRIPT]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                    timeout=ACTION_TIMEOUT_SECONDS, cwd="/")
        except subprocess.TimeoutExpired:
            return Observation("error", reason=f"Timed out ({ACTION_TIMEOUT_SECONDS}s limit)")

        stdout, stderr = result.stdout, result.stderr
        layer = detect_enforcement(result.returncode, stdout, stderr)
        if layer:
            reason = (stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
            if self.policy:
                self.policy._log_decision(kind, action.get("command") or action.get("path") or action.get("url", ""),
                                          "BLOCK", f"Blocked by {layer}: {reason}", layer=layer)
            return Observation("blocked", output=stdout, layer=layer, reason=reason)
        if result.returncode != 0:
            return Observation("error", output=stdout + stderr, reason=f"exit code {result.returncode}")
        return Observation("success", output=stdout)

    def close(self):
        # Never follow symlinks the agent may have planted in its staging area
        shutil.rmtree(self.stage, ignore_errors=True)
