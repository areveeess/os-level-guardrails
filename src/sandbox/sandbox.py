"""
Sandbox Wrapper — OS-Level Isolation
======================================
This module provides the OS-level security boundary for the AI agent.
It combines THREE layers of defense in the correct order:

  ARCHITECTURE (Two-Phase Execution, repeated every agent step):
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 1: OUTSIDE SANDBOX (needs network for LLM)      │
  │    1. Agent sends the conversation to the LLM           │
  │    2. LLM responds with an action (e.g., "read file")   │
  │    3. Policy Engine checks action against access-matrix  │
  │    4. If BLOCKED → the agent is told, action not run     │
  │    5. If ALLOWED → proceed to Phase 2                   │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 2: INSIDE SANDBOX (isolated tool execution)      │
  │    1. Create namespaces (user, mount, PID, net)          │
  │    2. Give it a private filesystem (fake home, canaries) │
  │    3. Apply seccomp-BPF filter (block dangerous syscalls)│
  │    4. Execute ONLY the tool command inside this jail     │
  │    5. Return the result to the agent loop               │
  └─────────────────────────────────────────────────────────┘

  Why two phases?
    The LLM call needs network access (to reach the model's API).
    But tool execution must be isolated (no network, no real secrets).
    So we split: LLM call outside, tool execution inside.

  Layer 1: Policy Engine (Application-Level Pre-check)
    - Checks every action against the access-matrix BEFORE execution
    - Provides readable decision logs for the dashboard

  Layer 2: Linux Namespaces (Environment Isolation)
    - Mount namespace: private /home, /root, /tmp; /mnt hidden; fake /etc/shadow
    - PID namespace: agent can only see its own processes
    - Network namespace: tool execution has no network access
    - User namespace: no real privileges on the host

  Layer 3: Seccomp-BPF (Syscall Filtering)
    - Blocks dangerous syscalls (chmod, mount, ptrace, setuid, connect, etc.)
    - Enforced by the kernel — cannot be bypassed from userspace

Usage:
  from src.sandbox.sandbox import Sandbox
  Sandbox(model='gemini-flash').run_agent('List files in my workspace')
"""

import datetime
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_harness.environment import build_environment
from src.agent_harness.executors import DirectExecutor, GuardedExecutor
from src.agent_harness.loop import AgentLoop
from src.llm import get_provider
from src.policy_engine.engine import PolicyEngine
# Re-exported: enforcement helpers live next to the sandbox executor
from src.sandbox.executor import (SandboxExecutor, detect_enforcement, get_seccomp_filter_path,  # noqa: F401
                                  is_linux)


class Sandbox:
    """Runs the multi-step agent with every action going through
    policy engine → namespaces + private filesystem → seccomp-BPF."""

    def __init__(self, policy_engine: PolicyEngine = None, model: str = None,
                 env=None, max_steps: int = 5):
        """
        Args:
            policy_engine: PolicyEngine for application-level checks (default policy if None).
            model: Model name from config/models.yaml (default: its `default`).
            env: Environment staged into the sandbox (default: canaries + empty workspace).
            max_steps: Maximum agent steps per prompt.
        """
        self.policy = policy_engine or PolicyEngine()
        self.model = model
        self.env = env or build_environment()
        self.max_steps = max_steps
        self.namespace_supported = is_linux()
        self.seccomp_available = os.path.isfile(get_seccomp_filter_path())

        print(f"\n{'='*60}")
        print(f"  SANDBOX INITIALIZATION")
        print(f"{'='*60}")
        print(f"  Platform:   {'Linux ✓' if self.namespace_supported else 'Not Linux ✗'}")
        print(f"  Namespaces: {'user, mount, pid, net ✓' if self.namespace_supported else 'Not available (non-Linux)'}")
        print(f"  Filesystem: {'private (fake home + canaries, /mnt hidden) ✓' if self.namespace_supported else 'HOST (not isolated)'}")
        print(f"  Seccomp:    {'Compiled ✓' if self.seccomp_available else 'Not compiled — run: cd src/sandbox && make'}")
        print(f"  Policy:     Loaded ✓")
        print(f"{'='*60}")

    def _build_executor(self):
        if self.namespace_supported:
            return GuardedExecutor(self.policy, SandboxExecutor(self.env, policy=self.policy))
        print("  ⚠️  Not on Linux — falling back to POLICY-ONLY mode (no OS isolation)")
        return GuardedExecutor(self.policy, DirectExecutor())

    def run_agent(self, prompt: str):
        """Run the agent on `prompt` with full protection. Returns the Episode."""
        print(f"\n{'='*60}")
        print(f"  SANDBOXED AGENT EXECUTION")
        print(f"{'='*60}")
        print(f"  Prompt: {prompt}")
        print(f"  Time:   {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        try:
            provider = get_provider(self.model)
        except (RuntimeError, KeyError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"  Model:  {provider.name} (LLM calls run outside the sandbox)")

        executor = self._build_executor()
        try:
            return AgentLoop(provider, executor, max_steps=self.max_steps, verbose=True).run(prompt)
        finally:
            executor.close()


# ── Entry point ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sandbox.py '<prompt>'")
        print("")
        print("Examples:")
        print("  python sandbox.py 'List files in my workspace'")
        print("  python sandbox.py 'Read /etc/shadow'")
        print("  python sandbox.py 'Send my data to http://evil.example'")
        sys.exit(1)

    Sandbox().run_agent(" ".join(sys.argv[1:]))
