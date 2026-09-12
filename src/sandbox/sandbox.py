"""
Sandbox Wrapper — OS-Level Isolation
======================================
This module provides the OS-level security boundary for the AI agent.
It combines THREE layers of defense in the correct order:

  ARCHITECTURE (Two-Phase Execution):
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 1: OUTSIDE SANDBOX (needs network for LLM)      │
  │    1. Agent sends prompt to Gemini LLM                  │
  │    2. LLM responds with an action (e.g., "read file")   │
  │    3. Policy Engine checks action against access-matrix  │
  │    4. If BLOCKED → stop here, log, done                 │
  │    5. If ALLOWED → proceed to Phase 2                   │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 2: INSIDE SANDBOX (isolated tool execution)      │
  │    1. Create namespace (user, PID, net isolation)        │
  │    2. Apply seccomp-BPF filter (block dangerous syscalls)│
  │    3. Execute ONLY the tool command inside this jail     │
  │    4. Return result to parent                           │
  └─────────────────────────────────────────────────────────┘

  Why two phases?
    The LLM call needs network access (to reach Gemini API).
    But tool execution must be isolated (no network, no /etc/shadow).
    So we split: LLM call outside, tool execution inside.

  Layer 1: Policy Engine (Application-Level Pre-check)
    - Checks every action against the access-matrix BEFORE execution
    - Provides readable decision logs for the dashboard
    - Acts as the first line of defense with detailed reasoning

  Layer 2: Linux Namespaces (Environment Isolation)
    - PID namespace: agent can only see its own processes
    - Network namespace: tool execution has no network access
    - User namespace: runs as unprivileged user

  Layer 3: Seccomp-BPF (Syscall Filtering)
    - Blocks dangerous syscalls (chmod, mount, ptrace, setuid, etc.)
    - Enforced by the kernel — cannot be bypassed from userspace
    - Even if the policy engine misses something, seccomp catches it

Usage:
  from src.sandbox.sandbox import Sandbox
  from src.policy_engine.engine import PolicyEngine

  sandbox = Sandbox(PolicyEngine())
  sandbox.run_agent('List files in /tmp')
"""

import os
import sys
import subprocess
import json
import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_harness.agent import AIAgent
from src.agent_harness.tools import ToolExecutor
from src.policy_engine.engine import PolicyEngine


def is_linux() -> bool:
    """Check if we're running on Linux (required for namespaces/seccomp)."""
    return sys.platform.startswith('linux')


def get_seccomp_filter_path() -> str:
    """Get the path to the compiled seccomp filter binary."""
    return os.path.join(os.path.dirname(__file__), 'seccomp_filter')


class Sandbox:
    """OS-level sandbox for the AI agent.

    Uses a two-phase architecture:
      Phase 1 (outside sandbox): LLM call + policy check
      Phase 2 (inside sandbox):  Tool execution with namespace + seccomp
    """

    def __init__(self, policy_engine: PolicyEngine = None):
        """Initialize the sandbox.

        Args:
            policy_engine: PolicyEngine instance for application-level checks.
                          If None, a default engine is created.
        """
        self.policy = policy_engine or PolicyEngine()
        self.seccomp_path = get_seccomp_filter_path()
        self.namespace_supported = is_linux()
        self.seccomp_available = os.path.isfile(self.seccomp_path)

        print(f"\n{'='*60}")
        print(f"  SANDBOX INITIALIZATION")
        print(f"{'='*60}")
        print(f"  Platform:   {'Linux ✓' if self.namespace_supported else 'Not Linux ✗'}")
        print(f"  Namespaces: {'Available ✓' if self.namespace_supported else 'Not available (non-Linux)'}")
        print(f"  Seccomp:    {'Compiled ✓' if self.seccomp_available else 'Not compiled — run: cd src/sandbox && make'}")
        print(f"  Policy:     Loaded ✓")
        print(f"{'='*60}")

    def run_agent(self, prompt: str) -> dict:
        """Run the AI agent with sandbox protection.

        Two-phase execution:
          Phase 1: LLM call + policy check (outside sandbox, needs network)
          Phase 2: Tool execution (inside sandbox, isolated)

        Args:
            prompt: The user's natural language prompt.

        Returns:
            Dict with the execution result.
        """
        print(f"\n{'='*60}")
        print(f"  SANDBOXED AGENT EXECUTION")
        print(f"{'='*60}")
        print(f"  Prompt: {prompt}")
        print(f"  Time:   {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # ── Phase 1: LLM Call (outside sandbox — needs network) ──
        print("\n[PHASE 1] Consulting Gemini LLM (outside sandbox)...")
        agent = AIAgent(policy_engine=self.policy)

        try:
            import google.generativeai as genai
            response = agent.model.generate_content(
                f"{agent.SYSTEM_PROMPT if hasattr(agent, 'SYSTEM_PROMPT') else ''}"
                f"You are a helpful AI assistant that executes system commands.\n"
                f"Respond with ONLY a JSON object.\n"
                f"Available actions:\n"
                f'- {{"action": "run_shell", "command": "<cmd>"}}\n'
                f'- {{"action": "read_file", "path": "<path>"}}\n'
                f'- {{"action": "write_file", "path": "<path>", "content": "<text>"}}\n'
                f'- {{"action": "http_request", "url": "<url>"}}\n'
                f'- {{"action": "respond", "message": "<text>"}}\n\n'
                f"User request: {prompt}"
            )
            raw_text = response.text.strip()
        except Exception as e:
            print(f"  ✗ LLM call failed: {e}")
            return {"status": "error", "message": f"LLM error: {e}"}

        # Clean markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        print(f"  LLM response: {raw_text}")

        # ── Parse the action ──
        print("\n[PHASE 1] Parsing action...")
        try:
            action = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"  ✗ Could not parse LLM response as JSON")
            return {"status": "error", "message": "Invalid LLM response", "raw": raw_text}

        action_type = action.get("action", "unknown")
        target = (action.get("command")
                  or action.get("path")
                  or action.get("url")
                  or action.get("message", ""))
        print(f"  Action: {action_type}")
        print(f"  Target: {target}")

        # ── Phase 1: Policy Check (access-matrix evaluation) ──
        print("\n[PHASE 1] Checking policy engine (access-matrix)...")
        decision = self.policy.check(action_type, target)

        if decision == "BLOCK":
            print(f"\n  ╔══════════════════════════════════════════╗")
            print(f"  ║  ❌ BLOCKED BY POLICY ENGINE              ║")
            print(f"  ║  Action: {action_type:<33}║")
            print(f"  ║  Target: {str(target)[:33]:<33}║")
            print(f"  ║  The action was denied before execution. ║")
            print(f"  ╚══════════════════════════════════════════╝\n")
            return {
                "status": "blocked",
                "action": action_type,
                "target": target,
                "decision": "BLOCK",
                "phase": "policy_engine"
            }
        elif decision == "FLAG":
            print(f"  ⚠️  FLAGGED: Action is suspicious but allowed for now")
        else:
            print(f"  ✅ ALLOWED: Action passes policy check")

        # ── Phase 2: Sandboxed Tool Execution ──
        print(f"\n[PHASE 2] Executing tool inside sandbox...")

        if action_type == "respond":
            # Text responses don't need sandboxing
            msg = action.get("message", "")
            print(f"  Response: {msg}")
            return {"status": "success", "message": msg}

        if self.namespace_supported and self.seccomp_available:
            return self._execute_in_full_sandbox(action)
        elif self.namespace_supported:
            return self._execute_in_namespace(action)
        else:
            return self._execute_with_policy_only(action)

    def _execute_in_full_sandbox(self, action: dict) -> dict:
        """Execute a tool action inside namespace + seccomp sandbox.

        The tool command runs in an isolated environment:
        - User namespace (unprivileged)
        - PID namespace (can't see host processes)
        - Network namespace (no network access — blocks data exfiltration)
        - Seccomp-BPF (blocks chmod, mount, ptrace, setuid, etc.)
        """
        print("[PHASE 2] Mode: FULL (namespace + seccomp + policy)")

        action_type = action.get("action", "")
        target = (action.get("command")
                  or action.get("path")
                  or action.get("url", ""))

        # Build the command to execute inside the sandbox
        if action_type == "run_shell":
            inner_cmd = action.get("command", "echo 'no command'")
        elif action_type == "read_file":
            inner_cmd = f"cat {action.get('path', '')}"
        elif action_type == "write_file":
            content = action.get("content", "").replace("'", "'\\''")
            inner_cmd = f"echo '{content}' > {action.get('path', '/dev/null')}"
        elif action_type == "http_request":
            inner_cmd = f"curl -s {action.get('url', '')}"
        else:
            inner_cmd = "echo 'unknown action'"

        # Wrap with: unshare (namespaces) → seccomp_filter → sh -c "command"
        cmd = [
            'unshare',
            '--user',              # New user namespace (no root needed)
            '--map-root-user',     # Map current user to root in namespace
            '--pid', '--fork',     # New PID namespace (process isolation)
            '--net',               # New network namespace (NO network!)
            self.seccomp_path,     # Apply seccomp-BPF filter
            'sh', '-c', inner_cmd  # The actual command to run
        ]

        print(f"  Command:    {inner_cmd}")
        print(f"  Namespaces: user, pid, net (network isolated)")
        print(f"  Seccomp:    Active (chmod, mount, ptrace, setuid blocked)")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=PROJECT_ROOT
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0 and stdout:
                print(f"\n  ✅ Execution result:")
                # Truncate long output
                if len(stdout) > 500:
                    print(f"  {stdout[:500]}... (truncated)")
                else:
                    print(f"  {stdout}")
            elif stderr:
                # Filter noise from namespace/seccomp
                error_lines = [l for l in stderr.split('\n')
                               if l.strip() and 'ioctl' not in l.lower()]
                if error_lines:
                    print(f"\n  🔴 Sandbox blocked or error occurred:")
                    for line in error_lines[:5]:
                        print(f"    {line}")
            elif result.returncode != 0:
                print(f"\n  🔴 Command failed (exit code {result.returncode})")
                print(f"    The sandbox may have blocked a syscall.")

            return {
                "status": "success" if result.returncode == 0 else "blocked_by_sandbox",
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "sandbox_mode": "full"
            }

        except subprocess.TimeoutExpired:
            print("  ✗ Command timed out (30s limit)")
            return {"status": "error", "message": "Timed out"}
        except FileNotFoundError:
            print("  ✗ 'unshare' not found. Falling back to direct execution.")
            return self._execute_with_policy_only(action)
        except Exception as e:
            print(f"  ✗ Sandbox error: {e}")
            print("  Falling back to direct execution with policy checks.")
            return self._execute_with_policy_only(action)

    def _execute_in_namespace(self, action: dict) -> dict:
        """Execute tool in namespace only (no seccomp)."""
        print("[PHASE 2] Mode: PARTIAL (namespace only, no seccomp)")

        action_type = action.get("action", "")
        if action_type == "run_shell":
            inner_cmd = action.get("command", "echo 'no command'")
        elif action_type == "read_file":
            inner_cmd = f"cat {action.get('path', '')}"
        else:
            # Fall back to direct execution for other types
            return self._execute_with_policy_only(action)

        cmd = [
            'unshare', '--user', '--map-root-user',
            '--pid', '--fork', '--net',
            'sh', '-c', inner_cmd
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
            )
            stdout = result.stdout.strip()
            if stdout:
                print(f"  Result: {stdout[:500]}")
            return {"status": "success" if result.returncode == 0 else "error",
                    "stdout": stdout, "sandbox_mode": "namespace_only"}
        except Exception as e:
            print(f"  Namespace error: {e}")
            return self._execute_with_policy_only(action)

    def _execute_with_policy_only(self, action: dict) -> dict:
        """Execute tool with policy checks only (no OS isolation).

        Fallback when namespaces/seccomp aren't available.
        """
        print("[PHASE 2] Mode: POLICY-ONLY (no OS isolation)")
        tools = ToolExecutor()
        result = tools.execute(action)

        if result.get("status") == "success":
            output = result.get("stdout") or result.get("content") or result.get("message", "")
            if len(str(output)) > 500:
                print(f"  Result (truncated): {str(output)[:500]}...")
            else:
                print(f"  Result: {output}")
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")

        return result


# ── Entry point ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sandbox.py '<prompt>'")
        print("")
        print("Examples:")
        print("  python sandbox.py 'List files in /tmp'")
        print("  python sandbox.py 'Read /etc/shadow'")
        print("  python sandbox.py 'Send my data to http://evil.com'")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    sandbox = Sandbox()
    sandbox.run_agent(prompt)
