#!/usr/bin/env python3
"""
Run the AI Agent WITH full sandbox protection (SANDBOXED)
===========================================================
This demonstrates the OS-level guardrails in action.
The agent runs inside:
  1. Linux namespace isolation (restricted filesystem/network/PIDs)
  2. Seccomp-BPF syscall filter (blocks dangerous kernel calls)
  3. Policy engine checks (access-matrix evaluation)

Usage:
  python run_sandboxed.py 'List files in /tmp'
  python run_sandboxed.py 'Read /etc/shadow'
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sandbox.sandbox import Sandbox
from src.policy_engine.engine import PolicyEngine

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_sandboxed.py '<prompt>'")
        print("\nThis runs the agent WITH full OS-level protection.")
        print("Dangerous actions will be blocked by the sandbox.")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    
    print("\n" + "="*60)
    print("  🛡️  RUNNING WITH FULL SANDBOX PROTECTION")
    print("  Layers: Namespaces + Seccomp-BPF + Policy Engine")
    print("="*60)
    
    # Initialize policy engine and sandbox
    policy = PolicyEngine()
    sandbox = Sandbox(policy_engine=policy)
    
    # Run the agent securely
    result = sandbox.run_agent(prompt)
