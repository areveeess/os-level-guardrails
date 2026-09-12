#!/usr/bin/env python3
"""
Run the AI Agent WITHOUT any sandbox (UNSANDBOXED)
====================================================
This demonstrates what happens when an AI agent has
unrestricted access to the system. A prompt-injected
agent can read sensitive files, exfiltrate data, etc.

Usage:
  python run_agent.py 'List files in /tmp'
  python run_agent.py 'Read /etc/shadow'
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent_harness.agent import AIAgent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_agent.py '<prompt>'")
        print("\nThis runs the agent WITHOUT any protection.")
        print("The agent can execute ANY command the LLM generates.")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    
    print("\n" + "!"*60)
    print("  ⚠️  RUNNING WITHOUT SANDBOX — NO PROTECTION")
    print("  Any command the LLM generates will be executed.")
    print("!"*60)
    
    agent = AIAgent()  # No policy engine!
    result = agent.run(prompt)
