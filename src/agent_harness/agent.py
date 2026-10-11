"""
AI Agent Harness — The Untrusted Principal
============================================
This module creates a tool-using AI agent powered by any configured LLM
(Gemini, Groq, Ollama, or the from-scratch model — see config/models.yaml).

How it works:
  1. User gives a natural language prompt (e.g., 'List files in /tmp')
  2. The agent sends this to the LLM
  3. The LLM responds with a structured action (e.g., {'action': 'run_shell', 'command': 'ls /tmp'})
  4. The agent executes that action using ToolExecutor
  5. The result is returned

Why this is the 'untrusted principal':
  The LLM can be tricked via prompt injection to generate ANY action,
  including malicious ones. Without the sandbox and policy engine,
  the agent will blindly execute whatever the LLM says.

Usage:
  python agent.py 'List all files in /tmp'
  python agent.py --model groq-llama-70b 'What is my username?'
"""

import os
import sys
import datetime

# Add the project root to Python's import path so we can import sibling modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_harness.tools import ToolExecutor
from src.agent_harness.prompting import build_messages, parse_action, action_target, ActionParseError
from src.llm import get_provider


class AIAgent:
    """A tool-using AI agent powered by a pluggable LLM provider.
    
    Attributes:
        provider: The LLMProvider that turns prompts into actions.
        tools: The ToolExecutor that actually runs system operations.
        policy_engine: Optional PolicyEngine for checking actions before execution.
    """

    def __init__(self, policy_engine=None, model: str = None, provider=None):
        """Initialize the agent with an LLM provider and optional policy engine.
        
        Args:
            policy_engine: If provided, every action is checked against
                          the policy before execution. Pass None to run
                          without any restrictions (dangerous!).
            model: Model name from config/models.yaml (default: its `default`).
            provider: A ready-made LLMProvider (overrides `model`; used in tests).
        """
        try:
            self.provider = provider or get_provider(model)
        except (RuntimeError, KeyError) as e:
            print(f"ERROR: {e}")
            print("Copy .env.example to .env and add the API key for this model.")
            sys.exit(1)
        self.tools = ToolExecutor()
        self.policy_engine = policy_engine

    def propose_action(self, user_prompt: str) -> dict:
        """Ask the LLM for an action. Returns the action dict, or an error result
        dict with status 'error' (LLM failure or unparseable reply)."""
        response = self.provider.generate(build_messages(user_prompt))
        if not response.ok:
            print(f"  ✗ LLM call failed: {response.error}")
            return {"status": "error", "message": f"LLM error: {response.error}"}
        print(f"  LLM says ({response.latency_ms:.0f} ms): {response.text}")
        try:
            return parse_action(response.text)
        except ActionParseError as e:
            print(f"  ✗ Could not parse LLM response: {e}")
            return {"status": "error", "message": "Invalid LLM response", "raw": response.text}

    def run(self, user_prompt: str) -> dict:
        """Process a user prompt end-to-end: LLM → parse → check policy → execute.
        
        Args:
            user_prompt: Natural language instruction from the user.
            
        Returns:
            Dict with the execution result or block decision.
        """
        print(f"\n{'='*60}")
        print(f"  AGENT HARNESS")
        print(f"{'='*60}")
        print(f"  Prompt: {user_prompt}")
        print(f"  Time:   {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"  Policy: {'ENABLED' if self.policy_engine else 'DISABLED (no sandbox)'}")
        print(f"{'='*60}")

        # ── Step 1 + 2: Ask the LLM what action to take, then parse it ──
        print(f"\n[1] Consulting LLM ({self.provider.name})...")
        action = self.propose_action(user_prompt)
        if action.get("status") == "error":
            return action

        print("\n[2] Parsed action:")
        action_type = action.get("action", "unknown")
        target = action_target(action)
        print(f"  Action: {action_type}")
        print(f"  Target: {target}")

        # ── Step 3: Check policy (if engine is attached) ──
        if self.policy_engine:
            print("\n[3] Checking policy engine...")
            decision = self.policy_engine.check(action_type, target)

            if decision == "BLOCK":
                print(f"  ❌ BLOCKED by policy engine")
                return {
                    "status": "blocked",
                    "action": action_type,
                    "target": target,
                    "decision": "BLOCK"
                }
            elif decision == "FLAG":
                print(f"  ⚠️  FLAGGED (allowed but logged as suspicious)")
            else:
                print(f"  ✅ ALLOWED by policy engine")
        else:
            print("\n[3] No policy engine — executing without checks ⚠️")

        # ── Step 4: Execute the action ──
        print(f"\n[4] Executing: {action_type}...")
        result = self.tools.execute(action)

        # Print result summary
        if result.get("status") == "success":
            output = result.get("stdout") or result.get("content") or result.get("message", "")
            # Truncate long output for display
            if len(str(output)) > 500:
                print(f"  Result (truncated): {str(output)[:500]}...")
            else:
                print(f"  Result: {output}")
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")

        print(f"\n{'='*60}")
        return result


# ── Entry point: run directly from command line ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<prompt>'")
        print("")
        print("Examples:")
        print("  python agent.py 'List all files in /tmp'")
        print("  python agent.py 'What is my username?'")
        print("  python agent.py --model ollama-llama3.2-3b 'Read the file /etc/hostname'")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Run the agent without protection")
    parser.add_argument("--model", help="Model name from config/models.yaml")
    parser.add_argument("prompt", nargs="+")
    args = parser.parse_args()
    agent = AIAgent(model=args.model)  # No policy engine = no restrictions!
    prompt = " ".join(args.prompt)
    agent.run(prompt)
