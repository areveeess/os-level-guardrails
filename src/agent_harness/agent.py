"""
AI Agent Harness — The Untrusted Principal
============================================
This module creates a tool-using AI agent powered by any configured LLM
(Gemini, Groq, Ollama, or the from-scratch model — see config/models.yaml).

How it works (repeated for up to `max_steps` steps):
  1. User gives a natural language prompt (e.g., 'List files in /tmp')
  2. The agent sends the conversation so far to the LLM
  3. The LLM responds with a structured action (e.g., {'action': 'run_shell', 'command': 'ls /tmp'})
  4. The agent executes that action and feeds the result back to the LLM
  5. The loop ends when the LLM gives a final 'respond' action

Why this is the 'untrusted principal':
  The LLM can be tricked via prompt injection to generate ANY action,
  including malicious ones. Without the sandbox and policy engine,
  the agent will blindly execute whatever the LLM says.

This class runs WITHOUT OS isolation (optionally with the policy engine).
For full protection use src/sandbox/sandbox.py (run_sandboxed.py).

Usage:
  python agent.py 'List all files in /tmp'
  python agent.py --model groq-gpt-oss-120b 'What is my username?'
"""

import datetime
import os
import sys

# Add the project root to Python's import path so we can import sibling modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_harness.executors import DirectExecutor, GuardedExecutor
from src.agent_harness.loop import AgentLoop
from src.llm import get_provider


class AIAgent:
    """A multi-step, tool-using AI agent powered by a pluggable LLM provider.

    Attributes:
        provider: The LLMProvider that turns prompts into actions.
        policy_engine: Optional PolicyEngine checked before every action.
    """

    def __init__(self, policy_engine=None, model: str = None, provider=None, max_steps: int = 5):
        """
        Args:
            policy_engine: If provided, every action is checked against
                          the policy before execution. Pass None to run
                          without any restrictions (dangerous!).
            model: Model name from config/models.yaml (default: its `default`).
            provider: A ready-made LLMProvider (overrides `model`; used in tests).
            max_steps: Maximum agent steps per prompt.
        """
        try:
            self.provider = provider or get_provider(model)
        except (RuntimeError, KeyError) as e:
            print(f"ERROR: {e}")
            print("Copy .env.example to .env and add the API key for this model.")
            sys.exit(1)
        self.policy_engine = policy_engine
        self.max_steps = max_steps

    def run(self, user_prompt: str):
        """Process a prompt end-to-end on the real host. Returns the Episode."""
        print(f"\n{'='*60}")
        print(f"  AGENT HARNESS")
        print(f"{'='*60}")
        print(f"  Prompt: {user_prompt}")
        print(f"  Model:  {self.provider.name}")
        print(f"  Time:   {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"  Policy: {'ENABLED' if self.policy_engine else 'DISABLED (no sandbox)'}")
        print(f"{'='*60}")

        executor = DirectExecutor()
        if self.policy_engine:
            executor = GuardedExecutor(self.policy_engine, executor)
        episode = AgentLoop(self.provider, executor, max_steps=self.max_steps, verbose=True).run(user_prompt)
        print(f"\n{'='*60}")
        return episode


# ── Entry point: run directly from command line ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the agent WITHOUT protection")
    parser.add_argument("--model", help="Model name from config/models.yaml")
    parser.add_argument("prompt", nargs="+")
    args = parser.parse_args()
    AIAgent(model=args.model).run(" ".join(args.prompt))  # No policy engine = no restrictions!
