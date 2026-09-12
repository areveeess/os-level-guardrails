"""
AI Agent Harness — The Untrusted Principal
============================================
This module creates a tool-using AI agent powered by Google Gemini.

How it works:
  1. User gives a natural language prompt (e.g., 'List files in /tmp')
  2. The agent sends this to the Gemini LLM
  3. Gemini responds with a structured action (e.g., {'action': 'run_shell', 'command': 'ls /tmp'})
  4. The agent executes that action using ToolExecutor
  5. The result is returned

Why this is the 'untrusted principal':
  The LLM can be tricked via prompt injection to generate ANY action,
  including malicious ones. Without the sandbox and policy engine,
  the agent will blindly execute whatever the LLM says.

Usage:
  python agent.py 'List all files in /tmp'
  python agent.py 'What is my username?'
"""

import os
import sys
import json
import datetime

# ── Load API keys from .env file (if present) ────────────────────────────────
# This lets you store GEMINI_API_KEY in a .env file instead of exporting it
# manually every time. The .env file is in .gitignore so it stays secret.
try:
    from dotenv import load_dotenv
    # Look for .env starting from this file, going up to project root
    _PROJECT_ROOT_FOR_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    load_dotenv(os.path.join(_PROJECT_ROOT_FOR_ENV, '.env'))
except ImportError:
    pass  # python-dotenv not installed yet; fall back to manual export

# Add the project root to Python's import path so we can import sibling modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_harness.tools import ToolExecutor

# Attempt to import the Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not installed.")
    print("Fix:   pip install google-generativeai")
    sys.exit(1)


# The system prompt tells the LLM how to behave and what format to respond in.
# It instructs the LLM to always return a JSON action object.
SYSTEM_PROMPT = """You are a helpful AI assistant that executes system commands to help the user.
When the user asks you to do something, respond with a JSON object describing the action.

Available actions:
- {"action": "run_shell", "command": "<shell command>"}
- {"action": "read_file", "path": "<file path>"}
- {"action": "write_file", "path": "<file path>", "content": "<content>"}
- {"action": "http_request", "url": "<url>"}
- {"action": "respond", "message": "<text response>"}

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no explanation, just the JSON.
"""


class AIAgent:
    """A tool-using AI agent powered by Google Gemini.
    
    Attributes:
        model: The Gemini generative model instance.
        tools: The ToolExecutor that actually runs system operations.
        policy_engine: Optional PolicyEngine for checking actions before execution.
    """

    def __init__(self, policy_engine=None):
        """Initialize the agent with Gemini API and optional policy engine.
        
        Args:
            policy_engine: If provided, every action is checked against
                          the policy before execution. Pass None to run
                          without any restrictions (dangerous!).
        """
        # Check for API key (loaded from .env file or environment)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not found.")
            print("")
            print("Option A — Use a .env file (recommended, stays secret):")
            print("  1. Copy .env.example to .env")
            print("  2. Open .env and paste your real key")
            print("")
            print("Option B — Export it manually (only lasts for this terminal session):")
            print("  export GEMINI_API_KEY='your-key-here'")
            print("")
            print("Get a free key at: https://aistudio.google.com/apikey")
            sys.exit(1)

        # Configure the Gemini SDK
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3.6-flash")
        self.tools = ToolExecutor()
        self.policy_engine = policy_engine

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

        # ── Step 1: Ask the LLM what action to take ──
        print("\n[1] Consulting Gemini LLM...")
        try:
            response = self.model.generate_content(
                f"{SYSTEM_PROMPT}\n\nUser request: {user_prompt}"
            )
            raw_text = response.text.strip()
        except Exception as e:
            print(f"  ✗ LLM call failed: {e}")
            return {"status": "error", "message": f"LLM error: {e}"}

        # Clean markdown code fences if the LLM wraps its response
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        print(f"  LLM says: {raw_text}")

        # ── Step 2: Parse the JSON action ──
        print("\n[2] Parsing action...")
        try:
            action = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"  ✗ Could not parse LLM response as JSON")
            return {"status": "error", "message": "Invalid LLM response", "raw": raw_text}

        action_type = action.get("action", "unknown")
        # Determine the 'target' based on the action type
        target = (action.get("command")
                  or action.get("path")
                  or action.get("url")
                  or action.get("message", ""))
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
        print("  python agent.py 'Read the file /etc/hostname'")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    agent = AIAgent()  # No policy engine = no restrictions!
    agent.run(prompt)
