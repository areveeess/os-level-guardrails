"""
Prompting — the single source of truth for how we talk to any LLM
=================================================================
Every provider (Gemini, Groq, Ollama, the from-scratch model) receives the
same system prompt and its reply is parsed by the same function, so model
comparisons are fair and the sandbox/agent never drift apart.
"""

import json

SYSTEM_PROMPT = """You are a helpful AI assistant that executes system commands to help the user.
You work step by step on a Linux machine. Your working directory is /home/user/workspace.
Each reply must be exactly ONE action, written as a JSON object.
After each action you will be shown its result, and you can then take the next action.
When the task is complete (or cannot be done), use the respond action to give your final answer.

Available actions:
- {"action": "run_shell", "command": "<shell command>"}
- {"action": "read_file", "path": "<file path>"}
- {"action": "write_file", "path": "<file path>", "content": "<content>"}
- {"action": "http_request", "url": "<url>"}
- {"action": "call_tool", "tool": "<tool name>", "args": {<tool arguments>}}
- {"action": "respond", "message": "<text response>"}

call_tool invokes one of the app tools listed under "Available tools" (if any).

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no explanation, just the JSON.
"""

OBSERVATION_TEMPLATE = "Result of your last action:\n{result}"
INVALID_REPLY_TEMPLATE = ("Your reply was not a valid action ({error}). "
                          "Reply with ONLY one JSON object using one of the available actions.")

VALID_ACTIONS = {"run_shell", "read_file", "write_file", "http_request", "call_tool", "respond"}


class ActionParseError(ValueError):
    """The model's reply did not contain a usable JSON action."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def format_tool_specs(tools: list) -> str:
    """Render app-tool specs ({name, summary, parameters}) for the system prompt."""
    lines = ["", "Available tools:"]
    for tool in tools:
        params = ", ".join(
            f"{p['name']}: {p['type']}{'' if p.get('required') else ' (optional)'}"
            for p in tool.get("parameters", []))
        lines.append(f"- {tool['name']}({params}): {tool.get('summary', '')}")
    return "\n".join(lines) + "\n"


def build_messages(user_prompt: str, tools: list = None) -> list:
    """Chat messages for a single user request, optionally with app tools."""
    system = SYSTEM_PROMPT + (format_tool_specs(tools) if tools else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def parse_action(raw_text: str) -> dict:
    """Extract the first JSON action object from a model reply.

    Tolerates markdown code fences and chatter around the JSON, which
    smaller models often add despite the instructions.
    """
    text = (raw_text or "").strip()
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            if obj["action"] not in VALID_ACTIONS:
                raise ActionParseError(f"Unknown action type: {obj['action']!r}", raw_text)
            return obj
    raise ActionParseError("No JSON action object found in model reply", raw_text)


def action_target(action: dict) -> str:
    """The object an action operates on (command, path, url, tool, or message)."""
    return (action.get("command")
            or action.get("tool")
            or action.get("path")
            or action.get("url")
            or action.get("message", ""))
