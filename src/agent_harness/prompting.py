"""
Prompting — the single source of truth for how we talk to any LLM
=================================================================
Every provider (Gemini, Groq, Ollama, the from-scratch model) receives the
same system prompt and its reply is parsed by the same function, so model
comparisons are fair and the sandbox/agent never drift apart.
"""

import json

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

VALID_ACTIONS = {"run_shell", "read_file", "write_file", "http_request", "respond"}


class ActionParseError(ValueError):
    """The model's reply did not contain a usable JSON action."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def build_messages(user_prompt: str) -> list:
    """Chat messages for a single user request."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    """The object an action operates on (command, path, url, or message)."""
    return (action.get("command")
            or action.get("path")
            or action.get("url")
            or action.get("message", ""))
