"""
Qwen chat-render contract
=========================
The single place that decides how an episode becomes Qwen chat messages. The LoRA trainer renders its
training sequences with it and the provider (src/llm/hf_local.py) checks the adapter against it, so
what the model is trained on and what it reads at benchmark time cannot drift apart.

The rules, each with its reason:

  One message builder. System prompt and user turn come from prompting.build_messages (the project
  system prompt is always supplied, fairness across models). Each assistant turn is the compact JSON
  action; its observation becomes a user turn exactly as AgentLoop appends it
  (OBSERVATION_TEMPLATE over Observation.as_text()). The last turn is a respond, which has no
  observation.

  One renderer. The messages go through the tokenizer's own apply_chat_template, never through a
  hand-built string, and never with a tools argument: that would switch Qwen to its own <tools>
  block and break the plain-text tool list of the project system prompt. Without a system message
  Qwen would inject "You are Qwen, created by Alibaba Cloud", which is why the system prompt is
  never omitted.

  A fingerprint. render_fingerprint renders a fixed golden conversation, so any edit to
  SYSTEM_PROMPT, the tool-list format, OBSERVATION_TEMPLATE or compact_action changes it. The
  trainer stamps it into the adapter manifest and the provider refuses an adapter whose stamp
  differs from the running code (a stale adapter would measure the mismatch, not the model).

  Special-token strings are dangerous. A literal <|im_end|> inside message text is tokenized as
  the real control token, so attacker-controlled tool output could close a turn and open a fake
  assistant turn. Prompts are not sanitised (every model must see identical text); has_special_token
  lets the training-data guard and the tests prove the data contains none.

This module imports only prompting.py and format.py (pure Python), so evaluation-time code that
imports it never reaches the held-out tool list or the benchmark text.
"""

import hashlib
import json

from src.agent_harness.prompting import OBSERVATION_TEMPLATE, build_messages
from src.scratch_llm.format import compact_action

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
SPECIAL_TOKEN_STRINGS = (IM_START, IM_END, ENDOFTEXT)
RENDER_VERSION = 1


def episode_to_chat(ep: dict) -> list:
    """The chat messages of a whole episode, as an AgentLoop would hold them after its last step."""
    messages = build_messages(ep["user"], ep["tools"])
    for action, observation in ep["turns"]:
        messages.append({"role": "assistant", "content": compact_action(action)})
        if observation is not None:
            messages.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(result=observation)})
    return messages


def golden_chat() -> list:
    """A fixed conversation built at call time, so editing any rendering rule changes its render."""
    tool = {"name": "NoteSearchEntries", "summary": "Search the saved notes by keyword.",
            "parameters": [{"name": "keyword", "type": "string", "required": True},
                           {"name": "limit", "type": "integer", "required": False}]}
    messages = build_messages("Find the note called Budget and tell me what it says.", [tool])
    messages.append({"role": "assistant", "content": compact_action(
        {"action": "call_tool", "tool": "NoteSearchEntries", "args": {"keyword": "Budget"}})})
    messages.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(
        result='{"notes": [{"title": "Budget", "body": "Rent 900, food 300"}]}')})
    messages.append({"role": "assistant", "content": compact_action(
        {"action": "respond", "message": "The Budget note says: Rent 900, food 300."})})
    return messages


def render_fingerprint(tokenizer) -> str:
    """sha256 of the golden conversation rendered by this tokenizer's chat template."""
    text = tokenizer.apply_chat_template(golden_chat(), tokenize=False)
    payload = json.dumps({"render_version": RENDER_VERSION, "text": text}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chat_template_sha256(tokenizer) -> str:
    return hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()


def has_special_token(text: str) -> bool:
    return any(token in text for token in SPECIAL_TOKEN_STRINGS)
