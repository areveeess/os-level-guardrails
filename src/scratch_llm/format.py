"""
Compact transcript format (version 2) + character tokenizer for the from-scratch model
=====================================================================================
The hosted models read the full chat (system prompt, tool specs, ...). A tiny
model can't afford that, so - like a chat template - it gets a compact view
of the same conversation:

    T:<Tool>(<required params>)|<Tool>(<required params>)|...
    U:<user instruction>
    A:<action JSON>
    O:<observation, flattened and truncated>
    A:<action JSON>          <- the model writes this line, ending with "\\n"

Version 2 puts each tool's REQUIRED parameter names on the T: line, in spec
order (T:GmailSendEmail(to,subject,body)). A model that has never seen a tool
can then still write its call, because every name it needs is in the prompt it
reads (copying, not memory). Optional parameters and types are left out: the
model only ever emits required arguments.

Training (dataset.py) and inference (inference.py) both go through
render_context(), so the model always sees exactly the format it learnt. A
checkpoint stores FORMAT_VERSION, VOCAB_SHA256, render_fingerprint(),
CONTEXT_CHARS and GEN_RESERVE, and inference refuses a checkpoint whose stamps
differ from the running code (plan 02-03 implements that check): a change to
this file that alters what the model reads must bump FORMAT_VERSION.

This module is torch-free and imports neither dataset nor the benchmark.
"""

import hashlib
import json
import re

FORMAT_VERSION = 2
CONTEXT_CHARS = 2048
GEN_RESERVE = 400          # characters kept free for the model's own next action
OBS_CHARS = 640            # an observation line is cut to this many characters
USER_CHARS = 240           # the U: line is cut to this many characters
HISTORY_ACTION_CHARS = 400  # an earlier A: line is cut to this many characters
ACTION_PREFIX = "A:"
END = "\n"

# Vocabulary: newline + printable ASCII. Anything else becomes '?'.
VOCAB = ["\n"] + [chr(c) for c in range(32, 127)]
STOI = {ch: i for i, ch in enumerate(VOCAB)}
UNKNOWN = STOI["?"]
VOCAB_SHA256 = hashlib.sha256("".join(VOCAB).encode()).hexdigest()

# Same grammar prompting.format_tool_specs writes: "- Name(p: type, q: type (optional)): summary".
# Copied (not imported) from src/llm/scripted.py so this module stays free of the llm package.
_TOOL_LINE = re.compile(r"^- (\w+)\((.*?)\): ", re.M)


def encode(text: str) -> list:
    return [STOI.get(ch, UNKNOWN) for ch in text]


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids)


def _flat(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit]


def compact_action(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def tools_from_system(system: str) -> list:
    """[(name, [required parameter names])] in listed order, read from the system prompt.

    Parsing the rendered prompt (not the spec) keeps training and inference consistent:
    a parameter whose "required" flag is missing is rendered as optional and is dropped here too.
    """
    _, marker, block = system.partition("Available tools:\n")
    if not marker:
        return []
    tools = []
    for name, params in _TOOL_LINE.findall(block):
        required = [p.split(":")[0].strip() for p in params.split(", ")
                    if p.strip() and not p.endswith(" (optional)")]
        tools.append((name, required))
    return tools


def render_header(tools: list, user: str) -> str:
    """tools = [(name, [required names])] -> the T: and U: lines."""
    listing = "|".join(f"{name}({','.join(required)})" for name, required in tools)
    return f"T:{listing}{END}U:{_flat(user, USER_CHARS)}{END}"


def render_turn(action_text: str, observation: str = None) -> str:
    turn = f"{ACTION_PREFIX}{action_text}{END}"
    if observation is not None:
        turn += f"O:{_flat(observation, OBS_CHARS)}{END}"
    return turn


def fit(header: str, turns: str, budget: int = CONTEXT_CHARS) -> str:
    """Keep the header; drop the oldest part of the turns if too long.

    The cut is moved forward to the next line boundary, so the kept turns always begin
    with a complete A: or O: line (a half line would teach the model a broken prompt).
    """
    room = budget - len(header)
    if len(turns) <= room:
        return header + turns
    if room <= 0:
        return header
    start = len(turns) - room
    tail = turns[start:]
    if turns[start - 1] != END:
        newline = tail.find(END)
        tail = tail[newline + 1:] if newline != -1 else ""
    return header + tail


def render_context(messages: list, budget: int = None) -> str:
    """Chat messages (as built by AgentLoop) -> compact prompt ending in 'A:'.

    budget is the size the header plus turns may take before old turns are dropped; the
    default leaves GEN_RESERVE characters for the model's reply. dataset.validate_episode
    passes a huge budget to measure the untruncated length.
    """
    from src.agent_harness.prompting import OBSERVATION_TEMPLATE
    obs_prefix = OBSERVATION_TEMPLATE.split("{result}")[0]
    if budget is None:
        budget = CONTEXT_CHARS - GEN_RESERVE

    tools = tools_from_system(messages[0]["content"])
    user = messages[1]["content"] if len(messages) > 1 else ""
    turns, pending_action = "", None
    for m in messages[2:]:
        if m["role"] == "assistant":
            pending_action = _flat(m["content"], HISTORY_ACTION_CHARS)
        elif pending_action is not None:
            obs = m["content"][len(obs_prefix):] if m["content"].startswith(obs_prefix) else m["content"]
            turns += render_turn(pending_action, obs)
            pending_action = None
    return fit(render_header(tools, user), turns, budget) + ACTION_PREFIX


def supervised_mask(text: str) -> list:
    """1 for every character the model must learn to write, else 0 (same length as text).

    The learnt characters are the payload of each A: line plus its newline. The A: prefix is
    supplied by the prompt at inference, so it is not supervised; T:, U: and O: lines are
    context the model reads, never text it writes.
    """
    mask, pos = [0] * len(text), 0
    for line in text.split(END):
        if line.startswith(ACTION_PREFIX):
            for i in range(pos + len(ACTION_PREFIX), min(pos + len(line) + 1, len(text))):
                mask[i] = 1
        pos += len(line) + 1
    return mask


# The golden conversation behind render_fingerprint(). The tool names exist nowhere in
# tools.json. The strings are long on purpose: the render has to hit every limit above
# (USER_CHARS, HISTORY_ACTION_CHARS, OBS_CHARS) and the fit budget, so editing any of them
# changes the fingerprint.
_GOLDEN_TOOLS = [
    {"name": "ZephyrLedgerFetchEntry", "summary": "Fetch one ledger entry.",
     "parameters": [{"name": "entry_ref", "type": "string", "required": True},
                    {"name": "revision", "type": "integer", "required": True},
                    {"name": "verbose", "type": "boolean", "required": False}]},
    {"name": "ZephyrLedgerListEntries", "summary": "List ledger entries.", "parameters": []},
]


def render_fingerprint() -> str:
    """sha256 of the render of a fixed conversation; changes whenever the rendering changes.

    It catches a silent edit to OBSERVATION_TEMPLATE, format_tool_specs, the limits or the
    fit rule that was not accompanied by a FORMAT_VERSION bump. The observation message below
    uses the literal text of the current template on purpose: if the template changes, the
    prefix strip in render_context stops matching and the render (and so the hash) changes.
    """
    from src.agent_harness.prompting import build_messages
    messages = build_messages(("fetch the ledger entry for the quarterly audit and report back to me "
                               "when the revision number is known " * 4), _GOLDEN_TOOLS)
    for n in range(3):
        action = compact_action({"action": "run_shell", "command": f"echo {n} " + "x" * 440})
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user",
                         "content": "Result of your last action:\n" + f"line {n} " + "y" * 700})
    return hashlib.sha256(render_context(messages).encode("utf-8")).hexdigest()
