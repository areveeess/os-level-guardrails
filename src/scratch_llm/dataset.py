"""
Training data for ScratchGPT - episode schema, loss text, leakage gate and generator
====================================================================================
generate_episodes() writes synthetic agent episodes; train.py turns each one into the
compact transcript (format.py) and trains the model to write the `A:` lines. This module
is torch-free and is the only place that knows the held-out benchmark list.

Episode (a plain JSON object):

    {
      "id":    "train-0-000001",
      "split": "train",                 # one of SPLITS
      "cls":   "benign",                # "benign" | "dh" | "ds"
      "tools": [                        # full specs, exactly what build_messages() formats,
          {"name": ..., "summary": ...,     # in the order they are offered
           "parameters": [{"name": ..., "type": ..., "required": true}, ...]}],
      "user":  "the user's request",    # the U: line
      "turns": [                        # in order
          [{"action": "call_tool", "tool": "...", "args": {...}}, "tool result text"],
          [{"action": "respond", "message": "..."}, null]],   # last turn: no observation
      "meta":  {"synthetic": false, "os_actions": false, "blocked_tail": false,
                "two_step": false, "template_ids": ["user:t03", ...]}
    }

Training text is the render of the conversation, never a second renderer:
    training_text(ep) = render_context(messages before the last step) + last action + "\\n"
so every step's inference prompt is an exact prefix of the text the model is trained on.

Leakage design (SCRATCH-02, decision D-04): the model must never have seen a benchmark
tool, address, canary, sentence or parameter-name-as-tool-parameter, otherwise a benchmark
score would measure recall, not behaviour. The gate (assert_clean) derives everything from
benchmarks/injecagent at run time and runs inside generate_episodes, so leaking data cannot
be written. Held-only parameter names (names that occur only in held-out tools) are rejected
as tool parameter names and as call_tool argument keys, never as ordinary words (see
HELD_ONLY_GATE_SCOPE).
"""

import copy
import functools
import hashlib
import json
import os

from src.agent_harness.executors import Observation
from src.agent_harness.prompting import OBSERVATION_TEMPLATE, build_messages, parse_action
from src.benchmark.injecagent import DATA_DIR, EXFIL_TOOL, load_tool_specs
from src.scratch_llm.format import (ACTION_PREFIX, CONTEXT_CHARS, END, GEN_RESERVE, HISTORY_ACTION_CHARS,
                                    OBS_CHARS, USER_CHARS, _flat, compact_action, render_context)

GENERATOR_VERSION = 1
SPLITS = ("train", "dev_tools", "dev_synthetic")
CLASSES = ("benign", "dh", "ds")
EPISODE_KEYS = ("id", "split", "cls", "tools", "user", "turns", "meta")
META_KEYS = ("synthetic", "os_actions", "blocked_tail", "two_step", "template_ids")
MAX_TURNS = 5   # AgentLoop's default max_steps: an episode longer than this could never be replayed


class LeakageError(ValueError):
    """Training data contains something from the held-out benchmark."""


# ── held-out sets (derived from the benchmark files, never typed in) ──

def held_out_tools() -> set:
    """Every tool the benchmark uses (InjecAgent user + attacker tools, and the exfiltration tool)."""
    return set(_held_out())


@functools.lru_cache(maxsize=None)
def _held_out() -> frozenset:
    names = {EXFIL_TOOL}
    with open(os.path.join(DATA_DIR, "user_cases.jsonl"), encoding="utf-8") as f:
        names |= {json.loads(line)["User Tool"] for line in f if line.strip()}
    for kind in ("dh", "ds"):
        with open(os.path.join(DATA_DIR, f"attacker_cases_{kind}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    names |= set(json.loads(line)["Attacker Tools"])
    return frozenset(names)


def held_out_sha256() -> str:
    """Stamp for manifests and checkpoints: which held-out list the data was gated against."""
    return hashlib.sha256("\n".join(sorted(_held_out())).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def _tool_extras() -> dict:
    """name -> {"returns": [...], "toolkit": str} read from tools.json (load_tool_specs drops them)."""
    with open(os.path.join(DATA_DIR, "tools.json"), encoding="utf-8") as f:
        toolkits = json.load(f)
    return {tk["toolkit"] + tool["name"]: {"returns": tool.get("returns", []), "toolkit": tk["toolkit"]}
            for tk in toolkits for tool in tk["tools"]}


def toolkit_of(name: str) -> str:
    """The toolkit prefix of a real tool name ('' for a name that is not in tools.json)."""
    return _tool_extras().get(name, {}).get("toolkit", "")


@functools.lru_cache(maxsize=None)
def _held_only_params() -> frozenset:
    held = _held_out()
    specs = load_tool_specs()
    in_held = {p["name"] for n in held if n in specs for p in specs[n]["parameters"]}
    in_free = {p["name"] for n, s in specs.items() if n not in held for p in s["parameters"]}
    return frozenset(in_held - in_free)


def held_only_param_names() -> set:
    """Parameter names that occur only in held-out tools (D-04: never a tool parameter in training)."""
    return set(_held_only_params())


def episode_tool(spec: dict) -> dict:
    """The part of a tool spec an episode stores: exactly what build_messages() renders."""
    return {"name": spec["name"], "summary": spec.get("summary", ""),
            "parameters": [{"name": p["name"], "type": p["type"], "required": bool(p.get("required"))}
                           for p in spec.get("parameters", [])]}


@functools.lru_cache(maxsize=None)
def _free_pool() -> tuple:
    """Sorted ((name, spec-with-returns), ...) for every real tool that is not held out."""
    held = _held_out()
    extras = _tool_extras()
    specs = load_tool_specs()
    return tuple((n, dict(specs[n], returns=extras[n]["returns"])) for n in sorted(specs) if n not in held)


def free_tool_specs() -> dict:
    """name -> spec (name, summary, parameters, returns) for the 251 real non-benchmark tools, sorted by name."""
    return {name: copy.deepcopy(spec) for name, spec in _free_pool()}


# ── episodes -> messages -> training text ──

def episode_to_messages(ep: dict, upto: int) -> list:
    """The chat messages an AgentLoop would hold just before step `upto` (0-based).

    Built with the harness's own build_messages and OBSERVATION_TEMPLATE, so this can never
    drift from what a real conversation looks like.
    """
    messages = build_messages(ep["user"], ep["tools"])
    for action, observation in ep["turns"][:upto]:
        messages.append({"role": "assistant", "content": compact_action(action)})
        messages.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(result=observation)})
    return messages


def training_text(ep: dict) -> str:
    """The single training sequence of an episode: the last step's prompt plus its gold action."""
    last = len(ep["turns"]) - 1
    return render_context(episode_to_messages(ep, last)) + compact_action(ep["turns"][last][0]) + END


# ── validation ──

def _fail(ep: dict, rule: str):
    raise ValueError(f"{ep.get('id', '?')}: {rule}")


def validate_episode(ep: dict):
    """Raise ValueError (naming the episode id and the rule) if an episode doesn't follow the format."""
    if not isinstance(ep, dict) or any(k not in ep for k in EPISODE_KEYS):
        raise ValueError(f"{ep.get('id', '?') if isinstance(ep, dict) else '?'}: episode needs the keys {EPISODE_KEYS}")
    if not isinstance(ep["id"], str) or not ep["id"]:
        _fail(ep, "id must be a non-empty string")
    if ep["split"] not in SPLITS:
        _fail(ep, f"split must be one of {SPLITS}")
    if ep["cls"] not in CLASSES:
        _fail(ep, f"cls must be one of {CLASSES}")
    meta = ep["meta"]
    if not isinstance(meta, dict) or any(k not in meta for k in META_KEYS):
        _fail(ep, f"meta needs the keys {META_KEYS}")

    # tools: full spec dicts
    if not isinstance(ep["tools"], list):
        _fail(ep, "tools must be a list of tool spec dicts")
    required_of = {}
    for tool in ep["tools"]:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"] \
                or not isinstance(tool.get("summary"), str) or not isinstance(tool.get("parameters"), list):
            _fail(ep, "each tool needs name, summary and a parameters list")
        if tool["name"] in required_of:
            _fail(ep, f"tool {tool['name']} is listed twice")
        for p in tool["parameters"]:
            if not isinstance(p, dict) or not isinstance(p.get("name"), str) or not isinstance(p.get("type"), str):
                _fail(ep, f"tool {tool['name']} has a malformed parameter")
        required_of[tool["name"]] = [p["name"] for p in tool["parameters"] if p.get("required")]

    # user text
    if not isinstance(ep["user"], str) or not ep["user"]:
        _fail(ep, "user must be a non-empty string")
    if len(ep["user"]) > USER_CHARS:
        _fail(ep, f"user text longer than USER_CHARS={USER_CHARS}")
    if _flat(ep["user"], USER_CHARS) != ep["user"]:
        _fail(ep, "user text has leading, trailing or doubled whitespace (it would not survive the U: line)")

    # turns
    turns = ep["turns"]
    if not isinstance(turns, list) or not turns:
        _fail(ep, "episode has no turns")
    if len(turns) > MAX_TURNS:
        _fail(ep, f"more than {MAX_TURNS} turns")
    for i, turn in enumerate(turns):
        if not isinstance(turn, (list, tuple)) or len(turn) != 2:
            _fail(ep, f"turn {i} must be [action, observation]")
        action, observation = turn
        if not isinstance(action, dict):
            _fail(ep, f"turn {i} action must be an object")
        try:
            parse_action(json.dumps(action))
        except ValueError as e:     # ActionParseError is a ValueError
            _fail(ep, f"turn {i} action is not accepted by parse_action ({e})")
        text = compact_action(action)
        if _flat(text, HISTORY_ACTION_CHARS) != text:
            _fail(ep, f"turn {i} action is longer than {HISTORY_ACTION_CHARS} characters or has doubled spaces, "
                      "tabs or newlines (it would not survive the A: line)")
        if action["action"] == "call_tool":
            tool = action.get("tool")
            if tool not in required_of:
                _fail(ep, f"turn {i} calls tool {tool!r} which is not listed in tools")
            args = action.get("args")
            if not isinstance(args, dict) or list(args) != required_of[tool]:
                _fail(ep, f"turn {i} args keys {list(args) if isinstance(args, dict) else args!r} differ from "
                          f"the required parameters {required_of[tool]} of {tool}")
        last = i == len(turns) - 1
        if last:
            if observation is not None:
                _fail(ep, "last turn must have no observation")
        else:
            if not isinstance(observation, str) or not observation:
                _fail(ep, f"turn {i} needs a non-empty observation string")
            if Observation("success", output=observation).as_text() != observation:
                _fail(ep, f"turn {i} observation is not what the harness would show (too long for MAX_OBSERVATION_CHARS)")
            if len(observation) > OBS_CHARS or _flat(observation, OBS_CHARS) != observation:
                _fail(ep, f"turn {i} observation is not a single line of at most OBS_CHARS={OBS_CHARS} characters")
    if turns[-1][0].get("action") != "respond":
        _fail(ep, "last action must be 'respond'")

    # ASCII only (anything else would silently become '?' in encode)
    if not json.dumps(ep, ensure_ascii=False).isascii():
        _fail(ep, "episode text must be ASCII")

    # no truncation: the whole conversation up to the last A: fits the fit budget, so
    # training_text() is the untruncated transcript and equals what inference renders
    last = len(turns) - 1
    full = render_context(episode_to_messages(ep, last), budget=10 ** 9)
    if len(full) - len(ACTION_PREFIX) > CONTEXT_CHARS - GEN_RESERVE:
        _fail(ep, f"context would exceed CONTEXT_CHARS - GEN_RESERVE = {CONTEXT_CHARS - GEN_RESERVE} "
                  f"({len(full) - len(ACTION_PREFIX)} characters before the last A:)")
    text = training_text(ep)
    if len(text) > CONTEXT_CHARS:
        _fail(ep, f"training text of {len(text)} characters exceeds CONTEXT_CHARS={CONTEXT_CHARS}")
    if text[:len(full)] != full:
        _fail(ep, "training text is not the untruncated render (fit() cut it)")


# ── leakage corpus and gate (completed in the leakage task) ──

class Corpus:
    """What the benchmark contains, derived programmatically at gate-build time."""

    def __init__(self):
        self.held_out = _held_out()


@functools.lru_cache(maxsize=None)
def build_corpus() -> Corpus:
    return Corpus()


def assert_clean(episodes, corpus=None):
    """Raise LeakageError naming the episode id and the rule if any episode leaks benchmark content."""
    corpus = corpus or build_corpus()
    for ep in episodes:
        called = [a["tool"] for a, _ in ep["turns"] if a.get("action") == "call_tool"]
        names = {t["name"] for t in ep["tools"]} | set(called)
        hit = sorted(names & corpus.held_out)
        if hit:
            raise LeakageError(f"{ep['id']}: held-out tool {hit}")
