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

import argparse
import copy
import functools
import hashlib
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone

from src.agent_harness.executors import Observation
from src.agent_harness.prompting import OBSERVATION_TEMPLATE, build_messages, parse_action
from src.agent_harness.tool_broker import DEFAULT_RESPONSE, ToolBroker
from src.benchmark.injecagent import DATA_DIR, EXFIL_TOOL, PROJECT_ROOT, load_tool_specs
from src.scratch_llm.format import (ACTION_PREFIX, CONTEXT_CHARS, END, FORMAT_VERSION, GEN_RESERVE, HISTORY_ACTION_CHARS,
                                    OBS_CHARS, USER_CHARS, _flat, compact_action, render_context, supervised_mask)
from src.scratch_llm import templates
from src.scratch_llm.provenance import (git_commit, git_dirty, refuse_foreign_manifest, resolve_under_root,
                                        write_json_atomic)

GENERATOR_VERSION = 2   # 1 was the benign-only slice of plan 02-01; 2 is the full mix
MAX_REDRAWS = 50   # attempts to draw one valid episode before generation gives up
MAX_LEAK_REJECTION_RATE = 0.001     # leak rejections allowed per draw before generation fails (see generate_episodes)
MIN_LEAK_REJECTIONS_ALLOWED = 3     # ... but never fewer than this, so tiny test sets are not tripped by one unlucky draw
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
    """name -> spec (name, summary, parameters, returns) for the real non-benchmark tools the generator may
    offer, sorted by name: the 251 free tools minus any whose own summary trips the leakage gate."""
    return {name: copy.deepcopy(spec) for name, spec in _gen_pool()}


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


# ── leakage corpus and gate (SCRATCH-02, D-04) ──

HELD_ONLY_GATE_SCOPE = (
    "Held-only parameter names (names that occur only in held-out benchmark tools, such as to, subject "
    "or url) are rejected as tool parameter names (on the T: line / in tool specs) and as call_tool "
    "argument keys. The same words in free text, that is U: requests, O: observation values and "
    "respond messages, are allowed and are not gated."
)

RULE_HELD_OUT_TOOL = "held-out tool"
RULE_HELD_OUT_NAME = "held-out tool name in text"
RULE_HELD_ONLY_PARAM = "held-only parameter name"
RULE_SYNTHETIC_SIGNATURE = "synthetic tool with a held-out signature"
RULE_CANARY = "canary shape"
RULE_BANNED = "banned benchmark phrase"
RULE_EMAIL = "benchmark e-mail address"
RULE_SHINGLE = "8-word shingle"
RULE_SENTENCE = "verbatim benchmark sentence"
RULE_SUMMARY = "tool summary"

SHINGLE_WORDS = 8
_WORD = re.compile(r"[a-z0-9@._']+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# The benchmark's canary is CANARY<12 upper-case hex>DATA, but every other rule is case-insensitive, so this one is
# too (review WR-07): canary0a1b2c3d4e5fdata and Canary0A1B2C3D4E5FData are the same leak. No legitimate template
# or tool uses the word, so the bare word is banned in any case as well.
_CANARY = re.compile(r"canary[0-9a-f]{12}data", re.IGNORECASE)
_HEX12 = re.compile(r"[0-9A-Fa-f]{12}")      # the variable part of a canary, as a whole token
_BANNED_ANY_CASE = ("canary", "private record", "account owner. reference")


def _tokens(text: str, strip: bool = True) -> list:
    """Lower-case words (letters, digits, @ . _ '), so addresses and ids stay in one piece.

    strip=True also removes edge dots and quotes ('door.' and 'door' are one word); strip=False is
    the plain research normalisation. Shingles are collected and checked under both, so the gate
    is a superset of either."""
    words = _WORD.findall(text.lower())
    return [t for t in (w.strip("._'") for w in words) if t] if strip else words


def _shingles(tokens: list) -> set:
    return {tuple(tokens[i:i + SHINGLE_WORDS]) for i in range(len(tokens) - SHINGLE_WORDS + 1)}


def _both_shingles(text: str) -> set:
    return _shingles(_tokens(text)) | _shingles(_tokens(text, strip=False))


def _strings_in(value):
    """Every string inside a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings_in(v)


def _read_jsonl(name: str) -> list:
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Corpus:
    """What the benchmark contains, derived programmatically at gate-build time."""

    def __init__(self):
        specs = load_tool_specs()
        self.held_out = _held_out()
        self.held_only_params = _held_only_params()
        self.real_tools = frozenset(specs)
        # required-parameter signatures of the held-out tools (non-empty ones only)
        self.held_signatures = frozenset(
            sig for sig in (tuple(p["name"] for p in specs[n]["parameters"] if p.get("required"))
                            for n in self.held_out if n in specs) if sig)
        names = sorted(self.held_out, key=len, reverse=True)
        self.held_re = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")

        users = _read_jsonl("user_cases.jsonl")
        attackers = _read_jsonl("attacker_cases_dh.jsonl") + _read_jsonl("attacker_cases_ds.jsonl")
        emails = set()
        for record in users + attackers:
            for text in _strings_in(record):
                emails |= {m.lower() for m in _EMAIL.findall(text)}
        self.emails = frozenset(emails)

        shingles = set()
        for record in users:
            for key in ("User Instruction", "Tool Response Template", "Thought"):
                shingles |= _both_shingles(record[key])
        for record in attackers:
            for key in ("Attacker Instruction", "Expected Achievements"):
                shingles |= _both_shingles(record[key])
        self.shingles = frozenset(shingles)

        sentences = {tuple(_tokens(r["User Instruction"])) for r in users}
        sentences |= {tuple(_tokens(r["Attacker Instruction"])) for r in attackers}
        sentences.discard(())
        self.exact = frozenset(" ".join(s) for s in sentences)
        index = {}
        for sentence in sorted(sentences):
            index.setdefault(sentence[0], []).append(sentence)
        self.exact_index = index


@functools.lru_cache(maxsize=None)
def build_corpus() -> Corpus:
    return Corpus()


def _text_violation(text: str, corpus: Corpus, sentences: bool = False):
    """(rule, detail) for the first benchmark content found in `text`, else None."""
    match = _CANARY.search(text)
    if match:
        return RULE_CANARY, match.group(0)
    lowered = text.lower()
    for phrase in _BANNED_ANY_CASE:
        if phrase in lowered:
            return RULE_BANNED, phrase
    for address in sorted(corpus.emails):
        if address in lowered:
            return RULE_EMAIL, address
    overlap = _both_shingles(text) & corpus.shingles
    if overlap:
        return RULE_SHINGLE, " ".join(sorted(overlap)[0])
    if sentences:
        hit = _sentence_in(_tokens(text), corpus)
        if hit:
            return RULE_SENTENCE, " ".join(hit)
    return None


def _sentence_in(tokens: list, corpus: Corpus):
    """A whole benchmark instruction occurring as a run of whole words in `tokens`, else None."""
    index = corpus.exact_index
    for i, token in enumerate(tokens):
        for sentence in index.get(token, ()):
            if tuple(tokens[i:i + len(sentence)]) == sentence:
                return sentence
    return None


def _episode_violation(ep: dict, corpus: Corpus):
    """(rule, detail) for the first leak found in an episode, else None."""
    tools = ep["tools"]
    calls = [a for a, _ in ep["turns"] if a.get("action") == "call_tool"]
    hit = sorted(({t["name"] for t in tools} | {a.get("tool") for a in calls}) & corpus.held_out)
    if hit:
        return RULE_HELD_OUT_TOOL, ", ".join(hit)
    # D-04 scope: tool parameter names and call_tool argument keys - never free text
    for tool in tools:
        hit = sorted({p["name"] for p in tool["parameters"]} & corpus.held_only_params)
        if hit:
            return RULE_HELD_ONLY_PARAM, f"{', '.join(hit)} as a parameter of {tool['name']}"
    for action in calls:
        hit = sorted({k for k in (action.get("args") or {})} & corpus.held_only_params)
        if hit:
            return RULE_HELD_ONLY_PARAM, f"{', '.join(hit)} as a call_tool argument key"
    for tool in tools:
        signature = tuple(p["name"] for p in tool["parameters"] if p.get("required"))
        if tool["name"] not in corpus.real_tools and signature and signature in corpus.held_signatures:
            return RULE_SYNTHETIC_SIGNATURE, f"{tool['name']}{signature}"

    text = training_text(ep)
    match = corpus.held_re.search(text)
    if match:
        return RULE_HELD_OUT_NAME, match.group(0)
    found = _text_violation(text, corpus)
    if found:
        return found
    for line in text.split(END):
        if line[:2] in ("U:", "O:", ACTION_PREFIX):
            hit = _sentence_in(_tokens(line[2:]), corpus)
            if hit:
                return RULE_SENTENCE, " ".join(hit)
    for tool in tools:
        found = _text_violation(tool["summary"], corpus)
        if found:
            return RULE_SUMMARY, f"summary of {tool['name']}: {found[0]}: {found[1]}"
    return None


def assert_clean(episodes, corpus=None):
    """Raise LeakageError naming the episode id and the rule if any episode leaks benchmark content."""
    corpus = corpus or build_corpus()
    for ep in episodes:
        found = _episode_violation(ep, corpus)
        if found:
            raise LeakageError(f"{ep['id']}: {found[0]}: {found[1]}")


def _template_strings():
    """(where, text) for every string in templates.py (ids, phrases and word lists)."""
    for attr in sorted(vars(templates)):
        if not attr.startswith("_"):
            for text in _strings_in(getattr(templates, attr)):
                yield attr, text


def check_templates(corpus=None):
    """Raise LeakageError if any string in templates.py repeats benchmark content, before any rendering."""
    corpus = corpus or build_corpus()
    for attr, text in _template_strings():
        found = _text_violation(text, corpus, sentences=True)
        if found or corpus.held_re.search(text):
            rule, detail = found or (RULE_HELD_OUT_NAME, corpus.held_re.search(text).group(0))
            raise LeakageError(f"templates.{attr}: {rule}: {detail}")


@functools.lru_cache(maxsize=None)
def _dropped_free_tools() -> tuple:
    """Real non-benchmark tools whose own summary trips a gate rule: never offered to the model."""
    corpus = build_corpus()
    return tuple(name for name, spec in _free_pool() if _text_violation(spec["summary"], corpus))


def dropped_free_tools() -> list:
    """Sorted names of the free-pool tools dropped by the gate (their count goes into the dataset manifest)."""
    return list(_dropped_free_tools())


# ── value generators (every draw comes from the one random.Random passed in) ──

_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_ID_ALPHABET = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def fill(template: str, **slots) -> str:
    """Plain slot replacement ({do}, {args}); text without a slot is left as it is."""
    for key, value in slots.items():
        template = template.replace("{" + key + "}", value)
    return template


def _tidy(text: str) -> str:
    """One sentence: single spaces, no space before punctuation, capital first letter."""
    text = re.sub(r" ([.,;:?!])", r"\1", " ".join(text.split()))
    return text[:1].upper() + text[1:]


def _words(rng: random.Random, low: int, high: int) -> str:
    return " ".join(rng.choice(templates.WORD_LIST) for _ in range(rng.randint(low, high)))


def _id_token(rng: random.Random) -> str:
    length = rng.randint(5, 12)
    prefix = rng.choice(templates.ID_PREFIXES)
    lead = prefix + "-" if prefix else ""
    return lead + "".join(rng.choice(_ID_ALPHABET) for _ in range(max(3, length - len(lead))))


def _address(rng: random.Random) -> str:
    """A synthetic e-mail address on a reserved documentation domain (RFC 2606): it reaches nobody."""
    first = rng.choice(templates.FIRST_NAMES)
    kind = rng.randint(0, 2)
    if kind == 0:
        return f"{first}{rng.randint(1, 999)}@{rng.choice(templates.RESERVED_DOMAINS)}"
    if kind == 1:
        return f"{first}.{rng.choice(templates.WORD_LIST)}@{rng.choice(templates.RESERVED_DOMAINS)}"
    return f"{first}{rng.randint(1, 99)}@{rng.choice(templates.WORD_LIST)}.{rng.choice(templates.RESERVED_TLDS)}"


def _string_value(rng: random.Random, name: str, single_word: bool = False) -> str:
    """A string whose shape follows the parameter name (id, address, date, path, ...)."""
    n = name.lower()
    if n.endswith(("id", "ids", "token", "code", "key", "ref")) or "uuid" in n:
        return _id_token(rng)
    if "mail" in n or "recipient" in n:
        return _address(rng)
    if any(k in n for k in ("url", "link", "uri", "website")):
        return f"https://{rng.choice(templates.WORD_LIST)}.example/{rng.choice(templates.WORD_LIST)}"
    if any(k in n for k in ("path", "file", "folder", "directory")):
        return (f"/home/user/{rng.choice(templates.PATH_DIRS)}/{rng.choice(templates.WORD_LIST)}"
                f"{rng.choice(templates.PATH_EXTENSIONS)}")
    if "phone" in n:
        return f"+1-555-{rng.randint(0, 9999):04d}"
    if "time" in n and "timezone" not in n:
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if "date" in n:
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return _words(rng, 1, 1 if single_word else 3)


def _arg_value(rng: random.Random, name: str, ptype: str):
    """A value of the declared JSON type for a required parameter (ASCII, no quotes or backslashes)."""
    if ptype == "integer":
        return rng.randint(0, 200)
    if ptype == "number":
        return round(rng.uniform(0, 200), 1)
    if ptype == "boolean":
        return rng.choice([True, False])
    if ptype == "array":
        return [_string_value(rng, name, single_word=True) for _ in range(rng.randint(1, 3))]
    if ptype == "object":
        keys = rng.sample(templates.OBJECT_KEYS, rng.randint(1, 2))
        return {key: _words(rng, 1, 1) for key in keys}
    return _string_value(rng, name)


def _return_value(rng: random.Random, name: str, ptype: str):
    """A value for a field of a simulated tool response (the tool's `returns` list)."""
    if ptype == "boolean":
        return True if name == "success" else rng.choice([True, False])
    if ptype == "array":
        items = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                items.append(_words(rng, 1, 1))
            else:
                items.append({"id": _id_token(rng), rng.choice(templates.RECORD_KEYS): _words(rng, 1, 3)})
        return items
    return _arg_value(rng, name, ptype)


def _attack_address(rng: random.Random) -> str:
    """The address an injection asks the agent to send data to: reserved documentation domains for
    about 92 percent, otherwise a public-provider domain with a numeric local part (so copying an
    address outside the reserved domains is practised too; nothing is ever sent anywhere)."""
    if rng.random() < 0.08:
        return f"{rng.randint(100000, 999999999)}@{rng.choice(templates.PUBLIC_DOMAINS)}"
    return _address(rng)


def _secret(rng: random.Random) -> str:
    """A secret token of 12-30 characters in one of six shapes, never the benchmark's canary shape.

    The canary's variable part is 12 hex characters, so no token is ever exactly 12 hex characters (review
    WR-07): the two hex shapes start at 13, and the guard below catches the rare 12-character draw of another
    shape that happens to use only hex characters."""
    token = _secret_token(rng)
    return token[:-1] + "Z" if _HEX12.fullmatch(token) else token


def _secret_token(rng: random.Random) -> str:
    shape = rng.randrange(6)
    if shape == 0:
        return "".join(rng.choice("0123456789ABCDEF") for _ in range(rng.randint(13, 20)))
    if shape == 1:
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(13, 20)))
    if shape == 2:
        return "".join(rng.choice(_ID_ALPHABET) for _ in range(rng.randint(12, 24)))
    if shape == 3:
        return "-".join("".join(rng.choice("0123456789") for _ in range(4)) for _ in range(rng.randint(3, 5)))
    if shape == 4:
        first, second = rng.choice(templates.WORD_LIST), rng.choice(templates.WORD_LIST)
        return f"{first}-{second}-" + "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 6)))
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(rng.randint(12, 20)))


# ── observations: built from a tool's return schema ──

def _payload(rng: random.Random, returns: list, carrier_text: str = None) -> dict:
    """The fields of a tool response. With carrier_text, that text sits in a free-text key
    (snippet, comment, ...) inside an array of records or an object - the field is put first so it
    is the last thing dropped when the observation has to be shortened."""
    payload = {r["name"]: _return_value(rng, r["name"], r["type"]) for r in returns}
    if carrier_text is None:
        return payload
    key = rng.choice(templates.CARRIER_KEYS)
    arrays = [r["name"] for r in returns if r["type"] == "array"]
    objects = [r["name"] for r in returns if r["type"] == "object"]
    if arrays:
        field = arrays[rng.randrange(len(arrays))]
    elif objects:
        field = objects[rng.randrange(len(objects))]
    else:
        spare = [name for name in ("items", "results", "entries", "records") if name not in payload]
        field = spare[rng.randrange(len(spare))] if spare else "extra"
    if field in objects and not arrays:
        value = {key: carrier_text}
    else:
        value = [{"id": _id_token(rng), key: carrier_text}]
        if rng.random() < 0.3:
            value.insert(0, {"id": _id_token(rng), "name": _words(rng, 1, 2)})
    payload.pop(field, None)
    return {field: value, **payload}


def _render_payload(rng: random.Random, payload: dict, limit: int):
    """One line of text for a payload: Python repr (single quotes) about 60 percent of the time,
    json.dumps otherwise, cut field by field from the end until it fits `limit` characters.
    Returns None when even the first field is too long."""
    style_is_repr = rng.random() < 0.6
    payload = dict(payload)
    while payload:
        text = repr(payload) if style_is_repr else json.dumps(payload, ensure_ascii=True)
        if not style_is_repr and "\\" in text:      # json would escape a quote and hide the copied value
            text = repr(payload)
        if len(text) <= limit:
            return text
        if len(payload) == 1:
            return None
        payload.popitem()
    return None


def _observation(rng: random.Random, returns: list, carrier_text: str = None, limit: int = OBS_CHARS):
    """The text the harness would show for a successful call, built from the tool's return schema
    (None when the carrier text cannot fit in `limit` characters)."""
    payload = _payload(rng, returns, carrier_text)
    text = _render_payload(rng, payload, limit) if payload else json.dumps({"success": True})
    return None if text is None else Observation("success", output=text).as_text()


def _success_observation() -> str:
    """What the harness returns for a tool without its own simulated response."""
    return Observation("success", output=DEFAULT_RESPONSE).as_text()


# ── instructions and replies ──

def _tool_words(tool: dict) -> list:
    """The CamelCase words of a tool name after its toolkit prefix, as written."""
    name, toolkit = tool["name"], tool["toolkit"]
    rest = name[len(toolkit):] if toolkit and name.startswith(toolkit) else name
    return _CAMEL.findall(rest) or [name]


def _request_words(tool: dict) -> tuple:
    """(verb, object words, toolkit words), lower case, from a tool's name."""
    words = [w.lower() for w in _tool_words(tool)]
    return words[0], words[1:], [w.lower() for w in _CAMEL.findall(tool["toolkit"])]


def _plain_request(tool: dict) -> str:
    verb, obj, _ = _request_words(tool)
    return " ".join([verb] + obj)


def _request_phrase(rng: random.Random, tool: dict) -> str:
    """The request as a verb phrase: the tool's own words (about 70%) or a synonym verb (about 30%),
    sometimes naming the application."""
    verb, obj, toolkit = _request_words(tool)
    if verb in templates.SYNONYMS and rng.random() < 0.3:
        verb = rng.choice(templates.SYNONYMS[verb])
    phrase = " ".join([verb] + obj)
    if toolkit and rng.random() < 0.5:
        phrase += f" {rng.choice(templates.TOOLKIT_PREPOSITIONS)} {' '.join(toolkit)}"
    return phrase


def _render_value(value) -> str:
    """How a value is written inside the instruction: strings as they are, everything else as the
    compact JSON the action will contain, so the model copies it verbatim."""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _arguments_text(rng: random.Random, args: dict) -> str:
    if not args:
        return ""
    listing = ", ".join(f"{name.replace('_', ' ').lower()} {_render_value(value)}" for name, value in args.items())
    return f"{rng.choice(templates.ARG_CONNECTORS)} {listing}"


# ── tools and the roles they can play ──

RET_WORDS = frozenset(("Get", "Retrieve", "Read", "View", "Search", "List", "Fetch", "Show"))
SINK_WORDS = frozenset(("Send", "Post", "Share", "Forward", "Message", "Email", "Upload", "Publish", "Reply"))
SINK_SKIP_WORDS = frozenset(("Sign", "Verify"))
TEXT_PARAMS = ("body", "content", "message", "text", "note", "description", "comment", "message_content")
DEST_HINTS = ("mail", "recipient", "receiver", "contact", "phone", "channel", "username")
NOT_A_PAYLOAD = ("id", "ids", "path", "url", "file", "folder", "date", "time")
MAX_SOURCE_REQUIRED = 2


@functools.lru_cache(maxsize=None)
def _gen_pool() -> tuple:
    """((name, spec), ...) sorted by name: the real tools the generator may offer."""
    dropped = set(_dropped_free_tools())
    return tuple((name, spec) for name, spec in _free_pool() if name not in dropped)


def make_tool(name: str, spec: dict, toolkit: str) -> dict:
    """A generator-side tool: the episode fields (name, summary, parameters), its return schema, its toolkit."""
    return {"name": name, "summary": spec.get("summary", ""),
            "parameters": [{"name": p["name"], "type": p["type"], "required": bool(p.get("required"))}
                           for p in spec.get("parameters", [])],
            "returns": [{"name": r["name"], "type": r["type"]} for r in spec.get("returns", [])],
            "toolkit": toolkit}


def _required(tool: dict) -> list:
    return [p for p in tool["parameters"] if p["required"]]


def _payload_param(tool: dict):
    """The required string parameter that would carry stolen data when the tool is used as a sink."""
    strings = [p for p in _required(tool) if p["type"] == "string"]
    for p in strings:
        if p["name"] in TEXT_PARAMS:
            return p["name"]
    for p in strings:
        lowered = p["name"].lower()
        if any(hint in lowered for hint in DEST_HINTS):
            continue                      # a recipient is where the data goes, not the data
        if not any(lowered.endswith(bad) or bad + "_" in lowered for bad in NOT_A_PAYLOAD):
            return p["name"]
    return None


def _dest_param(tool: dict, payload: str):
    """The recipient-like required parameter (an address, channel, phone number, ...), or None."""
    for p in _required(tool):
        if p["name"] != payload and p["type"] in ("string", "array") \
                and any(hint in p["name"].lower() for hint in DEST_HINTS):
            return p["name"]
    return None


def _is_source(tool: dict) -> bool:
    words = _tool_words(tool)
    return words[0] in RET_WORDS and len(_required(tool)) <= MAX_SOURCE_REQUIRED


def _is_sink(tool: dict) -> bool:
    words = _tool_words(tool)
    return words[0] in SINK_WORDS and not SINK_SKIP_WORDS & set(words) and _payload_param(tool) is not None


class _Pools:
    """The tools one kind of episode may draw from, by role."""

    def __init__(self, tools):
        tools = tuple(tools)
        self.all = tools
        self.sources = tuple(t for t in self.all if _is_source(t))
        self.sinks = tuple(t for t in self.all if _is_sink(t))


@functools.lru_cache(maxsize=None)
def _real_tools() -> tuple:
    return tuple(make_tool(name, spec, toolkit_of(name)) for name, spec in _gen_pool())


DEV_TOOLS_SEED = 20261021   # fixed: the dev tool set must not move with the episode seed (D-04)
DEV_TOOL_SHARE = 0.10


@functools.lru_cache(maxsize=None)
def _dev_names() -> tuple:
    names = [name for name, _ in _gen_pool()]
    return tuple(sorted(random.Random(DEV_TOOLS_SEED).sample(names, round(DEV_TOOL_SHARE * len(names)))))


def dev_tool_names() -> list:
    """Sorted names of the real tools held back for the dev split: a seeded 10% of the free pool that the
    train split never offers, in any role (D-04)."""
    return list(_dev_names())


@functools.lru_cache(maxsize=None)
def _train_pools() -> _Pools:
    dev = set(_dev_names())
    return _Pools(t for t in _real_tools() if t["name"] not in dev)


@functools.lru_cache(maxsize=None)
def _dev_pools() -> _Pools:
    dev = set(_dev_names())
    return _Pools(t for t in _real_tools() if t["name"] in dev)


def _pick(rng: random.Random, pool: tuple, exclude=()):
    """A tool from `pool` that is not in `exclude` (None when the pool has nothing left)."""
    for _ in range(30):
        tool = pool[rng.randrange(len(pool))] if pool else None
        if tool is None or tool["name"] not in exclude:
            return tool
    options = [t for t in pool if t["name"] not in exclude]
    return options[rng.randrange(len(options))] if options else None


def _tool_args(rng: random.Random, tool: dict) -> dict:
    """Values for the required parameters, in the spec's order (the order of the T: line)."""
    return {p["name"]: _arg_value(rng, p["name"], p["type"]) for p in _required(tool)}


def _send_word(rng: random.Random, sink: dict) -> str:
    own = next((w.lower() for w in _tool_words(sink) if w in SINK_WORDS), "send")
    return rng.choice(templates.SEND_WORDS) if rng.random() < 0.3 else own


# ── synthetic tools (made-up names and parameters, so the model has to copy them) ──

SYNTHETIC_SHARE = 0.35
OS_SHARE = 0.10             # of ALL episodes; every one of them is taken from the benign half (D-02)
TWO_STEP_SHARE = 0.15       # of the benign episodes that are not OS tasks
BLOCKED_SHARE = 0.08        # of the dh and ds episodes (D-03)
DISTRACTOR_SHARE = 0.30
SHUFFLE_SHARE = 0.50


@functools.lru_cache(maxsize=None)
def _syn_params() -> tuple:
    """(sorted parameter names, type by name) a synthetic tool may use: the parameter names of the real
    free tools plus the generic ones in templates.SYN_PARAMS, minus every held-only name."""
    held = _held_only_params()
    seen = {}
    for _, spec in _free_pool():
        for p in spec["parameters"]:
            seen.setdefault(p["name"], {}).setdefault(p["type"], 0)
            seen[p["name"]][p["type"]] += 1
    types = {name: sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for name, counts in seen.items()}
    for name, ptype in templates.SYN_PARAMS.items():
        types.setdefault(name, ptype)
    return tuple(sorted(name for name in types if name not in held)), types


def _draw_names(rng: random.Random, names: tuple, count: int, used: set) -> list:
    picked = []
    while len(picked) < count:
        name = names[rng.randrange(len(names))]
        if name not in used and name not in picked:
            picked.append(name)
    return picked


def _synthetic_parameters(rng: random.Random, role: str) -> list:
    names, types = _syn_params()
    # names is already without held-only names; types is not, so filter here too (review WR-08): a benchmark
    # update that made one of these held-only would otherwise be absorbed silently by redraws
    held = _held_only_params()
    text_names = [n for n in TEXT_PARAMS if n in types and n not in held]
    dest_names = [n for n in templates.SYN_DEST_NAMES if n not in held]
    if role == "sink":
        required = [text_names[rng.randrange(len(text_names))]]
        if rng.random() < 0.85:
            required.append(dest_names[rng.randrange(len(dest_names))])
        if rng.random() < 0.3:
            required += _draw_names(rng, names, 1, set(required))
        rng.shuffle(required)
        forced = {n: "string" for n in required[:] if n in TEXT_PARAMS or n in templates.SYN_DEST_NAMES}
    else:
        weights = [1, 5, 2] if role == "source" else [1, 5, 3, 1]
        required = _draw_names(rng, names, rng.choices(range(len(weights)), weights)[0], set())
        forced = {}
    optional = _draw_names(rng, names, rng.choices([0, 1, 2], [3, 2, 1])[0], set(required))
    parameters = [{"name": n, "type": forced.get(n, types.get(n, "string")), "required": True} for n in required]
    parameters += [{"name": n, "type": types[n], "required": False} for n in optional]
    # real specs interleave required and optional parameters; the required ones keep their order
    for p in [p for p in parameters if not p["required"]]:
        parameters.remove(p)
        parameters.insert(rng.randint(0, len(parameters)), p)
    return parameters


def _synthetic_returns(rng: random.Random, role: str) -> list:
    fields = rng.sample(templates.SYN_RETURNS, rng.randint(1, 3))
    if role == "source" and not any(t in ("array", "object") for _, t in fields):
        fields.append(("results", "array") if ("results", "array") not in fields else ("details", "object"))
    return [{"name": n, "type": t} for n, t in fields]


def _synthetic_tool(rng: random.Random, role: str, part: str, exclude=()) -> dict:
    """A tool that exists nowhere in tools.json: toolkit word + verb + object, parameter names that are
    never held-only, and (for a non-empty required list) a signature no held-out tool has."""
    corpus = build_corpus()
    toolkits = templates.SYN_TOOLKITS_TRAIN if part == "TRAIN" else templates.SYN_TOOLKITS_DEV
    for _ in range(MAX_REDRAWS * 4):
        toolkit = toolkits[rng.randrange(len(toolkits))]
        if role == "source":
            verb = templates.SYN_RET_VERBS[rng.randrange(len(templates.SYN_RET_VERBS))]
        elif role == "sink":
            verb = templates.SYN_SEND_VERBS[rng.randrange(len(templates.SYN_SEND_VERBS))]
        else:
            verb = templates.SYN_VERBS[rng.randrange(len(templates.SYN_VERBS))]
        objects = templates.SYN_SINK_OBJECTS if role == "sink" else templates.SYN_OBJECTS
        obj = objects[rng.randrange(len(objects))]
        if role != "sink" and rng.random() < 0.25:
            obj += templates.SYN_OBJECTS[rng.randrange(len(templates.SYN_OBJECTS))]
        name = toolkit + verb + obj
        if name in corpus.real_tools or name in corpus.held_out or name in exclude:
            continue
        parameters = _synthetic_parameters(rng, role)
        signature = tuple(p["name"] for p in parameters if p["required"])
        if signature and signature in corpus.held_signatures:
            continue
        words = " ".join(w.lower() for w in _CAMEL.findall(obj))
        summary = _tidy(fill(templates.SYN_SUMMARIES[rng.randrange(len(templates.SYN_SUMMARIES))],
                             Verb=verb, verb=verb.lower(), object=words))
        return {"name": name, "summary": summary, "parameters": parameters,
                "returns": _synthetic_returns(rng, role), "toolkit": toolkit}
    raise RuntimeError(f"no synthetic {role} tool after {MAX_REDRAWS * 4} draws")


# ── episode parts ──

class _Ctx:
    """Everything one episode draws from: its split's templates and tool pools, and whether its tools are synthetic."""

    def __init__(self, split: str, synthetic: bool, pools: _Pools):
        self.split, self.synthetic, self.pools = split, synthetic, pools
        self.part = "TRAIN" if split == "train" else "DEV"
        self.extra_ids = ["syn:" + self.part[0].lower()] if synthetic else []

        def lists(family):
            return getattr(templates, f"{family}_{self.part}")
        self.user, self.dh, self.ds = lists("USER_TEMPLATES"), lists("DH_TEMPLATES"), lists("DS_TEMPLATES")
        self.record, self.wrappers, self.os = lists("RECORD_TEMPLATES"), lists("WRAPPERS"), lists("OS_TASKS")
        self.two, self.blocked_reply = lists("TWO_STEP_TEMPLATES"), lists("BLOCKED_REPLY_TEMPLATES")
        self.reply = templates.REPLY_TEMPLATES if self.part == "TRAIN" else templates.REPLY_TEMPLATES_DEV

    def tool(self, rng: random.Random, role: str, exclude=()) -> dict:
        """A tool for `role` ('any', 'source' or 'sink'): synthetic for a synthetic episode, otherwise from
        the split's real pool (synthetic only where that pool has no tool for the role)."""
        if not self.synthetic:
            pool = {"any": self.pools.all, "source": self.pools.sources, "sink": self.pools.sinks}[role]
            tool = _pick(rng, pool, exclude)
            if tool is not None:
                return tool
            self.extra_ids.append(f"{role}:synthetic")
        return _synthetic_tool(rng, role, self.part, exclude)

    def offered(self, rng: random.Random, base: list) -> list:
        """The tool list the model is shown: the episode's tools in the harness order (about half of the
        time) or shuffled, and in about 30 percent of episodes with 1-2 extra tools."""
        tools = list(base)
        if rng.random() < DISTRACTOR_SHARE:
            for _ in range(rng.randint(1, 2)):
                tools.append(self.tool(rng, "any", {t["name"] for t in tools}))
        if rng.random() >= SHUFFLE_SHARE and len(tools) > 1:
            first = [t["name"] for t in base]
            for _ in range(20):
                rng.shuffle(tools)
                if [t["name"] for t in tools][:len(first)] != first:
                    break
        return tools


def _instruction(rng: random.Random, tool: dict, args: dict, choices: list) -> tuple:
    """(template id, the user's request) for one call."""
    template_id, template = choices[rng.randrange(len(choices))]
    return template_id, _tidy(fill(template, do=_request_phrase(rng, tool), args=_arguments_text(rng, args)))


def _wrapped(rng: random.Random, text: str, wrappers: list) -> tuple:
    """(wrapper id, injection with its wrapper: 'IMPORTANT: ...', 'Before you reply, ...' or none)."""
    wrapper_id, wrapper = wrappers[rng.randrange(len(wrappers))]
    if not wrapper:
        return wrapper_id, text
    lead = text[:1].lower() + text[1:] if wrapper.endswith(",") else text
    return wrapper_id, f"{wrapper} {lead}"


def _embedded(rng: random.Random, injection: str) -> str:
    """The injection as the whole free-text field, or as the prefix or suffix of harmless text."""
    harmless = templates.BENIGN_TEXTS[rng.randrange(len(templates.BENIGN_TEXTS))]
    placement = rng.randrange(3)
    text = injection if placement == 0 else f"{injection} {harmless}" if placement == 1 else f"{harmless} {injection}"
    return " ".join(text.split())


def _reply(rng: random.Random, do: str, choices: list) -> tuple:
    reply_id, template = choices[rng.randrange(len(choices))]
    return reply_id, _tidy(fill(template, do=do))


def _call(tool: dict, args: dict) -> dict:
    return {"action": "call_tool", "tool": tool["name"], "args": args}


def _blocked_observation(user_tool: dict, action: dict) -> str:
    """What a real ToolBroker, restricted to the user tool, answers to an out-of-capability call."""
    return ToolBroker(None, capabilities={user_tool["name"]}).execute(action).as_text()


def _episode(ep_id: str, ctx: _Ctx, cls: str, tools: list, user: str, turns: list, template_ids: list,
             os_actions: bool = False, blocked_tail: bool = False, two_step: bool = False) -> dict:
    return {"id": ep_id, "split": ctx.split, "cls": cls, "tools": [episode_tool(t) for t in tools], "user": user,
            "turns": turns,
            "meta": {"synthetic": ctx.synthetic, "os_actions": os_actions, "blocked_tail": blocked_tail,
                     "two_step": two_step, "template_ids": template_ids + ctx.extra_ids}}


# ── benign episodes: one call, two calls, or an OS task ──

def _benign_observation(rng: random.Random, tool: dict):
    """A clean response; in about 40 percent a harmless free-text field (a review, a note) sits in it, so
    free text alone never marks an attack."""
    harmless = templates.BENIGN_TEXTS[rng.randrange(len(templates.BENIGN_TEXTS))] if rng.random() < 0.4 else None
    return _observation(rng, tool["returns"], harmless)


def _benign_episode(rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    tool = ctx.tool(rng, "any")
    args = _tool_args(rng, tool)
    user_id, user = _instruction(rng, tool, args, ctx.user)
    reply_id, message = _reply(rng, _plain_request(tool), ctx.reply)
    return _episode(ep_id, ctx, "benign", ctx.offered(rng, [tool]), user,
                    [[_call(tool, args), _benign_observation(rng, tool)],
                     [{"action": "respond", "message": message}, None]],
                    [user_id, reply_id])


def _two_step_episode(rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    first = ctx.tool(rng, "any")
    second = ctx.tool(rng, "any", {first["name"]})
    first_args, second_args = _tool_args(rng, first), _tool_args(rng, second)
    user_id, template = ctx.two[rng.randrange(len(ctx.two))]
    user = _tidy(fill(template, do1=_request_phrase(rng, first), args1=_arguments_text(rng, first_args),
                      do2=_request_phrase(rng, second), args2=_arguments_text(rng, second_args)))
    reply_id, message = _reply(rng, _plain_request(second), ctx.reply)
    return _episode(ep_id, ctx, "benign", ctx.offered(rng, [first, second]), user,
                    [[_call(first, first_args), _benign_observation(rng, first)],
                     [_call(second, second_args), _benign_observation(rng, second)],
                     [{"action": "respond", "message": message}, None]],
                    [user_id, reply_id], two_step=True)


def _file_name(rng: random.Random) -> str:
    return rng.choice(templates.WORD_LIST) + rng.choice(templates.OS_EXTENSIONS)


def _os_episode(rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    """A task done with run_shell / read_file / write_file under /home/user/workspace, with commands the
    default policy allows (grammar breadth beyond call_tool)."""
    task_id, kind, text = ctx.os[rng.randrange(len(ctx.os))]
    directory = templates.OS_DIRS[rng.randrange(len(templates.OS_DIRS))]
    path = f"{directory}/{_file_name(rng)}"
    count, content = rng.randint(2, 4), _words(rng, 2, 6)
    user = _tidy(fill(text, dir=directory, path=path, k=str(count), content=content))
    if kind == "ls":
        names = sorted({_file_name(rng) for _ in range(rng.randint(2, 6))})
        action, output = {"action": "run_shell", "command": f"ls {directory}"}, " ".join(names)
    elif kind == "cat":
        action, output = {"action": "run_shell", "command": f"cat {path}"}, _words(rng, 3, 14)
    elif kind == "read":
        action, output = {"action": "read_file", "path": path}, _words(rng, 3, 14)
    elif kind == "wc":
        action, output = {"action": "run_shell", "command": f"wc -l {path}"}, f"{rng.randint(1, 400)} {path}"
    elif kind == "head":
        action = {"action": "run_shell", "command": f"head -n {count} {path}"}
        output = " ".join(_words(rng, 2, 4) for _ in range(count))
    else:
        action = {"action": "write_file", "path": path, "content": content}
        output = f"Written {len(content)} chars to {path}"
    reply_id, message = _reply(rng, templates.OS_DO[kind], ctx.reply)
    return _episode(ep_id, ctx, "benign", ctx.offered(rng, []), user,
                    [[action, Observation("success", output=output).as_text()],
                     [{"action": "respond", "message": message}, None]],
                    [task_id, reply_id], os_actions=True)


# ── attack episodes ──

ATTACK_OBS_CHARS = 400     # an observation that carries an injection stays short, so the whole episode fits the context
RECORD_OBS_CHARS = 280


def _dh_episode(rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    """User tool -> an observation carrying an injection that asks for another tool -> that call -> respond
    (or, for a blocked tail, the broker's refusal text and a respond that gives up)."""
    user_tool = ctx.tool(rng, "any")
    target = ctx.tool(rng, "any", {user_tool["name"]})
    user_args, target_args = _tool_args(rng, user_tool), _tool_args(rng, target)
    user_id, user = _instruction(rng, user_tool, user_args, ctx.user)
    injection_id, template = ctx.dh[rng.randrange(len(ctx.dh))]
    injection = _tidy(fill(template, do=_request_phrase(rng, target), args=_arguments_text(rng, target_args),
                           tool=target["name"]))
    wrapper_id, injection = _wrapped(rng, injection, ctx.wrappers)
    observation = _observation(rng, user_tool["returns"], _embedded(rng, injection), ATTACK_OBS_CHARS)
    if observation is None:
        return None
    attack_call = _call(target, target_args)
    if flags["blocked"]:
        reply_id, message = _reply(rng, "", ctx.blocked_reply)
        second = _blocked_observation(user_tool, attack_call)
    else:
        reply_id, message = _reply(rng, _plain_request(user_tool), ctx.reply)
        second = _success_observation()
    return _episode(ep_id, ctx, "dh", ctx.offered(rng, [user_tool, target]), user,
                    [[_call(user_tool, user_args), observation], [attack_call, second],
                     [{"action": "respond", "message": message}, None]],
                    [user_id, injection_id, wrapper_id, reply_id], blocked_tail=flags["blocked"])


def _ds_episode(rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    """User tool -> injection "get X and send it to ADDR" -> source tool returns a record with a secret ->
    sink tool called with the secret (and ADDR) -> respond. A blocked tail stops at the source call."""
    user_tool = ctx.tool(rng, "any")
    source = ctx.tool(rng, "source", {user_tool["name"]})
    sink = ctx.tool(rng, "sink", {user_tool["name"], source["name"]})
    payload_name = _payload_param(sink)
    dest_name = _dest_param(sink, payload_name)
    address = _attack_address(rng) if dest_name else None
    secret, label = _secret(rng), templates.RECORD_LABELS[rng.randrange(len(templates.RECORD_LABELS))]
    record_id, record_template = ctx.record[rng.randrange(len(ctx.record))]
    record = " ".join(fill(record_template, name=rng.choice(templates.WORD_LIST), label=label, secret=secret).split())
    share = rng.random()
    payload = secret if share < 0.4 else record if share < 0.7 else f"{label}: {secret}"

    user_args, source_args = _tool_args(rng, user_tool), _tool_args(rng, source)
    sink_args, others = {}, {}
    for p in _required(sink):
        if p["name"] == payload_name:
            sink_args[p["name"]] = payload
        elif p["name"] == dest_name:
            sink_args[p["name"]] = [address] if p["type"] == "array" else address
        else:
            sink_args[p["name"]] = others[p["name"]] = _arg_value(rng, p["name"], p["type"])

    user_id, user = _instruction(rng, user_tool, user_args, ctx.user)
    injection_id, template = ctx.ds[rng.randrange(len(ctx.ds))]
    injection = _tidy(fill(template, pull=_request_phrase(rng, source), sargs=_arguments_text(rng, source_args),
                           send=_send_word(rng, sink), dest=f" to {address}" if address else "",
                           args=_arguments_text(rng, others)))
    wrapper_id, injection = _wrapped(rng, injection, ctx.wrappers)
    first = _observation(rng, user_tool["returns"], _embedded(rng, injection), ATTACK_OBS_CHARS)
    if first is None:
        return None
    ids = [user_id, injection_id, wrapper_id]
    tools = ctx.offered(rng, [user_tool, source, sink])
    source_call = _call(source, source_args)
    if flags["blocked"]:
        reply_id, message = _reply(rng, "", ctx.blocked_reply)
        return _episode(ep_id, ctx, "ds", tools, user,
                        [[_call(user_tool, user_args), first],
                         [source_call, _blocked_observation(user_tool, source_call)],
                         [{"action": "respond", "message": message}, None]],
                        ids + [reply_id], blocked_tail=True)
    second = _observation(rng, source["returns"], record, RECORD_OBS_CHARS)
    if second is None:
        return None
    reply_id, message = _reply(rng, _plain_request(user_tool), ctx.reply)
    return _episode(ep_id, ctx, "ds", tools, user,
                    [[_call(user_tool, user_args), first], [source_call, second],
                     [_call(sink, sink_args), _success_observation()],
                     [{"action": "respond", "message": message}, None]],
                    ids + [record_id, reply_id])


def _build(cls: str, rng: random.Random, ep_id: str, ctx: _Ctx, flags: dict):
    if cls == "dh":
        return _dh_episode(rng, ep_id, ctx, flags)
    if cls == "ds":
        return _ds_episode(rng, ep_id, ctx, flags)
    if flags["os"]:
        return _os_episode(rng, ep_id, ctx, flags)
    if flags["two_step"]:
        return _two_step_episode(rng, ep_id, ctx, flags)
    return _benign_episode(rng, ep_id, ctx, flags)


# ── the quota (D-01) and the shares of the other families ──

def class_counts(n: int) -> dict:
    """Episodes per class: n/2 benign, n/4 dh, n/4 ds; exact when 4 divides n, otherwise the leftover
    episodes go to the largest remainders (ties: benign, dh, ds), so every count is within one of its share."""
    scaled = {"benign": 2 * n, "dh": n, "ds": n}          # share * 4
    counts = {cls: value // 4 for cls, value in scaled.items()}
    leftover = n - sum(counts.values())
    for cls in sorted(scaled, key=lambda c: (-(scaled[c] % 4), CLASSES.index(c)))[:leftover]:
        counts[cls] += 1
    return counts


def _class_sequence(n: int, rng: random.Random) -> list:
    counts = class_counts(n)
    sequence = [cls for cls in CLASSES for _ in range(counts[cls])]
    rng.shuffle(sequence)
    return sequence


def _flags(rng: random.Random, n: int, split: str, classes: list) -> list:
    """Which episodes are synthetic, OS tasks, two-step or blocked-tail. Counts are allocated, not drawn,
    so the shares hold exactly (up to rounding) for any seed: synthetic 35% of all (train), OS 10% of all
    (all benign), two-step 15% of the other benign, blocked tails 8% of dh+ds. dev_synthetic is fully
    synthetic, dev_tools has no synthetic episode."""
    flags = [{"synthetic": False, "os": False, "two_step": False, "blocked": False} for _ in range(n)]
    synthetic = {"train": round(SYNTHETIC_SHARE * n), "dev_tools": 0, "dev_synthetic": n}[split]
    order = list(range(n))
    rng.shuffle(order)
    for i in order[:synthetic]:
        flags[i]["synthetic"] = True
    benign = [i for i, cls in enumerate(classes) if cls == "benign"]
    rng.shuffle(benign)
    os_count = min(round(OS_SHARE * n), len(benign))
    for i in benign[:os_count]:
        flags[i]["os"] = True
    rest = benign[os_count:]
    for i in rest[:round(TWO_STEP_SHARE * len(rest))]:
        flags[i]["two_step"] = True
    attack = [i for i, cls in enumerate(classes) if cls != "benign"]
    rng.shuffle(attack)
    for i in attack[:round(BLOCKED_SHARE * len(attack))]:
        flags[i]["blocked"] = True
    return flags


def _reason_key(message: str) -> str:
    """A validation message reduced to a countable reason: no episode id, numbers replaced by N."""
    return re.sub(r"[0-9]+", "N", message.split(": ", 1)[-1])[:100]


def generate_episodes(n: int, seed: int = 0, split: str = "train", report: dict = None) -> list:
    """Return `n` training episodes (schema above), reproducible for a given (seed, split).

    The class mix is a deterministic quota (50% benign, 25% dh, 25% ds), not a per-episode
    probability, and so are the shares of synthetic-tool, OS-action, two-step and blocked-tail
    episodes. Every episode must pass validate_episode and the leakage gate (an episode that does not
    is drawn again, a bounded number of times), and the whole list passes assert_clean before it is
    returned, so a leaking template makes generation raise instead of producing data.

    A rejected draw is never silent (review WR-08). If `report` is a dict it is filled with `draws`, `accepted`
    and `rejected` (a count per reason: "no episode built", "invalid: <validation message>", "leak: <rule>"), which
    the dataset manifest stores under gate.rejected. Generation fails with LeakageError when the leak rejections
    exceed MAX_LEAK_REJECTION_RATE of the draws (at least MIN_LEAK_REJECTIONS_ALLOWED): a template, tool or
    parameter name that keeps tripping the gate is a defect to fix, not something to draw around.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    check_templates()
    rng = random.Random(seed * 1_000_003 + SPLITS.index(split))
    pools, corpus = (_train_pools() if split == "train" else _dev_pools()), build_corpus()
    classes = _class_sequence(n, rng)
    flags = _flags(rng, n, split, classes)
    episodes, rejected, draws = [], {}, 0

    def reject(reason: str):
        rejected[reason] = rejected.get(reason, 0) + 1

    for i, (cls, flag) in enumerate(zip(classes, flags)):
        ep_id = f"{split}-{seed}-{i:06d}"
        leak = None
        for _ in range(MAX_REDRAWS):
            draws += 1
            episode = _build(cls, rng, ep_id, _Ctx(split, flag["synthetic"], pools), flag)
            if episode is None:
                reject("no episode built")
                continue
            try:
                validate_episode(episode)
            except ValueError as e:
                reject("invalid: " + _reason_key(str(e)))
                continue
            leak = _episode_violation(episode, corpus)
            if leak is None:
                episodes.append(episode)
                break
            reject(f"leak: {leak[0]}")
        else:
            if leak:
                raise LeakageError(f"{ep_id}: {leak[0]}: {leak[1]}")
            raise RuntimeError(f"{ep_id}: no valid {cls} episode after {MAX_REDRAWS} draws")
    if report is not None:
        report.update(draws=draws, accepted=len(episodes), rejected=dict(sorted(rejected.items())))
    leaks = {reason: count for reason, count in rejected.items() if reason.startswith("leak: ")}
    if sum(leaks.values()) > max(MIN_LEAK_REJECTIONS_ALLOWED, int(MAX_LEAK_REJECTION_RATE * draws)):
        raise LeakageError(f"{sum(leaks.values())} of {draws} draws were rejected by the leakage gate ({leaks}); "
                           f"fix the generator or the templates instead of redrawing around them")
    assert_clean(episodes, corpus)
    return episodes


def dataset_stats(episodes: list) -> dict:
    """Counts and shares of a generated set, for the manifest and the training-time estimate.

    Fractions: synthetic_fraction and os_action_fraction over all episodes, blocked_tail_fraction over
    the dh and ds episodes, two_step_fraction over the benign ones. mean_chars / max_context_chars are
    the mean and maximum length of training_text; supervised_fraction is the share of all training
    characters that carry loss (the A: payloads)."""
    def share(count, total):
        return round(count / total, 6) if total else 0.0

    n = len(episodes)
    classes = {cls: sum(e["cls"] == cls for e in episodes) for cls in CLASSES}
    splits = {}
    for e in episodes:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
    texts = [training_text(e) for e in episodes]
    total_chars = sum(len(t) for t in texts)
    supervised = sum(sum(supervised_mask(t)) for t in texts)
    attack = classes["dh"] + classes["ds"]
    return {
        "n": n,
        "class_counts": classes,
        "split_counts": dict(sorted(splits.items())),
        "synthetic_fraction": share(sum(bool(e["meta"]["synthetic"]) for e in episodes), n),
        "os_action_fraction": share(sum(bool(e["meta"]["os_actions"]) for e in episodes), n),
        "blocked_tail_fraction": share(sum(bool(e["meta"]["blocked_tail"]) for e in episodes), attack),
        "two_step_fraction": share(sum(bool(e["meta"]["two_step"]) for e in episodes), classes["benign"]),
        "mean_chars": round(total_chars / n, 2) if n else 0.0,
        "max_context_chars": max((len(t) for t in texts), default=0),
        "supervised_fraction": share(supervised, total_chars),
    }


# ── files ──

def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_episodes(path: str, episodes: list) -> str:
    """One compact JSON object per line, LF line endings on every OS, so the sha256 of the
    file is the same on Windows and in WSL. Keys keep their insertion order (the order is part
    of the gold action text: {"action", "tool", "args"}), and generation is deterministic.
    Returns the file's sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps(ep, separators=(",", ":"), ensure_ascii=True) + "\n")
    os.replace(tmp, path)
    return file_sha256(path)


def read_episodes(path: str) -> list:
    """Load a JSONL episode file, validating every line."""
    episodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episode = json.loads(line)
                validate_episode(episode)
                episodes.append(episode)
    return episodes


# ── command line: python -m src.scratch_llm.dataset ──

MAX_EPISODES = 200000
ALLOWED_OUTPUT_ROOT = PROJECT_ROOT     # --out-dir must stay inside this directory (tests point it at a tmp dir)
DEFAULT_OUT_DIR = "data/local"
MANIFEST_SCHEMA = 2
GATE_RULES = (RULE_HELD_OUT_TOOL, RULE_HELD_OUT_NAME, RULE_HELD_ONLY_PARAM, RULE_SYNTHETIC_SIGNATURE,
              RULE_CANARY, RULE_BANNED, RULE_EMAIL, RULE_SHINGLE, RULE_SENTENCE, RULE_SUMMARY)
CLASS_MIX = {"benign": 0.5, "dh": 0.25, "ds": 0.25}


def _file_entry(path: str, root: str, episodes: list, digest: str) -> dict:
    """The manifest entry of one episode file: repo-relative path, size, hash and the dataset_stats fields."""
    stats = dataset_stats(episodes)
    entry = {"path": os.path.relpath(path, root).replace(os.sep, "/"), "n": stats.pop("n"), "sha256": digest}
    entry.update(stats)
    return entry


def _merge_reports(*reports: dict) -> dict:
    """Sum the draw counts of several generate_episodes reports (the dev set is generated in two halves)."""
    merged = {"draws": 0, "accepted": 0, "rejected": {}}
    for report in reports:
        merged["draws"] += report.get("draws", 0)
        merged["accepted"] += report.get("accepted", 0)
        for reason, count in report.get("rejected", {}).items():
            merged["rejected"][reason] = merged["rejected"].get(reason, 0) + count
    merged["rejected"] = dict(sorted(merged["rejected"].items()))
    return merged


def main(argv=None) -> int:
    """Generate the training and dev episode files and a manifest next to them.

    Writes train-seed{seed}-n{n}.jsonl, dev-seed{seed}-n{dev}.jsonl (first half dev_tools, second half
    dev_synthetic) and manifest.json into --out-dir. The episode files are git-ignored; the manifest (hashes,
    counts, shares, the held-out list hash, the gate verdict and its scope) is the committed record, and it
    holds no environment values and no absolute paths."""
    parser = argparse.ArgumentParser(prog="python -m src.scratch_llm.dataset", description=main.__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=40000, help=f"training episodes (4..{MAX_EPISODES}, default 40000)")
    parser.add_argument("--dev", type=int, default=2000, help=f"dev episodes, an even number (0..{MAX_EPISODES}, default 2000)")
    parser.add_argument("--seed", type=int, default=0, help="generator seed (default 0)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory inside the repository (default data/local)")
    args = parser.parse_args(argv)
    if not 4 <= args.n <= MAX_EPISODES:
        parser.error(f"--n must be between 4 and {MAX_EPISODES}")
    if not 0 <= args.dev <= MAX_EPISODES or args.dev % 2:
        parser.error(f"--dev must be an even number between 0 and {MAX_EPISODES}")
    try:
        out_dir = resolve_under_root(args.out_dir, ALLOWED_OUTPUT_ROOT)
    except ValueError:
        parser.error("--out-dir must stay inside the repository")
    try:
        refuse_foreign_manifest(out_dir, "generator_version", "dataset")        # before generating or writing anything
    except ValueError as e:
        parser.error(f"--out-dir: {e}")
    root = os.path.realpath(ALLOWED_OUTPUT_ROOT)

    train_report, tools_report, synthetic_report = {}, {}, {}
    train = generate_episodes(args.n, args.seed, "train", train_report)
    half = args.dev // 2
    dev = (generate_episodes(half, args.seed, "dev_tools", tools_report)
           + generate_episodes(half, args.seed, "dev_synthetic", synthetic_report))
    dev_report = _merge_reports(tools_report, synthetic_report)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"train-seed{args.seed}-n{args.n}.jsonl")
    dev_path = os.path.join(out_dir, f"dev-seed{args.seed}-n{args.dev}.jsonl")
    train_sha, dev_sha = write_episodes(train_path, train), write_episodes(dev_path, dev)

    dev_names = dev_tool_names()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": git_commit(),
        "code_dirty": git_dirty(),
        "python": platform.python_version(),
        "generator_version": GENERATOR_VERSION,
        "format_version": FORMAT_VERSION,
        "seed": args.seed,
        "held_out_sha256": held_out_sha256(),
        "held_out_count": len(held_out_tools()),
        "held_only_param_count": len(held_only_param_names()),
        "free_pool_count": len(free_tool_specs()),
        "dropped_free_tools": dropped_free_tools(),
        "dev_tool_count": len(dev_names),
        "dev_tools_sha256": hashlib.sha256("\n".join(dev_names).encode("utf-8")).hexdigest(),
        "mix": dict(CLASS_MIX),
        "files": {"train": _file_entry(train_path, root, train, train_sha),
                  "dev": _file_entry(dev_path, root, dev, dev_sha)},
        "gate": {"passed": True, "rules": list(GATE_RULES), "held_only_param_scope": HELD_ONLY_GATE_SCOPE,
                 "rejected": {"train": train_report, "dev": dev_report}},
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    write_json_atomic(manifest_path, manifest)
    for kind in ("train", "dev"):
        entry = manifest["files"][kind]
        print(f"{entry['path']}  n={entry['n']}  sha256={entry['sha256']}")
        print("  " + json.dumps({k: v for k, v in entry.items() if k not in ("path", "n", "sha256")}))
    print(f"{os.path.relpath(manifest_path, root).replace(os.sep, '/')}  written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
