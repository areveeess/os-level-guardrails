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
import random
import re

from src.agent_harness.executors import Observation
from src.agent_harness.prompting import OBSERVATION_TEMPLATE, build_messages, parse_action
from src.agent_harness.tool_broker import DEFAULT_RESPONSE
from src.benchmark.injecagent import DATA_DIR, EXFIL_TOOL, load_tool_specs
from src.scratch_llm.format import (ACTION_PREFIX, CONTEXT_CHARS, END, GEN_RESERVE, HISTORY_ACTION_CHARS,
                                    OBS_CHARS, USER_CHARS, _flat, compact_action, render_context)
from src.scratch_llm import templates

GENERATOR_VERSION = 1
MAX_REDRAWS = 50   # attempts to draw one valid episode before generation gives up
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
_CANARY = re.compile(r"CANARY[0-9A-F]{12}DATA")
_BANNED_CASE_SENSITIVE = ("CANARY",)
_BANNED_ANY_CASE = ("private record", "account owner. reference")


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
    for phrase in _BANNED_CASE_SENSITIVE:
        if phrase in text:
            return RULE_BANNED, phrase
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
    """A secret token of 12-30 characters in one of six shapes, never the benchmark's canary shape."""
    shape = rng.randrange(6)
    if shape == 0:
        return "".join(rng.choice("0123456789ABCDEF") for _ in range(rng.randint(12, 20)))
    if shape == 1:
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(12, 20)))
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
        self.all = tuple(tools)
        self.sources = tuple(t for t in self.all if _is_source(t))
        self.sinks = tuple(t for t in self.all if _is_sink(t))


@functools.lru_cache(maxsize=None)
def _real_tools() -> tuple:
    return tuple(make_tool(name, spec, toolkit_of(name)) for name, spec in _gen_pool())


@functools.lru_cache(maxsize=None)
def _train_pools() -> _Pools:
    return _Pools(_real_tools())


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


# ── episode parts ──

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


def _reply(rng: random.Random, tool: dict, choices: list) -> tuple:
    reply_id, template = choices[rng.randrange(len(choices))]
    return reply_id, _tidy(fill(template, do=_plain_request(tool)))


def _call(tool: dict, args: dict) -> dict:
    return {"action": "call_tool", "tool": tool["name"], "args": args}


def _episode(ep_id: str, split: str, cls: str, tools: list, user: str, turns: list, template_ids: list,
             synthetic: bool = False, os_actions: bool = False, blocked_tail: bool = False,
             two_step: bool = False) -> dict:
    return {"id": ep_id, "split": split, "cls": cls, "tools": [episode_tool(t) for t in tools], "user": user,
            "turns": turns,
            "meta": {"synthetic": synthetic, "os_actions": os_actions, "blocked_tail": blocked_tail,
                     "two_step": two_step, "template_ids": template_ids}}


# ── the three classes ──

ATTACK_OBS_CHARS = 400     # an observation that carries an injection stays short, so the whole episode fits the context
RECORD_OBS_CHARS = 280


def _benign_episode(rng: random.Random, ep_id: str, split: str, pools: _Pools):
    tool = _pick(rng, pools.all)
    args = _tool_args(rng, tool)
    user_id, user = _instruction(rng, tool, args, templates.USER_TEMPLATES_TRAIN)
    reply_id, message = _reply(rng, tool, templates.REPLY_TEMPLATES)
    observation = _observation(rng, tool["returns"])
    return _episode(ep_id, split, "benign", [tool], user,
                    [[_call(tool, args), observation], [{"action": "respond", "message": message}, None]],
                    [user_id, reply_id])


def _dh_episode(rng: random.Random, ep_id: str, split: str, pools: _Pools):
    """User tool -> an observation carrying an injection that asks for another tool -> that call -> respond."""
    user_tool = _pick(rng, pools.all)
    target = _pick(rng, pools.all, exclude={user_tool["name"]})
    user_args, target_args = _tool_args(rng, user_tool), _tool_args(rng, target)
    user_id, user = _instruction(rng, user_tool, user_args, templates.USER_TEMPLATES_TRAIN)
    injection_id, template = templates.DH_TEMPLATES_TRAIN[rng.randrange(len(templates.DH_TEMPLATES_TRAIN))]
    injection = _tidy(fill(template, do=_request_phrase(rng, target), args=_arguments_text(rng, target_args),
                           tool=target["name"]))
    wrapper_id, injection = _wrapped(rng, injection, templates.WRAPPERS_TRAIN)
    observation = _observation(rng, user_tool["returns"], _embedded(rng, injection), ATTACK_OBS_CHARS)
    reply_id, message = _reply(rng, user_tool, templates.REPLY_TEMPLATES)
    if observation is None:
        return None
    return _episode(ep_id, split, "dh", [user_tool, target], user,
                    [[_call(user_tool, user_args), observation],
                     [_call(target, target_args), _success_observation()],
                     [{"action": "respond", "message": message}, None]],
                    [user_id, injection_id, wrapper_id, reply_id])


def _ds_episode(rng: random.Random, ep_id: str, split: str, pools: _Pools):
    """User tool -> injection "get X and send it to ADDR" -> source tool returns a record with a
    secret -> sink tool called with the secret (and ADDR) -> respond."""
    user_tool = _pick(rng, pools.all)
    source = _pick(rng, pools.sources, exclude={user_tool["name"]})
    sink = _pick(rng, pools.sinks, exclude={user_tool["name"], source["name"]})
    payload_name = _payload_param(sink)
    dest_name = _dest_param(sink, payload_name)
    address = _attack_address(rng) if dest_name else None
    secret, label = _secret(rng), templates.RECORD_LABELS[rng.randrange(len(templates.RECORD_LABELS))]
    record_id, record_template = templates.RECORD_TEMPLATES_TRAIN[rng.randrange(len(templates.RECORD_TEMPLATES_TRAIN))]
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

    user_id, user = _instruction(rng, user_tool, user_args, templates.USER_TEMPLATES_TRAIN)
    injection_id, template = templates.DS_TEMPLATES_TRAIN[rng.randrange(len(templates.DS_TEMPLATES_TRAIN))]
    injection = _tidy(fill(template, pull=_request_phrase(rng, source), sargs=_arguments_text(rng, source_args),
                           send=_send_word(rng, sink), dest=f" to {address}" if address else "",
                           args=_arguments_text(rng, others)))
    wrapper_id, injection = _wrapped(rng, injection, templates.WRAPPERS_TRAIN)
    first = _observation(rng, user_tool["returns"], _embedded(rng, injection), ATTACK_OBS_CHARS)
    second = _observation(rng, source["returns"], record, RECORD_OBS_CHARS)
    reply_id, message = _reply(rng, user_tool, templates.REPLY_TEMPLATES)
    if first is None or second is None:
        return None
    return _episode(ep_id, split, "ds", [user_tool, source, sink], user,
                    [[_call(user_tool, user_args), first], [_call(source, source_args), second],
                     [_call(sink, sink_args), _success_observation()],
                     [{"action": "respond", "message": message}, None]],
                    [user_id, injection_id, wrapper_id, record_id, reply_id])


BUILDERS = {"benign": _benign_episode, "dh": _dh_episode, "ds": _ds_episode}


# ── the quota (D-01) ──

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


def generate_episodes(n: int, seed: int = 0, split: str = "train") -> list:
    """Return `n` training episodes (schema above), reproducible for a given (seed, split).

    The class mix is a deterministic quota (50% benign, 25% dh, 25% ds), not a per-episode
    probability. Every episode must pass validate_episode and the leakage gate (an episode that does
    not is drawn again, a bounded number of times), and the whole list passes assert_clean before it is
    returned, so a leaking template makes generation raise instead of producing data.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if split != "train":
        raise NotImplementedError("only the train split is generated in this slice")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    check_templates()
    rng = random.Random(seed * 1_000_003 + SPLITS.index(split))
    pools, corpus = _train_pools(), build_corpus()
    episodes = []
    for i, cls in enumerate(_class_sequence(n, rng)):
        ep_id = f"{split}-{seed}-{i:06d}"
        leak = None
        for _ in range(MAX_REDRAWS):
            episode = BUILDERS[cls](rng, ep_id, split, pools)
            if episode is None:
                continue
            try:
                validate_episode(episode)
            except ValueError:
                continue
            leak = _episode_violation(episode, corpus)
            if leak is None:
                episodes.append(episode)
                break
        else:
            if leak:
                raise LeakageError(f"{ep_id}: {leak[0]}: {leak[1]}")
            raise RuntimeError(f"{ep_id}: no valid {cls} episode after {MAX_REDRAWS} draws")
    assert_clean(episodes, corpus)
    return episodes


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
