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


def _observation(rng: random.Random, returns: list) -> str:
    """The text the harness would show for a successful call, built from the tool's return schema."""
    payload = {r["name"]: _return_value(rng, r["name"], r["type"]) for r in returns}
    while payload:
        text = json.dumps(payload, ensure_ascii=True)
        if len(text) <= OBS_CHARS:
            break
        payload.popitem()
    else:
        text = json.dumps({"success": True})
    return Observation("success", output=text).as_text()


# ── instructions and replies ──

def _request_words(name: str) -> tuple:
    """(verb, object words, toolkit words), lower case, from a CamelCase tool name."""
    toolkit = toolkit_of(name)
    rest = name[len(toolkit):] if toolkit and name.startswith(toolkit) else name
    words = [w.lower() for w in _CAMEL.findall(rest)] or [name.lower()]
    return words[0], words[1:], [w.lower() for w in _CAMEL.findall(toolkit)]


def _plain_request(name: str) -> str:
    verb, obj, _ = _request_words(name)
    return " ".join([verb] + obj)


def _request_phrase(rng: random.Random, name: str) -> str:
    """The request as a verb phrase: the tool's own words (about 70%) or a synonym verb (about 30%),
    sometimes naming the application."""
    verb, obj, toolkit = _request_words(name)
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


# ── the benign episode ──

def _gen_pool() -> tuple:
    """((name, spec), ...) sorted by name: the tools the generator may offer."""
    return _free_pool()


def _benign_episode(rng: random.Random, ep_id: str, split: str, pool: tuple) -> dict:
    name, spec = pool[rng.randrange(len(pool))]
    required = [p for p in spec["parameters"] if p.get("required")]
    args = {p["name"]: _arg_value(rng, p["name"], p["type"]) for p in required}
    user_id, user_template = rng.choice(templates.USER_TEMPLATES_TRAIN)
    reply_id, reply_template = rng.choice(templates.REPLY_TEMPLATES)
    user = _tidy(fill(user_template, do=_request_phrase(rng, name), args=_arguments_text(rng, args)))
    message = _tidy(fill(reply_template, do=_plain_request(name)))
    return {
        "id": ep_id,
        "split": split,
        "cls": "benign",
        "tools": [episode_tool(spec)],
        "user": user,
        "turns": [[{"action": "call_tool", "tool": name, "args": args}, _observation(rng, spec["returns"])],
                  [{"action": "respond", "message": message}, None]],
        "meta": {"synthetic": False, "os_actions": False, "blocked_tail": False, "two_step": False,
                 "template_ids": [user_id, reply_id]},
    }


def generate_episodes(n: int, seed: int = 0, split: str = "train") -> list:
    """Return `n` training episodes (schema above), reproducible for a given (seed, split).

    This slice emits benign episodes from real non-benchmark tools; the attack classes, synthetic
    tools and the dev splits are added behind the same signature. The leakage gate runs on the
    whole batch before it is returned, so a leaking template makes generation raise instead of
    producing data.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if split != "train":
        raise NotImplementedError("only the train split is generated in this slice")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    rng = random.Random(seed * 1_000_003 + SPLITS.index(split))
    pool = _gen_pool()
    episodes = []
    for i in range(n):
        ep_id = f"{split}-{seed}-{i:06d}"
        for _ in range(MAX_REDRAWS):
            episode = _benign_episode(rng, ep_id, split, pool)
            try:
                validate_episode(episode)
            except ValueError:
                continue
            episodes.append(episode)
            break
        else:
            raise RuntimeError(f"{ep_id}: no valid episode after {MAX_REDRAWS} draws")
    assert_clean(episodes)
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
