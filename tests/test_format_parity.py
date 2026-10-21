"""
Tests for the compact transcript format (version 2) and the train/inference parity of
ScratchGPT's data (SCRATCH-01 and SCRATCH-03). Torch-free: no network, no GPU, no model.

What is proved here:
  * the T: line carries each tool's required parameter names, parsed from the very system
    prompt the harness builds (all 330 vendored tool specs round-trip);
  * training text is the inference renderer's output plus the gold action, so for every step
    the prompt a model reads at inference is an exact prefix of its training text - also when
    a real AgentLoop builds the messages, including the harness's own blocked-tool text;
  * the loss mask covers exactly the payload of the A: lines;
  * the version, vocab hash and render fingerprint stamps exist and notice drift;
  * the generator is reproducible and writes LF-only JSONL whose sha256 matches across OSes.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

import src.agent_harness.prompting as prompting
from src.agent_harness.executors import Executor, Observation
from src.agent_harness.loop import AgentLoop
from src.agent_harness.prompting import OBSERVATION_TEMPLATE, build_messages
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark.injecagent import load_tool_specs
from src.llm.base import LLMProvider, LLMResponse
from src.scratch_llm import dataset, templates
from src.scratch_llm.format import (CONTEXT_CHARS, GEN_RESERVE, OBS_CHARS, STOI, UNKNOWN, VOCAB, VOCAB_SHA256,
                                    FORMAT_VERSION, compact_action, encode, render_context, render_fingerprint,
                                    render_header, supervised_mask, tools_from_system)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Recorded once from the implementation. If this changes, the text a model reads has changed:
# bump FORMAT_VERSION and retrain every checkpoint.
GOLDEN_FINGERPRINT = "650ca1c4193c2e07e9fdee4990c4fb37b2edd7964aa5026a64c53b9af8fadf04"


@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()


@pytest.fixture(scope="module")
def free_specs():
    """Real non-benchmark tools whose required parameters are all strings, at least two of them."""
    pool = dataset.free_tool_specs()
    chosen = [s for s in pool.values()
              if len([p for p in s["parameters"] if p.get("required")]) >= 2
              and all(p["type"] == "string" for p in s["parameters"] if p.get("required"))]
    assert len(chosen) >= 2
    return chosen


# ── helpers ──

def required_names(spec):
    return [p["name"] for p in spec["parameters"] if p.get("required")]


def make_benign_episode(spec, ep_id="train-0-000000", obs='{"status": "ok", "count": 2}'):
    """A schema-complete one-call episode built by hand from a real tool spec (not by the generator)."""
    names = required_names(spec)
    args = {name: f"val{n}x" for n, name in enumerate(names)}
    given = " and ".join(f"{name.replace('_', ' ')} {args[name]}" for name in names)
    return {
        "id": ep_id, "split": "train", "cls": "benign",
        "tools": [dataset.episode_tool(spec)],
        "user": f"Please take care of this request for me with {given}.",
        "turns": [[{"action": "call_tool", "tool": spec["name"], "args": args}, obs],
                  [{"action": "respond", "message": "Done, the request is complete."}, None]],
        "meta": {"synthetic": False, "os_actions": False, "blocked_tail": False, "two_step": False,
                 "template_ids": []},
    }


def prefix_ends(text):
    """Index just after each line-initial 'A:' (the end of each step's inference prompt)."""
    ends, pos = [], 0
    for line in text.split("\n"):
        if line.startswith("A:"):
            ends.append(pos + 2)
        pos += len(line) + 1
    return ends


class GoldProvider(LLMProvider):
    """Replies with the gold actions in order and records every conversation it is shown."""

    name = "gold"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, temperature=0.0, max_tokens=512):
        self.seen.append([dict(m) for m in messages])
        return LLMResponse(text=self.replies.pop(0), latency_ms=0.0, model=self.name)


class GoldExecutor(Executor):
    def __init__(self, observations):
        self.observations = list(observations)

    def execute(self, action):
        return Observation("success", output=self.observations.pop(0))


def replay(ep):
    """Run the episode through a real AgentLoop; return the messages of every provider call."""
    provider = GoldProvider([compact_action(a) for a, _ in ep["turns"]])
    executor = GoldExecutor([o for _, o in ep["turns"][:-1]])
    AgentLoop(provider, executor, max_steps=5, tools=ep["tools"]).run(ep["user"])
    return provider.seen


def assert_prefix_parity(ep):
    text = dataset.training_text(ep)
    ends = prefix_ends(text)
    assert len(ends) == len(ep["turns"])
    for k, end in enumerate(ends):
        assert render_context(dataset.episode_to_messages(ep, k)) == text[:end], f"step {k}"


def assert_replay_parity(ep):
    text = dataset.training_text(ep)
    ends = prefix_ends(text)
    seen = replay(ep)
    assert len(seen) == len(ep["turns"])
    for k, messages in enumerate(seen):
        assert render_context(messages) == text[:ends[k]], f"step {k}"


def run_python(code):
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=300)


# ── the T: line ──

def test_render_header_puts_required_parameters_on_the_t_line():
    header = render_header([("GmailSendEmail", ["to", "subject", "body"]), ("FooBarList", [])], "hi")
    assert header == "T:GmailSendEmail(to,subject,body)|FooBarList()\nU:hi\n"


def test_tools_from_system_reads_required_parameters_in_order():
    spec = {"name": "FooDoThing", "summary": "Do a thing.",
            "parameters": [{"name": "a", "type": "string", "required": True},
                           {"name": "c", "type": "integer", "required": False},
                           {"name": "b", "type": "array", "required": True}]}
    assert tools_from_system(build_messages("x", [spec])[0]["content"]) == [("FooDoThing", ["a", "b"])]
    assert tools_from_system(build_messages("x")[0]["content"]) == []
    assert render_context(build_messages("do it", [spec])) == "T:FooDoThing(a,b)\nU:do it\nA:"


def test_every_vendored_tool_spec_round_trips_through_the_t_line(specs):
    assert len(specs) == 330
    for name, spec in specs.items():
        parsed = tools_from_system(build_messages("x", [spec])[0]["content"])
        assert parsed == [(name, required_names(spec))], name


# ── the hand-built episode, parity and the loss mask ──

def test_hand_built_episode_is_valid(free_specs):
    ep = make_benign_episode(free_specs[0])
    dataset.validate_episode(ep)
    assert dataset.training_text(ep).startswith("T:" + free_specs[0]["name"] + "(")


def test_every_step_prompt_is_a_prefix_of_the_training_text(free_specs):
    assert_prefix_parity(make_benign_episode(free_specs[0]))


def test_supervised_mask_decodes_to_the_gold_actions(free_specs):
    ep = make_benign_episode(free_specs[0])
    text = dataset.training_text(ep)
    mask = supervised_mask(text)
    assert len(mask) == len(text)
    supervised = "".join(ch for ch, m in zip(text, mask) if m)
    assert supervised == "".join(compact_action(a) + "\n" for a, _ in ep["turns"])


def test_supervised_mask_is_zero_outside_action_payloads(free_specs):
    ep = make_benign_episode(free_specs[0])
    text = dataset.training_text(ep)
    mask = supervised_mask(text)
    pos = 0
    for line in text.split("\n"):
        for i in range(len(line)):
            if line.startswith("A:") and i >= 2:
                assert mask[pos + i] == 1
            else:
                assert mask[pos + i] == 0, f"{line[:2]} line char {i}"
        if pos + len(line) < len(text):      # the newline that ends the line
            assert mask[pos + len(line)] == (1 if line.startswith("A:") else 0)
        pos += len(line) + 1


def test_agent_loop_replay_matches_the_training_prefixes(free_specs):
    assert_replay_parity(make_benign_episode(free_specs[0]))


def test_agent_loop_replay_matches_with_a_real_tool_broker_denial(free_specs):
    first, second = free_specs[0], free_specs[1]
    broker = ToolBroker(None, capabilities={first["name"]})
    second_args = {name: f"val{n}x" for n, name in enumerate(required_names(second))}
    denial = broker.execute({"action": "call_tool", "tool": second["name"], "args": second_args}).as_text()
    assert denial.startswith("BLOCKED by tool_broker: Tool '")
    ep = make_benign_episode(first)
    ep["tools"].append(dataset.episode_tool(second))
    ep["turns"].insert(1, [{"action": "call_tool", "tool": second["name"], "args": second_args}, denial])
    dataset.validate_episode(ep)
    assert_prefix_parity(ep)
    assert_replay_parity(ep)


# ── validation (planner-authored edge predicates) ──

def _too_long_context(ep):
    call = ep["turns"][0]
    ep["turns"] = [[call[0], "x" * 600] for _ in range(4)] + [ep["turns"][-1]]


def _doubled_space(ep):
    ep["turns"][-1][0]["message"] = "two  spaces"


def _long_user(ep):
    ep["user"] = "word " * 60


def _six_turns(ep):
    call = ep["turns"][0]
    ep["turns"] = [[call[0], "ok"] for _ in range(5)] + [ep["turns"][-1]]


def _unlisted_tool(ep):
    ep["turns"][0][0]["tool"] = "NotListedAnywhere"


def _wrong_arg_keys(ep):
    ep["turns"][0][0]["args"] = {"wrong_key": "v"}


def _empty_observation(ep):
    ep["turns"][0][1] = ""


def _non_ascii(ep):
    ep["turns"][0][1] = "café ok"


def _last_not_respond(ep):
    ep["turns"][-1] = [ep["turns"][0][0], None]


@pytest.mark.parametrize("mutate", [_too_long_context, _doubled_space, _long_user, _six_turns, _unlisted_tool,
                                    _wrong_arg_keys, _empty_observation, _non_ascii, _last_not_respond])
def test_validate_episode_rejects(free_specs, mutate):
    ep = make_benign_episode(free_specs[0])
    dataset.validate_episode(ep)
    mutate(ep)
    with pytest.raises(ValueError, match=ep["id"]):
        dataset.validate_episode(ep)


def test_render_context_cuts_old_turns_at_a_line_boundary(free_specs):
    spec = free_specs[0]
    messages = build_messages("do it", [spec])
    call = {"action": "call_tool", "tool": spec["name"], "args": {n: "v" for n in required_names(spec)}}
    for _ in range(5):
        messages.append({"role": "assistant", "content": compact_action(call)})
        messages.append({"role": "user", "content": OBSERVATION_TEMPLATE.format(result="y" * 600)})
    context = render_context(messages)
    header = render_header(tools_from_system(messages[0]["content"]), "do it")
    assert len(context) <= CONTEXT_CHARS - GEN_RESERVE + 2
    assert context.startswith(header)
    assert context.endswith("A:")
    for line in context[len(header):].split("\n")[:-1]:
        assert line.startswith("A:") or line.startswith("O:")


def test_non_ascii_text_becomes_unknown_without_an_exception():
    assert encode("aé中") == [STOI["a"], UNKNOWN, UNKNOWN]
    messages = build_messages("café 中")
    assert isinstance(render_context(messages), str)


# ── stamps ──

def test_vocab_stamp_matches_the_vocabulary():
    assert len(VOCAB) == 96
    assert VOCAB_SHA256 == hashlib.sha256("".join(VOCAB).encode()).hexdigest()
    assert FORMAT_VERSION == 2 and CONTEXT_CHARS == 2048 and GEN_RESERVE == 400 and OBS_CHARS == 640


def test_render_fingerprint_is_pinned():
    assert render_fingerprint() == GOLDEN_FINGERPRINT, \
        "the rendered text changed: bump FORMAT_VERSION, retrain, then re-pin the fingerprint"


def test_render_fingerprint_notices_a_changed_observation_template(monkeypatch):
    before = render_fingerprint()
    monkeypatch.setattr(prompting, "OBSERVATION_TEMPLATE", "Output:\n{result}")
    assert render_fingerprint() != before


# ── the leakage gate on the hand-built episode ──

def test_hand_built_episode_is_clean_and_a_held_out_tool_is_rejected(free_specs, specs):
    ep = make_benign_episode(free_specs[0])
    dataset.assert_clean([ep])
    ep["tools"].append(dataset.episode_tool(specs["GmailSendEmail"]))
    with pytest.raises(dataset.LeakageError):
        dataset.assert_clean([ep])


# ── import hygiene ──

def test_format_and_dataset_import_without_torch():
    done = run_python("import sys\nimport src.scratch_llm.format, src.scratch_llm.dataset\n"
                      "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert done.returncode == 0, done.stderr


def test_format_alone_does_not_import_the_benchmark_or_dataset():
    done = run_python("import sys\nimport src.scratch_llm.format\n"
                      "bad = [m for m in ('src.benchmark', 'src.scratch_llm.dataset') if m in sys.modules]\n"
                      "sys.exit(1 if bad else 0)")
    assert done.returncode == 0, done.stderr


# ── the generator (SCRATCH-01): reproducible, valid, gated, LF JSONL ──

def string_values(value):
    """Every string inside a JSON-like value (call_tool argument values may be nested)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from string_values(v)
    elif isinstance(value, list):
        for v in value:
            yield from string_values(v)


def test_generated_episodes_follow_the_schema():
    episodes = dataset.generate_episodes(8, seed=0)
    assert len(episodes) == 8
    held = dataset.held_out_tools()
    for ep in episodes:
        assert set(ep) == set(dataset.EPISODE_KEYS)
        assert ep["split"] == "train" and ep["cls"] in dataset.CLASSES
        assert ep["turns"][-1][0]["action"] == "respond" and ep["turns"][-1][1] is None
        listed = {t["name"]: required_names(t) for t in ep["tools"]}
        assert not set(listed) & held
        for action, _ in ep["turns"]:
            if action["action"] == "call_tool":
                assert list(action["args"]) == listed[action["tool"]]
        dataset.validate_episode(ep)


def test_generated_argument_values_are_copyable_from_the_prompt():
    for ep in dataset.generate_episodes(60, seed=3):
        seen = ep["user"]
        for action, observation in ep["turns"]:
            if action["action"] == "call_tool":
                for value in string_values(action["args"]):
                    assert value in seen, (ep["id"], value, seen)
            if observation:
                seen += " " + observation


def test_generated_episodes_satisfy_parity_mask_and_replay():
    for ep in dataset.generate_episodes(50, seed=0):
        assert_prefix_parity(ep)
        assert_replay_parity(ep)
        text = dataset.training_text(ep)
        supervised = "".join(ch for ch, m in zip(text, supervised_mask(text)) if m)
        assert supervised == "".join(compact_action(a) + "\n" for a, _ in ep["turns"])


def test_write_and_read_round_trip_with_lf_only_and_a_file_hash(tmp_path):
    episodes = dataset.generate_episodes(8, seed=0)
    path = str(tmp_path / "eps.jsonl")
    digest = dataset.write_episodes(path, episodes)
    raw = open(path, "rb").read()
    assert digest == hashlib.sha256(raw).hexdigest() == dataset.file_sha256(path)
    assert b"\r" not in raw
    assert dataset.read_episodes(path) == json.loads(json.dumps(episodes))


def test_same_seed_gives_the_same_hash_and_another_seed_a_different_one(tmp_path):
    def digest(seed, name):
        return dataset.write_episodes(str(tmp_path / name), dataset.generate_episodes(8, seed=seed))
    assert digest(0, "a.jsonl") == digest(0, "b.jsonl")
    assert digest(1, "c.jsonl") != digest(0, "a.jsonl")


def test_generation_does_not_depend_on_the_hash_seed(tmp_path):
    code = ("import sys\n"
            "import src.scratch_llm.dataset as d\n"
            "d.write_episodes(sys.argv[1], d.generate_episodes(20, seed=0))\n")
    digests = []
    for hash_seed in ("1", "2"):
        out = str(tmp_path / f"h{hash_seed}.jsonl")
        done = subprocess.run([sys.executable, "-c", code, out], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=300, env=dict(os.environ, PYTHONHASHSEED=hash_seed))
        assert done.returncode == 0, done.stderr
        digests.append(dataset.file_sha256(out))
    assert digests[0] == digests[1]


def test_generated_set_passes_the_gate_and_an_unknown_split_is_refused():
    dataset.assert_clean(dataset.generate_episodes(8, seed=0))
    with pytest.raises(ValueError):
        dataset.generate_episodes(4, 0, "nonsense")


def test_template_inventory_is_large_enough():
    train = [t for t, _ in templates.USER_TEMPLATES_TRAIN if t.startswith("user:t")]
    dev = [t for t, _ in templates.USER_TEMPLATES_DEV if t.startswith("user:d")]
    replies = [t for t, _ in templates.REPLY_TEMPLATES if t.startswith("reply:")]
    assert len(train) >= 40 and len(set(train)) == len(train)
    assert len(dev) >= 8 and not set(dev) & set(train)
    assert len(replies) >= 15
