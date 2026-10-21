"""
Tests for the benchmark-leakage gate of ScratchGPT's training data (SCRATCH-02, decision D-04).
Torch-free: no network, no GPU.

Each planted-violation test builds one clean, hand-made episode (not through the generator, so
later generator changes cannot break it), breaks exactly one rule, and asserts that assert_clean
raises LeakageError naming that rule. A clean 2,000-episode generated set must pass.

D-04 scope of the held-only parameter-name rule: names that occur ONLY in held-out benchmark
tools (to, subject, url, ...) are rejected as tool parameter names (the T: line / tool specs)
and as call_tool argument keys - so the model can only produce them by copying from the prompt.
The same ordinary words in free text (U: requests, O: observation values, respond messages) are
allowed and are not gated; test_free_text_may_use_held_only_words pins that boundary.

Every benchmark string a test needs (addresses, attacker sentences, user instructions, held-out
signatures) is read from the benchmark files or from build_corpus() at test time, never pasted.
"""

import json
import os

import pytest

from src.benchmark.injecagent import DATA_DIR, EXFIL_TOOL, load_tool_specs
from src.scratch_llm import dataset, templates
from src.scratch_llm.dataset import (RULE_BANNED, RULE_CANARY, RULE_EMAIL, RULE_HELD_ONLY_PARAM,
                                     RULE_HELD_OUT_NAME, RULE_HELD_OUT_TOOL, RULE_SENTENCE, RULE_SHINGLE,
                                     RULE_SUMMARY, RULE_SYNTHETIC_SIGNATURE, LeakageError, assert_clean,
                                     build_corpus)


# ── helpers ──

def read_jsonl(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def required_names(spec):
    return [p["name"] for p in spec["parameters"] if p.get("required")]


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def base_spec():
    """One real free-pool tool with two or more required string parameters."""
    for spec in dataset.free_tool_specs().values():
        names = required_names(spec)
        if len(names) >= 2 and all(p["type"] == "string" for p in spec["parameters"] if p.get("required")):
            return spec
    raise AssertionError("no suitable free tool")


def make_episode(spec, **overrides):
    """A valid benign episode built by hand from a real tool spec, with top-level overrides applied."""
    names = required_names(spec)
    args = {name: f"val{n}x" for n, name in enumerate(names)}
    given = ", ".join(f"{name.replace('_', ' ')} {args[name]}" for name in names)
    ep = {
        "id": "train-0-000000", "split": "train", "cls": "benign",
        "tools": [dataset.episode_tool(spec)],
        "user": f"Please take care of this request for me with {given}.",
        "turns": [[{"action": "call_tool", "tool": spec["name"], "args": args}, '{"status": "ok", "count": 2}'],
                  [{"action": "respond", "message": "Done, the request is complete."}, None]],
        "meta": {"synthetic": False, "os_actions": False, "blocked_tail": False, "two_step": False,
                 "template_ids": []},
    }
    ep.update(overrides)
    return ep


def expect_rule(ep, rule):
    with pytest.raises(LeakageError) as caught:
        assert_clean([ep])
    assert rule in str(caught.value), str(caught.value)
    assert ep["id"] in str(caught.value)


def long_instruction(rows, key, minimum=9):
    return next(r[key] for r in rows if len(r[key].split()) >= minimum)


# ── the clean case and the corpus ──

def test_a_clean_generated_set_passes_the_gate():
    assert_clean(dataset.generate_episodes(2000, seed=0))


def test_the_hand_made_episode_is_clean(base_spec):
    ep = make_episode(base_spec)
    dataset.validate_episode(ep)
    assert_clean([ep])


def test_corpus_is_derived_from_the_benchmark_files(corpus):
    assert len(corpus.held_out) == 79 and EXFIL_TOOL in corpus.held_out
    assert len(corpus.held_only_params) == 50
    assert len(corpus.emails) >= 6 and all("@" in address for address in corpus.emails)
    benchmark_text = json.dumps([read_jsonl(n) for n in ("user_cases.jsonl", "attacker_cases_dh.jsonl",
                                                         "attacker_cases_ds.jsonl")]).lower()
    assert all(address in benchmark_text for address in corpus.emails)
    assert len(corpus.shingles) > 2000
    shortest = min((r["Attacker Instruction"] for r in read_jsonl("attacker_cases_dh.jsonl")), key=len)
    assert " ".join(dataset._tokens(shortest)) in corpus.exact
    clean = [s for s in corpus.held_signatures if not set(s) & corpus.held_only_params]
    assert len(clean) == 27


def test_gate_scope_constant_states_the_d04_boundary():
    scope = dataset.HELD_ONLY_GATE_SCOPE
    assert "call_tool" in scope and "free text" in scope
    assert "T: line" in scope and "tool specs" in scope


# ── one planted violation per rule ──

def test_rule_1_held_out_tool_in_the_tool_list(base_spec):
    ep = make_episode(base_spec)
    ep["tools"].append(dataset.episode_tool(load_tool_specs()[EXFIL_TOOL]))
    expect_rule(ep, RULE_HELD_OUT_TOOL)


def test_rule_2_call_tool_names_a_held_out_tool(base_spec, corpus):
    ep = make_episode(base_spec)
    ep["turns"][0][0] = {"action": "call_tool", "tool": sorted(corpus.held_out - {EXFIL_TOOL})[0], "args": {}}
    expect_rule(ep, RULE_HELD_OUT_TOOL)


def test_rule_3_benchmark_address_in_the_user_instruction(base_spec, corpus):
    address = sorted(corpus.emails)[0]
    expect_rule(make_episode(base_spec, user=f"Please write it down and keep {address} in mind."), RULE_EMAIL)


def test_rule_4_benchmark_address_in_an_observation(base_spec, corpus):
    ep = make_episode(base_spec)
    ep["turns"][0][1] = f"contact {sorted(corpus.emails)[-1]} for details"
    expect_rule(ep, RULE_EMAIL)


def test_rule_5_canary_shape_in_an_argument(base_spec):
    ep = make_episode(base_spec)
    first = required_names(base_spec)[0]
    ep["turns"][0][0]["args"][first] = "CANARY" + "0123456789AB" + "DATA"
    expect_rule(ep, RULE_CANARY)


def test_rule_6_private_record_phrasing_in_an_observation(base_spec):
    ep = make_episode(base_spec)
    ep["turns"][0][1] = "Private record for somebody else. Number 12"
    expect_rule(ep, RULE_BANNED)


def test_rule_7_eight_words_of_an_attacker_instruction(base_spec):
    attackers = read_jsonl("attacker_cases_dh.jsonl") + read_jsonl("attacker_cases_ds.jsonl")
    words = long_instruction(attackers, "Attacker Instruction").split()[:8]
    expect_rule(make_episode(base_spec, user="Hello there, " + " ".join(words) + " thanks"), RULE_SHINGLE)


def test_rule_8_a_short_benchmark_sentence_verbatim(base_spec):
    attackers = read_jsonl("attacker_cases_dh.jsonl") + read_jsonl("attacker_cases_ds.jsonl")
    shortest = min((r["Attacker Instruction"] for r in attackers), key=lambda s: len(s.split()))
    assert len(shortest.split()) < 8
    ep = make_episode(base_spec)
    ep["turns"][0][1] = "note: " + shortest
    expect_rule(ep, RULE_SENTENCE)


def test_rule_9_held_only_name_as_a_tool_parameter(base_spec, corpus):
    held_only = sorted(corpus.held_only_params)[0]
    ep = make_episode(base_spec)
    ep["tools"].append({"name": "QuillHeldOnlyProbe", "summary": "Probe.",
                        "parameters": [{"name": held_only, "type": "string", "required": True}]})
    expect_rule(ep, RULE_HELD_ONLY_PARAM)


def test_rule_10_held_only_name_as_a_call_tool_argument_key(base_spec, corpus):
    held_only = sorted(corpus.held_only_params)[0]
    ep = make_episode(base_spec)
    ep["turns"][0][0]["args"] = {held_only: "val0x"}
    expect_rule(ep, RULE_HELD_ONLY_PARAM)


def test_rule_11_synthetic_tool_with_a_held_out_signature(base_spec, corpus):
    signature = sorted(s for s in corpus.held_signatures if not set(s) & corpus.held_only_params)[0]
    ep = make_episode(base_spec)
    ep["tools"].append({"name": "QuillSynthSignature", "summary": "Does a thing.",
                        "parameters": [{"name": n, "type": "string", "required": True} for n in signature]})
    assert "QuillSynthSignature" not in corpus.real_tools
    expect_rule(ep, RULE_SYNTHETIC_SIGNATURE)


def test_rule_12_tool_summary_shares_a_benchmark_shingle(base_spec):
    users = read_jsonl("user_cases.jsonl")
    words = long_instruction(users, "User Instruction").split()[:8]
    ep = make_episode(base_spec)
    ep["tools"][0]["summary"] = " ".join(words)
    expect_rule(ep, RULE_SUMMARY)


def test_rule_13_held_out_tool_name_as_a_word_in_an_observation(base_spec):
    ep = make_episode(base_spec)
    ep["turns"][0][1] = f"next you could use {EXFIL_TOOL} for that"
    expect_rule(ep, RULE_HELD_OUT_NAME)


# ── D-04 boundaries ──

def test_free_text_may_use_held_only_words(base_spec, corpus):
    assert {"to", "subject", "url"} <= corpus.held_only_params
    ep = make_episode(base_spec)
    ep["user"] = "Please keep the subject, the url and the to field in mind for me."
    ep["turns"][0][1] = '{"subject": "hello", "to": "somebody", "url": "page"}'
    ep["turns"][1][0]["message"] = "Done with the subject, the url and the to part."
    assert_clean([ep])


def test_a_real_tool_may_share_a_signature_with_a_held_out_tool(corpus):
    real = [s for s in dataset.free_tool_specs().values()
            if tuple(required_names(s)) in corpus.held_signatures and required_names(s)]
    assert real, "expected real free tools whose signature equals a held-out one"
    empty = [s for s in dataset.free_tool_specs().values() if not required_names(s)]
    assert empty
    for spec in (real[0], empty[0]):
        assert spec["name"] in corpus.real_tools
        assert_clean([make_episode(spec)])


def test_free_pool_tools_never_carry_a_held_only_parameter(corpus):
    for spec in dataset.free_tool_specs().values():
        assert not {p["name"] for p in spec["parameters"]} & corpus.held_only_params


def test_dropped_free_tools_are_exposed_and_never_offered():
    dropped = dataset.dropped_free_tools()
    assert isinstance(dropped, list)
    offered = {t["name"] for ep in dataset.generate_episodes(200, seed=1) for t in ep["tools"]}
    assert not offered & set(dropped)
    assert not set(dataset.free_tool_specs()) & set(dropped)


# ── the templates and the gate inside generation ──

def test_every_template_string_passes_the_gate():
    dataset.check_templates()


def test_a_leaking_template_is_caught_before_rendering(monkeypatch, corpus):
    address = sorted(corpus.emails)[0]
    monkeypatch.setattr(templates, "REPLY_TEMPLATES",
                        templates.REPLY_TEMPLATES + [("reply:99", f"Sent it to {address} as asked.")])
    with pytest.raises(LeakageError):
        dataset.check_templates()


def test_generation_raises_when_a_template_leaks(monkeypatch, corpus):
    address = sorted(corpus.emails)[0]
    leaking = [(tid, text + f" Reach me at {address}.") for tid, text in templates.USER_TEMPLATES_TRAIN]
    monkeypatch.setattr(templates, "USER_TEMPLATES_TRAIN", leaking)
    with pytest.raises(LeakageError):
        dataset.generate_episodes(20, seed=0)


def test_the_gate_inside_generation_does_not_rely_on_the_template_scan(monkeypatch, corpus):
    address = sorted(corpus.emails)[0]
    leaking = [(tid, text + f" Reach me at {address}.") for tid, text in templates.USER_TEMPLATES_TRAIN]
    monkeypatch.setattr(templates, "USER_TEMPLATES_TRAIN", leaking)
    monkeypatch.setattr(dataset, "check_templates", lambda corpus=None: None)
    with pytest.raises(LeakageError):
        dataset.generate_episodes(20, seed=0)
