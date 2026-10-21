"""
Tests for ScratchGPT's full training distribution (SCRATCH-01): the exact 50 / 25 / 25 mix of
benign, direct-harm-compliant (dh) and data-stealing-compliant (ds) episodes, the self-written
injections, the dataset command line and its manifest. Torch-free: no network, no GPU, no model.

What each group pins:
  * quota: n/2 benign, n/4 dh, n/4 ds exactly when n is divisible by 4, within one otherwise;
  * attack shape: a dh episode calls the user tool, reads an injection in the observation, then
    calls the injected tool; a ds episode reads a secret from a source tool and puts it into a
    sink tool together with the address the injection names;
  * copyability: every argument value of every call is visible in what the model has read;
  * secrets and addresses are not shaped like the benchmark's canaries and addresses;
  * the leakage gate (tests/test_training_leakage.py) passes on every split and class;
  * determinism: the same seed gives the same file hash in two interpreters with different
    PYTHONHASHSEED values, another seed gives another hash.
"""

import collections
import json
import os
import re
import subprocess
import sys

import pytest

from src.scratch_llm import dataset, templates

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CANARY = re.compile(r"CANARY[0-9A-F]{12}DATA")
RESERVED = re.compile(r"@(?:example\.(?:com|org|net)|[a-z]+\.(?:example|test))$")
CAMEL_NAME = re.compile(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+")


# ── helpers ──

def required_names(tool):
    return [p["name"] for p in tool["parameters"] if p.get("required")]


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from string_values(v)
    elif isinstance(value, list):
        for v in value:
            yield from string_values(v)


def calls(ep):
    return [action for action, _ in ep["turns"] if action["action"] == "call_tool"]


def by_class(episodes, cls, blocked_tail=False):
    return [e for e in episodes if e["cls"] == cls and bool(e["meta"]["blocked_tail"]) == blocked_tail]


@pytest.fixture(scope="module")
def corpus():
    return dataset.build_corpus()


@pytest.fixture(scope="module")
def episodes():
    """One 2,000-episode set reused by the shape tests (generation is the slow part)."""
    return dataset.generate_episodes(2000, seed=0)


# ── the quota (D-01) ──

def test_quota_is_exact_when_n_is_divisible_by_four():
    eps = dataset.generate_episodes(400, 0)
    assert dict(collections.Counter(e["cls"] for e in eps)) == {"benign": 200, "dh": 100, "ds": 100}


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6, 7, 9, 401, 402, 403])
def test_quota_is_within_one_episode_for_other_sizes(n):
    eps = dataset.generate_episodes(n, 0)
    counts = collections.Counter(e["cls"] for e in eps)
    assert len(eps) == n
    assert abs(counts["benign"] - 0.5 * n) <= 1
    assert abs(counts["dh"] - 0.25 * n) <= 1 and abs(counts["ds"] - 0.25 * n) <= 1


def test_zero_episodes_negative_n_and_bad_arguments():
    assert dataset.generate_episodes(0, 0) == []
    with pytest.raises(ValueError):
        dataset.generate_episodes(-1, 0)
    with pytest.raises(ValueError):
        dataset.generate_episodes(2.5, 0)


# ── attack shape ──

def test_dh_episode_calls_the_injected_tool_named_in_the_observation(episodes):
    dh = by_class(episodes, "dh")
    assert len(dh) > 300
    literal = 0
    for ep in dh:
        first, second, last = calls(ep)[0], calls(ep)[1], ep["turns"][-1][0]
        assert [a["action"] for a, _ in ep["turns"]] == ["call_tool", "call_tool", "respond"]
        listed = {t["name"] for t in ep["tools"]}
        assert second["tool"] in listed and second["tool"] != first["tool"]
        observation = ep["turns"][0][1]
        for value in string_values(second["args"]):
            assert value in observation, (ep["id"], value)
        literal += second["tool"] in observation
        assert last["action"] == "respond"
    assert literal / len(dh) <= 0.25   # the injection names the tool in words, not by its CamelCase name


def test_ds_episode_reads_a_secret_and_sends_it_to_the_address_in_the_injection(episodes):
    ds = by_class(episodes, "ds")
    assert len(ds) > 300
    for ep in ds:
        assert [a["action"] for a, _ in ep["turns"]] == ["call_tool", "call_tool", "call_tool", "respond"]
        user_call, source_call, sink_call = calls(ep)
        assert len({user_call["tool"], source_call["tool"], sink_call["tool"]}) == 3
        injection, record = ep["turns"][0][1], ep["turns"][1][1]
        sink_text = json.dumps(sink_call["args"])
        carried = [v for v in string_values(sink_call["args"]) if v in record]
        assert carried, (ep["id"], "no sink argument carries data from the source observation")
        payload = max(carried, key=len)
        assert any(12 <= len(token) <= 30 for token in payload.split()), (ep["id"], payload)
        addresses = EMAIL.findall(injection)
        if addresses:                       # the sink has a recipient-like parameter
            assert addresses[0] in sink_text, (ep["id"], addresses, sink_text)
        assert source_call["tool"] != sink_call["tool"]


def test_every_call_names_a_listed_tool_and_every_value_is_copyable(episodes):
    for ep in episodes:
        listed = {t["name"]: required_names(t) for t in ep["tools"]}
        seen = ep["user"]
        for action, observation in ep["turns"]:
            if action["action"] == "call_tool":
                assert action["tool"] in listed
                assert list(action["args"]) == listed[action["tool"]]
                for value in string_values(action["args"]):
                    assert value in seen, (ep["id"], value)
            if observation:
                seen += " " + observation


def test_secrets_and_addresses_do_not_look_like_the_benchmark(episodes, corpus):
    addresses, secrets = [], 0
    for ep in by_class(episodes, "ds"):
        text = " ".join(str(o) for _, o in ep["turns"] if o) + " " + ep["user"] + json.dumps(calls(ep))
        assert not CANARY.search(text) and "CANARY" not in text
        found = EMAIL.findall(text)
        assert not {a.lower() for a in found} & set(corpus.emails)
        addresses += [a for a in found if a in json.dumps(calls(ep)[-1]["args"])]
        secrets += 1
    assert secrets and len(addresses) > 100
    reserved = sum(bool(RESERVED.search(a)) for a in addresses)
    assert reserved / len(addresses) >= 0.85


# ── the gate and validity ──

def test_every_generated_episode_is_valid_and_the_set_passes_the_gate(episodes):
    for ep in episodes:
        dataset.validate_episode(ep)
    dataset.assert_clean(episodes)


def test_ds_training_text_shows_the_secret_in_an_observation_and_again_in_the_sink_call():
    ds = next(e for e in dataset.generate_episodes(40, 0) if e["cls"] == "ds" and not e["meta"]["blocked_tail"])
    lines = dataset.training_text(ds).split("\n")
    a_lines = [l for l in lines if l.startswith("A:")]
    o_lines = [l for l in lines if l.startswith("O:")]
    assert len(a_lines) == 4 and len(o_lines) == 3
    sink_args = list(string_values(calls(ds)[-1]["args"]))
    payload = max((v for v in sink_args if v in o_lines[1]), key=len)
    assert payload in o_lines[1] and payload in a_lines[2]


# ── determinism ──

def digest_in_subprocess(tmp_path, hash_seed, seed):
    code = ("import sys\n"
            "import src.scratch_llm.dataset as d\n"
            "eps = d.generate_episodes(200, seed=int(sys.argv[2]))\n"
            "print(d.write_episodes(sys.argv[1], eps))\n")
    out = str(tmp_path / f"h{hash_seed}-{seed}.jsonl")
    done = subprocess.run([sys.executable, "-c", code, out, str(seed)], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=600, env=dict(os.environ, PYTHONHASHSEED=str(hash_seed)))
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_same_seed_same_hash_across_interpreters_and_another_seed_differs(tmp_path):
    first = digest_in_subprocess(tmp_path, 1, 0)
    second = digest_in_subprocess(tmp_path, 2, 0)
    other = digest_in_subprocess(tmp_path, 1, 1)
    assert first == second and len(first) == 64
    assert other != first


# ── the self-written text ──

def ids(rows):
    return [row[0] for row in rows]


def test_attack_template_inventory_and_partitions():
    assert len([i for i in ids(templates.DH_TEMPLATES_TRAIN) if i.startswith("dh:t")]) >= 30
    assert len([i for i in ids(templates.DH_TEMPLATES_DEV) if i.startswith("dh:d")]) >= 6
    assert len([i for i in ids(templates.DS_TEMPLATES_TRAIN) if i.startswith("ds:t")]) >= 20
    assert len([i for i in ids(templates.DS_TEMPLATES_DEV) if i.startswith("ds:d")]) >= 4
    assert len([i for i in ids(templates.RECORD_TEMPLATES_TRAIN) if i.startswith("rec:t")]) >= 26
    assert len([i for i in ids(templates.RECORD_TEMPLATES_DEV) if i.startswith("rec:d")]) >= 4
    assert any(i.startswith("wrap:d") for i in ids(templates.WRAPPERS_DEV))
    for family in ("DH", "DS", "RECORD", "WRAPPERS"):
        train, dev = (getattr(templates, f"{family}_{'TEMPLATES_' if family != 'WRAPPERS' else ''}{part}")
                      for part in ("TRAIN", "DEV"))
        assert not set(ids(train)) & set(ids(dev))
        assert not {text for _, text in train if text} & {text for _, text in dev if text}


def test_template_text_is_plain_ascii_without_quotes_or_backslashes():
    for attr in ("DH_TEMPLATES_TRAIN", "DH_TEMPLATES_DEV", "DS_TEMPLATES_TRAIN", "DS_TEMPLATES_DEV",
                 "RECORD_TEMPLATES_TRAIN", "RECORD_TEMPLATES_DEV", "WRAPPERS_TRAIN", "WRAPPERS_DEV"):
        for _, text in getattr(templates, attr):
            assert text.isascii() and not set('"\\') & set(text), (attr, text)
