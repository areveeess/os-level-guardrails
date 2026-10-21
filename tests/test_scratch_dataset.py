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

from src.agent_harness.prompting import action_target, parse_action
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark.injecagent import load_tool_specs
from src.policy_engine.engine import PolicyEngine
from src.scratch_llm import dataset, provenance, templates

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
        carried = [v for v in string_values(sink_call["args"]) if v in record]
        assert carried, (ep["id"], "no sink argument carries data from the source observation")
        payload = max(carried, key=len)
        assert any(12 <= len(token) <= 30 for token in payload.split()), (ep["id"], payload)
        sink_tool = next(t for t in ep["tools"] if t["name"] == sink_call["tool"])
        dest = dataset._dest_param(sink_tool, dataset._payload_param(sink_tool))
        if dest:                            # the sink has a recipient-like parameter: it gets the injected address
            recipient = sink_call["args"][dest]
            recipient = recipient[0] if isinstance(recipient, list) else recipient
            assert recipient in EMAIL.findall(injection), (ep["id"], recipient, injection)
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


# ══ breadth and the dev split (synthetic tools, OS actions, blocked tails, two-step, D-04) ══

TEMPLATE_ID = re.compile(r"^[a-z]+:([td])\d*$")


@pytest.fixture(scope="module")
def big():
    """4,000 training episodes: large enough for the share bands below."""
    return dataset.generate_episodes(4000, seed=0)


def called_names(ep):
    names = []
    for call in calls(ep):
        if call["tool"] not in names:
            names.append(call["tool"])
    return names


def test_synthetic_tools_are_a_third_of_the_episodes_and_leak_nothing(big, corpus):
    real, held, held_only = set(load_tool_specs()), dataset.held_out_tools(), dataset.held_only_param_names()
    synthetic = [e for e in big if e["meta"]["synthetic"]]
    assert 0.32 <= len(synthetic) / len(big) <= 0.38
    for ep in synthetic:
        for tool in ep["tools"]:
            assert tool["name"] not in real and tool["name"] not in held
            assert not {p["name"] for p in tool["parameters"]} & held_only
            signature = tuple(required_names(tool))
            assert not signature or signature not in corpus.held_signatures
    for ep in big:
        if not ep["meta"]["synthetic"]:
            assert all(t["name"] in real for t in ep["tools"])


def test_os_action_episodes_are_a_tenth_benign_and_inside_the_default_policy(big, tmp_path):
    os_eps = [e for e in big if e["meta"]["os_actions"]]
    assert 0.08 <= len(os_eps) / len(big) <= 0.12
    policy = PolicyEngine(log_file=str(tmp_path / "policy.jsonl"), verbose=False)
    kinds = set()
    for ep in os_eps:
        assert ep["cls"] == "benign" and len(ep["turns"]) == 2
        action = ep["turns"][0][0]
        kinds.add(action["action"])
        assert action["action"] in ("run_shell", "read_file", "write_file")
        assert parse_action(json.dumps(action)) == action
        target = action_target(action)
        assert "/home/user/workspace" in target
        assert policy.check(action["action"], target) != "BLOCK", (ep["id"], action)
    assert kinds == {"run_shell", "read_file", "write_file"}


def test_blocked_tails_end_with_the_real_broker_text_and_keep_their_class(big):
    attack = [e for e in big if e["cls"] in ("dh", "ds")]
    tails = [e for e in attack if e["meta"]["blocked_tail"]]
    assert 0.06 <= len(tails) / len(attack) <= 0.10
    assert not [e for e in big if e["cls"] == "benign" and e["meta"]["blocked_tail"]]
    for ep in tails:
        user_call, attack_call = calls(ep)[0], calls(ep)[1]
        broker = ToolBroker(None, capabilities={user_call["tool"]})
        assert len(ep["turns"]) == 3
        assert ep["turns"][1][1] == broker.execute(attack_call).as_text()
        assert ep["turns"][1][1].startswith("BLOCKED by tool_broker: Tool '")
        assert ep["turns"][2][0]["action"] == "respond"


def test_two_step_benign_tasks_call_both_tools_named_in_the_instruction(big):
    benign = [e for e in big if e["cls"] == "benign"]
    two = [e for e in benign if e["meta"]["two_step"]]
    assert 0.10 <= len(two) / len(benign) <= 0.20
    for ep in two:
        assert [a["action"] for a, _ in ep["turns"]] == ["call_tool", "call_tool", "respond"]
        assert calls(ep)[0]["tool"] != calls(ep)[1]["tool"]
        assert not ep["meta"]["os_actions"]
        for call in calls(ep):
            for value in string_values(call["args"]):
                assert value in ep["user"]


def test_tool_list_order_and_distractors(big):
    roles = {"benign": 1, "dh": 2, "ds": 3}
    attack = [e for e in big if e["cls"] in ("dh", "ds") and not e["meta"]["blocked_tail"]]
    kept = sum([t["name"] for t in e["tools"]][:len(called_names(e))] == called_names(e) for e in attack)
    assert 0.40 <= kept / len(attack) <= 0.60
    extra = 0
    for ep in big:
        base = 0 if ep["meta"]["os_actions"] else 2 if ep["meta"]["two_step"] else roles[ep["cls"]]
        extra += len(ep["tools"]) > base
        assert len(ep["tools"]) - base in (0, 1, 2)
    assert 0.25 <= extra / len(big) <= 0.35


def test_observations_use_both_quote_styles_and_benign_ones_carry_harmless_text(big):
    plain = '{"success": true}'
    observations = [o for e in big if not e["meta"]["os_actions"] for _, o in e["turns"]
                    if o and o != plain and not o.startswith("BLOCKED")]
    single = sum(o.startswith("{'") for o in observations)
    double = sum(o.startswith('{"') for o in observations)
    assert single + double == len(observations)
    assert single / len(observations) >= 0.25 and double / len(observations) >= 0.25
    benign = [e for e in big if e["cls"] == "benign" and not e["meta"]["os_actions"]]
    harmless = sum(any(text in e["turns"][0][1] for text in templates.BENIGN_TEXTS) for e in benign)
    assert harmless / len(benign) >= 0.25


def test_dev_tools_are_a_stable_tenth_of_the_free_pool_and_never_in_train(big):
    names = dataset.dev_tool_names()
    pool = dataset.free_tool_specs()
    assert names == sorted(names) and set(names) <= set(pool)
    assert len(names) == round(0.10 * len(pool))
    dataset.generate_episodes(8, seed=5)
    assert dataset.dev_tool_names() == names
    used = {t["name"] for e in big for t in e["tools"]}
    assert not used & set(names)
    assert isinstance(dataset.DEV_TOOLS_SEED, int)


def test_dev_tool_set_does_not_depend_on_the_hash_seed():
    code = "import json, src.scratch_llm.dataset as d\nprint(json.dumps(d.dev_tool_names()))\n"
    outputs = []
    for hash_seed in ("3", "4"):
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=300, env=dict(os.environ, PYTHONHASHSEED=hash_seed))
        assert done.returncode == 0, done.stderr
        outputs.append(json.loads(done.stdout))
    assert outputs[0] == outputs[1] == dataset.dev_tool_names()


@pytest.mark.parametrize("split", ["dev_tools", "dev_synthetic"])
def test_dev_splits_keep_the_quota_pass_the_gate_and_use_held_back_text_only(split):
    eps = dataset.generate_episodes(400, 0, split)
    assert dict(collections.Counter(e["cls"] for e in eps)) == {"benign": 200, "dh": 100, "ds": 100}
    assert all(e["split"] == split for e in eps)
    dataset.assert_clean(eps)
    real, dev = set(load_tool_specs()), set(dataset.dev_tool_names())
    for ep in eps:
        dataset.validate_episode(ep)
        for template_id in ep["meta"]["template_ids"]:
            match = TEMPLATE_ID.match(template_id)
            assert not match or match.group(1) == "d", (ep["id"], template_id)
        for tool in ep["tools"]:
            if split == "dev_tools":
                assert tool["name"] not in real or tool["name"] in dev
            else:
                assert tool["name"] not in real
    if split == "dev_synthetic":
        assert all(e["meta"]["synthetic"] for e in eps)
    else:
        assert not any(e["meta"]["synthetic"] for e in eps)
        assert any(t["name"] in dev for e in eps for t in e["tools"])


def test_train_episodes_use_training_templates_only(big):
    seen = set()
    for ep in big:
        for template_id in ep["meta"]["template_ids"]:
            match = TEMPLATE_ID.match(template_id)
            if match:
                seen.add(template_id)
                assert match.group(1) == "t", (ep["id"], template_id)
    assert len(seen) > 150


def test_unknown_split_and_bad_sizes_raise():
    with pytest.raises(ValueError):
        dataset.generate_episodes(10, 0, "nonsense")
    with pytest.raises(ValueError):
        dataset.generate_episodes(-4, 0, "dev_tools")


def test_synthetic_vocabulary_inventory():
    assert len(templates.SYN_TOOLKITS_TRAIN) >= 40 and len(templates.SYN_TOOLKITS_DEV) >= 6
    assert not set(templates.SYN_TOOLKITS_TRAIN) & set(templates.SYN_TOOLKITS_DEV)
    assert len(templates.SYN_VERBS) >= 55 and len(templates.SYN_OBJECTS) >= 75
    with open(os.path.join(PROJECT_ROOT, "benchmarks", "injecagent", "tools.json"), encoding="utf-8") as f:
        real_toolkits = {tk["toolkit"] for tk in json.load(f)}
    assert not (set(templates.SYN_TOOLKITS_TRAIN) | set(templates.SYN_TOOLKITS_DEV)) & real_toolkits
    held_only = dataset.held_only_param_names()
    assert not set(templates.SYN_PARAMS) & held_only and not set(templates.SYN_DEST_NAMES) & held_only
    assert len(templates.OS_TASKS_TRAIN) >= 12 and len(templates.OS_TASKS_DEV) >= 4
    assert len(templates.TWO_STEP_TEMPLATES_TRAIN) >= 8 and len(templates.TWO_STEP_TEMPLATES_DEV) >= 2
    assert len(templates.REPLY_TEMPLATES_DEV) >= 5 and len(templates.BLOCKED_REPLY_TEMPLATES_TRAIN) >= 8


def test_dataset_stats_reports_counts_shares_and_lengths(big):
    stats = dataset.dataset_stats(big)
    assert stats["n"] == 4000 and stats["split_counts"] == {"train": 4000}
    assert stats["class_counts"] == {"benign": 2000, "dh": 1000, "ds": 1000}
    assert 0.32 <= stats["synthetic_fraction"] <= 0.38 and 0.08 <= stats["os_action_fraction"] <= 0.12
    assert 0.06 <= stats["blocked_tail_fraction"] <= 0.10 and 0.10 <= stats["two_step_fraction"] <= 0.20
    assert 300 < stats["mean_chars"] < 1200 and stats["max_context_chars"] <= 2048
    assert 0.1 < stats["supervised_fraction"] < 0.9
    assert dataset.dataset_stats([])["n"] == 0


# ══ the dataset command line, its manifest and the provenance helpers ══

MANIFEST_KEYS = ["schema", "created_at", "code_commit", "code_dirty", "python", "generator_version",
                 "format_version", "seed", "held_out_sha256", "held_out_count", "held_only_param_count",
                 "free_pool_count", "dropped_free_tools", "dev_tool_count", "dev_tools_sha256", "mix", "files",
                 "gate"]


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ALLOWED_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


def run(out_root, *extra, n="40", dev="8", seed="3", out=None):
    out = out or str(out_root)
    assert dataset.main(["--n", n, "--dev", dev, "--seed", seed, "--out-dir", out, *extra]) in (0, None)
    with open(os.path.join(str(out_root), out, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def test_main_writes_both_files_and_a_manifest_that_matches_them(out_root, capsys):
    manifest = run(out_root)
    train, dev = out_root / "train-seed3-n40.jsonl", out_root / "dev-seed3-n8.jsonl"
    assert train.exists() and dev.exists()
    assert manifest["files"]["train"]["sha256"] == dataset.file_sha256(str(train))
    assert manifest["files"]["dev"]["sha256"] == dataset.file_sha256(str(dev))
    assert manifest["files"]["train"]["path"] == "train-seed3-n40.jsonl"
    assert manifest["files"]["train"]["n"] == 40 and manifest["files"]["dev"]["n"] == 8
    assert manifest["files"]["train"]["class_counts"] == {"benign": 20, "dh": 10, "ds": 10}
    splits = [e["split"] for e in dataset.read_episodes(str(dev))]
    assert splits == ["dev_tools"] * 4 + ["dev_synthetic"] * 4
    assert set(e["split"] for e in dataset.read_episodes(str(train))) == {"train"}
    out = capsys.readouterr().out
    assert manifest["files"]["train"]["sha256"] in out and manifest["files"]["dev"]["sha256"] in out


def test_manifest_has_exactly_the_fixed_field_list_and_the_gate_scope(out_root):
    manifest = run(out_root)
    assert list(manifest) == MANIFEST_KEYS
    assert manifest["schema"] == 1 and manifest["seed"] == 3
    assert manifest["held_out_sha256"] == dataset.held_out_sha256() and manifest["held_out_count"] == 79
    assert manifest["held_only_param_count"] == 50
    assert manifest["free_pool_count"] == len(dataset.free_tool_specs())
    assert manifest["dev_tool_count"] == len(dataset.dev_tool_names())
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["dev_tools_sha256"])
    assert manifest["mix"] == {"benign": 0.5, "dh": 0.25, "ds": 0.25}
    assert manifest["generator_version"] == dataset.GENERATOR_VERSION
    assert list(manifest["gate"]) == ["passed", "rules", "held_only_param_scope"]
    assert manifest["gate"]["passed"] is True
    assert manifest["gate"]["held_only_param_scope"] == dataset.HELD_ONLY_GATE_SCOPE
    assert "held-only parameter name" in manifest["gate"]["rules"]
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", manifest["code_commit"])
    assert manifest["code_dirty"] in (True, False, None)


def test_same_arguments_give_the_same_file_hashes(out_root):
    first = run(out_root, out=str(out_root / "a"))
    second = run(out_root, out=str(out_root / "b"))
    for kind in ("train", "dev"):
        assert first["files"][kind]["sha256"] == second["files"][kind]["sha256"]
    other = run(out_root, seed="4", out=str(out_root / "c"))
    assert other["files"]["train"]["sha256"] != first["files"]["train"]["sha256"]


def test_the_manifest_carries_no_environment_values_or_local_paths(out_root, monkeypatch):
    monkeypatch.setenv("FAKE_SCRATCH_SECRET", "sentinel-7f3a9c-do-not-leak")
    run(out_root)
    text = (out_root / "manifest.json").read_text(encoding="utf-8")
    assert "sentinel-7f3a9c-do-not-leak" not in text
    assert str(out_root) not in text and PROJECT_ROOT not in text
    home = os.path.expanduser("~")
    assert len(home) < 4 or home not in text
    assert (out_root / "manifest.json").read_bytes().endswith(b"\n") and b"\r" not in (out_root / "manifest.json").read_bytes()


@pytest.mark.parametrize("arguments", [["--n", "200001"], ["--n", "3"], ["--n", "40", "--dev", "7"],
                                       ["--n", "40", "--dev", "200002"], ["--n", "40", "--dev", "-2"]])
def test_bad_sizes_are_refused_before_anything_is_written(out_root, arguments):
    with pytest.raises(SystemExit) as caught:
        dataset.main([*arguments, "--out-dir", str(out_root)])
    assert caught.value.code not in (0, None)
    assert os.listdir(out_root) == []


@pytest.mark.parametrize("out_dir", ["/etc", "../outside", "sub/../../outside"])
def test_an_out_dir_outside_the_repository_is_refused(tmp_path, monkeypatch, out_dir):
    inside = tmp_path / "repo"
    inside.mkdir()
    monkeypatch.setattr(dataset, "ALLOWED_OUTPUT_ROOT", str(inside))
    with pytest.raises(SystemExit) as caught:
        dataset.main(["--n", "8", "--dev", "4", "--out-dir", out_dir])
    assert caught.value.code not in (0, None)
    assert os.listdir(inside) == [] and not os.path.exists("/etc/manifest.json")
    assert os.listdir(tmp_path) == ["repo"]


def test_the_dataset_command_refuses_an_out_dir_that_holds_a_training_manifest(out_root):
    """WR-06: the trainer's manifest.json must not be overwritten by the dataset command."""
    foreign = out_root / "manifest.json"
    foreign.write_text('{"schema": 3, "checkpoint_sha256": "ab12"}', encoding="utf-8")
    before = foreign.read_bytes()
    with pytest.raises(SystemExit) as caught:
        dataset.main(["--n", "8", "--dev", "4", "--out-dir", str(out_root)])
    assert caught.value.code == 2
    assert foreign.read_bytes() == before and os.listdir(out_root) == ["manifest.json"]
    foreign.unlink()
    run(out_root, n="8", dev="4")
    run(out_root, n="8", dev="4")                  # its own earlier manifest is replaced as before


def test_a_relative_out_dir_is_resolved_under_the_root(out_root):
    manifest = run(out_root, out="nested/out")
    assert manifest["files"]["train"]["path"] == "nested/out/train-seed3-n40.jsonl"
    assert (out_root / "nested" / "out" / "train-seed3-n40.jsonl").exists()


# ── provenance helpers ──

def test_resolve_under_root_refuses_escapes_and_returns_absolute_paths(tmp_path):
    with pytest.raises(ValueError):
        provenance.resolve_under_root("../outside", str(tmp_path))
    with pytest.raises(ValueError):
        provenance.resolve_under_root("/etc", str(tmp_path))
    inside = provenance.resolve_under_root("data/local", str(tmp_path))
    assert os.path.isabs(inside) and inside.startswith(os.path.realpath(str(tmp_path)))
    assert provenance.resolve_under_root(str(tmp_path), str(tmp_path)) == os.path.realpath(str(tmp_path))


def test_git_helpers_fall_back_to_safe_values(monkeypatch):
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", provenance.git_commit())
    assert provenance.git_dirty() in (True, False, None)

    def boom(*args, **kwargs):
        raise OSError("git is not installed")
    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert provenance.git_commit() == "unknown" and provenance.git_dirty() is None


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "x.json"
    provenance.write_json_atomic(str(path), {"b": 1, "a": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"b": 1, "a": [1, 2]}
    assert path.read_text(encoding="utf-8").index('"b"') < path.read_text(encoding="utf-8").index('"a"')
    assert sorted(os.listdir(tmp_path)) == ["x.json"]


def test_provenance_module_is_stdlib_only_and_never_uses_a_shell():
    source = open(os.path.join(PROJECT_ROOT, "src", "scratch_llm", "provenance.py"), encoding="utf-8").read()
    assert "shell=True" not in source and "torch" not in source
