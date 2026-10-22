"""
Tests for the Qwen LoRA trainer (src/qwen_lora/train.py) and the data code it uses (src/qwen_lora/data.py).
CPU only: a tiny random-init Qwen2 (hidden size 64, 2 layers) next to the real cached Qwen tokenizer, outputs
under tmp_path, no network, no GPU.

What is pinned here, cheapest first:
  - the tracer: a handful of optimizer steps of our own loop on the tiny model write an adapter plus manifest,
    and src.llm.hf_local serves that adapter through run_benchmark (the whole path from Phase 2 episodes to a
    stamped benchmark row, so a wrong mask, render skew or manifest field shows up here and not after a
    41-minute GPU run)
  - the leakage gate: a training file with a held-out tool stops the run before any adapter file exists

Importing transformers and peft costs about 75 s once per pytest process, so the tokenizer, the tiny base model
and the trained tracer run are module-scoped fixtures and every test keeps its runs to a few steps.
"""

import json
import math
import os
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.qwen_fixtures import build_tiny_qwen, qwen_snapshot_or_skip, require_hf

require_hf()

from src.benchmark.injecagent import load_tool_specs
from src.llm.hf_local import sha256_file
from src.qwen_lora import chat
from src.qwen_lora.data import build_example, gold_supervised_text
from src.qwen_lora.train import TrainConfig, train
from src.scratch_llm.dataset import LeakageError, episode_tool, generate_episodes, write_episodes
from tests.test_hf_local_provider import point_runner_at, raw_rows, run_one_case, write_hf_models_yaml_entries


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(qwen_snapshot_or_skip(), local_files_only=True)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return build_tiny_qwen(str(tmp_path_factory.mktemp("tiny_base")))


@pytest.fixture(scope="module")
def data_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("episodes")
    train_path, dev_path = str(directory / "train.jsonl"), str(directory / "dev.jsonl")
    write_episodes(train_path, generate_episodes(16, 0))
    write_episodes(dev_path, generate_episodes(8, 1, "dev_tools"))
    return train_path, dev_path


def quick_config(**changes):
    values = dict(max_steps=3, micro_batch=2, accum=2, warmup=1, log_every=1, eval_every=1000, dev_eval_n=8)
    values.update(changes)
    return TrainConfig(**values)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tracer_run(tiny_dir, data_files, tmp_path_factory):
    """One real run of train() on the CPU: three optimizer steps of micro-batch 2 x accumulation 2."""
    out_dir = str(tmp_path_factory.mktemp("tracer_out") / "adapter")
    logged = []
    manifest = train(data_files[0], data_files[1], out_dir, base_model=tiny_dir, cfg=quick_config(), device="cpu",
                     log=logged.append)
    return SimpleNamespace(out_dir=out_dir, manifest=manifest, log=logged)


# ── tracer: episodes -> chat render -> offset mask -> LoRA steps -> adapter + manifest -> served by hf_local ──

def test_tracer_episode_mask_supervises_exactly_the_gold_actions(tokenizer):
    episodes = generate_episodes(6, 3)
    for ep in episodes:
        example = build_example(tokenizer, ep)
        supervised = [i for i, label in zip(example.input_ids, example.labels) if label != -100]
        assert tokenizer.decode(supervised) == gold_supervised_text(ep)
        assert example.episode_id == ep["id"] and example.cls == ep["cls"]


def test_tracer_tiny_lora_adapter_is_served_in_a_benchmark_episode(tracer_run, tiny_dir, tokenizer, tmp_path, monkeypatch):
    out_dir, manifest = tracer_run.out_dir, tracer_run.manifest
    for name in ("adapter_config.json", "adapter_model.safetensors", "manifest.json"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    assert not os.path.exists(os.path.join(out_dir, "README.md"))
    config = read_json(os.path.join(out_dir, "adapter_config.json"))
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (16, 32, 0.05)
    assert config["base_model_name_or_path"] == os.path.realpath(tiny_dir)

    on_disk = read_json(os.path.join(out_dir, "manifest.json"))
    assert on_disk == manifest
    assert manifest["adapter_sha256"] == sha256_file(os.path.join(out_dir, "adapter_model.safetensors"))
    assert manifest["base_model"] == {"id": os.path.realpath(tiny_dir), "revision": "local"}
    assert manifest["render"]["fingerprint"] == chat.render_fingerprint(tokenizer)

    steps = [line for line in tracer_run.log if line.startswith("step ")]
    assert len(steps) == 3
    for s, line in enumerate(steps, start=1):
        match = re.match(rf"^step {s}/3 loss (\S+) lr (\S+) tok/s (\S+) elapsed (\S+)s", line)
        assert match, line
        assert math.isfinite(float(match.group(1)))

    config_path = tmp_path / "models.yaml"
    write_hf_models_yaml_entries(config_path, {"tiny-lora": dict(base_model=tiny_dir, adapter=out_dir, device="cpu",
                                                                 max_new_tokens=8)})
    point_runner_at(monkeypatch, config_path)
    bench_out = str(tmp_path / "bench")
    run_one_case("tiny-lora", bench_out, lambda *_: None)
    (row,) = raw_rows(bench_out)
    assert row["resolved_model"].endswith("+tiny-lora@" + manifest["adapter_sha256"][:12])
    assert row["llm_error"] is None


# ── leakage gate (D-02) ──

def test_leak_a_held_out_tool_in_the_training_file_stops_the_run_before_any_adapter_exists(tiny_dir, data_files,
                                                                                         tmp_path):
    spec = load_tool_specs()["GmailSendEmail"]
    names = [p["name"] for p in spec["parameters"] if p.get("required")]
    leaking = dict(generate_episodes(1, 0)[0], id="train-0-leak", tools=[episode_tool(spec)],
                   user="Please send the quarterly note for me.",
                   turns=[[{"action": "call_tool", "tool": "GmailSendEmail", "args": {n: "x1" for n in names}},
                           '{"status": "ok"}'],
                          [{"action": "respond", "message": "Done."}, None]])
    bad = str(tmp_path / "leaking.jsonl")
    write_episodes(bad, [leaking])
    out_dir = str(tmp_path / "leak_out")
    with pytest.raises(LeakageError):
        train(bad, data_files[1], out_dir, base_model=tiny_dir, cfg=quick_config(), device="cpu", log=lambda *_: None)
    for name in ("adapter_config.json", "adapter_model.safetensors", "manifest.json"):
        assert not os.path.exists(os.path.join(out_dir, name))
