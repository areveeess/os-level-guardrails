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
  - the training maths: our supervised-position loss equals the Hugging Face labels= loss, the LoRA parameter
    count is the one the viva quotes (8,798,208 for the real dimensions), the seven target modules, gradient
    checkpointing still gives every LoRA weight a gradient
  - evaluate(): the action-level metrics, and the kept adapter is the one with the lowest dev_loss_action
    (never the last step), with early stopping after `patience` evaluations without improvement

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
from src.qwen_lora import train as train_module
from src.qwen_lora.data import build_example, collate, gold_supervised_text
from src.qwen_lora.train import (EVAL_KEYS, LORA_TARGETS, PAD_ID, SELECTION_METRIC, TrainConfig, build_lora_model,
                                 evaluate, lora_param_count, supervised_loss, train)
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


# ── the training maths ──

@pytest.fixture(scope="module")
def tiny_lora(tiny_dir):
    """A LoRA model on the tiny base with dropout 0 (so two forward passes agree) and non-zero B matrices."""
    peft_model, _, _ = build_lora_model(tiny_dir, TrainConfig(lora_dropout=0.0), "cpu")
    generator = torch.Generator().manual_seed(0)
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            param.data = torch.randn(param.shape, generator=generator) * 0.02
    return peft_model


@pytest.fixture(scope="module")
def tiny_examples(tokenizer):
    return [build_example(tokenizer, ep) for ep in generate_episodes(6, 7)]


def test_loss_equivalence_our_supervised_position_loss_equals_the_huggingface_labels_loss(tiny_lora, tiny_examples):
    input_ids, attention, labels, _ = collate(tiny_examples, [0, 1, 2], PAD_ID, "cpu")
    tiny_lora.eval()
    with torch.no_grad():
        loss_sum, n = supervised_loss(tiny_lora, input_ids, attention, labels)
        reference = tiny_lora(input_ids=input_ids, attention_mask=attention, labels=labels).loss
    ours = float(loss_sum) / n
    print(f"\nloss equivalence: ours {ours:.8f} huggingface {float(reference):.8f} diff {abs(ours - float(reference)):.2e}")
    assert n == int((labels[:, 1:] != -100).sum())
    assert abs(ours - float(reference)) < 1e-4


def test_param_count_trainable_parameters_equal_the_formula_on_the_tiny_model(tiny_lora):
    trainable = sum(p.numel() for p in tiny_lora.parameters() if p.requires_grad)
    assert trainable == lora_param_count(64, 128, 4, 2, 2, 16) == 32768
    assert not any(p.requires_grad for name, p in tiny_lora.named_parameters() if "lora_" not in name)


def test_param_count_for_the_real_qwen_dimensions_is_8798208():
    assert lora_param_count(896, 4864, 14, 2, 1, 16) == 366_592
    assert lora_param_count(896, 4864, 14, 2, 24, 16) == 8_798_208
    config = read_json(os.path.join(qwen_snapshot_or_skip(), "config.json"))
    assert lora_param_count(config["hidden_size"], config["intermediate_size"], config["num_attention_heads"],
                            config["num_key_value_heads"], config["num_hidden_layers"], 16) == 8_798_208


def test_targets_resolve_to_the_seven_projections_and_the_saved_config_records_the_recipe(tiny_lora, tracer_run):
    def leaves(names):
        return {name.rsplit(".", 1)[-1] for name in names}

    resolved = tiny_lora.peft_config["default"].target_modules      # full names: one per layer and projection
    assert leaves(resolved) == set(LORA_TARGETS) and len(resolved) == 7 * 2       # the tiny model has 2 layers
    assert not any("lm_head" in name or "embed" in name for name in resolved)
    assert LORA_TARGETS == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    saved = read_json(os.path.join(tracer_run.out_dir, "adapter_config.json"))
    assert leaves(saved["target_modules"]) == set(LORA_TARGETS)
    assert (saved["r"], saved["lora_alpha"], saved["lora_dropout"]) == (16, 32, 0.05)


def test_checkpointing_every_lora_parameter_gets_a_finite_gradient_and_the_base_gets_none(tiny_lora, tiny_examples):
    input_ids, attention, labels, _ = collate(tiny_examples, [0, 1], PAD_ID, "cpu")
    tiny_lora.train()
    assert tiny_lora.base_model.model.model.gradient_checkpointing
    loss_sum, n = supervised_loss(tiny_lora, input_ids, attention, labels)
    (loss_sum / n).backward()
    for name, param in tiny_lora.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and bool(torch.isfinite(param.grad).all()), name
        else:
            assert param.grad is None, name
    assert any(float(p.grad.abs().sum()) > 0 for p in tiny_lora.parameters() if p.requires_grad)
    tiny_lora.zero_grad(set_to_none=True)


# ── evaluate(): action-level metrics ──

def standin_logits_fn(examples):
    """Logits whose argmax is the gold token everywhere except inside respond message text (where it is wrong)."""
    by_ids = {tuple(e.input_ids): e for e in examples}

    def logits_fn(peft_model, input_ids, attention_mask, labels):
        targets = labels[:, 1:]
        positions = targets != -100
        rows, cols = positions.nonzero(as_tuple=True)
        logits = torch.zeros(int(rows.numel()), 151936)
        for k, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
            n = int(attention_mask[row].sum())
            example = by_ids[tuple(input_ids[row, :n].tolist())]
            gold = int(targets[row, col])
            if example.action_labels[col + 1] == -100:                  # respond message text: argmax is wrong
                logits[k, (gold + 1) % 151936] = 10.0
            else:
                logits[k, gold] = 10.0
        return logits, targets[positions], positions

    return logits_fn


def test_evaluate_returns_exactly_the_eval_keys_and_ignores_respond_message_text(tiny_lora, tokenizer):
    dev = [build_example(tokenizer, ep) for ep in generate_episodes(12, 5, "dev_synthetic")]
    result = evaluate(tiny_lora, dev, 2, "cpu", logits_fn=standin_logits_fn(dev))
    assert tuple(result) == EVAL_KEYS
    assert result["dev_action_exact"] == 1.0                              # the wrong argmax sits only in respond text
    assert result["dev_first_action_exact"] == 1.0
    assert result["dev_line_exact"] < 1.0                                 # the respond lines are not exact as a whole
    present = {e.cls for e in dev}
    assert set(result["dev_action_exact_by_cls"]) == set(result["dev_line_exact_by_cls"]) == present
    assert all(v == 1.0 for v in result["dev_action_exact_by_cls"].values())
    assert result["dev_loss_action"] < result["dev_loss_respond_text"]
    assert result["dev_loss_action"] < result["dev_loss"] < result["dev_loss_respond_text"]
    assert all(math.isfinite(v) for v in result.values() if isinstance(v, float))


def test_evaluate_counts_a_wrong_action_token_and_restores_the_training_mode(tiny_lora, tokenizer):
    dev = [build_example(tokenizer, ep) for ep in generate_episodes(4, 6, "dev_tools")]

    def always_wrong(peft_model, input_ids, attention_mask, labels):
        targets = labels[:, 1:]
        positions = targets != -100
        logits = torch.zeros(int(positions.sum()), 151936)
        logits[torch.arange(logits.shape[0]), (targets[positions] + 1) % 151936] = 10.0
        return logits, targets[positions], positions

    tiny_lora.train()
    result = evaluate(tiny_lora, dev, 2, "cpu", logits_fn=always_wrong)
    assert tiny_lora.training
    assert result["dev_action_exact"] == 0.0 and result["dev_line_exact"] == 0.0
    assert result["dev_first_action_exact"] == 0.0
    tiny_lora.eval()
    evaluate(tiny_lora, dev, 2, "cpu", logits_fn=always_wrong)
    assert not tiny_lora.training


def test_evaluate_runs_the_real_model_and_every_value_is_finite(tiny_lora, tokenizer):
    dev = [build_example(tokenizer, ep) for ep in generate_episodes(4, 6, "dev_tools")]
    result = evaluate(tiny_lora, dev, 2, "cpu")
    assert tuple(result) == EVAL_KEYS
    assert all(math.isfinite(result[k]) for k in ("dev_loss", "dev_loss_action", "dev_loss_respond_text"))
    assert 0.0 <= result["dev_line_exact"] <= result["dev_action_exact"] <= 1.0      # action-exact is the looser test


# ── selection and early stopping on dev_loss_action (D-06) ──

def scripted_eval(values, snapshots):
    """An evaluate() whose dev_loss_action follows `values` while the plain dev_loss falls at every call, so
    selecting on the plain loss (or on the last step) would keep a different adapter than the right metric does."""
    remaining = iter(values)

    def fake_evaluate(peft_model, examples, micro_batch, device, logits_fn=None):
        from peft import get_peft_model_state_dict
        snapshots.append({k: v.detach().clone() for k, v in get_peft_model_state_dict(peft_model).items()})
        action = next(remaining)
        return {"dev_loss": 5.0 - 0.1 * len(snapshots), "dev_loss_action": action, "dev_loss_respond_text": 9.0,
                "dev_action_exact": 0.5, "dev_action_exact_by_cls": {"benign": 0.5}, "dev_line_exact": 0.1,
                "dev_line_exact_by_cls": {"benign": 0.1}, "dev_first_action_exact": 0.5}

    return fake_evaluate


def test_selection_keeps_the_lowest_dev_loss_action_and_stops_after_patience(tiny_dir, data_files, tmp_path,
                                                                            monkeypatch):
    from safetensors.torch import load_file
    snapshots = []
    values =[0.9, 0.5, 0.7, 0.8, 0.6, 0.6, 0.6, 0.6]
    monkeypatch.setattr(train_module, "evaluate", scripted_eval(values, snapshots))
    out_dir = str(tmp_path / "selected")
    manifest = train(data_files[0], data_files[1], out_dir, base_model=tiny_dir,
                     cfg=quick_config(max_steps=8, eval_every=1, patience=2, micro_batch=2, accum=1), device="cpu",
                     log=lambda *_: None)
    assert SELECTION_METRIC == "dev_loss_action"
    assert manifest["stop_reason"] == "early_stop" and manifest["steps"] == 4
    assert manifest["best_step"] == 2 and manifest["best"]["step"] == 2
    assert manifest["best"]["dev_loss_action"] == 0.5 and manifest["last_eval"]["step"] == 4
    assert len(snapshots) == 4
    kept = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    assert set(kept) == set(snapshots[1])
    assert all(torch.equal(kept[k], snapshots[1][k]) for k in kept)              # the step-2 weights
    assert any(not torch.equal(kept[k], snapshots[3][k]) for k in kept)          # not the last step's
    assert manifest["adapter_sha256"] == sha256_file(os.path.join(out_dir, "adapter_model.safetensors"))
    assert not os.path.exists(os.path.join(out_dir, train_module.IN_RUN_ADAPTER))


def test_selection_runs_to_max_steps_when_the_metric_keeps_improving(tiny_dir, data_files, tmp_path, monkeypatch):
    snapshots = []
    monkeypatch.setattr(train_module, "evaluate", scripted_eval([0.9, 0.8, 0.7, 0.6, 0.5], snapshots))
    manifest = train(data_files[0], data_files[1], str(tmp_path / "improving"), base_model=tiny_dir,
                     cfg=quick_config(max_steps=3, eval_every=1, patience=2, micro_batch=2, accum=1), device="cpu",
                     log=lambda *_: None)
    assert manifest["stop_reason"] == "max_steps" and manifest["steps"] == 3 and manifest["best_step"] == 3


def test_evaluate_a_real_tiny_run_writes_one_log_record_and_one_eval_line_per_evaluation(tiny_dir, data_files, tmp_path):
    out_dir = str(tmp_path / "logged")
    logged = []
    manifest = train(data_files[0], data_files[1], out_dir, base_model=tiny_dir,
                     cfg=quick_config(max_steps=4, eval_every=2, micro_batch=2, accum=1), device="cpu",
                     log=logged.append)
    with open(os.path.join(out_dir, "train_log.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert [r["step"] for r in records] == [2, 4]
    for record in records:
        for key in ("step", "train_loss", "lr", "tokens_seen", "elapsed_s", *EVAL_KEYS):
            assert key in record, key
        assert math.isfinite(record["train_loss"]) and math.isfinite(record["dev_loss_action"])
    evals = [line for line in logged if line.startswith("eval step ")]
    assert len(evals) == 2
    pattern = r"^eval step (\d+) dev_loss (\S+) dev_loss_action (\S+) dev_action_exact (\S+) best (\S+)$"
    assert all(re.match(pattern, line) for line in evals), evals
    assert manifest["best"]["step"] in (2, 4) and manifest["last_eval"]["step"] == 4
    assert set(EVAL_KEYS) <= set(manifest["best"]) and set(EVAL_KEYS) <= set(manifest["last_eval"])
