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
import shutil
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
    # gen_check_n / gen_max_new_tokens keep the final generation check of every train() call to two short replies
    values = dict(max_steps=3, micro_batch=2, accum=2, warmup=1, log_every=1, eval_every=1000, dev_eval_n=8,
                  gen_check_n=2, gen_max_new_tokens=8)
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
    assert (manifest["base_model"]["id"], manifest["base_model"]["revision"]) == (os.path.realpath(tiny_dir), "local")
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
    assert len(snapshots) == 5                                   # four in-training evaluations plus the final full-dev pass
    kept = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    assert set(kept) == set(snapshots[1])
    assert all(torch.equal(kept[k], snapshots[1][k]) for k in kept)              # the step-2 weights
    assert any(not torch.equal(kept[k], snapshots[3][k]) for k in kept)          # not the last step's
    assert all(torch.equal(kept[k], snapshots[4][k]) for k in kept)              # the final pass saw the kept weights
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


# ── the command line end to end (tracer of plan 03-03): manifest, polling lines, final dev pass, gen_check ──

SECRET = "qwen-test-secret-5d1f0c7e9a"


def cli_args(train_path, dev_path, out_dir, base, *extra):
    return ["--train", train_path, "--dev", dev_path, "--out-dir", out_dir, "--base-model", base, "--device", "cpu",
            "--max-steps", "4", "--micro-batch", "2", "--accum", "2", "--warmup", "1", "--eval-every", "2",
            "--log-every", "1", "--dev-eval-n", "6", "--final-dev-n", "0", "--gen-check-n", "2",
            "--gen-max-new-tokens", "8", *extra]


def run_main(argv, root):
    """main(argv) with ALLOWED_ROOT pointed at `root`; returns (exit code, stdout lines)."""
    import contextlib
    import io
    buffer = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_module, "ALLOWED_ROOT", str(root))
        mp.setenv("QWEN_TEST_SECRET", SECRET)
        with contextlib.redirect_stdout(buffer):
            code = train_module.main(argv)
    return code, buffer.getvalue().splitlines()


@pytest.fixture(scope="module")
def cli_run(tiny_dir, data_files, tmp_path_factory):
    root = tmp_path_factory.getbasetemp()
    out_dir = str(tmp_path_factory.mktemp("cli_out") / "adapter")
    code, lines = run_main(cli_args(*data_files, out_dir, tiny_dir, "--run-note", "tiny cli test"), root)
    return SimpleNamespace(code=code, lines=lines, out_dir=out_dir, manifest=read_json(os.path.join(out_dir, "manifest.json")),
                           root=root)


def test_cli_prints_the_polling_lines_the_gpu_plan_reads(cli_run):
    assert cli_run.code == 0
    for prefix in ("device cpu trainable ", "step 1/4 ", "eval step 2 ", "eval step 4 ", "final_dev_full n ",
                   "gen_check tuned parse ", "DONE best_step "):
        assert any(line.startswith(prefix) for line in cli_run.lines), (prefix, cli_run.lines)
    assert re.match(r"^device cpu trainable \d+ total \d+$", cli_run.lines[0]), cli_run.lines[0]
    step = next(line for line in cli_run.lines if line.startswith("step 1/4 "))
    assert re.match(r"^step 1/4 loss \S+ lr \S+ tok/s \S+ elapsed \S+s vram \d+/\d+MiB rss \d+MB$", step), step
    final = next(line for line in cli_run.lines if line.startswith("final_dev_full "))
    assert re.match(r"^final_dev_full n 8 dev_action_exact \S+ dev_loss_action \S+$", final), final
    gen = next(line for line in cli_run.lines if line.startswith("gen_check "))
    assert re.match(r"^gen_check tuned parse \d/2 gold \d/2 base parse \d/2 gold \d/2$", gen), gen
    assert cli_run.lines[-1].startswith("DONE best_step ")
    assert cli_run.lines.index(final) < cli_run.lines.index(gen) < len(cli_run.lines) - 1


def test_cli_manifest_has_exactly_the_manifest_keys_in_order(cli_run):
    manifest = cli_run.manifest
    assert tuple(manifest) == train_module.MANIFEST_KEYS
    assert train_module.MANIFEST_KEYS[:4] == ("schema", "created_at", "code_commit", "code_dirty")
    assert manifest["schema"] == train_module.MANIFEST_SCHEMA and manifest["device"] == "cpu"
    assert manifest["adapter_sha256"] == sha256_file(os.path.join(cli_run.out_dir, "adapter_model.safetensors"))
    assert set(manifest["packages"]) == {"transformers", "peft", "accelerate", "tokenizers", "safetensors",
                                         "huggingface_hub"}
    assert all(isinstance(v, str) and v for v in manifest["packages"].values())
    assert manifest["run_notes"][0] == "tiny cli test" and train_module.D01_NOTE in manifest["run_notes"]
    assert manifest["resumed_from_steps"] == [] and manifest["stop_reason"] in ("max_steps", "early_stop")


def test_cli_manifest_records_the_lora_dataset_render_and_selection_values(cli_run, data_files, tokenizer):
    from src.agent_harness.prompting import OBSERVATION_TEMPLATE, SYSTEM_PROMPT
    from src.qwen_lora import data
    from src.scratch_llm.dataset import file_sha256, held_out_sha256, read_episodes
    m = cli_run.manifest
    train_eps, dev_eps = read_episodes(data_files[0]), read_episodes(data_files[1])
    assert m["lora"]["trainable_params"] == lora_param_count(64, 128, 4, 2, 2, 16) == 32768
    assert m["lora"]["r"] == 16 and m["lora"]["alpha"] == 32 and m["lora"]["dropout"] == 0.05
    assert m["lora"]["target_modules"] == sorted(LORA_TARGETS)
    assert m["lora"]["total_params"] > m["lora"]["trainable_params"]
    assert m["lora"]["trainable_fraction"] == round(m["lora"]["trainable_params"] / m["lora"]["total_params"], 6)
    d = m["dataset"]
    assert d["train_path"] == "train.jsonl" and d["dev_path"] == "dev.jsonl"      # outside the repository: the bare name
    assert d["train_sha256"] == file_sha256(data_files[0]) and d["dev_sha256"] == file_sha256(data_files[1])
    assert d["n_train_available"] == len(train_eps) and d["n_dev"] == len(dev_eps) == 8
    assert d["n_train_used"] == sum(d["class_counts"].values()) == 16 and d["dev_eval_n"] == 6
    assert d["held_out_sha256"] == held_out_sha256()
    assert m["render"] == {"fingerprint": chat.render_fingerprint(tokenizer), "version": chat.RENDER_VERSION,
                           "system_prompt_sha256": hashlib_sha256(SYSTEM_PROMPT),
                           "observation_template": OBSERVATION_TEMPLATE, "max_len": data.MAX_LEN}
    assert m["tokenizer"]["chat_template_sha256"] == chat.chat_template_sha256(tokenizer)
    assert m["tokenizer"]["pad_id"] == PAD_ID and m["tokenizer"]["eos_id"] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert m["base_model"]["revision"] == "local" and m["base_model"]["model_safetensors_bytes"] > 0
    assert set(m["peak_vram_mib"]) == {"allocated", "reserved", "total"}
    assert m["config"] == {"train": TrainConfig(max_steps=4, micro_batch=2, accum=2, warmup=1, eval_every=2, log_every=1,
                                                dev_eval_n=6, final_dev_n=0, gen_check_n=2, gen_max_new_tokens=8).to_dict()}
    selection = m["selection"]
    assert selection["metric"] == "dev_loss_action" and selection["direction"] == "min"
    assert selection["remediation_rule"] == train_module.REMEDIATION_RULE
    assert selection["lora04_gate"] == train_module.LORA04_GATE_RULE
    assert "respond" in selection["description"]
    full = m["final_dev_full"]
    assert full["n"] == 8 and set(EVAL_KEYS) <= set(full)
    gen = m["gen_check"]
    assert gen["n"] == 2
    for who in ("tuned", "base"):
        assert set(gen[who]) == {"parse_rate", "gold_first_action_rate"}
        assert all(0.0 <= gen[who][k] <= 1.0 for k in gen[who])
    passed, reason = train_module.lora04_gate(m)
    assert isinstance(passed, bool) and "parse_rate" in reason and "0.90" in reason


def hashlib_sha256(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_cli_manifest_holds_no_environment_value_and_no_dataset_path_outside_the_repository(cli_run):
    with open(os.path.join(cli_run.out_dir, "manifest.json"), encoding="utf-8") as f:
        text = f.read()
    assert SECRET not in text
    assert str(cli_run.root) not in json.dumps(cli_run.manifest["dataset"])      # base_model.id of the tiny dir is a path by design


def test_cli_leaves_the_curve_the_log_and_no_scratch_files(cli_run):
    out = cli_run.out_dir
    assert os.path.getsize(os.path.join(out, "loss_curve.png")) > 1000
    with open(os.path.join(out, "train_log.jsonl"), encoding="utf-8") as f:
        assert len([line for line in f if line.strip()]) == 2
    for name in (train_module.IN_RUN_ADAPTER, "qwen_lora.resume.tmp", "README.md", "manifest.json.tmp"):
        assert not os.path.exists(os.path.join(out, name)), name


def test_cli_output_is_served_by_the_provider_with_every_stamp_check(cli_run, tiny_dir):
    from src.llm.hf_local import HFLocalProvider
    provider = HFLocalProvider("tiny-lora", tiny_dir, adapter=cli_run.out_dir, device="cpu", max_new_tokens=8)
    assert provider.resolved_model.endswith("+tiny-lora@" + cli_run.manifest["adapter_sha256"][:12])


# ── restart-proof and fail-closed (D-06): resume, identity refusal, exit codes, remediation ──

def resume_config(**changes):
    return quick_config(**{**dict(max_steps=6, eval_every=2, micro_batch=2, accum=1, patience=10), **changes})


def interrupt_at(call_number):
    """An evaluate() that raises KeyboardInterrupt at its `call_number`-th call (the real one otherwise)."""
    real, state = train_module.evaluate, {"calls": 0}

    def wrapper(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == call_number:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    return wrapper


def read_log(out_dir):
    with open(os.path.join(out_dir, "train_log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-8)
    return a == b


@pytest.fixture(scope="module")
def straight_run(tiny_dir, data_files, tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("straight") / "adapter")
    manifest = train(data_files[0], data_files[1], out_dir, base_model=tiny_dir, cfg=resume_config(), device="cpu",
                     log=lambda *_: None)
    return SimpleNamespace(out_dir=out_dir, manifest=manifest)


@pytest.fixture(scope="module")
def interrupted(tiny_dir, data_files, tmp_path_factory):
    """A run killed (KeyboardInterrupt) inside its second evaluation, i.e. after the resume file of step 2 exists."""
    out_dir = str(tmp_path_factory.mktemp("interrupted") / "adapter")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_module, "evaluate", interrupt_at(2))
        with pytest.raises(KeyboardInterrupt):
            train(data_files[0], data_files[1], out_dir, base_model=tiny_dir, cfg=resume_config(), device="cpu",
                  log=lambda *_: None)
    return out_dir


def test_resume_a_killed_run_keeps_the_resume_file_and_the_in_run_adapter(interrupted):
    assert os.path.isfile(os.path.join(interrupted, train_module.RESUME_FILE))
    assert os.path.isfile(os.path.join(interrupted, train_module.IN_RUN_ADAPTER, "adapter_model.safetensors"))
    assert not os.path.exists(os.path.join(interrupted, "manifest.json"))
    assert not os.path.exists(os.path.join(interrupted, "adapter_model.safetensors"))


def test_resume_continues_at_the_next_step_and_equals_an_uninterrupted_run(tiny_dir, data_files, interrupted,
                                                                         straight_run, tmp_path):
    from safetensors.torch import load_file
    out_dir = str(tmp_path / "resumed")
    shutil.copytree(interrupted, out_dir)
    logged = []
    manifest = train(data_files[0], data_files[1], out_dir, base_model=tiny_dir, cfg=resume_config(), device="cpu",
                     log=logged.append, resume=True)
    assert "RESUMED from step 2" in logged
    assert [line for line in logged if line.startswith("step ")][0].startswith("step 3/6 ")
    assert manifest["resumed_from_steps"] == [2] and "resumed from step 2" in manifest["run_notes"]
    reference = straight_run.manifest
    assert reference["resumed_from_steps"] == []
    for key in ("steps", "best_step", "stop_reason", "tokens_seen"):
        assert manifest[key] == reference[key], key
    resumed_log, straight_log = read_log(out_dir), read_log(straight_run.out_dir)
    assert [r["step"] for r in resumed_log] == [r["step"] for r in straight_log] == [2, 4, 6]
    for mine, theirs in zip(resumed_log, straight_log):
        assert close({k: v for k, v in mine.items() if k != "elapsed_s"},
                     {k: v for k, v in theirs.items() if k != "elapsed_s"}), (mine, theirs)
    a = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    b = load_file(os.path.join(straight_run.out_dir, "adapter_model.safetensors"))
    assert set(a) == set(b) and all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)
    for name in (train_module.RESUME_FILE, train_module.IN_RUN_ADAPTER):
        assert not os.path.exists(os.path.join(out_dir, name))


def test_resume_refuses_a_different_dev_file_config_or_base_revision_naming_the_field(tiny_dir, data_files, interrupted,
                                                                                    tmp_path):
    other_dev = str(tmp_path / "other_dev.jsonl")
    write_episodes(other_dev, generate_episodes(8, 9, "dev_tools"))
    with pytest.raises(ValueError, match="dev_sha256"):
        train(data_files[0], other_dev, interrupted, base_model=tiny_dir, cfg=resume_config(), device="cpu",
              log=lambda *_: None, resume=True)
    with pytest.raises(ValueError, match="max_steps"):
        train(data_files[0], data_files[1], interrupted, base_model=tiny_dir, cfg=resume_config(max_steps=8),
              device="cpu", log=lambda *_: None, resume=True)
    # the refusals changed nothing: the killed run can still be resumed
    assert os.path.isfile(os.path.join(interrupted, train_module.RESUME_FILE))
    edited = str(tmp_path / "edited")
    shutil.copytree(interrupted, edited)
    path = os.path.join(edited, train_module.RESUME_FILE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["identity"]["base_model"]["revision"] = "somewhere-else"
    torch.save(state, path)
    with pytest.raises(ValueError, match="base_model.revision"):
        train(data_files[0], data_files[1], edited, base_model=tiny_dir, cfg=resume_config(), device="cpu",
              log=lambda *_: None, resume=True)
    assert os.path.isfile(path)


def test_resume_refuses_when_the_resume_file_or_the_in_run_adapter_is_missing(tiny_dir, data_files, interrupted,
                                                                            tmp_path):
    empty = str(tmp_path / "empty")
    with pytest.raises(ValueError, match="qwen_lora.resume.tmp"):
        train(data_files[0], data_files[1], empty, base_model=tiny_dir, cfg=resume_config(), device="cpu",
              log=lambda *_: None, resume=True)
    no_adapter = str(tmp_path / "no_adapter")
    shutil.copytree(interrupted, no_adapter)
    shutil.rmtree(os.path.join(no_adapter, train_module.IN_RUN_ADAPTER))
    with pytest.raises(ValueError, match="adapter.best.tmp"):
        train(data_files[0], data_files[1], no_adapter, base_model=tiny_dir, cfg=resume_config(), device="cpu",
              log=lambda *_: None, resume=True)
    assert os.path.isfile(os.path.join(no_adapter, train_module.RESUME_FILE))


def test_resume_a_fresh_run_removes_stale_files_but_never_the_launchers(tiny_dir, data_files, tmp_path):
    out_dir = tmp_path / "stale"
    (out_dir / train_module.IN_RUN_ADAPTER).mkdir(parents=True)
    (out_dir / train_module.IN_RUN_ADAPTER / "adapter_model.safetensors").write_bytes(b"old")
    for name, content in ((train_module.RESUME_FILE, b"old"), (train_module.RESUME_FILE + ".tmp", b"old"),
                          ("adapter_model.safetensors", b"old"), ("adapter_config.json", b"{}"),
                          ("loss_curve.png", b"old"), ("manifest.json", b'{"adapter_sha256": "old"}')):
        (out_dir / name).write_bytes(content)
    launcher = {"train_progress.log": b"progress\n", "train.done": b"", "gpu_preflight.log": b"reading\n"}
    for name, content in launcher.items():
        (out_dir / name).write_bytes(content)
    manifest = train(data_files[0], data_files[1], str(out_dir), base_model=tiny_dir, cfg=quick_config(max_steps=2),
                     device="cpu", log=lambda *_: None)
    assert manifest["adapter_sha256"] != "old"
    assert (out_dir / "adapter_model.safetensors").read_bytes() != b"old"
    for name in (train_module.IN_RUN_ADAPTER, train_module.RESUME_FILE, train_module.RESUME_FILE + ".tmp"):
        assert not (out_dir / name).exists(), name
    for name, content in launcher.items():
        assert (out_dir / name).read_bytes() == content, name


# failure exits: each leaves nothing loadable

def assert_nothing_loadable(out_dir):
    for name in ("adapter_model.safetensors", "adapter_config.json", "manifest.json", train_module.IN_RUN_ADAPTER,
                 train_module.RESUME_FILE, train_module.RESUME_FILE + ".tmp"):
        assert not os.path.exists(os.path.join(out_dir, name)), name


def failing_main(tiny_dir, data_files, tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("failing") / "adapter")
    code, lines = run_main(cli_args(*data_files, out_dir, tiny_dir), tmp_path_factory.getbasetemp())
    return code, lines, out_dir


def from_call(real, first_bad_call, replacement):
    """`real`, except that the `first_bad_call`-th and every later call returns/raises through `replacement`."""
    state = {"calls": 0}

    def wrapper(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] >= first_bad_call:
            return replacement(real, *args, **kwargs)
        return real(*args, **kwargs)

    return wrapper


def test_non_finite_training_loss_exits_3_and_leaves_nothing_loadable(tiny_dir, data_files, tmp_path_factory,
                                                                    monkeypatch):
    def nan_loss(real, peft_model, input_ids, attention, labels):
        return torch.tensor(float("nan")), 1

    monkeypatch.setattr(train_module, "supervised_loss", from_call(train_module.supervised_loss, 3, nan_loss))
    code, lines, out_dir = failing_main(tiny_dir, data_files, tmp_path_factory)
    assert code == 3 and "FAILED non_finite_loss step 2" in lines
    assert not any(line.startswith("DONE") for line in lines)
    assert_nothing_loadable(out_dir)
    assert read_log(out_dir)[-1] == {"step": 2, "event": "non_finite_loss"}


def test_non_finite_dev_loss_exits_3_and_leaves_nothing_loadable(tiny_dir, data_files, tmp_path_factory, monkeypatch):
    def nan_dev(real, *args, **kwargs):
        return {**real(*args, **kwargs), "dev_loss_action": float("nan")}

    monkeypatch.setattr(train_module, "evaluate", from_call(train_module.evaluate, 2, nan_dev))
    code, lines, out_dir = failing_main(tiny_dir, data_files, tmp_path_factory)
    assert code == 3 and "FAILED non_finite_dev_loss step 4" in lines
    assert_nothing_loadable(out_dir)             # the resume file of step 2 was removed too
    assert read_log(out_dir)[-1] == {"step": 4, "event": "non_finite_dev_loss"}


def test_vram_guard_exits_4_when_reserved_memory_passes_95_percent(tiny_dir, data_files, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(train_module, "read_vram_mib", lambda device: {"allocated": 900, "reserved": 960, "total": 1000})
    code, lines, out_dir = failing_main(tiny_dir, data_files, tmp_path_factory)
    assert code == 4
    assert "FAILED vram_budget reserved 960 total 1000" in lines
    assert_nothing_loadable(out_dir)


def test_vram_guard_stays_quiet_at_the_limit_and_records_the_peak(tiny_dir, data_files, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(train_module, "read_vram_mib", lambda device: {"allocated": 900, "reserved": 950, "total": 1000})
    code, lines, out_dir = failing_main(tiny_dir, data_files, tmp_path_factory)
    assert code == 0 and not any(line.startswith("FAILED") for line in lines)
    assert read_json(os.path.join(out_dir, "manifest.json"))["peak_vram_mib"] == {"allocated": 900, "reserved": 950,
                                                                                  "total": 1000}


def test_cuda_out_of_memory_exits_4_like_the_vram_guard(tiny_dir, data_files, tmp_path_factory, monkeypatch):
    def out_of_memory(real, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(train_module, "supervised_loss", from_call(train_module.supervised_loss, 3, out_of_memory))
    code, lines, out_dir = failing_main(tiny_dir, data_files, tmp_path_factory)
    assert code == 4 and "FAILED cuda_oom step 2" in lines
    assert_nothing_loadable(out_dir)


def test_leak_through_the_command_line_stops_before_any_adapter_file_exists(tiny_dir, data_files, tmp_path_factory):
    spec = load_tool_specs()["GmailSendEmail"]
    names = [p["name"] for p in spec["parameters"] if p.get("required")]
    leaking = dict(generate_episodes(1, 0)[0], id="train-0-leak", tools=[episode_tool(spec)],
                   user="Please send the quarterly note for me.",
                   turns=[[{"action": "call_tool", "tool": "GmailSendEmail", "args": {n: "x1" for n in names}},
                           '{"status": "ok"}'],
                          [{"action": "respond", "message": "Done."}, None]])
    directory = tmp_path_factory.mktemp("leaking_cli")
    bad = str(directory / "leaking.jsonl")
    write_episodes(bad, [leaking])
    out_dir = str(directory / "adapter")
    with pytest.raises(LeakageError):
        run_main(cli_args(bad, data_files[1], out_dir, tiny_dir), tmp_path_factory.getbasetemp())
    assert not os.path.exists(os.path.join(out_dir, "adapter_model.safetensors"))
    assert not os.path.exists(os.path.join(out_dir, "manifest.json"))


def exit_code_of(argv, root):
    with pytest.raises(SystemExit) as stopped:
        run_main(argv, root)
    return stopped.value.code


def test_cli_refuses_paths_outside_the_repository_missing_inputs_and_foreign_manifests(tiny_dir, data_files, tmp_path):
    hub_id = "Qwen/Qwen2.5-0.5B-Instruct"                      # never loaded: every refusal happens before the model
    jail = tmp_path / "jail"
    jail.mkdir()
    shutil.copy(data_files[0], jail / "train.jsonl")
    shutil.copy(data_files[1], jail / "dev.jsonl")
    train_in, dev_in = str(jail / "train.jsonl"), str(jail / "dev.jsonl")
    outside = str(tmp_path / "outside" / "adapter")
    assert exit_code_of(cli_args(train_in, dev_in, outside, hub_id), jail) == 2                  # --out-dir outside the root
    assert not os.path.exists(outside)
    assert exit_code_of(cli_args(data_files[0], dev_in, str(jail / "out"), hub_id), jail) == 2   # --train outside the root
    assert exit_code_of(cli_args(train_in, dev_in, str(jail / "out"), tiny_dir), jail) == 2      # directory --base-model outside
    assert exit_code_of(cli_args(str(jail / "missing.jsonl"), dev_in, str(jail / "out"), hub_id), jail) == 2
    assert not os.path.exists(jail / "out")
    foreign = jail / "foreign"
    foreign.mkdir()
    (foreign / "manifest.json").write_text('{"checkpoint_sha256": "x"}', encoding="utf-8")
    assert exit_code_of(cli_args(train_in, dev_in, str(foreign), hub_id), jail) == 2
    assert (foreign / "manifest.json").read_text(encoding="utf-8") == '{"checkpoint_sha256": "x"}'
    assert os.listdir(foreign) == ["manifest.json"]


# remediation (pre-registered, dev metrics only) and the LORA-04 dev gate

def synthetic_manifest(exact, stop_reason="max_steps", best_step=600, steps=600, every=100):
    return {"final_dev_full": {"dev_action_exact": exact}, "stop_reason": stop_reason, "best_step": best_step,
            "steps": steps, "config": {"train": {"eval_every": every}}}


def test_remediation_fires_only_below_the_threshold_while_still_improving():
    needed, reason = train_module.remediation_needed(synthetic_manifest(0.85))
    assert needed is True
    assert "0.8500" in reason and "0.90" in reason and "max_steps" in reason
    assert train_module.remediation_needed(synthetic_manifest(0.85, stop_reason="early_stop"))[0] is False
    assert train_module.remediation_needed(synthetic_manifest(0.85, best_step=300))[0] is False
    assert train_module.remediation_needed(synthetic_manifest(0.93))[0] is False
    assert train_module.remediation_needed(synthetic_manifest(0.85, best_step=400))[0] is True       # 400 >= 600 - 2 x 100
    assert train_module.remediation_needed(synthetic_manifest(0.85, best_step=399))[0] is False
    converged = train_module.remediation_needed(synthetic_manifest(0.85, stop_reason="early_stop"))[1]
    assert "early_stop" in converged and "0.90" in converged and "0.8500" in converged


def test_remediation_rule_is_pre_registered_text_in_the_code():
    rule = train_module.REMEDIATION_RULE
    assert train_module.REMEDIATION_THRESHOLD == 0.90
    for phrase in ("dev metrics only", "final_dev_full.dev_action_exact", "max_steps", "doubled", "early_stop"):
        assert phrase in rule, phrase


def test_lora04_gate_passes_at_or_above_090_parse_rate_and_fails_when_missing():
    def with_rate(rate):
        return {"gen_check": {"n": 100, "tuned": {"parse_rate": rate, "gold_first_action_rate": 0.5},
                              "base": {"parse_rate": 0.1, "gold_first_action_rate": 0.0}}}

    assert train_module.lora04_gate(with_rate(0.95))[0] is True
    assert train_module.lora04_gate(with_rate(0.90))[0] is True
    passed, reason = train_module.lora04_gate(with_rate(0.85))
    assert passed is False and "0.8500" in reason and "0.90" in reason
    passed, reason = train_module.lora04_gate({"gen_check": None})
    assert passed is False and "gen_check is missing" in reason
