"""
Tests for the ScratchGPT training script (src/scratch_llm/train.py).
CPU only, tiny models (2 layers, d_model 32-64), outputs under tmp_path, no network, no GPU.

What is pinned here, cheapest first:
  - the learning-rate schedule and the token-budget batching (pure maths)
  - the loss mask: decoding the supervised targets of a batch gives exactly the agent's own actions
  - the model is causal and right padding never changes a real position's logits
  - a 60-step micro-run has a finite, falling loss; four episodes can be memorised to the point
    that greedy decoding through the saved checkpoint reproduces every gold action line
  - the manifest has a fixed field list and no environment values
  - failure paths: a leaking training file, a path outside the repository, a non-finite loss
"""

import hashlib
import json
import math
import os
import random

import pytest

torch = pytest.importorskip("torch")

from src.scratch_llm import train as train_module
from src.scratch_llm.dataset import (LeakageError, episode_tool, episode_to_messages, generate_episodes,
                                     write_episodes)
from src.benchmark.injecagent import load_tool_specs
from src.scratch_llm.format import compact_action, decode
from src.scratch_llm.inference import load_generator
from src.scratch_llm.model import GPTConfig, ScratchGPT
from src.scratch_llm.train import (MANIFEST_KEYS, NonFiniteLossError, TrainConfig, collate, encode_episode, evaluate,
                                   lr_at, main, make_batches, train)

TINY = dict(n_layer=2, n_head=2, d_model=32, dropout=0.0)
CLI_TINY = ["--n-layer", "2", "--n-head", "2", "--d-model", "32", "--dropout", "0.0", "--device", "cpu",
            "--max-steps", "10", "--eval-every", "5", "--log-every", "5", "--batch-tokens", "4096", "--warmup", "3"]


@pytest.fixture(scope="module")
def episodes():
    return generate_episodes(64, 0)


@pytest.fixture(scope="module")
def dev_episodes():
    return generate_episodes(8, 1, "dev_tools")


@pytest.fixture
def files(tmp_path, episodes, dev_episodes):
    train_path, dev_path = str(tmp_path / "train.jsonl"), str(tmp_path / "dev.jsonl")
    write_episodes(train_path, episodes)
    write_episodes(dev_path, dev_episodes)
    return train_path, dev_path


def quiet(*_):
    pass


def quick_config(**changes):
    values = dict(max_steps=20, eval_every=10, log_every=5, batch_tokens=4096, warmup=5)
    values.update(changes)
    return TrainConfig(**values)


def read_log(out_dir):
    with open(os.path.join(out_dir, "train_log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── schedule and batching ──

def test_learning_rate_warms_up_peaks_and_decays_to_the_floor():
    peak, floor = 1e-3, 1e-4
    assert lr_at(0, 1000, peak, floor, 200) == pytest.approx(peak / 200)
    assert lr_at(199, 1000, peak, floor, 200) == pytest.approx(peak)
    assert lr_at(200, 1000, peak, floor, 200) == pytest.approx(peak)
    assert abs(lr_at(999, 1000, peak, floor, 200) - floor) < 1e-9
    values = [lr_at(s, 1000, peak, floor, 200) for s in range(200, 1000)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(floor - 1e-12 <= v <= peak + 1e-12 for v in values)


def test_batches_respect_the_token_budget_cover_every_sequence_once_and_follow_the_seed():
    rng = random.Random(5)
    lengths = [rng.randint(60, 1700) for _ in range(50)] + [5000]       # one sequence larger than the budget
    batches = make_batches(lengths, 4096, seed=3, epoch=0)
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 4096
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    assert [lengths[i] for i in next(b for b in batches if 50 in b)] == [5000]
    assert make_batches(lengths, 4096, seed=3, epoch=0) == batches
    assert make_batches(lengths, 4096, seed=3, epoch=1) != batches
    assert make_batches(lengths, 4096, seed=4, epoch=0) != batches


def test_collate_right_pads_ids_and_mask_with_zero(episodes):
    encoded = [encode_episode(e) for e in episodes[:4]]
    x, y, m = collate(encoded, [0, 1, 2, 3], "cpu")
    longest = max(len(e[0]) for e in encoded)
    assert x.shape == y.shape == m.shape == (4, longest)
    for row, (ex, ey, em, _) in enumerate(encoded):
        n = len(ex)
        assert x[row, :n].tolist() == ex.tolist() and y[row, :n].tolist() == ey.tolist()
        assert m[row, :n].tolist() == em.tolist()
        assert not x[row, n:].any() and not y[row, n:].any() and not m[row, n:].any()


# ── the loss mask ──

def test_decoded_supervised_targets_are_exactly_the_compact_actions(episodes):
    encoded = [encode_episode(e) for e in episodes[:8]]
    x, y, m = collate(encoded, list(range(4)), "cpu")
    for row in range(4):
        supervised = decode(y[row][m[row] == 1].tolist())
        assert supervised == "".join(compact_action(a) + "\n" for a, _ in episodes[row]["turns"])
    assert int(m.sum()) > 0


# ── model invariants ──

def test_the_model_is_causal():
    torch.manual_seed(0)
    model = ScratchGPT(GPTConfig(**TINY)).eval()
    a = torch.randint(0, 96, (2, 40))
    b = a.clone()
    b[:, 21:] = torch.randint(0, 96, (2, 19))
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :21], lb[:, :21], atol=1e-6)
    assert not torch.allclose(la[:, 21:], lb[:, 21:], atol=1e-6)


def test_right_padding_does_not_change_the_logits_of_real_positions():
    torch.manual_seed(0)
    model = ScratchGPT(GPTConfig(**TINY)).eval()
    real = torch.randint(0, 96, (1, 30))
    padded = torch.cat([real, torch.zeros(1, 15, dtype=torch.long)], dim=1)
    with torch.no_grad():
        plain, _ = model(real)
        wide, _ = model(padded)
    assert torch.allclose(plain, wide[:, :30], atol=1e-5)


# ── runs ──

def test_a_micro_run_has_a_finite_falling_loss(files, tmp_path):
    lines = []
    out = str(tmp_path / "micro")
    train(*files, out, model_cfg=GPTConfig(**TINY), train_cfg=quick_config(max_steps=60, eval_every=20, log_every=10),
          device="cpu", log=lines.append)
    records = read_log(out)
    assert len(records) == 3
    assert all(math.isfinite(r["train_loss"]) and math.isfinite(r["dev_loss"]) for r in records)
    assert records[-1]["train_loss"] < records[0]["train_loss"]
    step_lines = [line for line in lines if line.startswith("step ")]
    assert len(step_lines) == 6
    assert all(math.isfinite(float(line.split()[3])) for line in step_lines)


def test_four_episodes_can_be_memorised_and_the_saved_checkpoint_regenerates_every_gold_line(episodes, tmp_path):
    by_class = {}
    for ep in sorted(episodes, key=lambda e: len(json.dumps(e))):
        by_class.setdefault(ep["cls"], []).append(ep)
    chosen = [by_class["benign"][0], by_class["dh"][0], by_class["ds"][0], by_class["benign"][1]]
    path = str(tmp_path / "four.jsonl")
    write_episodes(path, chosen)
    out = str(tmp_path / "overfit")
    cfg = TrainConfig(max_steps=1500, eval_every=50, log_every=10, batch_tokens=8192, lr=3e-3, min_lr=3e-4,
                      warmup=20, patience=1000, target_loss=0.02)
    manifest = train(path, path, out, model_cfg=GPTConfig(n_layer=2, n_head=2, d_model=64, dropout=0.0),
                     train_cfg=cfg, device="cpu", log=quiet)
    assert manifest["stop_reason"] == "target_loss"
    assert manifest["steps"] < 1500
    assert manifest["final"]["dev_line_exact"] == 1.0
    gen = load_generator(os.path.join(out, "scratch_gpt.pt"), device="cpu")
    for ep in chosen:
        for k, (action, _) in enumerate(ep["turns"]):
            assert gen(episode_to_messages(ep, k), max_tokens=320) == compact_action(action)
            assert gen.last_finish_reason == "stop"


def test_evaluate_reports_loss_and_line_exact_overall_per_class_and_for_the_first_action(episodes):
    torch.manual_seed(0)
    model = ScratchGPT(GPTConfig(**TINY))
    dev = [encode_episode(e) for e in episodes[:12]]
    result = evaluate(model, dev, 4096, "cpu")
    assert model.training            # evaluate puts the model back into the mode it found it in
    assert math.isfinite(result["dev_loss"]) and result["dev_loss"] > 0
    for key in ("dev_line_exact", "dev_first_line_exact"):
        assert 0.0 <= result[key] <= 1.0
    assert set(result["dev_line_exact_by_cls"]) == {e["cls"] for e in episodes[:12]}
    assert all(0.0 <= v <= 1.0 for v in result["dev_line_exact_by_cls"].values())
    assert result["dev_loss"] == pytest.approx(math.log(96), rel=0.25)    # an untrained model is near uniform


# ── manifest ──

def test_manifest_has_the_fixed_fields_no_environment_values_and_the_checkpoint_hash(files, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SCRATCH_SECRET", "sentinel-9f3a61c2")
    out = str(tmp_path / "manifest_run")
    train(*files, out, model_cfg=GPTConfig(**TINY), train_cfg=quick_config(), device="cpu", log=quiet)
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        text = f.read()
    manifest = json.loads(text)
    assert tuple(manifest) == MANIFEST_KEYS
    assert "sentinel-9f3a61c2" not in text
    assert str(tmp_path) not in text and "\\" not in text
    with open(os.path.join(out, "scratch_gpt.pt"), "rb") as f:
        assert manifest["checkpoint_sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert manifest["run_notes"] == [] and manifest["stop_reason"] == "max_steps"
    assert set(manifest["final"]) == {"train_loss", "dev_loss", "dev_line_exact", "dev_line_exact_by_cls",
                                      "dev_first_line_exact"}
    assert manifest["dataset"]["n_train"] == 64 and manifest["dataset"]["n_dev"] == 8


def test_a_run_whose_dev_loss_stops_improving_ends_with_early_stop(files, tmp_path):
    out = str(tmp_path / "early")
    manifest = train(*files, out, model_cfg=GPTConfig(**TINY), device="cpu", log=quiet,
                     train_cfg=quick_config(max_steps=20, eval_every=2, log_every=2, patience=2, lr=0.0, min_lr=0.0))
    assert manifest["stop_reason"] == "early_stop"
    assert manifest["steps"] < 20


# ── failure paths ──

def test_a_leaking_training_file_stops_training_before_any_checkpoint(files, tmp_path, episodes):
    spec = load_tool_specs()["GmailSendEmail"]
    names = [p["name"] for p in spec["parameters"] if p.get("required")]
    leaking = dict(episodes[0], id="train-0-leak",
                   tools=[episode_tool(spec)],
                   user="Please send the quarterly note for me.",
                   turns=[[{"action": "call_tool", "tool": "GmailSendEmail", "args": {n: "x1" for n in names}},
                           '{"status": "ok"}'],
                          [{"action": "respond", "message": "Done."}, None]])
    bad = str(tmp_path / "leaking.jsonl")
    write_episodes(bad, [leaking])
    out = str(tmp_path / "leak_out")
    with pytest.raises(LeakageError):
        train(bad, files[1], out, model_cfg=GPTConfig(**TINY), train_cfg=quick_config(), device="cpu", log=quiet)
    assert not os.path.exists(os.path.join(out, "scratch_gpt.pt"))
    assert not os.path.exists(os.path.join(out, "manifest.json"))


def test_the_command_line_refuses_paths_outside_the_repository_and_missing_inputs(files, tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "ALLOWED_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as outside:
        main(["--train", files[0], "--dev", files[1], "--out-dir", "/etc"] + CLI_TINY)
    assert outside.value.code == 2
    with pytest.raises(SystemExit) as missing:
        main(["--train", str(tmp_path / "nope.jsonl"), "--dev", files[1], "--out-dir", str(tmp_path / "o")] + CLI_TINY)
    assert missing.value.code == 2
    assert not os.path.exists("/etc/scratch_gpt.pt")


def test_the_command_line_trains_and_prints_the_lines_a_background_run_is_polled_for(files, tmp_path, monkeypatch,
                                                                                   capsys):
    monkeypatch.setattr(train_module, "ALLOWED_ROOT", str(tmp_path))
    out = str(tmp_path / "cli_out")
    code = main(["--train", files[0], "--dev", files[1], "--out-dir", out, "--run-note", "oom retry 1: batch_tokens "
                 "32768 -> 16384", "--run-note", "second note"] + CLI_TINY)
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("device cpu params ")
    assert any(line.startswith("step 5/10 loss ") and " tok/s " in line for line in lines)
    assert any(line.startswith("eval step 5 dev_loss ") and " dev_line_exact " in line and " best " in line
               for line in lines)
    assert lines[-1].startswith("DONE best_step ") and lines[-1].endswith(" stop max_steps")
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        assert json.load(f)["run_notes"] == ["oom retry 1: batch_tokens 32768 -> 16384", "second note"]


def test_a_non_finite_loss_fails_the_run_with_exit_code_3_and_no_final_manifest(files, tmp_path, monkeypatch, capsys):
    original = ScratchGPT.forward
    calls = {"n": 0}

    def poisoned(self, idx, targets=None, mask=None):
        logits, loss = original(self, idx, targets, mask)
        if self.training and loss is not None:
            calls["n"] += 1
            if calls["n"] >= 3:
                loss = loss * float("nan")
        return logits, loss

    monkeypatch.setattr(ScratchGPT, "forward", poisoned)
    out = str(tmp_path / "nan_run")
    with pytest.raises(NonFiniteLossError):
        train(*files, out, model_cfg=GPTConfig(**TINY), train_cfg=quick_config(), device="cpu", log=quiet)
    assert not os.path.exists(os.path.join(out, "manifest.json"))
    assert read_log(out)[-1] == {"step": 3, "event": "non_finite_loss"}

    calls["n"] = 0
    monkeypatch.setattr(train_module, "ALLOWED_ROOT", str(tmp_path))
    code = main(["--train", files[0], "--dev", files[1], "--out-dir", str(tmp_path / "nan_cli")] + CLI_TINY)
    assert code == 3
    assert any(line.startswith("FAILED non_finite_loss step 3") for line in capsys.readouterr().out.splitlines())
    assert not os.path.exists(str(tmp_path / "nan_cli" / "manifest.json"))
