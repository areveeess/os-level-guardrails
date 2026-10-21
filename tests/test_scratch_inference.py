"""
Tests for ScratchGPT checkpoints at evaluation time: training stamps them, inference loads them,
LocalScratchProvider serves them to the benchmark.
CPU only, tiny model (2 layers, d_model 32), no network, no GPU.

One tiny checkpoint is trained once per file (module-scoped fixture); every test reads it.
Torch is imported once per interpreter (about a minute cold from the Windows drive), so this file
keeps its training run to 20 steps on 16 episodes.
"""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.agent_harness.prompting import build_messages
from src.agent_harness.environment import build_environment
from src.agent_harness.executors import MockExecutor
from src.agent_harness.loop import AgentLoop
from src.benchmark import runner as runner_module
from src.benchmark.injecagent import PROJECT_ROOT, benign_cases, load_tool_specs
from src.benchmark.runner import run_benchmark, run_episode
from src.llm import registry
from src.llm.scratch import LocalScratchProvider
from src.policy_engine.engine import PolicyEngine
from src.scratch_llm.dataset import generate_episodes, write_episodes
from src.scratch_llm.format import (CONTEXT_CHARS, FORMAT_VERSION, GEN_RESERVE, HISTORY_ACTION_CHARS, VOCAB_SHA256,
                                    compact_action, encode, render_context, render_fingerprint)
from src.scratch_llm.inference import CHECKPOINT_KEYS, MAX_NEW_CHARS, ScratchGenerator, load_generator
from src.scratch_llm.model import GPTConfig
from src.scratch_llm.train import TrainConfig, train

TINY_MODEL = dict(n_layer=2, n_head=2, d_model=32, dropout=0.0)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The provider picks CUDA when it exists; these tests stay on the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny_scratch")
    train_path, dev_path, out = str(work / "train.jsonl"), str(work / "dev.jsonl"), str(work / "out")
    write_episodes(train_path, generate_episodes(16, 0))
    write_episodes(dev_path, generate_episodes(8, 1, "dev_tools"))
    manifest = train(train_path, dev_path, out, model_cfg=GPTConfig(**TINY_MODEL),
                     train_cfg=TrainConfig(max_steps=20, eval_every=10, log_every=5, batch_tokens=4096, warmup=5),
                     device="cpu", log=lambda *_: None)
    return SimpleNamespace(out=out, path=os.path.join(out, "scratch_gpt.pt"), manifest=manifest, work=work)


def messages_for_a_notes_request():
    tools = benign_cases(load_tool_specs())[0].tools
    return build_messages("list my notes", tools)


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── training output ──

def test_training_writes_the_four_artifacts_and_a_stamped_checkpoint(tiny):
    for name in ("scratch_gpt.pt", "manifest.json", "train_log.jsonl", "loss_curve.png"):
        assert os.path.getsize(os.path.join(tiny.out, name)) > 0, name
    ckpt = torch.load(tiny.path, map_location="cpu", weights_only=True)
    assert tuple(ckpt) == CHECKPOINT_KEYS
    assert ckpt["format_version"] == FORMAT_VERSION == 2
    assert ckpt["vocab_sha256"] == VOCAB_SHA256
    assert ckpt["render_fingerprint"] == render_fingerprint()
    assert ckpt["context_chars"] == CONTEXT_CHARS and ckpt["gen_reserve"] == GEN_RESERVE
    assert ckpt["cfg"]["context"] == CONTEXT_CHARS


def test_default_model_context_is_the_format_context():
    assert GPTConfig().context == CONTEXT_CHARS


# ── the generator ──

def test_generator_returns_one_line_and_exposes_its_provenance(tiny):
    gen = load_generator(tiny.path, device="cpu")
    messages = messages_for_a_notes_request()
    text = gen(messages)
    assert isinstance(text, str) and "\n" not in text and len(text) <= MAX_NEW_CHARS
    assert gen.last_prompt_chars == len(render_context(messages))
    assert gen.last_finish_reason in ("stop", "length")
    assert gen.checkpoint_sha256 == sha256_of(tiny.path)
    assert gen.identity["checkpoint_sha256"] == gen.checkpoint_sha256
    assert gen.identity["format_version"] == FORMAT_VERSION


def test_provider_stamps_every_provenance_field(tiny):
    provider = LocalScratchProvider("scratch-gpt", tiny.path)
    response = provider.generate(messages_for_a_notes_request())
    gen = provider._generate
    assert response.ok
    assert response.prompt_tokens == gen.last_prompt_chars
    assert response.completion_tokens is not None and response.completion_tokens >= 0
    assert response.finish_reason in ("stop", "length")
    assert response.resolved_model == "scratch-gpt@" + sha256_of(tiny.path)[:12]


def test_a_tiny_model_drives_a_real_benchmark_episode_end_to_end(tiny):
    provider = LocalScratchProvider("scratch-gpt", tiny.path)
    case = benign_cases(load_tool_specs())[0]
    policy = PolicyEngine(log_file=str(tiny.work / "decisions.jsonl"), verbose=False)
    episode, score = run_episode(provider, case, "unprotected", policy, 3)
    assert episode.llm_error is None
    assert len(episode.steps) >= 1
    assert "parse_errors" in score and "steps" in score
    assert episode.steps[0].resolved_model.startswith("scratch-gpt@")


# ── isolation and refusal ──

def test_inference_imports_neither_the_dataset_nor_the_benchmark():
    code = ("import sys; import src.scratch_llm.inference; "
            "bad = [m for m in ('src.scratch_llm.dataset', 'src.benchmark.injecagent') if m in sys.modules]; "
            "print('BAD', bad) if bad else print('CLEAN')")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "CLEAN", done.stdout


def test_a_checkpoint_with_another_format_version_is_refused(tiny, tmp_path):
    ckpt = torch.load(tiny.path, map_location="cpu", weights_only=True)
    ckpt["format_version"] = 1
    stale = str(tmp_path / "stale.pt")
    torch.save(ckpt, stale)
    with pytest.raises(RuntimeError, match="format_version"):
        load_generator(stale, device="cpu")


# ── every stamp is enforced, and the runner skips a refused model ──

def tampered_copy(tiny, tmp_path, mutate):
    ckpt = torch.load(tiny.path, map_location="cpu", weights_only=True)
    mutate(ckpt)
    path = str(tmp_path / "tampered.pt")
    torch.save(ckpt, path)
    return path


@pytest.mark.parametrize("field, value", [("format_version", 1), ("vocab_sha256", "0" * 64),
                                          ("render_fingerprint", "f" * 64), ("context_chars", 1536),
                                          ("gen_reserve", 300)])
def test_each_changed_stamp_is_refused_and_named(tiny, tmp_path, field, value):
    path = tampered_copy(tiny, tmp_path, lambda c: c.__setitem__(field, value))
    with pytest.raises(RuntimeError, match=field):
        load_generator(path, device="cpu")


def test_an_extra_or_a_missing_checkpoint_key_is_refused_by_name(tiny, tmp_path):
    extra = tampered_copy(tiny, tmp_path, lambda c: c.__setitem__("pickled_extra", 1))
    with pytest.raises(RuntimeError, match="pickled_extra"):
        load_generator(extra, device="cpu")
    missing = tampered_copy(tiny, tmp_path, lambda c: c.pop("train_meta"))
    with pytest.raises(RuntimeError, match="train_meta"):
        load_generator(missing, device="cpu")


def test_a_missing_checkpoint_says_to_train_it_first(tmp_path):
    with pytest.raises(RuntimeError, match="Train it first"):
        LocalScratchProvider("scratch-gpt", str(tmp_path / "missing.pt"))


def write_models_yaml(path, entries):
    lines = ["default: scratch-tiny", "models:"]
    for name, checkpoint in entries.items():
        lines += [f"  {name}:", "    provider: scratch", f"    model: {name}", f"    checkpoint: {checkpoint}",
                  "    family: test", "    benchmark: true"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def point_runner_at(monkeypatch, config_path):
    monkeypatch.setattr(runner_module, "get_provider", lambda name: registry.get_provider(name, path=str(config_path)))
    monkeypatch.setattr(runner_module, "load_model_config", lambda: registry.load_model_config(str(config_path)))


def test_the_runner_skips_a_refused_checkpoint_and_writes_no_row_for_it(tiny, tmp_path, monkeypatch):
    stale = tampered_copy(tiny, tmp_path, lambda c: c.__setitem__("render_fingerprint", "f" * 64))
    config = tmp_path / "models.yaml"
    write_models_yaml(config, {"scratch-tampered": stale})
    point_runner_at(monkeypatch, config)
    lines = []
    out = str(tmp_path / "run_tampered")
    run_benchmark(["scratch-tampered"], benign_cases(load_tool_specs())[:1], out, runs=1, arms=("unprotected",),
                  log=lines.append)
    assert any(line.startswith("[skip] scratch-tampered:") for line in lines), lines
    raw = os.path.join(out, "raw.jsonl")
    assert not os.path.exists(raw) or os.path.getsize(raw) == 0


def test_runner_rows_of_a_good_checkpoint_carry_the_scratch_stamps(tiny, tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    write_models_yaml(config, {"scratch-tiny": tiny.path})
    point_runner_at(monkeypatch, config)
    out = str(tmp_path / "run_good")
    run_benchmark(["scratch-tiny"], benign_cases(load_tool_specs())[:1], out, runs=1, arms=("unprotected",),
                  max_steps=2, log=lambda *_: None)
    with open(os.path.join(out, "raw.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "scratch" and row["config_model"] == "scratch-tiny"
    assert row["resolved_model"] == "scratch-tiny@" + sha256_of(tiny.path)[:12]
    assert row["finish_reasons"] and all(r in ("stop", "length") for r in row["finish_reasons"])
    assert all(isinstance(n, int) for n in row["prompt_tokens"] + row["completion_tokens"])


# ── generation cap, errors and garbage as values ──

class ScriptedModel:
    """A stand-in for ScratchGPT.generate: appends a fixed reply character by character, honouring max_new and stop_id."""

    def __init__(self, reply_ids, context=CONTEXT_CHARS):
        self.reply_ids = list(reply_ids)
        self.cfg = SimpleNamespace(context=context)

    def num_params(self):
        return 0

    def generate(self, idx, max_new, stop_id, temperature=0.0):
        new = []
        for token in self.reply_ids[:max_new]:
            new.append(token)
            if token == stop_id:
                break
        return torch.cat([idx, torch.tensor([new], dtype=idx.dtype)], dim=1)


def test_a_long_legitimate_action_is_not_cut_by_the_generation_cap():
    """WR-02: gold actions of 335 characters exist in the training data; the cap was 320."""
    action = compact_action({"action": "call_tool", "tool": "CiscoUmbrellaCreatePolicy",
                             "args": {"policy_name": "p" * 150, "description": "d" * 120}})
    assert 335 <= len(action) + 1 <= HISTORY_ACTION_CHARS
    generator = ScratchGenerator(ScriptedModel(encode(action + "\n")), "cpu", "0" * 64)
    assert MAX_NEW_CHARS >= HISTORY_ACTION_CHARS + 1
    assert generator(messages_for_a_notes_request(), max_tokens=512) == action
    assert generator.last_finish_reason == "stop" and generator.last_new_chars == len(action) + 1


def test_generation_is_capped_by_max_tokens_and_by_the_hard_limit(tiny):
    gen = load_generator(tiny.path, device="cpu")
    messages = messages_for_a_notes_request()
    short = gen(messages, max_tokens=5)
    assert len(short) <= 5 and gen.last_new_chars <= 5
    if "\n" not in short and gen.last_new_chars == 5:
        assert gen.last_finish_reason == "length"
    long = gen(messages, max_tokens=10000)
    assert len(long) <= MAX_NEW_CHARS and gen.last_new_chars <= MAX_NEW_CHARS
    assert gen(messages, max_tokens=0) == "" and gen.last_finish_reason == "length"


def test_a_generator_that_raises_becomes_an_error_response(tiny):
    provider = LocalScratchProvider("scratch-gpt", tiny.path)

    def broken(messages, temperature=0.0, max_tokens=512):
        raise ValueError("boom")

    provider._generate = broken
    response = provider.generate(messages_for_a_notes_request())
    assert not response.ok and response.error.startswith("ValueError") and response.text == ""
    assert response.resolved_model is None


class Replies:
    """A stand-in generator that answers from a fixed list."""

    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, messages, temperature=0.0, max_tokens=512):
        return self.replies.pop(0)


def test_empty_and_garbage_replies_become_parse_errors_the_loop_reports_back(tiny):
    provider = LocalScratchProvider("scratch-gpt", tiny.path)
    provider._generate = Replies(["", "not json at all", '{"action": "respond", "message": "done"}'])
    episode = AgentLoop(provider, MockExecutor(build_environment()), max_steps=3).run("say hi")
    assert [bool(s.parse_error) for s in episode.steps] == [True, True, False]
    assert episode.stop_reason == "responded" and episode.llm_error is None


def test_a_non_ascii_prompt_character_is_encoded_as_a_question_mark(tiny):
    gen = load_generator(tiny.path, device="cpu")
    messages = build_messages("list my notes — café ☃", benign_cases(load_tool_specs())[0].tools)
    assert isinstance(gen(messages), str)
    assert gen.last_prompt_chars == len(render_context(messages))


def test_models_yaml_stamps_the_config_model_for_scratch_gpt():
    assert registry.load_model_config()["models"]["scratch-gpt"]["model"] == "scratch-gpt"
