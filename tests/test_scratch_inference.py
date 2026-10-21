"""
Tests for ScratchGPT checkpoints at evaluation time: training stamps them, inference loads them,
LocalScratchProvider serves them to the benchmark.
CPU only, tiny model (2 layers, d_model 32), no network, no GPU.

One tiny checkpoint is trained once per file (module-scoped fixture); every test reads it.
Torch is imported once per interpreter (about a minute cold from the Windows drive), so this file
keeps its training run to 20 steps on 16 episodes.
"""

import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.agent_harness.prompting import build_messages
from src.benchmark.injecagent import PROJECT_ROOT, benign_cases, load_tool_specs
from src.benchmark.runner import run_episode
from src.llm.scratch import LocalScratchProvider
from src.policy_engine.engine import PolicyEngine
from src.scratch_llm.dataset import generate_episodes, write_episodes
from src.scratch_llm.format import (CONTEXT_CHARS, FORMAT_VERSION, GEN_RESERVE, VOCAB_SHA256, render_context,
                                    render_fingerprint)
from src.scratch_llm.inference import CHECKPOINT_KEYS, MAX_NEW_CHARS, load_generator
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
