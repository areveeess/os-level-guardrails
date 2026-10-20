"""
Tests for result provenance (EVAL-06): every raw.jsonl record is stamped with the
code commit, policy hash, provider and model ids, per-step finish_reason and token
counts, and the report's "Provenance and data quality" section shows them.
Providers are scripted or faked (no network).
"""

import json
import re

import pytest

from src.agent_harness.executors import Executor, Observation
from src.agent_harness.loop import AgentLoop
from src.benchmark import runner as runner_module
from src.benchmark.injecagent import benign_cases, load_tool_specs
from src.benchmark.report import generate_report
from src.llm.base import LLMProvider, LLMResponse
from src.policy_engine.engine import PolicyEngine

RESPOND = '{"action": "respond", "message": "done"}'
MODEL = "ollama-llama3.2-3b"       # in config/models.yaml: provider openai_compat, model llama3.2:3b
STAMP_FIELDS = ("code_commit", "code_dirty", "policy_sha256", "provider", "config_model",
                "resolved_model", "finish_reasons", "prompt_tokens", "completion_tokens")


@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()


class StampedProvider(LLMProvider):
    """Calls the case's user tool once, then responds; reports full response metadata."""

    def __init__(self, user_tool, name=MODEL, finish_reason="stop", resolved_model="stub-model-2026-10",
                 prompt_tokens=123, completion_tokens=7):
        self.name = name
        self.user_tool = user_tool
        self.meta = dict(finish_reason=finish_reason, resolved_model=resolved_model,
                         prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def generate(self, messages, temperature=0.0, max_tokens=512):
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        text = json.dumps({"action": "call_tool", "tool": self.user_tool, "args": {}}) if n_assistant == 0 else RESPOND
        return LLMResponse(text=text, latency_ms=1.0, model=self.name, **self.meta)


class BareProvider(StampedProvider):
    """Never sets the new response fields (what an old provider looks like)."""

    def generate(self, messages, temperature=0.0, max_tokens=512):
        response = super().generate(messages, temperature, max_tokens)
        return LLMResponse(text=response.text, latency_ms=1.0, model=self.name)


class OkExecutor(Executor):
    def execute(self, action):
        return Observation("success", output="ok")


def run_one(tmp_path, monkeypatch, specs, provider_factory, name="out"):
    case = benign_cases(specs)[:1]
    monkeypatch.setattr(runner_module, "get_provider", lambda _name: provider_factory(case[0]))
    out = tmp_path / name
    runner_module.run_benchmark([MODEL], case, str(out), runs=1, arms=("unprotected",), log=lambda *_: None)
    return out, [json.loads(line) for line in open(out / "raw.jsonl")]


# ── Provider response fields travel through the agent loop ──

def test_agent_loop_copies_response_fields_onto_each_step():
    case_tool = "AnyTool"
    provider = StampedProvider(case_tool)
    episode = AgentLoop(provider, OkExecutor()).run("do it")
    assert len(episode.steps) == 2
    for step in episode.steps:
        assert (step.finish_reason, step.prompt_tokens, step.completion_tokens, step.resolved_model) == \
               ("stop", 123, 7, "stub-model-2026-10")


def test_step_fields_default_to_none_for_providers_that_do_not_report_them():
    episode = AgentLoop(BareProvider("AnyTool"), OkExecutor()).run("do it")
    step = episode.steps[0]
    assert step.finish_reason is None and step.resolved_model is None
    assert step.prompt_tokens is None and step.completion_tokens is None


# ── Row stamps ──

def test_row_carries_every_stamp(tmp_path, monkeypatch, specs):
    _, rows = run_one(tmp_path, monkeypatch, specs, lambda case: StampedProvider(case.user_tool))
    assert len(rows) == 1
    row = rows[0]
    assert row["code_commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", row["code_commit"])
    assert row["code_dirty"] in (True, False, None)
    assert row["policy_sha256"] == PolicyEngine(verbose=False).policy_sha256
    assert row["provider"] == "openai_compat" and row["config_model"] == "llama3.2:3b"
    assert row["resolved_model"] == "stub-model-2026-10"
    steps = row["steps"]
    assert steps >= 1
    assert row["finish_reasons"] == ["stop"] * steps
    assert row["prompt_tokens"] == [123] * steps
    assert row["completion_tokens"] == [7] * steps


def test_provider_without_the_new_fields_does_not_crash(tmp_path, monkeypatch, specs):
    _, rows = run_one(tmp_path, monkeypatch, specs, lambda case: BareProvider(case.user_tool))
    row = rows[0]
    assert row["resolved_model"] is None
    assert row["finish_reasons"] == [None] * row["steps"]
    assert row["prompt_tokens"] == [None] * row["steps"]


def test_git_helpers_never_use_a_shell_and_fall_back(monkeypatch):
    commit = runner_module._git_commit()
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert runner_module._git_dirty() in (True, False, None)

    def boom(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(runner_module.subprocess, "run", boom)
    assert runner_module._git_commit() == "unknown" and runner_module._git_dirty() is None


# ── Report provenance section ──

def test_report_shows_provenance(tmp_path, monkeypatch, specs):
    out, rows = run_one(tmp_path, monkeypatch, specs, lambda case: StampedProvider(case.user_tool))
    generate_report(str(out), charts=False)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Provenance and data quality" in md
    section = md.split("## Provenance and data quality", 1)[1]
    assert rows[0]["code_commit"][:8] in section
    assert rows[0]["policy_sha256"][:12] in section
    assert "stub-model-2026-10" in section


def test_report_marks_records_without_stamps_as_unstamped(tmp_path, monkeypatch, specs):
    _, rows = run_one(tmp_path, monkeypatch, specs, lambda case: StampedProvider(case.user_tool))
    old = {k: v for k, v in rows[0].items() if k not in STAMP_FIELDS}
    out = tmp_path / "old"
    out.mkdir()
    (out / "raw.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    generate_report(str(out), charts=False)
    section = (out / "summary.md").read_text(encoding="utf-8").split("## Provenance and data quality", 1)[1]
    assert "unstamped (recorded before EVAL-06)" in section
