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
from src.llm.openai_compat import OpenAICompatibleProvider
from src.llm.registry import load_model_config
from src.policy_engine.engine import PolicyEngine
from tests.test_llm_layer import FakeClient

RESPOND = '{"action": "respond", "message": "done"}'
MODEL = "ollama-llama3.2-3b"       # in config/models.yaml: provider openai_compat, model llama3.2-3b-ctx8192
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
    assert row["provider"] == "openai_compat"
    assert row["config_model"] == load_model_config()["models"][MODEL]["model"]   # the 8192-context variant (Phase 4)
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


# ── Hosted provider through the runner ──

def test_openai_compatible_provider_stamps_rows_end_to_end(tmp_path, monkeypatch, specs):
    client = FakeClient(reply=RESPOND, model="gpt-oss-120b-2026-09", finish_reason="stop")
    _, rows = run_one(tmp_path, monkeypatch, specs,
                      lambda case: OpenAICompatibleProvider(MODEL, "llama3.2:3b", "http://x", client=client))
    row = rows[0]
    assert row["resolved_model"] == "gpt-oss-120b-2026-09"
    assert row["finish_reasons"] == ["stop"] * row["steps"]
    assert row["prompt_tokens"] == [10] * row["steps"]


# ── Provenance table: truncation, filtering, mixed provenance ──

def provenance_table(md: str) -> dict:
    """{model: {column: cell}} parsed from the provenance section's table."""
    lines = [l for l in md.split("## Provenance and data quality", 1)[1].splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return {cells[0]: dict(zip(header, cells))
            for cells in ([c.strip() for c in l.strip("|").split("|")] for l in lines[2:])}


def report_for(tmp_path, row_edits, specs, monkeypatch):
    """Report on a hand-written raw.jsonl: the real stamped row, copied once per edit."""
    _, rows = run_one(tmp_path, monkeypatch, specs, lambda case: StampedProvider(case.user_tool), name="base")
    out = tmp_path / "hand"
    out.mkdir()
    with open(out / "raw.jsonl", "w", encoding="utf-8") as f:
        for i, edit in enumerate(row_edits):
            f.write(json.dumps(dict(rows[0], case_id=f"case-{i}", **edit)) + "\n")
    generate_report(str(out), charts=False)
    return provenance_table((out / "summary.md").read_text(encoding="utf-8"))


def test_provenance_counts_length_and_content_filter_stops_and_max_prompt_tokens(tmp_path, monkeypatch, specs):
    table = report_for(tmp_path, [
        {"finish_reasons": ["length", "stop"], "prompt_tokens": [100, 4096]},
        {"finish_reasons": ["length", "content_filter"], "prompt_tokens": [200, None]},
    ], specs, monkeypatch)[MODEL]
    assert table["Length stops"] == "2" and table["Content-filter stops"] == "1"
    assert table["Max prompt_tokens"] == "4096"


def test_provenance_reports_prompt_tokens_not_reported(tmp_path, monkeypatch, specs):
    table = report_for(tmp_path, [{"finish_reasons": [None], "prompt_tokens": [None]}], specs, monkeypatch)[MODEL]
    assert table["Max prompt_tokens"] == "not reported" and table["Length stops"] == "0"


def test_provenance_marks_mixed_commits_and_policy_hashes(tmp_path, monkeypatch, specs):
    table = report_for(tmp_path, [
        {"code_commit": "a" * 40, "policy_sha256": "1" * 64},
        {"code_commit": "b" * 40, "policy_sha256": "2" * 64},
    ], specs, monkeypatch)[MODEL]
    assert table["Code commit(s)"].startswith("MIXED") and "aaaaaaaa" in table["Code commit(s)"]
    assert table["Policy hash(es)"].startswith("MIXED") and "222222222222" in table["Policy hash(es)"]


def test_provenance_single_commit_is_not_mixed(tmp_path, monkeypatch, specs):
    table = report_for(tmp_path, [{}, {}], specs, monkeypatch)[MODEL]
    assert "MIXED" not in table["Code commit(s)"] and "MIXED" not in table["Policy hash(es)"]


def test_provenance_escapes_model_supplied_ids(tmp_path, monkeypatch, specs):
    _, rows = run_one(tmp_path, monkeypatch, specs,
                      lambda case: StampedProvider(case.user_tool, resolved_model="evil|model\nsecond-line"),
                      name="evil")
    out = tmp_path / "evil"
    generate_report(str(out), charts=False)
    section = (out / "summary.md").read_text(encoding="utf-8").split("## Provenance and data quality", 1)[1]
    assert "evil\\|model second-line" in section
