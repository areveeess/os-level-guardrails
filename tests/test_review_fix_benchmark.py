"""
Regression tests for the Phase 1 code review (01-REVIEW.md) benchmark findings
(WR-04 .. WR-08). Each test fails on the pre-fix code.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmark.outcomes import classify_llm_error, classify_outcome, drop_infra_pairs


# ── WR-06: provider refusal markers ──────────────────────────────────────────

REFUSAL_ERRORS = [
    "BlockedPromptException: block_reason: PROHIBITED_CONTENT",
    "finish_reason: PROHIBITED_CONTENT",
    "prompt_feedback block_reason=BLOCKLIST",
    "Candidate was blocked due to RECITATION",
    "FinishReason.SAFETY",
    "finish_reason=SAFETY",
    "block_reason: SPII",
    "Error code: 400 - content_policy_violation",
    "Error: the response was filtered due to the prompt triggering Azure OpenAI's content filter",
    "ResponsibleAIPolicyViolation",
    "prohibited content detected",
    "content filter fired",
    "Request blocked by a block list",
]


@pytest.mark.parametrize("error", REFUSAL_ERRORS)
def test_wr06_provider_refusals_are_not_infra(error):
    assert classify_llm_error(error) == "provider_refusal"
    assert classify_outcome({"llm_error": error, "kind": "ds", "attempted": False}) == "refused"


@pytest.mark.parametrize("error", [
    "InternalServerError: Error code: 500",
    "ReadTimeout: timed out",
    "RateLimitError: 429 Too Many Requests",
    "ConnectionError: connection refused",
    "ServiceUnavailable: 503",
])
def test_wr06_real_outages_stay_infra(error):
    assert classify_llm_error(error) == "infra"


def test_wr06_refused_pair_is_kept_not_dropped():
    rows = [
        {"model": "m", "case_id": "c", "run": 0, "arm": "unprotected", "llm_error": "block_reason: PROHIBITED_CONTENT"},
        {"model": "m", "case_id": "c", "run": 0, "arm": "protected", "llm_error": None},
    ]
    kept, dropped = drop_infra_pairs(rows)
    assert dropped == 0 and len(kept) == 2


# ── WR-07: --resume survives a truncated raw.jsonl ───────────────────────────

from src.benchmark import runner as runner_module
from src.benchmark.injecagent import benign_cases, load_tool_specs
from src.llm.base import LLMProvider, LLMResponse


class _Respond(LLMProvider):
    def __init__(self, name="m"):
        self.name = name

    def generate(self, messages, temperature=0.0, max_tokens=512):
        return LLMResponse(text='{"action": "respond", "message": "done"}', latency_ms=1.0, model=self.name)


def _rows(out):
    return (out / "raw.jsonl").read_text(encoding="utf-8").split("\n")


def _first_run(tmp_path, monkeypatch, count=2):
    monkeypatch.setattr(runner_module, "get_provider", lambda _n: _Respond())
    cases = benign_cases(load_tool_specs())[:count]
    out = tmp_path / "r"
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=lambda *_: None)
    return out, cases


def test_wr07_resume_skips_and_counts_a_truncated_last_line(tmp_path, monkeypatch):
    out, cases = _first_run(tmp_path, monkeypatch)
    lines = (out / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # the run was killed in the middle of writing the second row
    (out / "raw.jsonl").write_text(lines[0] + "\n" + lines[1][:60], encoding="utf-8")
    logs = []
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=logs.append)
    assert any("1 undecodable" in line for line in logs)
    decoded = []
    undecodable = 0
    for line in _rows(out):
        if not line.strip():
            continue
        try:
            decoded.append(json.loads(line))
        except ValueError:
            undecodable += 1
    # the fragment stays on its own line; the re-run row is not glued to it
    assert undecodable == 1
    assert sorted(r["case_id"] for r in decoded) == sorted(c.id for c in cases)


def test_wr07_resume_tolerates_blank_and_garbage_lines(tmp_path, monkeypatch):
    out, cases = _first_run(tmp_path, monkeypatch)
    with open(out / "raw.jsonl", "a", encoding="utf-8") as f:
        f.write("\n   \nnot json at all\n[1, 2]\n{\"model\": \"m\"}\n")
    logs = []
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=logs.append)
    assert any("3 undecodable" in line for line in logs)


def test_wr07_missing_final_newline_is_repaired_before_appending(tmp_path, monkeypatch):
    out, cases = _first_run(tmp_path, monkeypatch)
    text = (out / "raw.jsonl").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    (out / "raw.jsonl").write_text(first, encoding="utf-8")        # complete row, no trailing newline
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=lambda *_: None)
    decoded = [json.loads(line) for line in _rows(out) if line.strip()]
    assert len(decoded) == 2                                       # nothing glued, nothing lost
