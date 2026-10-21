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


# ── WR-08: arms are validated; resume refuses a different plan ───────────────

import hashlib  # noqa: E402

import run_benchmark as cli  # noqa: E402


def _plan_run(out, cases, monkeypatch, **kw):
    monkeypatch.setattr(runner_module, "get_provider", lambda _n: _Respond())
    args = dict(runs=1, arms=("unprotected",), log=lambda *_: None)
    args.update(kw)
    return runner_module.run_benchmark(["m"], cases, str(out), **args)


def test_wr08_unknown_arm_is_rejected_before_anything_runs(tmp_path, monkeypatch):
    cases = benign_cases(load_tool_specs())[:1]
    out = tmp_path / "r"
    for arms in [("unprotcted", "protected"), ("protected", "bogus"), (), ("protected", "protected")]:
        with pytest.raises(ValueError, match="arm"):
            _plan_run(out, cases, monkeypatch, arms=arms)
    assert not (out / "raw.jsonl").exists() and not (out / "manifest.json").exists()


@pytest.mark.parametrize("change,needle", [
    (dict(runs=2), "runs"),
    (dict(arms=("unprotected", "protected")), "arms"),
    (dict(max_steps=9), "max_steps"),
    (dict(queue_seed=5), "seed"),
    (dict(manifest_extra={"seed": 5}), "seed"),
])
def test_wr08_resume_with_a_different_plan_is_refused(tmp_path, monkeypatch, change, needle):
    cases = benign_cases(load_tool_specs())[:2]
    out = tmp_path / "r"
    base = dict(manifest_extra={"seed": 0})
    _plan_run(out, cases, monkeypatch, **base)
    raw_before = (out / "raw.jsonl").read_bytes()
    manifest_before = (out / "manifest.json").read_bytes()
    with pytest.raises(runner_module.ResumeMismatch, match=needle):
        _plan_run(out, cases, monkeypatch, **{**base, **change})
    assert (out / "raw.jsonl").read_bytes() == raw_before
    assert (out / "manifest.json").read_bytes() == manifest_before


def test_wr08_resume_with_a_different_case_set_is_refused(tmp_path, monkeypatch):
    specs = benign_cases(load_tool_specs())
    out = tmp_path / "r"
    _plan_run(out, specs[:2], monkeypatch)
    with pytest.raises(runner_module.ResumeMismatch, match="cases"):
        _plan_run(out, specs[1:3], monkeypatch)      # same number of cases, different ones
    with pytest.raises(runner_module.ResumeMismatch, match="cases"):
        _plan_run(out, specs[:3], monkeypatch)       # more cases


def test_wr08_matching_resume_appends_an_invocation_and_keeps_the_plan(tmp_path, monkeypatch):
    cases = benign_cases(load_tool_specs())[:2]
    out = tmp_path / "r"
    _plan_run(out, cases, monkeypatch, manifest_extra={"seed": 3})
    first = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    _plan_run(out, cases, monkeypatch, manifest_extra={"seed": 3})
    second = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(first["invocations"]) == 1 and len(second["invocations"]) == 2
    assert second["invocations"][0] == first["invocations"][0]
    for key in ("expected_episodes_per_model", "cases", "runs", "arms", "max_steps", "extra"):
        assert second[key] == first[key]
    assert second["case_set_sha256"] == hashlib.sha256(
        "\n".join(sorted(c.id for c in cases)).encode()).hexdigest()


def test_wr08_a_resume_may_add_a_model(tmp_path, monkeypatch):
    cases = benign_cases(load_tool_specs())[:1]
    out = tmp_path / "r"
    _plan_run(out, cases, monkeypatch)
    monkeypatch.setattr(runner_module, "get_provider", lambda _n: _Respond(name="m2"))
    runner_module.run_benchmark(["m2"], cases, str(out), runs=1, arms=("unprotected",), log=lambda *_: None)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"] == ["m", "m2"]


def test_wr08_cli_rejects_unknown_arms_and_missing_resume_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def must_not_run(*_a, **_k):
        raise AssertionError("the benchmark must not start with invalid arguments")
    monkeypatch.setattr(cli, "run_benchmark", must_not_run)
    for argv in (["run_benchmark.py", "--arms", "unprotcted,protected"],
                 ["run_benchmark.py", "--resume", str(tmp_path / "typo-dir")]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "arm" in err and "does not exist" in err
    assert not (tmp_path / "typo-dir").exists() and not (tmp_path / "results").exists()


def test_wr08_cli_reports_a_resume_mismatch_as_a_usage_error(tmp_path, monkeypatch, capsys):
    cases = benign_cases(load_tool_specs())[:1]
    out = tmp_path / "r"
    _plan_run(out, cases, monkeypatch, manifest_extra={"seed": 0, "dh": 1, "ds": 1, "benign": 1})
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--resume", str(out), "--models", "m",
                                       "--runs", "3", "--arms", "unprotected"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "runs" in capsys.readouterr().err
