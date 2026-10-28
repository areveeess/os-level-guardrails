"""
Tests for src/benchmark/run_checks.py on synthetic result directories (no model, no
network). Each check must pass on a clean directory and name the problem otherwise.
"""

import json

import pytest

from src.benchmark import run_checks
from src.benchmark.scoring import SCORING_VERSION

MODEL = "ollama-qwen2.5-3b"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def _row(case_id, run=0, arm="unprotected", **extra):
    row = {
        "model": MODEL, "family": "open / small / local", "case_id": case_id, "kind": "benign",
        "attack_type": "", "run": run, "arm": arm, "attempted": False, "succeeded": False,
        "steps": 2, "parse_errors": 0, "stop_reason": "responded", "user_task_done": True,
        "llm_error": None, "blocked_by": [], "actions": [{"action": "respond", "message": "done"}],
        "final_message": "done", "scoring": SCORING_VERSION, "code_commit": COMMIT_A, "code_dirty": False,
        "policy_sha256": run_checks._policy_sha256(), "provider": "openai_compat",
        "config_model": "qwen2.5-3b-ctx8192", "resolved_model": "qwen2.5-3b-ctx8192",
        "finish_reasons": ["stop", "stop"], "prompt_tokens": [900, 1100], "completion_tokens": [20, 10],
    }
    row.update(extra)
    return row


def _six():
    return [_row(c, 0, arm) for c in ("benign-00", "dh-u01-a02", "ds-u03-a04") for arm in ("unprotected", "protected")]


def _write(tmp_path, rows, expected=6):
    with open(tmp_path / "raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"expected_episodes_per_model": expected}))
    return str(tmp_path)


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(run_checks, "_git_diff_names", lambda a, b: [])


def test_complete_ok(tmp_path, capsys):
    d = _write(tmp_path, _six())
    assert run_checks.main(["complete", "--results", d, "--model", MODEL, "--expect", "6",
                            "--resolved-contains", "ctx8192"]) == 0
    out = capsys.readouterr().out
    assert "COMPLETE model ollama-qwen2.5-3b finished 6/6 pending_infra 0 duplicates 0" in out
    assert "complete-ok" in out


def test_expected_comes_from_manifest(tmp_path):
    d = _write(tmp_path, _six(), expected=342)
    status = run_checks.complete_status(d, MODEL)
    assert status["expected"] == 342 and any("below expected 342" in p for p in status["problems"])


def test_missing_key_fails(tmp_path, capsys):
    d = _write(tmp_path, _six()[:5])
    assert run_checks.main(["complete", "--results", d, "--model", MODEL, "--expect", "6"]) == 1
    assert "FAIL complete: finished 5 below expected 6" in capsys.readouterr().out


def test_duplicated_finished_key_fails(tmp_path):
    rows = _six() + [_row("benign-00", 0, "unprotected")]
    status = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)
    assert status["duplicate_finished"] == 1
    assert any("finished more than once" in p for p in status["problems"])


def test_infra_retry_is_not_a_duplicate_and_infra_only_key_is_pending(tmp_path):
    rows = _six()[:5] + [_row("ds-u03-a04", 0, "protected", llm_error="APIConnectionError: boom")]
    rows.append(_row("benign-00", 0, "unprotected", llm_error="APITimeoutError: slow"))  # retried key
    status = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)
    assert status["finished"] == 5 and status["pending_infra"] == 1 and status["duplicate_finished"] == 0


def test_wrong_policy_hash_fails(tmp_path):
    rows = _six()
    rows[2]["policy_sha256"] = "0" * 64
    status = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)
    assert any("policy_sha256" in p for p in status["problems"])


def test_second_resolved_model_fails(tmp_path, capsys):
    rows = _six()
    rows[4]["resolved_model"] = "qwen2.5:3b"
    d = _write(tmp_path, rows)
    assert run_checks.main(["complete", "--results", d, "--model", MODEL, "--expect", "6"]) == 1
    assert "2 distinct resolved_model values" in capsys.readouterr().out


def test_resolved_contains_requires_the_variant(tmp_path, capsys):
    rows = _six()
    for r in rows:
        r["resolved_model"] = "qwen2.5:3b"
    d = _write(tmp_path, rows)
    assert run_checks.main(["complete", "--results", d, "--model", MODEL, "--expect", "6",
                            "--resolved-contains", "ctx8192"]) == 1
    assert "does not contain 'ctx8192'" in capsys.readouterr().out


def test_bad_stamps_fail(tmp_path):
    rows = _six()
    rows[0]["code_commit"] = "unknown"
    rows[1]["scoring"] = 1
    rows[3]["config_model"] = "qwen2.5:3b"
    problems = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)["problems"]
    assert any("40-hex code_commit" in p for p in problems)
    assert any(f"without scoring {SCORING_VERSION}" in p for p in problems)
    assert any("config_model" in p for p in problems)


def test_run_affecting_commit_difference_fails(tmp_path, monkeypatch):
    rows = _six()
    rows[5]["code_commit"] = COMMIT_B
    calls = []

    def fake_diff(a, b):
        calls.append((a, b))
        return ["src/benchmark/runner.py"]
    monkeypatch.setattr(run_checks, "_git_diff_names", fake_diff)
    problems = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)["problems"]
    assert calls == [(COMMIT_A, COMMIT_B)]
    assert any("run-affecting files differ" in p and "runner.py" in p for p in problems)


def test_commit_difference_outside_run_paths_passes(tmp_path):
    rows = _six()
    rows[5]["code_commit"] = COMMIT_B      # fake git diff (autouse) reports no run-affecting change
    status = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)
    assert status["problems"] == [] and len(status["commits"]) == 2


def test_run_affecting_paths_cover_the_engine_but_not_the_report():
    paths = run_checks.RUN_AFFECTING_PATHS
    for p in ("src/benchmark/runner.py", "src/llm", "src/policy_engine", "src/sandbox", "config", "run_benchmark.py"):
        assert p in paths
    for p in ("src/benchmark/report.py", "src/benchmark/run_checks.py", "src/benchmark/jobs.py", "docs"):
        assert p not in paths
