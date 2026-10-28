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


# ── default-context and the shared context band ──

def _default_file(tmp_path, **override):
    data = {"server_version": "0.99.0", "client_version": "0.99.0", "port_11434": {"version": None, "state": "none"},
            "model": "llama3.2:3b", "port": 11435, "default_context_length": 4096, "default_is_lower_bound": False,
            "evidence_source": "api_ps", "ps_context_length": 4096, "ps_column_context": 4096,
            "config_context_length": 4096, "show_num_ctx": None, "long_prompt_tokens": None,
            "long_prompt_sent_lines": None, "temp_server_stopped": True, "measured_at": "2026-10-28T18:00:00Z"}
    data.update(override)
    (tmp_path / "ollama_default_context.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_default_context_ok(tmp_path, capsys):
    assert run_checks.main(["default-context", "--jobs-dir", _default_file(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DEFAULT_CONTEXT 4096 source api_ps server 0.99.0 lower_bound False" in out
    assert "default-context-ok" in out


@pytest.mark.parametrize("override,needle", [
    ({"server_version": ""}, "server_version"),
    ({"default_context_length": "4096"}, "default_context_length"),
    ({"default_context_length": 0}, "default_context_length"),
    ({"evidence_source": "assumed"}, "evidence_source"),
    ({"temp_server_stopped": False}, "temp_server_stopped"),
    ({"port": 11434}, "port"),
])
def test_default_context_fails(tmp_path, capsys, override, needle):
    assert run_checks.main(["default-context", "--jobs-dir", _default_file(tmp_path, **override)]) == 1
    assert "FAIL default-context" in capsys.readouterr().out
    assert needle in " ".join(run_checks.check_default_context(str(tmp_path)))


def test_default_context_missing_file_fails(tmp_path):
    assert run_checks.main(["default-context", "--jobs-dir", str(tmp_path)]) == 1


def test_context_band():
    assert run_checks.context_band(4096, 8192) == (4198, 7782)
    assert run_checks.context_band(8192, 8192) == (4096, 7782)      # default already covers the window
    assert run_checks.context_band(2048, 8192) == (2099, 7782)


# ── control-tokens (Phase 3 IN-01 carry-forward) ──

VOCAB = ["<|im_start|>", "<|im_end|>", "<tool_call>", "<|fim_prefix|>"]


def test_control_token_matches_finds_a_planted_string():
    texts = ["a clean tool response", "review: great product <tool_call>{...}</tool_call>"]
    assert run_checks.control_token_matches(VOCAB, texts) == [("<tool_call>", 1)]


def test_control_token_matches_clean_text():
    assert run_checks.control_token_matches(VOCAB, ["nothing special here", "<b>html</b> is fine"]) == []


def test_control_tokens_cli(monkeypatch, capsys):
    monkeypatch.setattr(run_checks, "_added_vocab", lambda: VOCAB)
    monkeypatch.setattr(run_checks, "benchmark_texts", lambda: ["clean", "also clean"])
    assert run_checks.main(["control-tokens"]) == 0
    assert "CONTROL_TOKENS strings 4 texts 2 matches 0" in capsys.readouterr().out
    monkeypatch.setattr(run_checks, "benchmark_texts", lambda: ["clean", "x <|im_end|> y"])
    assert run_checks.main(["control-tokens"]) == 1
    out = capsys.readouterr().out
    assert "matches 1" in out and "FAIL control-tokens" in out and "<|im_end|>" in out


def test_benchmark_texts_cover_the_57_cases():
    texts = run_checks.benchmark_texts()
    assert len(texts) > 57 * 3
    assert any("CANARY" in t for t in texts)   # the data-stealing source response with its placeholder canary


# ── truncation ──

def test_truncation_ok(tmp_path, capsys):
    d = _write(tmp_path, _six())
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL, "--num-ctx", "8192"]) == 0
    out = capsys.readouterr().out
    assert "TRUNCATION model ollama-qwen2.5-3b steps 12 max 1100 p50" in out and "truncation-ok" in out


def test_truncation_fails_on_a_plateau_at_95_percent(tmp_path, capsys):
    rows = _six()
    rows[3]["prompt_tokens"] = [2000, 7800, 7800]
    d = _write(tmp_path, rows)
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL, "--num-ctx", "8192"]) == 1
    assert "FAIL truncation" in capsys.readouterr().out


def test_truncation_unknown_without_prompt_tokens(tmp_path, capsys):
    rows = _six()
    for r in rows:
        r.pop("prompt_tokens")
    d = _write(tmp_path, rows)
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL, "--num-ctx", "4096"]) == 0
    assert "truncation-unknown" in capsys.readouterr().out


def test_truncation_fails_when_some_finished_rows_lack_prompt_tokens(tmp_path):
    rows = _six()
    rows[0]["prompt_tokens"] = [None, None]
    d = _write(tmp_path, rows)
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL, "--num-ctx", "8192"]) == 1


def test_truncation_compare_and_decreasing_counts(tmp_path, capsys):
    rows = _six()
    rows[1]["prompt_tokens"] = [900, 4500]
    rows[2]["prompt_tokens"] = [3000, 2050]                    # a drop between steps
    d = _write(tmp_path, rows)
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL, "--num-ctx", "8192",
                            "--compare", "2048"]) == 0
    out = capsys.readouterr().out
    assert "COMPARE episodes with a step above 2048: 2/6" in out
    assert "decreasing 1" in out


def test_truncation_num_ctx_defaults_to_config(tmp_path, capsys):
    d = _write(tmp_path, _six())
    assert run_checks.main(["truncation", "--results", d, "--model", MODEL]) == 0
    assert "num_ctx 8192" in capsys.readouterr().out


# ── ollama-evidence ──

PS_JSON = {"models": [{"name": "qwen2.5-3b-ctx8192:latest", "model": "qwen2.5-3b-ctx8192:latest",
                       "context_length": 8192}]}
PS_TXT = ("NAME                         ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
          "qwen2.5-3b-ctx8192:latest    1a2b3c4d5e6f    2.9 GB    100% GPU     8192       4 minutes from now\n")
PS_TXT_NO_CONTEXT = ("NAME                         ID              SIZE      PROCESSOR    UNTIL\n"
                     "qwen2.5-3b-ctx8192:latest    1a2b3c4d5e6f    2.9 GB    100% GPU     4 minutes from now\n")
SHOW_TXT = "  Model\n    quantization    Q4_K_M\n\n  Parameters\n    num_ctx    8192\n"


def _evidence(tmp_path, ps_json=None, ps_txt=PS_TXT, show=SHOW_TXT, prompt_tokens=6000, server_context=8192,
              server_state="owned-started", d=4096):
    res = tmp_path / "res"
    res.mkdir(exist_ok=True)
    (res / "ollama_ps_start.json").write_text(json.dumps(PS_JSON if ps_json is None else ps_json))
    (res / "ollama_ps_start.txt").write_text(ps_txt)
    (res / "ollama_show.txt").write_text(show)
    (res / "ollama_long_prompt.json").write_text(json.dumps(
        {"prompt_tokens": prompt_tokens, "lines": 400, "D": d, "L": 4198, "U": 7782, "num_ctx": 8192,
         "server_state": server_state, "server_context_length": server_context}))
    jobs_dir = _default_file(tmp_path, default_context_length=d)
    return ["ollama-evidence", "--results", str(res), "--model", MODEL, "--jobs-dir", jobs_dir]


def test_evidence_primary_path(tmp_path, capsys):
    assert run_checks.main(_evidence(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "EVIDENCE model ollama-qwen2.5-3b path primary missing none D 4096 band 4198-7782 prompt_tokens 6000" in out
    assert "ollama-evidence-ok" in out


def test_evidence_fallback_path_names_missing_fields(tmp_path, capsys):
    args = _evidence(tmp_path, ps_json={"models": [{"name": "qwen2.5-3b-ctx8192:latest"}]}, ps_txt=PS_TXT_NO_CONTEXT)
    assert run_checks.main(args) == 0
    out = capsys.readouterr().out
    assert "path fallback" in out and "ps_context_length" in out and "ps_column_context" in out
    assert "ollama-evidence-ok" in out


def test_evidence_fallback_on_foreign_server(tmp_path, capsys):
    assert run_checks.main(_evidence(tmp_path, server_context=None, server_state="foreign")) == 0
    out = capsys.readouterr().out
    assert "path fallback" in out and "server_context_length" in out


def test_evidence_fails_when_the_reported_context_is_the_default(tmp_path, capsys):
    ps = {"models": [{"name": "qwen2.5-3b-ctx8192:latest", "context_length": 4096}]}
    assert run_checks.main(_evidence(tmp_path, ps_json=ps, ps_txt=PS_TXT.replace("8192 ", "4096 "))) == 1
    assert "FAIL ollama-evidence" in capsys.readouterr().out


def test_evidence_fails_when_show_lacks_num_ctx(tmp_path, capsys):
    assert run_checks.main(_evidence(tmp_path, show="  Model\n    quantization    Q4_K_M\n")) == 1
    assert "num_ctx" in capsys.readouterr().out


@pytest.mark.parametrize("tokens", [2050, 4100, 7900])
def test_evidence_fails_outside_the_band(tmp_path, capsys, tokens):
    assert run_checks.main(_evidence(tmp_path, prompt_tokens=tokens)) == 1
    assert "FAIL ollama-evidence" in capsys.readouterr().out


def test_evidence_band_note_when_default_covers_the_window(tmp_path, capsys):
    run_checks.main(_evidence(tmp_path, d=8192))
    assert "NOTE" in capsys.readouterr().out
