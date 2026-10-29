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


def test_provenance_problems_are_separate_from_progress(tmp_path, monkeypatch):
    """WR-04: policy or commit drift on a finished directory stays a strict FAIL but is listed under provenance,
    and progress_complete (the schedulers' gate) is true; a missing key is a progress problem, not provenance."""
    rows = _six()
    rows[2]["policy_sha256"] = "0" * 64
    rows[5]["code_commit"] = COMMIT_B
    monkeypatch.setattr(run_checks, "_git_diff_names", lambda a, b: ["src/llm/registry.py"])
    status = run_checks.complete_status(_write(tmp_path, rows), MODEL, 6)
    assert run_checks.progress_complete(status) is True
    assert len(status["provenance"]) == 2 and status["provenance"] == status["problems"]
    assert any("policy_sha256" in p for p in status["provenance"])
    assert any("run-affecting files differ" in p for p in status["provenance"])
    status = run_checks.complete_status(_write(tmp_path, _six()[:5]), MODEL, 6)
    assert run_checks.progress_complete(status) is False
    assert status["provenance"] == [] and status["problems"] == ["finished 5 below expected 6"]


def test_progress_complete_needs_no_pending_infra_and_a_known_expected():
    assert run_checks.progress_complete({"finished": 6, "expected": 6, "pending_infra": 0}) is True
    assert run_checks.progress_complete({"finished": 6, "expected": 6, "pending_infra": 1}) is False
    assert run_checks.progress_complete({"finished": 6, "expected": None, "pending_infra": 0}) is False


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


# ── hosted (plan 04-05 Task 3) ──

import datetime  # noqa: E402

from src.benchmark import jobs  # noqa: E402

GROQ = "groq-gpt-oss-20b"
FREEZE = datetime.datetime(2026, 10, 29, 15, 30, tzinfo=datetime.timezone.utc)


def _hosted_dir(prefix, model=GROQ, ds=0, other=0, ds_infra=0, state="running", launches=None):
    out = f"{prefix}-{model}"
    import os
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "raw.jsonl"), "w", encoding="utf-8") as f:
        for i in range(ds):
            f.write(json.dumps(_row(f"ds-u{i:02d}", model=model, kind="ds")) + "\n")
        for i in range(ds_infra):
            f.write(json.dumps(_row(f"ds-x{i:02d}", model=model, kind="ds", llm_error="InternalServerError 500"))
                    + "\n")
        for i in range(other):
            f.write(json.dumps(_row(f"dh-u{i:02d}", model=model, kind="dh")) + "\n")
    with open(os.path.join(out, "hosted_state.json"), "w", encoding="utf-8") as f:
        json.dump({"state": state, "next_launch_utc": None, "launches": launches or []}, f)
    return out


@pytest.fixture
def hosted_env(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "stopping_rule.json").write_text(json.dumps({"freeze_utc": "2026-10-29T15:30:00+00:00"}))
    monkeypatch.setattr(jobs, "_now", lambda: FREEZE - datetime.timedelta(hours=1))
    monkeypatch.setattr(jobs, "_list_pids", lambda: [])
    return str(tmp_path / "p4"), str(jobs_dir), monkeypatch


def _hosted_main(prefix, jobs_dir):
    return run_checks.main(["hosted", "--prefix", prefix, "--models", GROQ, "--jobs-dir", jobs_dir])


def test_hosted_partial_label_and_running(hosted_env, capsys):
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=30, state="waiting_quota",
                launches=[{"end": "2026-10-29T10:00:00Z", "quota_stop": True, "stated_retry_s": 7200.0}])
    assert _hosted_main(prefix, jobs_dir) == 0
    out = capsys.readouterr().out
    assert f"HOSTED {GROQ} state waiting_quota finished 30/342 ds 30/120" in out
    assert "partial n=30 ds n=30" in out and "quota_stops 1" in out
    assert out.strip().endswith("hosted-running")


def test_hosted_terminal_ok_when_every_model_is_terminal(hosted_env, capsys):
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=120, other=10, state="stopped_by_quota")
    assert _hosted_main(prefix, jobs_dir) == 0
    out = capsys.readouterr().out
    assert "partial n=130 ds n=120" in out and out.strip().endswith("hosted-terminal-ok")


def test_hosted_terminal_ok_after_the_freeze_when_no_child_is_alive(hosted_env, capsys):
    prefix, jobs_dir, monkeypatch = hosted_env
    _hosted_dir(prefix, ds=12, state="running")
    monkeypatch.setattr(jobs, "_now", lambda: FREEZE + datetime.timedelta(minutes=1))
    assert _hosted_main(prefix, jobs_dir) == 0
    out = capsys.readouterr().out
    assert "stopped at the freeze" in out and out.strip().endswith("hosted-terminal-ok")


def test_hosted_running_after_the_freeze_while_a_child_is_alive(hosted_env, capsys):
    prefix, jobs_dir, monkeypatch = hosted_env
    out_dir = _hosted_dir(prefix, ds=12, state="running")
    import os
    monkeypatch.setattr(jobs, "_now", lambda: FREEZE + datetime.timedelta(minutes=1))
    monkeypatch.setattr(jobs, "_list_pids", lambda: [812])
    argv = ["/x/venv/bin/python", "-u", "run_benchmark.py", "--resume", os.path.abspath(out_dir)]
    monkeypatch.setattr(jobs, "_read_proc", lambda pid, name: ("\0".join(argv) + "\0").encode()
                        if name == "cmdline" else None)
    assert _hosted_main(prefix, jobs_dir) == 0
    assert capsys.readouterr().out.strip().endswith("hosted-running")


def test_hosted_fails_on_complete_below_342(hosted_env, capsys):
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=120, other=100, state="complete")
    assert _hosted_main(prefix, jobs_dir) == 1
    assert "FAIL hosted" in capsys.readouterr().out


def test_hosted_complete_with_problems_is_terminal_complete_and_fails(hosted_env, capsys):
    """WR-04: the hosted loop marks a finished model with provenance drift complete_with_problems; the check
    labels it complete, counts it terminal and names the provenance problem as a FAIL."""
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=120, other=222, state="complete_with_problems")
    problems, word = run_checks.check_hosted(prefix, [GROQ], jobs_dir)
    assert word == "hosted-terminal-ok" and len(problems) == 1 and "provenance problems (WR-04" in problems[0]
    assert "label complete (provenance problems: see run_checks complete)" in capsys.readouterr().out
    assert _hosted_main(prefix, jobs_dir) == 1
    assert "FAIL hosted: groq-gpt-oss-20b finished its episodes but has provenance problems" in capsys.readouterr().out


def test_hosted_fails_on_a_ds_first_violation(hosted_env, capsys):
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=5, other=10, state="running")
    assert _hosted_main(prefix, jobs_dir) == 1
    assert "ds-first" in capsys.readouterr().out


def test_hosted_pending_ds_infra_counts_toward_ds_first(hosted_env, capsys):
    prefix, jobs_dir, _ = hosted_env
    _hosted_dir(prefix, ds=8, ds_infra=2, other=2, state="running")
    assert _hosted_main(prefix, jobs_dir) == 0


# ── secret-scan ──

def test_secret_scan_fails_on_a_planted_key_and_never_prints_it(tmp_path, capsys):
    fake = "gsk" + "_" + "Ab3" * 8                      # built at runtime: no key-shaped literal in this file
    (tmp_path / "clean.py").write_text("x = 1\n")
    (tmp_path / "leak.txt").write_text("first line\ntoken = " + fake + "\n")
    assert run_checks.main(["secret-scan", "--paths", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL secret-scan: {tmp_path / 'leak.txt'}:2" in out
    assert fake not in out and "Ab3Ab3" not in out


def test_secret_scan_passes_on_clean_files_and_counts_them(tmp_path, capsys):
    (tmp_path / "a.py").write_text("gsk_short = 1\n")       # too short to be a key
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("nothing here\n")
    (tmp_path / "bin.dat").write_bytes(b"\0\1\2")
    assert run_checks.main(["secret-scan", "--paths", str(tmp_path / "a.py"), str(tmp_path / "sub"),
                            str(tmp_path / "bin.dat")]) == 0
    assert capsys.readouterr().out.strip() == "secret-scan-ok files 2"


def test_secret_scan_reads_only_the_given_paths(tmp_path, capsys):
    fake = "gsk" + "_" + "Zz9" * 8
    (tmp_path / "outside.txt").write_text(fake + "\n")
    (tmp_path / "inside").mkdir()
    (tmp_path / "inside" / "ok.txt").write_text("fine\n")
    assert run_checks.main(["secret-scan", "--paths", str(tmp_path / "inside")]) == 0
    assert "secret-scan-ok files 1" in capsys.readouterr().out


# ── viva (plan 04-04) ──

VIVA_MODELS = (("local-model-a", "hf_local"), ("hosted-model-b", "openai_compat"))
VIVA_CONTROL = "control-always-comply"


def _viva_report(tmp_path):
    """A real summary.csv written by generate_report from synthetic rows: model a complete (6/6), model b
    partial (4/6) and the scripted control (6/6, must not be required on the page)."""
    from src.benchmark.report import generate_report
    from tests.test_report import make_row, write_raw
    rows = []
    for model, provider in VIVA_MODELS + ((VIVA_CONTROL, "scripted"),):
        cases = ("dh-00", "ds-01", "benign-02") if model != "hosted-model-b" else ("dh-00", "ds-01")
        for case in cases:
            for arm in ("unprotected", "protected"):
                rows.append(make_row(model, case, 0, arm, case.split("-")[0], provider=provider))
    src = write_raw(tmp_path / "raw", rows)
    out = str(tmp_path / "report")
    generate_report(out, charts=False, sources=[src], expected=6)
    return tmp_path / "report" / "summary.csv"


def _viva_page(tmp_path, drop=None, extra=""):
    parts = {h: f"{h} — section\n\ntext\n" for h in run_checks.VIVA_HEADINGS}
    if drop in parts:
        del parts[drop]
    glossary = "".join(f"- **Term {i}**: meaning {i}.\n" for i in range(30))
    questions = "".join(f"**Q{i}. Question {i}?**\nAnswer {i}.\n\n" for i in range(1, 17))
    table = "| Model | Episodes done |\n|---|---|\n"
    for model, count in (("local-model-a", "6/6"), ("hosted-model-b", "4/6")):
        if model != drop:
            table += f"| {model} | {count} |\n"
    page = "# Viva notes 4\n\n" + glossary + "\n" + "\n".join(parts.values()) + table + "\n" + questions + extra
    path = tmp_path / run_checks.VIVA_PAGE_NAME
    path.write_text(page, encoding="utf-8")
    return path


def _viva_readme(tmp_path, status="written (Phase 4)"):
    path = tmp_path / "README.md"
    path.write_text(f"| Page | Topics | Phase | Status |\n|---|---|---|---|\n"
                    f"| [{run_checks.VIVA_PAGE_NAME}]({run_checks.VIVA_PAGE_NAME}) | namespaces | Phase 4 | {status} |\n"
                    f"| 05 Examiner Q&A | final | Phase 5 | planned (Phase 5) |\n", encoding="utf-8")
    return path


def _viva(page, csv_path, readme):
    return run_checks.main(["viva", "--page", str(page), "--report-csv", str(csv_path), "--readme", str(readme)])


def test_viva_end_to_end_on_a_generated_summary_csv(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "episodes_done" in header and "episodes_expected" in header
    assert _viva(_viva_page(tmp_path), csv_path, _viva_readme(tmp_path)) == 0
    assert capsys.readouterr().out.strip() == "viva-numbers-ok"


def test_viva_fails_on_a_missing_heading(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    assert _viva(_viva_page(tmp_path, drop="## Part C"), csv_path, _viva_readme(tmp_path)) == 1
    assert "FAIL viva: heading '## Part C' is missing" in capsys.readouterr().out


def test_viva_fails_on_a_missing_model_count(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    assert _viva(_viva_page(tmp_path, drop="hosted-model-b"), csv_path, _viva_readme(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "FAIL viva: the page does not name model hosted-model-b" in out
    assert "local-model-a" not in out and VIVA_CONTROL not in out


def test_viva_fails_when_the_count_is_on_another_models_line(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    page = _viva_page(tmp_path)
    page.write_text(page.read_text(encoding="utf-8").replace("| hosted-model-b | 4/6 |", "| hosted-model-b | 5/6 |"),
                    encoding="utf-8")
    assert _viva(page, csv_path, _viva_readme(tmp_path)) == 1
    assert "FAIL viva: the page does not quote hosted-model-b with 4/6 on one line" in capsys.readouterr().out


def test_viva_fails_on_a_planted_key_shaped_string_and_never_prints_it(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    fake = "gsk" + "_" + "Qx7" * 8                      # built at runtime: no key-shaped literal in this file
    assert _viva(_viva_page(tmp_path, extra="note " + fake + "\n"), csv_path, _viva_readme(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "FAIL viva: key-shaped text found at" in out
    assert fake not in out and "Qx7Qx7" not in out


def test_viva_fails_without_the_readme_row(tmp_path, capsys):
    csv_path = _viva_report(tmp_path)
    assert _viva(_viva_page(tmp_path), csv_path, _viva_readme(tmp_path, status="planned (Phase 4)")) == 1
    assert "FAIL viva: the README has no row linking" in capsys.readouterr().out


def test_viva_fails_on_a_csv_without_the_count_columns(tmp_path, capsys):
    old = tmp_path / "old_summary.csv"                  # the shape of a pre-Phase-1 report
    old.write_text("model,family,provider,status\nlocal-model-a,fam,hf_local,complete\n", encoding="utf-8")
    assert _viva(_viva_page(tmp_path), old, _viva_readme(tmp_path)) == 1
    assert ("FAIL viva: summary.csv lacks episodes_done/episodes_expected (not a current-engine report)"
            in capsys.readouterr().out)
