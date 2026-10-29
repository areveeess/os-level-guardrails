"""
Tests for the Phase 5 deliverables build: the read-only freeze of the Phase 4 directories,
the report rendered from the frozen copy only, numbers.json and the CLI's failure path.
All data is synthetic and every file lives under tmp_path; nothing reads results/.
"""

import ast
import csv
import datetime
import hashlib
import json
import os
import stat
import subprocess
import sys

import pytest

from src import deliverables
from src.deliverables import build as build_module
from src.deliverables.__main__ import main
from tests.test_report import make_row, section, write_raw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEFORE = datetime.datetime(2026, 10, 29, 7, 0, 0, tzinfo=datetime.timezone.utc)
AFTER = datetime.datetime(2026, 10, 29, 15, 30, 0, tzinfo=datetime.timezone.utc)

LOCAL = ("qwen2.5-0.5b-agentlora", "scratch-gpt", "qwen2.5-0.5b-base", "ollama-qwen2.5-3b", "ollama-llama3.2-3b")
HOSTED = ("groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.8-27b")
EXPECTED = 12          # 2 dh + 2 ds + 2 benign cases x 1 run x 2 arms


# ── Synthetic results root ──

def _row(model, case_id, arm, kind, provider="hf_local", **over):
    base = dict(provider=provider, scoring=2, pre_injection_attempted=False, pre_injection_attacker_calls=0,
                attack_blocked=False, code_commit="c" * 40, prompt_tokens=[100, 200])
    base.update(over)
    return make_row(model, case_id, 0, arm, kind, **base)


def model_rows(model, provider="hf_local", attack_success=(), drop=0, overflow=0):
    """EXPECTED rows: unprotected attacks attempt and succeed for the case ids in attack_success, the protected
    arm attempts the same ones and is blocked by the policy engine. The last `drop` rows are left out, and
    the first `overflow` benign rows carry a context-overflow error."""
    rows = []
    for kind in ("dh", "ds", "benign"):
        for i in range(2):
            case = f"{kind}-{i:02d}"
            hit = case in attack_success
            rows.append(_row(model, case, "unprotected", kind, provider, attempted=hit, succeeded=hit,
                             refused_or_ignored=not hit))
            rows.append(_row(model, case, "protected", kind, provider, attempted=hit, succeeded=False,
                             attack_blocked=hit, blocked_by={"policy_engine": 1} if hit else {}))
    benign = [r for r in rows if r["kind"] == "benign"]
    for r in benign[:overflow]:
        r["llm_error"] = "context_overflow: prompt longer than the window"
    return rows[:len(rows) - drop] if drop else rows


def write_manifest(directory, expected=EXPECTED):
    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "expected_episodes_per_model": expected,
                   "invocations": [{"started_at": "x", "models": None}]}, f)


OLLAMA_SHOW = ("  Model\n    architecture        qwen2     \n    parameters          3.1B      \n"
               "    context length      32768     \n    quantization        Q4_K_M    \n\n"
               "  Parameters\n    num_ctx    8192    \n")


def make_results_root(tmp_path, attack_success=("dh-00", "ds-00"), hosted_drop=4, extra_rows=None):
    """A results root with the nine current Phase 4 directories and the two legacy ones."""
    root = tmp_path / "results"
    for model in LOCAL:
        provider = "openai_compat" if model.startswith("ollama") else ("scratch" if model == "scratch-gpt"
                                                                      else "hf_local")
        d = root / f"phase4-{model}"
        write_raw(d, model_rows(model, provider, attack_success))
        write_manifest(d)
        if model.startswith("ollama"):
            (d / "ollama_show.txt").write_text(OLLAMA_SHOW, encoding="utf-8")
    for model in HOSTED:
        d = root / f"phase4-{model}"
        write_raw(d, model_rows(model, "openai_compat", attack_success, drop=hosted_drop))
        write_manifest(d)
        (d / "hosted_state.json").write_text(json.dumps({
            "state": "waiting_quota", "next_launch_utc": None,
            "launches": [{"quota_stop": False}, {"quota_stop": True}, {"quota_stop": True}]}), encoding="utf-8")
    control = model_rows("control-always-comply", "scripted",
                         attack_success=("dh-00", "dh-01", "ds-00", "ds-01"))
    for r in control:
        if r["arm"] == "protected" and r["attempted"]:
            r["blocked_by"] = {"tool_broker": 1}
    write_raw(root / "phase4-control", control)
    write_manifest(root / "phase4-control")
    for model in ("gemini-flash-lite", "gemma-4-31b"):
        write_raw(root / "final-20261011" / model, model_rows(model, "openai_compat", attack_success))
    if extra_rows:
        with open(root / "phase4-scratch-gpt" / "raw.jsonl", "a", encoding="utf-8") as f:
            for r in extra_rows:
                f.write(json.dumps(r) + "\n")
    return root


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tree_bytes(directory):
    out = {}
    for dirpath, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                out[os.path.relpath(path, directory)] = f.read()
    return out


@pytest.fixture(autouse=True)
def no_chart_drawing(monkeypatch):
    """Chart drawing is tested in test_deliverables_sections.py; here it is stubbed (text layout in the WSL
    matplotlib takes about 20 s per chart set)."""
    monkeypatch.setattr(build_module.charts, "write_phase5_charts", lambda numbers, chart_dir: {})


@pytest.fixture
def built(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-final"
    lines = build_module.build(str(root), str(frozen), now=BEFORE)
    return root, frozen, lines


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Constants mirror the run-affecting modules (read from source, no import of jobs or run_checks) ──

def _constant(rel_path, name):
    with open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {rel_path}")


def test_constants_mirror_jobs_and_run_checks():
    assert deliverables.FREEZE_UTC == _constant("src/benchmark/jobs.py", "FREEZE_UTC")
    assert deliverables.DS_EXPECTED == _constant("src/benchmark/run_checks.py", "HOSTED_DS_EXPECTED")
    assert deliverables.MODEL_ORDER[:5] == LOCAL and deliverables.MODEL_ORDER[5:] == HOSTED
    assert deliverables.CURRENT_SOURCES[-1] == "phase4-control"
    assert len(deliverables.CURRENT_SOURCES) == 9 and len(deliverables.LEGACY_SOURCES) == 2


# ── Freeze ──

def test_freeze_copies_sources_and_records_hashes(built):
    _, frozen, lines = built
    record = load_json(frozen / "freeze.json")
    assert record["schema"] == 1
    assert record["snapshot_utc"] == "2026-10-29T07:00:00Z"
    assert record["snapshot_display"] == "2026-10-29T07:00Z"
    assert record["freeze_display"] == "2026-10-29T15:30Z"
    assert record["data_state"] == "snapshot"
    names = [s["name"] for s in record["sources"]]
    assert names == list(deliverables.CURRENT_SOURCES) + ["legacy-gemini-flash-lite", "legacy-gemma-4-31b"]
    groups = {s["name"]: s["group"] for s in record["sources"]}
    assert groups["phase4-control"] == "control" and groups["legacy-gemma-4-31b"] == "legacy"
    assert groups["phase4-scratch-gpt"] == "current"
    for s in record["sources"]:
        raw = frozen / "sources" / s["name"] / "raw.jsonl"
        assert s["raw_sha256"] == sha256(raw)
        assert s["raw_lines"] == raw.read_bytes().count(b"\n")
    assert (frozen / "sources" / "phase4-groq-gpt-oss-120b" / "hosted_state.json").exists()
    assert (frozen / "sources" / "phase4-ollama-qwen2.5-3b" / "ollama_show.txt").exists()
    assert (frozen / "sources" / "phase4-scratch-gpt" / "manifest.json").exists()
    assert lines[0] == "FREEZE 2026-10-29T07:00Z state snapshot sources 11"
    assert sum(1 for line in lines if line.startswith("SOURCE ")) == 11


def test_partial_last_line_is_not_copied(tmp_path):
    root = make_results_root(tmp_path)
    raw = root / "phase4-groq-gpt-oss-20b" / "raw.jsonl"
    complete = raw.read_bytes()
    with open(raw, "ab") as f:
        f.write(b'{"model": "groq-gpt-oss-20b", "case_')
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.freeze(str(root), str(frozen), now=BEFORE)
    assert (frozen / "sources" / "phase4-groq-gpt-oss-20b" / "raw.jsonl").read_bytes() == complete


def test_data_state_follows_the_freeze(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-frozen"
    assert build_module.freeze(str(root), str(frozen), now=BEFORE)["data_state"] == "snapshot"
    assert build_module.freeze(str(root), str(frozen), now=AFTER)["data_state"] == "final"
    assert load_json(frozen / "freeze.json")["data_state"] == "final"


def test_build_reads_read_only_sources_without_changing_them(tmp_path):
    root = make_results_root(tmp_path)
    before = {}
    for dirpath, dirnames, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            before[path] = sha256(path)
            os.chmod(path, 0o444)
    dirs = [d for d, _, _ in os.walk(root) if d != str(root)]   # the root stays writable: the frozen dir lives in it
    listings = {d: sorted(os.listdir(d)) for d in dirs}
    for d in dirs:
        os.chmod(d, 0o555)
    try:
        build_module.build(str(root), str(tmp_path / "results" / "phase5-frozen"), now=BEFORE)
        assert {p: sha256(p) for p in before} == before
        assert {d: sorted(os.listdir(d)) for d in dirs} == listings
    finally:
        for d in dirs:
            os.chmod(d, 0o755)
        for p in before:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def test_missing_source_fails_and_keeps_the_frozen_directory(tmp_path, capsys):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.build(str(root), str(frozen), now=BEFORE)
    snapshot = tree_bytes(frozen)
    for path in (root / "phase4-groq-qwen3.8-27b").iterdir():
        path.unlink()
    (root / "phase4-groq-qwen3.8-27b").rmdir()
    code = main(["build", "--results-root", str(root), "--frozen", str(frozen)])
    out = capsys.readouterr().out.splitlines()
    assert code == 1
    assert out[-1].startswith("FAIL build:") and "phase4-groq-qwen3.8-27b" in out[-1]
    assert tree_bytes(frozen) == snapshot


@pytest.mark.parametrize("target", ["phase4-jobs", "phase4-scratch-gpt", "final-20261011", "../docs",
                                    "phase5-final/inner", "."])
def test_frozen_target_outside_results_phase5_is_refused_and_untouched(tmp_path, target, capsys):
    root = make_results_root(tmp_path)
    for name in ("phase4-jobs", "../docs"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "keep.txt").write_text("not a frozen directory", encoding="utf-8")
    before = tree_bytes(tmp_path)
    frozen = root / target
    with pytest.raises(build_module.DeliverablesError, match="refusing"):
        build_module.freeze(str(root), str(frozen), now=BEFORE)
    with pytest.raises(build_module.DeliverablesError, match="refusing"):
        build_module.build(str(root), str(frozen), now=BEFORE)
    code = main(["build", "--results-root", str(root), "--frozen", str(frozen), "--no-docs", "--no-deck"])
    assert code == 1 and capsys.readouterr().out.strip().splitlines()[-1].startswith("FAIL build: ")
    assert tree_bytes(tmp_path) == before


def test_existing_phase5_directory_without_freeze_json_is_refused(tmp_path):
    root = make_results_root(tmp_path)
    other = root / "phase5-notes"
    other.mkdir()
    (other / "draft.md").write_text("hand-written notes", encoding="utf-8")
    with pytest.raises(build_module.DeliverablesError, match="not a frozen directory"):
        build_module.build(str(root), str(other), now=BEFORE)
    assert (other / "draft.md").read_text(encoding="utf-8") == "hand-written notes"
    empty = root / "phase5-empty"
    empty.mkdir()
    build_module.build(str(root), str(empty), now=BEFORE)          # an empty directory is a fine target
    assert (empty / "freeze.json").is_file()
    build_module.build(str(root), str(empty), now=BEFORE)          # and so is a frozen directory, again


@pytest.mark.skipif(not hasattr(os, "symlink") or sys.platform == "win32", reason="symlinks need POSIX")
def test_a_symlinked_frozen_target_is_resolved_before_the_checks(tmp_path):
    root = make_results_root(tmp_path)
    jobs = root / "phase4-jobs"
    jobs.mkdir()
    (jobs / "loop.pid").write_text("1", encoding="utf-8")
    link = root / "phase5-final"
    os.symlink(str(jobs), str(link))
    with pytest.raises(build_module.DeliverablesError, match="refusing"):
        build_module.build(str(root), str(link), now=BEFORE)
    assert (jobs / "loop.pid").read_text(encoding="utf-8") == "1"


def test_unknown_model_fails_the_build(tmp_path):
    root = make_results_root(tmp_path, extra_rows=[_row("mystery-model", "dh-00", "unprotected", "dh")])
    with pytest.raises(build_module.DeliverablesError, match="mystery-model"):
        build_module.build(str(root), str(tmp_path / "results" / "phase5-frozen"), now=BEFORE)


# ── Report from the frozen copy ──

def test_report_has_data_line_and_no_legacy_rows(built):
    _, frozen, lines = built
    text = (frozen / "summary.md").read_text(encoding="utf-8")
    first = text.splitlines()
    assert first[0] == "# Benchmark summary" and first[1] == ""
    assert "2026-10-29T07:00Z" in first[2] and "snapshot" in first[2]
    ranked = text.split("## Headline: complete runs")[1].split("## Confounds between the models")[0]
    for legacy in ("gemini-flash-lite", "gemma-4-31b"):
        assert legacy in section(text, "Legacy rows (older engine, not compared)")     # own table only (D-02)
        assert legacy not in ranked
        assert legacy not in (frozen / "summary.csv").read_text(encoding="utf-8")
        assert legacy not in (frozen / "layer_blocks.csv").read_text(encoding="utf-8")
    headline = text.split("## Headline: complete runs")[1].split("## Outcomes per episode")[0]
    assert "control-always-comply" not in headline
    assert "REPORT " in "\n".join(lines) and "NUMBERS " in "\n".join(lines)


def test_generate_report_preamble_is_optional(tmp_path):
    from src.benchmark.report import generate_report
    d = write_raw(tmp_path / "plain", model_rows("m1"))
    generate_report(d, charts=False)
    plain = (tmp_path / "plain" / "summary.md").read_text(encoding="utf-8")
    generate_report(d, charts=False, preamble=["Data: a line."])
    with_preamble = (tmp_path / "plain" / "summary.md").read_text(encoding="utf-8")
    assert with_preamble.splitlines()[:4] == ["# Benchmark summary", "", "Data: a line.", ""]
    assert with_preamble.replace("Data: a line.\n\n", "", 1) == plain


# ── numbers.json ──

def test_numbers_match_the_report_tables(built):
    _, frozen, _ = built
    numbers = load_json(frozen / "numbers.json")
    assert numbers["schema"] == 1
    assert numbers["snapshot_display"] == "2026-10-29T07:00Z" and numbers["data_state"] == "snapshot"
    assert len(numbers["sources"]) == 11
    models = numbers["models"]
    assert [m["model"] for m in models] == list(deliverables.MODEL_ORDER) + ["control-always-comply"]
    assert [m["group"] for m in models] == ["local"] * 5 + ["hosted"] * 3 + ["control"]
    with open(frozen / "summary.csv", newline="", encoding="utf-8") as f:
        csv_rows = {r["model"]: r for r in csv.DictReader(f)}
    for m in models:
        row = csv_rows[m["model"]]
        assert str(m["episodes_done"]) == row["episodes_done"]
        for prefix, values in m["metrics"].items():
            for key in ("k", "n", "pct", "lo", "hi"):
                expected = row[f"{prefix}_{key}"]
                assert ("" if values[key] is None else str(values[key])) == expected, (m["model"], prefix, key)
        assert "success_unprotected" in m["metrics"] and "blocked_given_attempt" in m["metrics"]
    by_model = {m["model"]: m for m in models}
    assert by_model["scratch-gpt"]["label"] == "complete" and by_model["scratch-gpt"]["complete"] is True
    assert by_model["groq-gpt-oss-20b"]["label"] == "partial, 8/12 episodes"
    assert by_model["groq-gpt-oss-20b"]["hosted_state"] == {"state": "waiting_quota", "launches": 3,
                                                            "quota_stops": 2}
    assert by_model["scratch-gpt"]["finished_by_kind"] == {"dh": 4, "ds": 4, "benign": 4}
    assert all(m["context_overflow_episodes"] == 0 for m in models)
    llm = [m for m in models if m["group"] != "control"]
    pooled = numbers["pooled"]["blocked_given_attempt"]
    assert pooled["k"] == sum(m["metrics"]["blocked_given_attempt"]["k"] for m in llm)
    assert pooled["n"] == sum(m["metrics"]["blocked_given_attempt"]["n"] for m in llm)
    text = (frozen / "summary.md").read_text(encoding="utf-8")
    from src.benchmark.stats import format_prop
    assert f"| Pooled (all LLM models) | {pooled['n']} | {format_prop(pooled['k'], pooled['n'])} |" in text
    with open(frozen / "layer_blocks.csv", newline="", encoding="utf-8") as f:
        pooled_rows = [r for r in csv.DictReader(f) if r["status"] == "pooled"]
    assert pooled_rows
    for r in pooled_rows:
        item = numbers["pooled"]["layers"][r["group"]][r["layer"]]
        assert (str(item["k"]), str(item["n"]), str(item["actions"])) == (r["k"], r["n"], r["actions"])


def test_context_overflow_and_error_dominated_labels(tmp_path):
    root = make_results_root(tmp_path)
    write_raw(root / "phase4-qwen2.5-0.5b-base", model_rows("qwen2.5-0.5b-base", overflow=1))
    rows = model_rows("groq-gpt-oss-120b", "openai_compat")
    for r in rows[:8]:
        r["llm_error"] = "HTTP 503 upstream"
    write_raw(root / "phase4-groq-gpt-oss-120b", rows)
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.build(str(root), str(frozen), now=BEFORE)
    by_model = {m["model"]: m for m in load_json(frozen / "numbers.json")["models"]}
    assert by_model["qwen2.5-0.5b-base"]["context_overflow_episodes"] == 1
    assert by_model["groq-gpt-oss-120b"]["label"] == "partial, 4/12 episodes (error-dominated)"


def test_reuse_frozen_copies_nothing_and_is_byte_identical(built):
    root, frozen, _ = built
    numbers = (frozen / "numbers.json").read_bytes()
    record = (frozen / "freeze.json").read_bytes()
    with open(root / "phase4-scratch-gpt" / "raw.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_row("scratch-gpt", "dh-09", "unprotected", "dh")) + "\n")
    frozen_raw = sha256(frozen / "sources" / "phase4-scratch-gpt" / "raw.jsonl")
    lines = build_module.build(str(root), str(frozen), reuse=True,
                               now=datetime.datetime(2026, 10, 29, 9, 0, tzinfo=datetime.timezone.utc))
    assert lines[0].startswith("REUSE 2026-10-29T07:00Z state snapshot sources 11")
    assert (frozen / "numbers.json").read_bytes() == numbers
    assert (frozen / "freeze.json").read_bytes() == record
    assert sha256(frozen / "sources" / "phase4-scratch-gpt" / "raw.jsonl") == frozen_raw
    assert b"\r\n" not in numbers


# ── Import hygiene: the build loads no keys ──

def test_build_imports_no_registry_jobs_or_run_checks():
    code = ("import sys, src.deliverables.build, src.deliverables.__main__; "
            "print(sorted(m for m in sys.modules if m.startswith('src.llm') or m in "
            "('src.benchmark.jobs', 'src.benchmark.run_checks')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
