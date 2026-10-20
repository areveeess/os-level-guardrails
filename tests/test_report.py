"""
Tests for the honest benchmark report: Wilson intervals, k/n cells, completeness
gate, outcome taxonomy, pair dropping and the conditional block rate.
All rows are synthetic and all output goes under tmp_path.
"""

import json
import re

import pytest

from src.benchmark.report import (DEFAULT_EXPECTED_EPISODES, _error_bars, generate_report,
                                  load_rows, load_rows_counted, summarize)
from src.benchmark.stats import Z_95, format_prop, prop_fields, wilson_interval


def make_row(model="m", case_id="dh-00", run=0, arm="unprotected", kind="dh", **over):
    """One raw.jsonl-shaped episode row (every field the report reads)."""
    row = {
        "model": model, "family": "fam", "case_id": case_id, "kind": kind, "attack_type": "",
        "run": run, "arm": arm, "attempted": False, "succeeded": False, "refused_or_ignored": True,
        "canary_exposed": False, "os_tags": {}, "blocked_by": {}, "retries_after_block": 0,
        "user_task_done": True, "false_positive": False, "tool_calls": [], "steps": 2,
        "parse_errors": 0, "latency_ms": 10.0, "stop_reason": "responded", "llm_error": None,
        "actions": [{"action": "respond", "message": "done"}], "final_message": "done",
    }
    row.update(over)
    return row


def write_raw(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(directory)


def full_model(model, cases=3):
    """cases x 1 run x 2 arms = 2*cases error-free rows (dh, ds, benign mix)."""
    kinds = ["dh", "ds", "benign"] * cases
    rows = []
    for i in range(cases):
        for arm in ("unprotected", "protected"):
            rows.append(make_row(model, f"{kinds[i]}-{i:02d}", 0, arm, kinds[i]))
    return rows


def section(text, heading):
    """Body of one '## ' section of a markdown document."""
    m = re.search(r"^## " + re.escape(heading) + r"\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


# ── Wilson interval and k/n formatting ──

def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 47)
    assert lo == 0.0 and hi == pytest.approx(0.0756, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_interval(47, 47)
    assert lo == pytest.approx(0.9244, abs=1e-4) and hi == 1.0
    assert wilson_interval(0, 0) is None
    assert Z_95 == 1.96


@pytest.mark.parametrize("k,n", [(3, 2), (-1, 5), (1, -1)])
def test_wilson_interval_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_format_prop_and_prop_fields():
    assert format_prop(14, 120) == "11.7% (14/120) [7.1, 18.6]"
    assert format_prop(0, 120) == "0.0% (0/120) [0.0, 3.1]"
    assert format_prop(0, 0) == "— (0/0)"
    assert prop_fields("attempted", 14, 120) == {
        "attempted_pct": 11.7, "attempted_k": 14, "attempted_n": 120,
        "attempted_lo": 7.1, "attempted_hi": 18.6}
    empty = prop_fields("x", 0, 0)
    assert empty["x_pct"] is None and empty["x_lo"] is None and empty["x_hi"] is None
    assert empty["x_k"] == 0 and empty["x_n"] == 0


# ── Completeness gate ──

def test_summarize_completeness_status():
    rows = full_model("m")
    s = summarize(rows, expected=6)[0]
    assert (s["episodes_done"], s["episodes_expected"], s["status"]) == (6, 6, "complete")
    assert s["pairs_complete"] == 3 and s["infra_errors"] == 0

    partial = summarize(rows[:4], expected=6)[0]
    assert partial["status"] == "partial" and partial["episodes_done"] == 4

    errored = [dict(r, llm_error="InternalServerError: 500") for r in rows[:4]] + rows[4:]
    s = summarize(errored, expected=6)[0]
    assert s["infra_errors"] == 4 and s["episodes_done"] == 2 and s["status"] == "error-dominated"


def test_summarize_without_expected_uses_default():
    s = summarize(full_model("m"))[0]
    assert DEFAULT_EXPECTED_EPISODES == 342
    assert s["episodes_expected"] == 342 and s["status"] == "partial"


def test_every_model_summary_has_the_same_keys():
    a, b = summarize(full_model("a") + full_model("b")[:2], expected=6)
    assert list(a.keys()) == list(b.keys())


def test_report_segregates_partial_runs(tmp_path):
    src_a = write_raw(tmp_path / "a", full_model("complete-model"))
    src_b = write_raw(tmp_path / "b", full_model("partial-model")[:4])
    out = tmp_path / "out"
    generate_report(str(out), charts=False, sources=[src_a, src_b], expected=6)
    text = (out / "summary.md").read_text(encoding="utf-8")
    headline = section(text, "Headline: complete runs")
    partial = section(text, "Partial or error-dominated runs (not compared)")
    assert "complete-model" in headline and "6/6" in headline
    assert "partial-model" not in headline
    assert "partial-model" in partial and "4/6" in partial and "complete-model" not in partial


def test_report_without_any_complete_run_says_so(tmp_path):
    src = write_raw(tmp_path / "a", full_model("p")[:2])
    out = tmp_path / "out"
    generate_report(str(out), charts=False, sources=[src], expected=6)
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "No model has a complete run yet." in section(text, "Headline: complete runs")


# ── Robust loading ──

def test_load_rows_skips_truncated_line_and_report_warns(tmp_path, capsys):
    rows = full_model("m")
    src = write_raw(tmp_path / "a", rows)
    with open(tmp_path / "a" / "raw.jsonl", "a", encoding="utf-8") as f:
        f.write('{"model": "m", "case_id": "dh-0')          # killed mid-write
    loaded, skipped = load_rows_counted(str(tmp_path / "a" / "raw.jsonl"))
    assert len(loaded) == 6 and skipped == 1
    assert len(load_rows(str(tmp_path / "a" / "raw.jsonl"))) == 6

    out = tmp_path / "out"
    generate_report(str(out), charts=False, sources=[src], expected=6)
    assert "Skipped 1 undecodable line(s)" in capsys.readouterr().out
    assert (out / "summary.md").exists()


def test_rows_from_before_the_stamp_fields_are_reported(tmp_path):
    src = write_raw(tmp_path / "a", full_model("old-run"))        # no finish_reasons, code_commit, ...
    out = tmp_path / "out"
    summary = generate_report(str(out), charts=False, sources=[src], expected=6)
    assert summary[0]["status"] == "complete"


# ── Markdown safety ──

def test_pipe_in_model_name_keeps_table_columns(tmp_path):
    src = write_raw(tmp_path / "a", full_model("evil|name"))
    out = tmp_path / "out"
    generate_report(str(out), charts=False, sources=[src], expected=6)
    text = (out / "summary.md").read_text(encoding="utf-8")
    body = section(text, "Headline: complete runs").strip().splitlines()
    table = [line for line in body if line.startswith("|")]
    split = lambda line: re.split(r"(?<!\\)\|", line.strip())
    assert "evil\\|name" in table[2]
    assert len({len(split(line)) for line in table}) == 1


# ── Chart error bars ──

def test_error_bars_come_from_the_prop_fields():
    summary = [dict(prop_fields("attempted", 14, 120)), dict(prop_fields("attempted", 0, 0))]
    lower, upper = _error_bars(summary, "attempted")
    assert lower[0] == pytest.approx(4.6, abs=0.05) and upper[0] == pytest.approx(6.9, abs=0.05)
    assert (lower[1], upper[1]) == (0, 0)


def test_charts_are_written_with_error_bars(tmp_path):
    src = write_raw(tmp_path / "a", full_model("m"))
    out = tmp_path / "out"
    generate_report(str(out), charts=True, sources=[src], expected=6)
    for name in ("attack_outcomes.png", "blocks_by_layer.png", "benign_utility.png"):
        assert (out / "charts" / name).exists()
