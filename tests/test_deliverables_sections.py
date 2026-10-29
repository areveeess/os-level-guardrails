"""
Tests for the Phase 5 report sections (confounds, legacy rows, threats to validity, chart links)
and the three D-05 charts. Synthetic data under tmp_path; fake manifests and config for the
confounds table.
"""

import ast
import os
import re
import shutil

import pytest

from src import deliverables
from src.benchmark.report import generate_report, legacy_lines
from src.deliverables import build as build_module
from src.deliverables import charts as charts_module
from src.deliverables import sections
from tests.test_deliverables_build import (BEFORE, OLLAMA_SHOW, REPO_ROOT, fake_repo, hermetic_repo,  # noqa: F401
                                           load_json, make_results_root, model_rows, write_raw)
from tests.test_report import section

REAL_CHARTS = charts_module.write_phase5_charts


@pytest.fixture(autouse=True)
def no_chart_drawing(monkeypatch):
    """The build's chart drawing is stubbed except where a test restores it (text layout in the WSL matplotlib
    takes about 20 s per chart set)."""
    monkeypatch.setattr(charts_module, "write_phase5_charts", lambda numbers, chart_dir: {})


NEW_HEADINGS = ("## Confounds between the models", "## Legacy rows (older engine, not compared)", "## Charts",
                "## Threats to validity")


@pytest.fixture
def frozen(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-final"
    build_module.freeze(str(root), str(frozen), now=BEFORE)
    return frozen


def _frozen_rows(frozen):
    return build_module.load_frozen_rows([build_module.source_dir(str(frozen), n)
                                          for n in deliverables.CURRENT_SOURCES[:-1]])


# ── Confounds ──

def test_confound_rows_cover_every_model_from_recorded_sources(frozen, tmp_path):
    rows = sections.confound_rows(str(frozen), _frozen_rows(frozen), repo_root=str(fake_repo(tmp_path)))
    assert [r["model"] for r in rows] == list(deliverables.MODEL_ORDER)
    by = {r["model"]: r for r in rows}
    assert by["qwen2.5-0.5b-base"]["parameters"] == "494.0M"
    assert by["qwen2.5-0.5b-agentlora"]["parameters"] == "494.0M + 8.8M LoRA"
    assert by["scratch-gpt"]["parameters"] == "4.8M"
    assert by["ollama-qwen2.5-3b"]["parameters"] == "3.1B"
    assert by["groq-qwen3.8-27b"]["parameters"] == "27B (per the model id, not verified here)"
    assert by["ollama-llama3.2-3b"]["precision"] == "Q4_K_M"
    assert by["qwen2.5-0.5b-base"]["precision"] == "bf16 on the GPU"
    assert by["scratch-gpt"]["precision"] == "fp32"
    assert by["groq-gpt-oss-20b"]["precision"] == "not disclosed"
    assert by["ollama-qwen2.5-3b"]["context"] == "8192 tokens (num_ctx)"
    assert by["qwen2.5-0.5b-agentlora"]["context"] == f"{sections.HF_LOCAL_CONTEXT} tokens"
    assert by["scratch-gpt"]["context"] == "2048 characters"
    assert by["groq-gpt-oss-120b"]["context"] == "provider limit, not measured"
    assert by["scratch-gpt"]["provenance"] == "trained from scratch here"
    assert by["qwen2.5-0.5b-agentlora"]["provenance"] == "LoRA-tuned here on the Phase 2 synthetic mix"
    assert by["ollama-llama3.2-3b"]["provenance"].startswith("pretrained instruct, pulled Ollama tag")
    assert by["groq-gpt-oss-120b"]["provenance"] == "hosted vendor model"
    assert by["groq-gpt-oss-120b"]["placement"] == "hosted (Groq)"
    assert "temperature 0" in by["groq-gpt-oss-120b"]["decoding"] and "not guaranteed" in by["groq-gpt-oss-120b"]["decoding"]
    assert by["scratch-gpt"]["max_prompt"] == "200"


def test_missing_ollama_show_is_not_recorded(frozen, tmp_path):
    os.remove(os.path.join(build_module.source_dir(str(frozen), "phase4-ollama-qwen2.5-3b"), "ollama_show.txt"))
    rows = sections.confound_rows(str(frozen), _frozen_rows(frozen), repo_root=str(fake_repo(tmp_path)))
    row = {r["model"]: r for r in rows}["ollama-qwen2.5-3b"]
    assert row["parameters"] == row["precision"] == row["context"] == "not recorded"


def test_hf_local_precision_and_context_follow_the_config(frozen, tmp_path):
    repo = fake_repo(tmp_path)
    config = repo / "config" / "models.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "qwen2.5-0.5b-base: {provider: hf_local,", "qwen2.5-0.5b-base: {provider: hf_local, device: cpu, "
        "max_context_tokens: 2048,"), encoding="utf-8")
    by = {r["model"]: r for r in sections.confound_rows(str(frozen), _frozen_rows(frozen), repo_root=str(repo))}
    assert by["qwen2.5-0.5b-base"]["precision"] == "fp32 on the CPU"
    assert by["qwen2.5-0.5b-base"]["context"] == "2048 tokens"
    assert by["qwen2.5-0.5b-agentlora"]["precision"] == "bf16 on the GPU"


# ── Frozen inputs (WR-04) ──

def test_the_freeze_copies_the_inputs_and_the_render_reads_only_them(tmp_path):
    root = make_results_root(tmp_path)
    repo = fake_repo(tmp_path)
    frozen = root / "phase5-final"
    build_module.build(str(root), str(frozen), now=BEFORE, repo_root=str(repo))
    record = load_json(frozen / "freeze.json")
    assert [i["path"] for i in record["inputs"]] == list(build_module.INPUT_FILES)
    for item in record["inputs"]:
        copy = frozen / "inputs" / item["path"]
        assert copy.read_bytes() == (repo / item["path"]).read_bytes()
    numbers = (frozen / "numbers.json").read_bytes()
    assert b"4.8M" in numbers and b"494.0M + 8.8M LoRA" in numbers
    for item in record["inputs"]:                       # the working tree changes after the freeze
        (repo / item["path"]).write_text("{}", encoding="utf-8")
    build_module.build(str(root), str(frozen), reuse=True, repo_root=str(repo))
    assert (frozen / "numbers.json").read_bytes() == numbers
    (frozen / "inputs" / "models" / "scratch" / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(build_module.DeliverablesError, match="frozen input .* does not match"):
        build_module.build(str(root), str(frozen), reuse=True, repo_root=str(repo))


def test_a_freeze_from_before_wr04_adopts_the_inputs_once(tmp_path):
    root = make_results_root(tmp_path)
    repo = fake_repo(tmp_path)
    frozen = root / "phase5-final"
    build_module.build(str(root), str(frozen), now=BEFORE, repo_root=str(repo))
    record = load_json(frozen / "freeze.json")
    numbers = (frozen / "numbers.json").read_bytes()
    old = dict(record)
    del old["inputs"]
    (frozen / "freeze.json").write_bytes(build_module._json_bytes(old))
    shutil.rmtree(frozen / "inputs")
    lines = build_module.build(str(root), str(frozen), reuse=True, repo_root=str(repo))
    assert lines[1] == "INPUTS adopted 3 from the working tree (the freeze predates WR-04)"
    assert load_json(frozen / "freeze.json") == record and (frozen / "numbers.json").read_bytes() == numbers
    lines = build_module.build(str(root), str(frozen), reuse=True, repo_root=str(repo))
    assert not any(line.startswith("INPUTS") for line in lines)


def test_hf_local_context_mirrors_the_provider():
    with open(os.path.join(REPO_ROOT, "src", "llm", "hf_local.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MAX_CONTEXT_TOKENS" for t in n.targets))
    assert sections.HF_LOCAL_CONTEXT == value


def test_local_training_copies_only_the_named_keys(tmp_path):
    figures = sections.local_training(str(fake_repo(tmp_path)))
    assert figures["scratch-gpt"] == {"params": 4763648, "config_model": {"vocab_size": 96, "context": 2048,
                                                                          "n_layer": 6, "n_head": 8, "d_model": 256},
                                      "steps": 13750, "best_step": 11750, "dev_action_exact": 0.8,
                                      "dev_loss": 0.58, "stop_reason": "early_stop"}
    assert figures["qwen2.5-0.5b-agentlora"] == {"r": 16, "alpha": 32, "trainable_params": 8798208,
                                                 "total_params": 502830976, "steps": 600,
                                                 "dev_action_exact": 0.9396,
                                                 "base_model": "Qwen/Qwen2.5-0.5B-Instruct"}


# ── Legacy rows ──

def test_legacy_summary_is_only_the_legacy_sources(frozen):
    legacy = sections.legacy_summary(str(frozen))
    assert [s["model"] for s in legacy] == ["gemini-flash-lite", "gemma-4-31b"]
    assert all(s["episodes_expected"] == 342 for s in legacy)
    lines = "\n".join(legacy_lines(legacy))
    assert "older engine" in lines and "not compared, pooled or charted" in lines


# ── Threats ──

def _numbers(tmp_path, **kw):
    root = make_results_root(tmp_path, **kw)
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.build(str(root), str(frozen), now=BEFORE)
    return root, frozen, load_json(frozen / "numbers.json")


def test_threat_items_from_the_current_data(tmp_path):
    _, _, numbers = _numbers(tmp_path)
    text = "\n".join(numbers["threats"])
    assert sections.WR01_NONE in text
    assert "groq-gpt-oss-120b" in text and "8/12" in text
    assert "lower bound" in text and "0/" in text
    assert "gemini-flash-lite" in text and "older engine" in text
    assert "InjecAgent" in text and "57 cases" in text
    assert "temperature 0" in text
    assert "Confounds" in text or "confound" in text


def test_threat_items_follow_the_data(tmp_path):
    root = make_results_root(tmp_path)
    rows = model_rows("scratch-gpt", "scratch", attack_success=("dh-00",), overflow=1)
    for r in rows:
        if r["arm"] == "protected" and r["case_id"] == "dh-00":
            r.update(succeeded=True, attack_blocked=False, blocked_by={"sandbox": 1})
        if r["case_id"] == "ds-01" and r["arm"] == "unprotected":
            r.update(actions=[], steps=0)          # a format failure
    write_raw(root / "phase4-scratch-gpt", rows)
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.build(str(root), str(frozen), now=BEFORE)
    numbers = load_json(frozen / "numbers.json")
    text = "\n".join(numbers["threats"])
    assert sections.WR01_NONE not in text
    assert "scratch-gpt 1" in text                         # the overflow count is listed per model
    assert "blocks were recorded" in text                  # the sandbox sentence changed
    scratch = {m["model"]: m for m in numbers["models"]}["scratch-gpt"]
    k = scratch["metrics"]["success_protected"]["k"]
    n = scratch["metrics"]["success_protected"]["n"]
    assert k == 1 and f"{k}/{n}" in text


# ── Report sections ──

def test_report_sections_in_order(tmp_path):
    _, frozen, _ = _numbers(tmp_path)
    text = (frozen / "summary.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    for h in NEW_HEADINGS:
        assert headings.count(h) == 1, h
    order = ["## Headline: complete runs", "## Partial or error-dominated runs (not compared)",
             "## Confounds between the models", "## Legacy rows (older engine, not compared)",
             "## Blocks by layer, per model and pooled (EVAL-08)", "## By attack kind", "## Charts",
             "## Threats to validity", "## Provenance and data quality"]
    positions = [headings.index(h) for h in order]
    assert positions == sorted(positions)
    assert positions[3] == positions[2] + 1 and positions[2] == positions[1] + 1
    assert headings.index("## Threats to validity") + 1 == headings.index("## Provenance and data quality")
    assert "these differences are confounds, not results" in section(text, "Confounds between the models")
    charts = section(text, "Charts")
    for name in ("charts/asr_ci.png", "charts/capability_vs_vulnerability.png", "charts/utility_vs_asr_arrows.png"):
        assert f"]({name})" in charts
    assert "gemini-flash-lite" in section(text, "Legacy rows (older engine, not compared)")


def test_generate_report_without_new_arguments_has_no_new_sections(tmp_path):
    d = write_raw(tmp_path / "plain", model_rows("m1"))
    generate_report(d, charts=False)
    text = (tmp_path / "plain" / "summary.md").read_text(encoding="utf-8")
    assert not any(h in text for h in NEW_HEADINGS)
    assert not (tmp_path / "plain" / "charts").exists()


# ── Charts ──

def test_asr_chart_error_bars_are_the_wilson_bounds_and_partial_rows_are_hollow(tmp_path, monkeypatch):
    """The drawn geometry, not only the labels (WR-06): each error bar runs from pct - (pct - lo) to
    pct + (hi - pct) at its model's x, and a partial model's marker is hollow (face = the surface colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import same_color
    from matplotlib.lines import Line2D

    from src.benchmark.report import SURFACE

    _, _, numbers = _numbers(tmp_path)
    drawn = {}
    real_save = charts_module._save

    def keep(fig, plt_, footer, path, notes=()):
        ax = fig.axes[0]
        drawn["bars"] = [seg for c in ax.containers for col in c.lines[2] for seg in col.get_segments()]
        drawn["points"] = [line for line in ax.lines if line.get_marker() in ("o", "s")]       # not the caps
        real_save(fig, plt_, footer, path, notes)
    monkeypatch.setattr(charts_module, "_save", keep)
    models = [m for m in numbers["models"] if m["group"] in ("local", "hosted")]
    charts_module._asr_ci(plt, Line2D, numbers, models, str(tmp_path / "asr_ci.png"))
    expected = []
    for i, m in enumerate(models):
        for arm, offset in (("unprotected", -0.17), ("protected", 0.17)):
            metric = m["metrics"][f"success_{arm}"]
            if metric["n"]:
                expected.append((i + offset, metric, m["complete"]))
    assert len(drawn["bars"]) == len(expected) == len(drawn["points"])
    for (x, metric, complete), seg, point in zip(expected, drawn["bars"], drawn["points"]):
        (x0, y0), (x1, y1) = seg
        assert x0 == pytest.approx(x) and x1 == pytest.approx(x)
        assert sorted((y0, y1)) == pytest.approx([metric["pct"] - (metric["pct"] - metric["lo"]),
                                                  metric["pct"] + (metric["hi"] - metric["pct"])])
        assert point.get_ydata()[0] == pytest.approx(metric["pct"])
        assert same_color(point.get_markerfacecolor(), SURFACE) == (not complete)
    assert any(not complete for _, _, complete in expected) and any(complete for _, _, complete in expected)


def test_charts_label_every_model_and_never_legacy(tmp_path):
    _, frozen, numbers = _numbers(tmp_path)
    labels = REAL_CHARTS(numbers, str(tmp_path / "charts"))
    assert sorted(labels) == sorted(charts_module.CHART_FILES)
    for name, drawn in labels.items():
        assert (tmp_path / "charts" / name).stat().st_size > 1000
        text = "\n".join(drawn)
        for m in numbers["models"]:
            if m["group"] == "control":
                continue
            assert m["model"] in text, (name, m["model"])
            k, n = m["metrics"]["success_unprotected"]["k"], m["metrics"]["success_unprotected"]["n"]
            assert f"{k}/{n}" in text
            if not m["complete"]:
                assert f"{m['episodes_done']}/{m['episodes_expected']}" in text
        assert "gemini" not in text.lower() and "gemma" not in text.lower()
        assert numbers["snapshot_display"] in text
        control_lines = [line for line in drawn if "control" in line]
        if name == "asr_ci.png":
            assert control_lines and all(line.startswith("scripted control, no guardrails:") for line in control_lines)
        else:
            assert not control_lines


def test_build_writes_the_charts_and_new_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(charts_module, "write_phase5_charts", REAL_CHARTS)
    _, frozen, numbers = _numbers(tmp_path)
    for name in charts_module.CHART_FILES:
        assert (frozen / "charts" / name).exists()
    assert len(numbers["confounds"]) == 8 and len(numbers["legacy"]) == 2 and len(numbers["charts"]) == 3
    assert set(numbers["local_training"]) == {"scratch-gpt", "qwen2.5-0.5b-agentlora"}
    assert numbers["threats"] and all(isinstance(t, str) for t in numbers["threats"])
    assert [c["file"] for c in numbers["charts"]] == [f"charts/{n}" for n in charts_module.CHART_FILES]
    assert not re.search(r"attack_outcomes|benign_utility", " ".join(os.listdir(frozen / "charts")))
