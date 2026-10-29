"""
Tests for the Review-3 slide deck (DOC-05, D-08): the pure deck spec built from numbers.json, its numbers and
legacy rules, and (only where python-pptx is importable) the render on the template and the deck check.
Synthetic numbers from a deliverables build under tmp_path, as in tests/test_deliverables_check.py. The WSL venv
has no python-pptx, so the render and check tests are skipped there and run on Windows through the CLI.
"""

import os
import subprocess
import sys

import pytest

from src.deliverables import consistency, slides
from src.deliverables import build as build_module
from tests.test_deliverables_build import BEFORE, REPO_ROOT, load_json, make_results_root

LEGACY_IDS = ("gemini-flash-lite", "gemma-4-31b")


@pytest.fixture(autouse=True)
def no_chart_drawing(monkeypatch):
    monkeypatch.setattr(build_module.charts, "write_phase5_charts", lambda numbers, chart_dir: {})


@pytest.fixture
def frozen_numbers(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "phase5-final"
    build_module.build(str(root), str(frozen), now=BEFORE)
    return frozen, load_json(frozen / "numbers.json")


def spec_lines(slide: dict) -> list:
    return slides.slide_text(slide)


# ── The pure spec ──

def test_title_slide_holds_the_project_review_author_and_data_line(frozen_numbers):
    _, numbers = frozen_numbers
    spec = slides.deck_spec(numbers)
    first = spec[0]
    assert first["key"] == "title" and first["kind"] == "title"
    assert first["title"] == slides.SLIDE_TITLES[0]
    text = "\n".join(spec_lines(first))
    assert "OS-Level Guardrails for Autonomous AI Agents" in text and "Review 3" in text
    assert "Raj Vardhan Singh" in text and "BCSE303P" in text
    assert numbers["snapshot_display"] in text and numbers["freeze_display"] in text
    assert "snapshot" in text
    assert numbers["snapshot_display"] in first["footer"]


def test_every_slide_has_the_contract_keys_and_notes(frozen_numbers):
    _, numbers = frozen_numbers
    spec = slides.deck_spec(numbers)
    assert [s["title"] for s in spec] == list(slides.SLIDE_TITLES)
    for s in spec:
        assert set(s) >= {"key", "title", "kind", "body", "notes", "footer"}
        assert len(s["notes"]) >= 200, s["key"]


def test_spec_text_passes_the_consistency_rules_and_has_no_legacy_id(frozen_numbers):
    _, numbers = frozen_numbers
    for s in slides.deck_spec(numbers):
        lines = spec_lines(s) + s["notes"].splitlines()
        assert consistency.scan_text(lines, numbers) == [], s["key"]
        assert not any(legacy in line.lower() for line in lines for legacy in LEGACY_IDS), s["key"]
        assert consistency.scan_text([s["footer"]], numbers) == []


def test_importing_slides_does_not_import_pptx():
    code = "import sys, src.deliverables.slides, src.deliverables.__main__; print('pptx' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_deck_command_without_pptx_exits_3(frozen_numbers, monkeypatch, capsys):
    from src.deliverables.__main__ import main

    frozen, _ = frozen_numbers

    def missing():
        raise slides.PptxUnavailable("python-pptx is not importable")
    monkeypatch.setattr(slides, "_import_pptx", missing)
    code = main(["deck", "--frozen", str(frozen), "--out", str(frozen / "deck.pptx")])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 3
    assert out[-1].startswith("FAIL deck: python-pptx is not importable by ")


# ── Render and check (python-pptx only) ──

def test_render_on_the_template_then_check(frozen_numbers, tmp_path):
    pytest.importorskip("pptx")
    from pptx import Presentation

    frozen, numbers = frozen_numbers
    spec = slides.deck_spec(numbers)
    out = tmp_path / "deck.pptx"
    slides.render_deck(spec, str(out), str(frozen), os.path.join(REPO_ROOT, "docs", "template.pptx"))
    prs = Presentation(str(out))
    assert len(prs.slides) == len(spec)
    for slide in prs.slides:
        assert slide.has_notes_slide and len(slide.notes_slide.notes_text_frame.text) >= 200
    problems, counts = slides.check_deck(str(out), numbers, str(frozen))
    assert problems == []
    assert counts["slides"] == len(spec) and counts["notes"] == len(spec)
