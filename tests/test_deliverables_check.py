"""
Tests for the generated doc blocks and the D-03 consistency check: block rendering, applying blocks
to CRLF and LF files, --doc scoping, every consistency rule and the key scan of the frozen directory.
Synthetic numbers from a Task 1 build under tmp_path; docs are small files written with explicit bytes.
"""

import ast
import copy
import itertools
import json
import os

import pytest

from src.deliverables import MODEL_ORDER
from src.deliverables import blocks, consistency
from src.deliverables import build as build_module
from src.deliverables.__main__ import main
from tests.test_deliverables_build import (BEFORE, REPO_ROOT, hermetic_repo, load_json,  # noqa: F401
                                           make_results_root, model_rows, write_raw)

DOC_04 = "docs/viva/04-namespaces-seccomp-and-local-inference.md"
DOC_05 = "docs/viva/05-evaluation-statistics-and-examiner-qa.md"
KEY_PREFIX = "gsk" + "_"                      # built from pieces so this file itself never matches the key regex


@pytest.fixture(autouse=True)
def no_chart_drawing(monkeypatch):
    monkeypatch.setattr(build_module.charts, "write_phase5_charts", lambda numbers, chart_dir: {})


def marker_doc(names, eol="\n", before="# Title", after="Tail text."):
    lines = [before, ""]
    for name in names:
        lines += [f"<!-- deliverables:begin {name} -->", f"<!-- deliverables:end {name} -->", ""]
    lines.append(after)
    return (eol.join(lines) + eol).encode("utf-8")


def make_repo(tmp_path, readme=b"# Readme\r\n\r\nNo numbers here.\r\n", sdd=b"# SDD\n\nPlain text.\n",
              page04=b"# Page 04\n\nPlain text.\n", page05=None):
    repo = tmp_path / "repo"
    (repo / "docs" / "viva").mkdir(parents=True)
    (repo / "README.md").write_bytes(readme)
    (repo / "docs" / "SDD.md").write_bytes(sdd)
    (repo / DOC_04).write_bytes(page04)
    if page05 is not None:
        (repo / DOC_05).write_bytes(page05)
    return repo


@pytest.fixture
def frozen_numbers(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-final"
    build_module.build(str(root), str(frozen), now=BEFORE)
    return frozen, load_json(frozen / "numbers.json")


def _check(repo, frozen, docs):
    return consistency.run_check(str(repo), str(frozen), docs)


# ── Rendering ──

def test_every_block_renders_deterministically_from_numbers_only(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    doc = str(tmp_path / "repo" / "docs" / "SDD.md")
    for name in blocks.BLOCK_NAMES:
        first = blocks.render_block(name, numbers, doc, str(frozen))
        assert first == blocks.render_block(name, copy.deepcopy(numbers), doc, str(frozen))
        assert first and all(isinstance(line, str) and "\n" not in line for line in first)


def test_status_block_rows_have_the_contract_form(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    lines = blocks.render_block("status", numbers, str(tmp_path / "README.md"), str(frozen))
    assert numbers["snapshot_display"] in lines[0]
    for m in numbers["models"]:
        row = f"| {m['model']} | {m['group']} | {m['episodes_done']}/{m['episodes_expected']} | {m['label']} |"
        assert row in lines


def test_headline_block_is_fixed_order_with_partial_labels_and_the_control_last(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    lines = blocks.render_block("headline", numbers, str(tmp_path / "README.md"), str(frozen))
    rows = [line for line in lines if line.startswith("| ") and not line.startswith("| Model")]
    assert [r.split(" | ")[0][2:].split(" (")[0] for r in rows] == list(MODEL_ORDER) + ["control-always-comply"]
    assert "(partial 8/12)" in rows[5] and "scripted control" in rows[-1]
    assert not any("rank" in line.lower() for line in lines if line.startswith("| Model"))


def test_findings_are_plain_bullets_that_follow_the_data(tmp_path):
    root = make_results_root(tmp_path)
    frozen = tmp_path / "results" / "phase5-frozen"
    build_module.build(str(root), str(frozen), now=BEFORE)
    numbers = load_json(frozen / "numbers.json")
    lines = blocks.render_block("findings", numbers, str(tmp_path / "x.md"), str(frozen))
    assert all(line.startswith("- ") and not line.startswith("- **") for line in lines if line)
    zero_text = "\n".join(lines)
    assert "no current-engine model succeeded" in zero_text
    rows = model_rows("scratch-gpt", "scratch", attack_success=("dh-00",))
    for r in rows:
        if r["arm"] == "protected" and r["case_id"] == "dh-00":
            r.update(succeeded=True, attack_blocked=False)
    write_raw(root / "phase4-scratch-gpt", rows)
    build_module.build(str(root), str(frozen), now=BEFORE)
    text = "\n".join(blocks.render_block("findings", load_json(frozen / "numbers.json"), str(tmp_path / "x.md"),
                                         str(frozen)))
    assert "no current-engine model succeeded" not in text and "scratch-gpt 1/" in text


def _claims(numbers, tmp_path):
    return {name: "\n".join(blocks.render_block(name, numbers, str(tmp_path / "x.md")))
            for name in blocks.CLAIM_NAMES}


def test_claim_blocks_state_the_clean_result(frozen_numbers, tmp_path):
    _, numbers = frozen_numbers
    claims = _claims(numbers, tmp_path)
    assert all(len(blocks.render_block(n, numbers, "x.md")) == 1 for n in blocks.CLAIM_NAMES)
    assert claims["claim-protected"].startswith("With guardrails, no attack succeeded for any of the 8 language")
    assert "scripted control" in claims["claim-protected"] and "no successful attack" in claims["claim-protected"]
    pooled = numbers["pooled"]["blocked_given_attempt"]
    assert claims["claim-blocked"].startswith("Every attack a language model really attempted")
    assert f"{pooled['k']}/{pooled['n']}" in claims["claim-blocked"]
    assert claims["claim-kernel"].startswith("The sandbox and the seccomp filter recorded no block")
    assert "blocked every attempted attack" in claims["claim-kernel"] and "lower bound" in claims["claim-kernel"]
    assert claims["claim-overflow"].startswith("No current-engine episode had a context overflow")
    state = claims["claim-state"]
    assert state.startswith(f"The data is a snapshot of {numbers['snapshot_display']}") and "before the" in state
    assert "groq-gpt-oss-20b (partial 8/12)" in state and "not final" in state


def test_claim_blocks_follow_the_data(tmp_path):
    root = make_results_root(tmp_path)
    rows = model_rows("scratch-gpt", "scratch", attack_success=("dh-00",), overflow=1)
    for r in rows:
        if r["arm"] == "protected" and r["case_id"] == "dh-00":
            r.update(succeeded=True, attack_blocked=False, blocked_by={"sandbox": 1})
    write_raw(root / "phase4-scratch-gpt", rows)
    frozen = root / "phase5-final"
    build_module.build(str(root), str(frozen), now=BEFORE)
    numbers = load_json(frozen / "numbers.json")
    claims = _claims(numbers, tmp_path)
    assert claims["claim-protected"].startswith("With guardrails, attacks still succeeded for 1 of the 8")
    assert "scratch-gpt 1/" in claims["claim-protected"] and "no attack succeeded" not in claims["claim-protected"]
    assert claims["claim-blocked"].startswith("Not every attempted attack was blocked")
    assert "scratch-gpt 0/1 blocked" in claims["claim-blocked"]
    assert claims["claim-kernel"].startswith("Kernel-layer blocks were recorded: the sandbox in 1/")
    assert claims["claim-overflow"].startswith("Context overflows occurred") and "scratch-gpt 1" in claims[
        "claim-overflow"]
    final = dict(numbers, data_state="final", snapshot_display="2026-10-29T15:40Z")
    state = "\n".join(blocks.render_block("claim-state", final, "x.md"))
    assert state.startswith("The data is final") and "stopped at the freeze or on their quota" in state
    late = dict(numbers, snapshot_display="2026-10-29T15:40Z")
    assert "after the D-09 freeze" in "\n".join(blocks.render_block("claim-state", late, "x.md"))


def test_claim_blocks_may_repeat_but_other_blocks_may_not(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    path = tmp_path / "doc.md"
    path.write_bytes(marker_doc(["claim-protected", "status", "claim-protected"]))
    assert blocks.apply_blocks(str(path), numbers, str(frozen)) == (3, True)
    assert path.read_text(encoding="utf-8").count("With guardrails, no attack succeeded") == 2
    with pytest.raises(blocks.BlockError, match="duplicate"):
        blocks.find_blocks(marker_doc(["headline", "headline"]).decode(), "x.md")


@pytest.mark.parametrize("prose", [
    "With guardrails, no attack succeeded for any model.",
    "The Success (guardrails) column is zero in every row.",
    "The sandbox and seccomp recorded zero blocks across the run.",
    "The tool broker and the policy engine stopped every\nattempted attack first.",      # wrapped across lines
    "No current episode had a context\noverflow, so nothing changes.",
    "The hosted rows were still running when the data was frozen.",
])
def test_an_outcome_stated_outside_a_block_fails(frozen_numbers, tmp_path, prose):
    frozen, numbers = frozen_numbers
    repo = make_repo(tmp_path, page04=f"# Page 04\n\nIntro line.\n{prose}\n".encode())
    lines, code = _check(repo, frozen, [DOC_04])
    fails = [line for line in lines if line.startswith("FAIL check:")]
    assert code == 1 and len(fails) == 1 and fails[0].startswith(f"FAIL check: {DOC_04}:4: outcome stated")
    assert prose.splitlines()[0] not in fails[0]              # the reason never echoes the text


def test_outcome_words_inside_a_claim_block_and_neutral_prose_pass(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    page = marker_doc(["claim-protected", "claim-kernel"], before="# Page 04\n\nThe sentence below says whether "
                      "any attack succeeded with guardrails.", after="A zero there means no attack action reached them.")
    repo = make_repo(tmp_path, page04=page)
    blocks.apply_blocks(str(repo / DOC_04), numbers, str(frozen))
    lines, code = _check(repo, frozen, [DOC_04])
    assert code == 0, lines


def test_only_the_legacy_block_has_legacy_rows(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    for name in blocks.BLOCK_NAMES:
        rows = [line for line in blocks.render_block(name, numbers, str(tmp_path / "a.md"), str(frozen))
                if line.startswith("| ")]
        has_legacy = any("gemini-flash-lite" in r or "gemma-4-31b" in r for r in rows)
        assert has_legacy == (name == "legacy"), name


def test_hosted_block_rows(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    lines = blocks.render_block("hosted", numbers, str(tmp_path / "a.md"), str(frozen))
    row = next(line for line in lines if line.startswith("| groq-gpt-oss-20b |"))
    assert "| 8/12 |" in row and "| 4/120 |" in row and "waiting_quota" in row and row.endswith("| 2 |")
    assert any("hosted_state.json" in line for line in lines)


def test_charts_block_links_are_relative_to_the_doc(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    doc = tmp_path / "repo" / "docs" / "SDD.md"
    lines = blocks.render_block("charts", numbers, str(doc), str(frozen))
    rel = os.path.relpath(str(frozen / "charts" / "asr_ci.png"), str(doc.parent)).replace("\\", "/")
    assert any(f"]({rel})" in line for line in lines)
    assert any("not committed" in line for line in lines)


# ── Applying blocks ──

def test_apply_keeps_crlf_and_every_byte_outside_the_blocks(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    path = tmp_path / "doc.md"
    original = marker_doc(["status", "threats"], eol="\r\n")
    path.write_bytes(original)
    count, changed = blocks.apply_blocks(str(path), numbers, str(frozen))
    assert (count, changed) == (2, True)
    data = path.read_bytes()
    assert b"\n" not in data.replace(b"\r\n", b"")
    assert data.startswith(b"# Title\r\n\r\n<!-- deliverables:begin status -->\r\n")
    assert data.endswith(b"<!-- deliverables:end threats -->\r\n\r\nTail text.\r\n")
    status = "\r\n".join(blocks.render_block("status", numbers, str(path), str(frozen))).encode("utf-8")
    assert status in data


def test_apply_lf_and_second_run_writes_nothing(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    path = tmp_path / "doc.md"
    path.write_bytes(marker_doc(["headline"]))
    assert blocks.apply_blocks(str(path), numbers, str(frozen)) == (1, True)
    data = path.read_bytes()
    assert b"\r" not in data
    os.utime(path, (1_000_000, 1_000_000))
    assert blocks.apply_blocks(str(path), numbers, str(frozen)) == (1, False)
    assert path.read_bytes() == data and os.stat(path).st_mtime == 1_000_000


@pytest.mark.parametrize("text, reason", [
    ("<!-- deliverables:begin nope -->\n<!-- deliverables:end nope -->\n", "unknown"),
    ("<!-- deliverables:begin status -->\n<!-- deliverables:end status -->\n"
     "<!-- deliverables:begin status -->\n<!-- deliverables:end status -->\n", "duplicate"),
    ("<!-- deliverables:begin status -->\n<!-- deliverables:begin headline -->\n"
     "<!-- deliverables:end headline -->\n<!-- deliverables:end status -->\n", "nested"),
    ("x\n<!-- deliverables:begin status -->\n", "unbalanced"),
    ("<!-- deliverables:end status -->\n", "unbalanced"),
])
def test_bad_markers_raise_with_path_and_line(text, reason):
    with pytest.raises(blocks.BlockError) as err:
        blocks.find_blocks(text, "docs/x.md")
    assert str(err.value).startswith("docs/x.md:") and reason in str(err.value)


# ── Check ──

def test_check_clean_docs_and_scoping(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    repo = make_repo(tmp_path, sdd=marker_doc(["status", "headline"]))
    blocks.apply_blocks(str(repo / "docs" / "SDD.md"), numbers, str(frozen))
    lines, code = _check(repo, frozen, ["README.md", "docs/SDD.md"])
    assert code == 0
    assert lines[:2] == ["DOC README.md blocks 0 ok", "DOC docs/SDD.md blocks 2 ok"]
    assert lines[-2].startswith(f"SECRETS {build_module.posix(str(frozen))} files ") and lines[-2].endswith(" ok")
    assert lines[-1] == "deliverables-check-ok docs 2"
    lines, code = _check(repo, frozen, None)
    assert code == 0 and f"SKIP {DOC_05} absent" in lines and lines[-1] == "deliverables-check-ok docs 3"


def test_stale_block_fails(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    repo = make_repo(tmp_path, sdd=marker_doc(["status"]))
    path = repo / "docs" / "SDD.md"
    blocks.apply_blocks(str(path), numbers, str(frozen))
    path.write_bytes(path.read_bytes().replace(b"| complete |", b"| done |", 1))
    lines, code = _check(repo, frozen, ["docs/SDD.md"])
    assert code == 1
    assert any(line.startswith("FAIL check: docs/SDD.md:3: ") and "stale" in line for line in lines)


def _scan(numbers, *texts):
    return consistency.scan_text(list(texts), numbers)


def test_fraction_rules(frozen_numbers):
    _, numbers = frozen_numbers
    by = {m["model"]: m for m in numbers["models"]}
    good = _kn(by["scratch-gpt"]["metrics"]["success_unprotected"])
    assert _scan(numbers, f"scratch-gpt succeeded in {good} episodes.") == []
    assert _scan(numbers, f"ScratchGPT succeeded in {good} episodes.") == []
    problems = _scan(numbers, "SCRATCHGPT succeeded in 999/1000 episodes.")
    assert problems and problems[0][0] == 1 and "999/1000" in problems[0][1]
    hosted = f"{by['groq-gpt-oss-20b']['episodes_done']}/{by['groq-gpt-oss-20b']['episodes_expected']}"
    assert _scan(numbers, f"scratch-gpt {good} and gpt-oss-20b {hosted}") == []
    assert _scan(numbers, f"scratch-gpt alone has {hosted}x") == []          # not a fraction token
    pooled = numbers["pooled"]["blocked_given_attempt"]
    assert _scan(numbers, f"Pooled block rate {pooled['k']}/{pooled['n']}") == []
    assert _scan(numbers, "pooled block rate 3/7")
    assert _scan(numbers, "the Gemini legacy row had 999/342") and _scan(numbers, "Gemma 999/1000")


def _kn(metric):
    return f"{metric['k']}/{metric['n']}"


_DOC_IDS = itertools.count()


def _doc_fails(frozen, tmp_path, body):
    """The FAIL lines of a page-04 document holding `body` (the check of docs, paragraph rules included)."""
    base = tmp_path / f"doc{next(_DOC_IDS)}"
    base.mkdir()
    repo = make_repo(base, page04=("# Page 04\n\n" + body + "\n").encode())
    lines, code = _check(repo, frozen, [DOC_04])
    fails = [line for line in lines if line.startswith("FAIL check:")]
    assert (code == 1) == bool(fails)
    return fails


def test_docs_check_fractions_per_paragraph_and_without_a_model(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    by = {m["model"]: m for m in numbers["models"]}
    good = _kn(by["scratch-gpt"]["metrics"]["success_unprotected"])
    hosted = f"{by['groq-gpt-oss-20b']['episodes_done']}/{by['groq-gpt-oss-20b']['episodes_expected']}"
    assert _doc_fails(frozen, tmp_path, f"The re-check of scratch-gpt found\n{good} episodes.") == []
    fails = _doc_fails(frozen, tmp_path, "The re-check of scratch-gpt found\n999/1000 episodes.")   # wrapped
    assert len(fails) == 1 and fails[0].startswith(f"FAIL check: {DOC_04}:4: fraction 999/1000")
    assert "scratch-gpt" in fails[0]
    fails = _doc_fails(frozen, tmp_path, f"scratch-gpt found {hosted} episodes.")          # another model's value
    assert fails and f"fraction {hosted}" in fails[0]
    fails = _doc_fails(frozen, tmp_path, "Some text with 999/1000 and no model.")             # no model named
    assert fails and "no model is named" in fails[0]
    assert _doc_fails(frozen, tmp_path, f"Some text with {good} and no model.") == []        # a generated value


def test_docs_check_k_of_n_wording(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    m = {m["model"]: m for m in numbers["models"]}["scratch-gpt"]["metrics"]["success_protected"]
    assert _doc_fails(frozen, tmp_path, f"scratch-gpt: {m['k']} of {m['n']} attack episodes.") == []
    assert _doc_fails(frozen, tmp_path, f"scratch-gpt: {m['k']} of its {m['n']} attack episodes.") == []
    fails = _doc_fails(frozen, tmp_path, "scratch-gpt: 111 of 1,000 attack episodes.")
    assert fails and "fraction 111/1000" in fails[0]
    assert _doc_fails(frozen, tmp_path, "Rows 3 of the table and the D-09 freeze of 2026-10-29.") == []


def test_docs_check_bare_percentages(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    hi = numbers["models"][0]["metrics"]["success_protected"]["hi"]
    assert _doc_fails(frozen, tmp_path, f"An upper bound of {hi} % and a 95% interval.") == []
    fails = _doc_fails(frozen, tmp_path, "Attack success fell to 92.6%.")
    assert len(fails) == 1 and "percentage 92.6% is not a generated value" in fails[0]
    assert _doc_fails(frozen, tmp_path, "Attack success fell to 92.6 %.")


def test_timestamp_and_percentage_rules(frozen_numbers):
    _, numbers = frozen_numbers
    assert _scan(numbers, f"Snapshot {numbers['snapshot_display']}, freeze 2026-10-29T15:30:00+00:00") == []
    assert _scan(numbers, "an older snapshot at 2026-10-29T03:06Z")
    assert _scan(numbers, "rate 50.0% (2/4) [15.0, 85.0]") == []
    assert _scan(numbers, "rate 51.0% (2/4) [15.0, 85.0]")
    assert _scan(numbers, "rate 33% (1/3)") == []


def test_key_shaped_text_fails_inside_and_outside_blocks_without_echo(frozen_numbers, tmp_path):
    frozen, numbers = frozen_numbers
    key = KEY_PREFIX + "A" * 24
    repo = make_repo(tmp_path, sdd=marker_doc(["status"]))
    path = repo / "docs" / "SDD.md"
    blocks.apply_blocks(str(path), numbers, str(frozen))
    data = path.read_bytes().replace(b"# Title", f"# Title {key}".encode(), 1)
    path.write_bytes(data)
    lines, code = _check(repo, frozen, ["docs/SDD.md"])
    assert code == 1 and any(line.startswith("FAIL check: docs/SDD.md:1: key-shaped") for line in lines)
    assert key not in "\n".join(lines)


def test_missing_required_doc_fails(frozen_numbers, tmp_path):
    frozen, _ = frozen_numbers
    repo = make_repo(tmp_path)
    os.remove(repo / DOC_04)
    lines, code = _check(repo, frozen, None)
    assert code == 1 and any(line.startswith(f"FAIL check: {DOC_04}:") for line in lines)


def test_frozen_directory_key_scan(frozen_numbers, tmp_path):
    frozen, _ = frozen_numbers
    repo = make_repo(tmp_path)
    (frozen / "notes.txt").write_text(f"line one\nvalue {KEY_PREFIX}{'B' * 30}\n", encoding="utf-8")
    (frozen / "blob.bin").write_bytes(b"\0" + (KEY_PREFIX + "C" * 30).encode())
    lines, code = _check(repo, frozen, ["README.md"])
    posix_frozen = build_module.posix(str(frozen))
    assert code == 1 and f"FAIL check: {posix_frozen}/notes.txt:2: key-shaped text" in lines
    assert not any("blob.bin" in line for line in lines)
    assert KEY_PREFIX + "B" not in "\n".join(lines)


# key shapes built from pieces, so this file itself never matches the key regex
OTHER_KEYS = ("AI" + "za" + "x1Y2z3" * 6 + "abcde",                       # Google: AIza + 35
              "sk" + "-or-v1-" + "0a" * 20,                                  # OpenRouter
              "s" + "k-" + "proj" + "Ab3" * 8,                               # any other sk- key
              "h" + "f_" + "Hf9" * 11)                                       # Hugging Face token


@pytest.mark.parametrize("key", OTHER_KEYS)
def test_the_key_regex_knows_the_project_key_shapes(frozen_numbers, tmp_path, key):
    frozen, numbers = frozen_numbers
    assert consistency.SECRET_RE.search(f"value {key} here")
    assert _scan(numbers, f"a line with {key}") == [(1, "key-shaped text")]
    (frozen / "provider_error.txt").write_text(f"ok\nerror text {key}\n", encoding="utf-8")
    lines, code = _check(make_repo(tmp_path), frozen, ["README.md"])
    assert code == 1 and f"FAIL check: {build_module.posix(str(frozen))}/provider_error.txt:2: key-shaped text" in lines
    assert key not in "\n".join(lines)


@pytest.mark.parametrize("text", ["a risk-assessment-framework-for-agents-in-general", "hf_local_provider_module_name",
                                  "task-oriented-dialogue-systems-are-common", "the gsk_ prefix alone"])
def test_the_key_regex_ignores_ordinary_words(text):
    assert not consistency.SECRET_RE.search(text)


def test_frozen_scan_reads_utf16_and_reports_unreadable_files(frozen_numbers, tmp_path):
    frozen, _ = frozen_numbers
    (frozen / "wide.txt").write_bytes(f"line\nkey {OTHER_KEYS[0]}\n".encode("utf-16"))
    lines, code = _check(make_repo(tmp_path), frozen, ["README.md"])
    posix_frozen = build_module.posix(str(frozen))
    assert code == 1 and f"FAIL check: {posix_frozen}/wide.txt:2: key-shaped text" in lines
    os.remove(frozen / "wide.txt")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        locked = frozen / "locked.txt"
        locked.write_text("text", encoding="utf-8")
        os.chmod(locked, 0)
        try:
            lines, code = _check(make_repo(tmp_path / "again"), frozen, ["README.md"])
        finally:
            os.chmod(locked, 0o644)
        assert code == 1 and any(line.startswith(f"FAIL check: {posix_frozen}/locked.txt:0: unreadable")
                                 for line in lines)


def test_check_refuses_a_numbers_json_that_does_not_belong_to_the_freeze(frozen_numbers, capsys):
    frozen, numbers = frozen_numbers
    original = (frozen / "numbers.json").read_bytes()
    edited = dict(numbers, snapshot_utc="2026-10-29T08:00:00Z")
    (frozen / "numbers.json").write_text(json.dumps(edited), encoding="utf-8")
    assert main(["check", "--frozen", str(frozen)]) == 1
    assert "numbers.json does not belong to freeze.json" in capsys.readouterr().out
    (frozen / "numbers.json").write_bytes(original)
    raw = frozen / "sources" / "phase4-scratch-gpt" / "raw.jsonl"
    raw.write_bytes(raw.read_bytes() + b"{}\n")
    for command in ("check", "blocks"):
        assert main([command, "--frozen", str(frozen)]) == 1
        out = capsys.readouterr().out
        assert out.startswith(f"FAIL {command}: frozen copy ") and "does not match freeze.json" in out


def test_secret_regex_mirrors_run_checks():
    with open(os.path.join(REPO_ROOT, "src", "benchmark", "run_checks.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SECRET_RE" for t in n.targets))
    assert consistency.SECRET_RE.pattern == ast.literal_eval(node.value.args[0])


# ── CLI and build wiring ──

def test_cli_rejects_a_doc_outside_the_targets(frozen_numbers, capsys):
    frozen, _ = frozen_numbers
    for command in ("check", "blocks"):
        assert main([command, "--frozen", str(frozen), "--doc", "notes.md"]) == 1
        assert capsys.readouterr().out.strip().startswith(f"FAIL {command}: notes.md is not")


def test_build_applies_blocks_and_checks_only_the_selected_docs(tmp_path):
    root = make_results_root(tmp_path)
    repo = make_repo(tmp_path, readme=marker_doc(["status"], eol="\r\n"), sdd=marker_doc(["headline"]),
                     page04=b"# Page 04\n\nOld snapshot 2026-10-29T03:06Z.\n")
    frozen = tmp_path / "results" / "phase5-frozen"
    lines = build_module.build(str(root), str(frozen), now=BEFORE, docs=["README.md", "docs/SDD.md"],
                               repo_root=str(repo))
    assert "BLOCKS README.md 1 updated" in lines and "BLOCKS docs/SDD.md 1 updated" in lines
    assert lines[-2] == "deliverables-check-ok docs 2" and lines[-1] == "BUILD ok"
    assert b"\r\n| Model | Group |" in (repo / "README.md").read_bytes()
    lines = build_module.build(str(root), str(frozen), reuse=True, docs=["README.md"], repo_root=str(repo))
    assert "BLOCKS README.md 1 unchanged" in lines and not any("SDD" in line for line in lines)
    with pytest.raises(build_module.DeliverablesError, match="check"):
        build_module.build(str(root), str(frozen), reuse=True, docs=None or list(blocks.DOC_TARGETS),
                           repo_root=str(repo))
    lines = build_module.build(str(root), str(frozen), reuse=True, repo_root=str(repo))
    assert not any(line.startswith(("BLOCKS", "DOC")) for line in lines) and lines[-1] == "BUILD ok"
    assert json.loads((frozen / "numbers.json").read_text(encoding="utf-8"))["schema"] == 1
