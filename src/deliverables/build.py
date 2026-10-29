"""
Deliverables build — freeze the Phase 4 data, regenerate the report and numbers.json
====================================================================================
freeze  Copies every source directory into a sibling staging directory, then swaps it
        in place of the frozen directory. Sources are opened only in binary read mode
        and raw.jsonl is copied up to and including its last newline byte, so a line
        the Groq loop is still writing is never copied half-way and nothing is ever
        written into results/phase4-* or results/final-20261011 (T-05-01). On any
        error the staging directory is removed and the old frozen directory is kept.
        The frozen directory itself must be a phase5-* directory directly under the
        results root (symlinks resolved) that overlaps no source and, if it exists and
        is not empty, holds a freeze.json; anything else is refused before any write,
        because the swap deletes the directory it replaces (CR-02).
render  Regenerates summary.md, summary.csv and layer_blocks.csv through
        src/benchmark/report.py from the frozen current-engine copies only (the legacy
        copies reach only their own labelled table, D-02), with the confounds, legacy,
        charts and threats-to-validity sections (D-04), draws the three D-05 charts and
        writes numbers.json, the one source of numbers for every document (D-03). numbers.json is a pure function of
        the frozen directory: nothing in it depends on the clock or git state at render
        time, so re-rendering a frozen directory rewrites it byte-identically.
build   freeze (or, with reuse, copy the verified frozen sources) into a staging sibling, render
        into it, and only then swap it in place of the frozen directory (WR-03): a failed render
        leaves the previous frozen directory, its numbers.json and charts untouched. Then, for
        the selected target docs, rewrite their
        generated blocks and run the consistency check (D-03); then (when a deck path is given)
        render and check the Review-3 deck (D-08). An interpreter without python-pptx still writes
        everything else and then fails loudly with DeckUnavailable (exit 3 in the CLI), naming the
        existing deck as stale, so an old deck is never left looking current (D-01).
        Returns the printed lines.
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

from src.benchmark.outcomes import classify_llm_error, classify_outcome
from src.benchmark.report import (CONTROL_PROVIDER, DEFAULT_EXPECTED_EPISODES, LAYERS, POOLED_LABEL,
                                  control_summary, generate_report, kind_breakdown, layer_block_summary,
                                  load_rows_counted, read_manifest, summarize)
from src.benchmark.stats import prop_fields
from src.deliverables import charts, sections, slides
from src.deliverables.blocks import BlockError, apply_blocks, data_line
from src.deliverables.consistency import load_numbers, run_check, select_docs
from src.deliverables import (CONTROL_MODEL, COPY_FILES, CURRENT_SOURCES, FREEZE_UTC, FROZEN_DIR, HOSTED_COMPLETE,
                              HOSTED_MODELS, HOSTED_STATE_FILE, HOSTED_TERMINAL, LEGACY_SOURCES, LOCAL_MODELS,
                              MODEL_ORDER, RESULTS_ROOT)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = 1
FREEZE_FILE = "freeze.json"
NUMBERS_FILE = "numbers.json"
SOURCES_DIR = "sources"
FROZEN_PREFIX = "phase5-"
INPUTS_DIR = "inputs"
# The tracked files the render reads besides the run data (WR-04): copied into frozen/inputs/ by the freeze and
# read only from there, so re-rendering a frozen directory never depends on the working tree
INPUT_FILES = ("config/models.yaml", "models/scratch/manifest.json", "models/qwen-agentlora/manifest.json")
CONTROL_SOURCE = "phase4-control"
KINDS = ("dh", "ds", "benign")
PROP_KEYS = ("k", "n", "pct", "lo", "hi")


class DeliverablesError(Exception):
    """Any failure of the deliverables build; the CLI prints it as one 'FAIL build:' line."""


class DeckUnavailable(DeliverablesError):
    """python-pptx is not importable, so the deck step could not run; the message is the 'FAIL deck:' line."""


# ── Time and paths ──

def _utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        raise DeliverablesError("the snapshot clock must be timezone-aware")
    return dt.astimezone(datetime.timezone.utc)


def parse_utc(text: str) -> datetime.datetime:
    return _utc(datetime.datetime.fromisoformat(text.replace("Z", "+00:00")))


def utc_seconds(dt: datetime.datetime) -> str:
    """2026-10-29T07:00:00Z"""
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_display(dt: datetime.datetime) -> str:
    """2026-10-29T07:00Z, the form printed on every deliverable."""
    return _utc(dt).strftime("%Y-%m-%dT%H:%MZ")


def posix(path: str) -> str:
    """A path as printed: forward slashes on every platform."""
    return str(path).replace("\\", "/")


def _real(path: str) -> str:
    """The path with every symlink resolved (IN-02), case-normalised, for comparisons only."""
    return os.path.normcase(os.path.realpath(path))


def _under(path: str, root: str) -> bool:
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def source_dir(frozen_dir: str, name: str) -> str:
    return os.path.join(frozen_dir, SOURCES_DIR, name)


def model_source(model: str) -> str:
    """Frozen source name of a current-engine model or the control."""
    return CONTROL_SOURCE if model == CONTROL_MODEL else f"phase4-{model}"


# ── Freeze ──

def source_specs(results_root: str) -> list:
    """(name, origin path, origin label, group) of the eleven sources, current first, legacy last."""
    label_root = os.path.basename(os.path.normpath(results_root)) or RESULTS_ROOT
    specs = [(name, os.path.join(results_root, name), f"{label_root}/{name}",
              "control" if name == CONTROL_SOURCE else "current") for name in CURRENT_SOURCES]
    specs += [(name, os.path.join(results_root, *origin.split("/")), f"{label_root}/{origin}", "legacy")
              for name, origin in LEGACY_SOURCES]
    return specs


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:          # read-only: the live directories are never opened for writing
        return f.read()


def _write_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _write_atomic(path: str, data: bytes):
    tmp = path + ".tmp"
    _write_bytes(tmp, data)
    os.replace(tmp, path)


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = (out.stdout or "").strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def _remove(path: str):
    if os.path.lexists(path):
        shutil.rmtree(path)


def check_frozen_target(results_root: str, frozen_dir: str):
    """Refuse a frozen directory the build must not replace (CR-02).

    The build moves the old frozen directory aside and deletes it, so the target must be a direct child of the
    results root named phase5-* (after resolving symlinks, so a link cannot point it elsewhere), must not overlap
    any source directory, and, when it already exists and is not empty, must hold a freeze.json. That keeps the
    build away from results/phase4-*, results/final-20261011, docs/ and anything else."""
    real = os.path.realpath(frozen_dir)
    if not os.path.basename(real).startswith(FROZEN_PREFIX) or _real(os.path.dirname(real)) != _real(results_root):
        raise DeliverablesError(f"the frozen directory {posix(frozen_dir)} must be a {FROZEN_PREFIX}* directory "
                                f"directly under the results root {posix(results_root)}; refusing it")
    for _, origin, label, _ in source_specs(results_root):
        if _under(frozen_dir, origin) or _under(origin, frozen_dir):
            raise DeliverablesError(f"the frozen directory {posix(frozen_dir)} overlaps the source {label}")
    if os.path.lexists(frozen_dir):
        if os.path.islink(frozen_dir) or not os.path.isdir(frozen_dir):
            raise DeliverablesError(f"{posix(frozen_dir)} is not a plain directory; refusing to replace it")
        if os.listdir(frozen_dir) and not os.path.isfile(os.path.join(frozen_dir, FREEZE_FILE)):
            raise DeliverablesError(f"{posix(frozen_dir)} exists and is not a frozen directory (no {FREEZE_FILE}); "
                                    f"refusing to replace it")


_ISO_SECONDS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _state_times(obj) -> list:
    """Every ISO UTC timestamp (seconds, Z) in a hosted_state.json, stop reasons included, as datetimes."""
    out = []
    if isinstance(obj, dict):
        for value in obj.values():
            out += _state_times(value)
    elif isinstance(obj, list):
        for value in obj:
            out += _state_times(value)
    elif isinstance(obj, str):
        out += [parse_utc(t) for t in _ISO_SECONDS_RE.findall(obj)]
    return out


def data_state(snapshot: datetime.datetime, freeze_at: datetime.datetime, hosted: dict) -> tuple:
    """("final" or "snapshot", reason or None) of a freeze (CR-03); `hosted` is {model: parsed frozen
    hosted_state.json or None}.

    "final" needs all three: the snapshot clock is at or after the freeze; every hosted model's frozen state is
    terminal (jobs.HOSTED_TERMINAL), so its loop has stopped the model's children and will append no row; and the
    frozen states themselves show the freeze was reached on the loop's own clock (a recorded time at or after it,
    or a model the loop ended as stopped_at_freeze), unless every hosted model completed its run. The build clock
    (Windows) and the loop clock (WSL) can differ, so the wall clock alone never makes the data final. The reason
    says why a post-freeze snapshot is not final; it is None before the freeze."""
    if snapshot < freeze_at:
        return "snapshot", None
    states = {m: (h.get("state") if isinstance(h, dict) else None) for m, h in hosted.items()}
    open_models = [f"{m} {states[m] or 'no hosted_state.json'}" for m in hosted if states[m] not in HOSTED_TERMINAL]
    if open_models:
        return "snapshot", "hosted loop not terminal: " + ", ".join(open_models)
    reached = any(t >= freeze_at for h in hosted.values() for t in _state_times(h))         or any(s == "stopped_at_freeze" for s in states.values())         or all(s in HOSTED_COMPLETE for s in states.values())
    if not reached:
        return "snapshot", "no frozen hosted_state.json shows that the loop reached the freeze"
    return "final", None


def _read_state_bytes(data: bytes):
    try:
        value = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _staging_paths(frozen_dir: str) -> tuple:
    """(frozen, staging, aside) absolute paths: the staging and aside directories are hidden siblings."""
    frozen_abs = os.path.abspath(frozen_dir)
    parent, base = os.path.dirname(frozen_abs), os.path.basename(frozen_abs)
    return frozen_abs, os.path.join(parent, f".{base}.staging"), os.path.join(parent, f".{base}.old")


def _swap_in(staging: str, frozen_dir: str):
    """Put the finished staging directory in place of the frozen directory: the old one is moved aside, the
    staging directory renamed in, and only then the old one deleted (it is moved back if the rename fails)."""
    frozen_abs, _, aside = _staging_paths(frozen_dir)
    _remove(aside)
    moved = False
    if os.path.exists(frozen_abs):
        os.rename(frozen_abs, aside)
        moved = True
    try:
        os.rename(staging, frozen_abs)
    except OSError:
        if moved:
            os.rename(aside, frozen_abs)
        raise
    _remove(aside)


def _copy_inputs(repo_root: str, staging: str) -> list:
    """Copy the present INPUT_FILES from the repository into staging/inputs; returns [{path, sha256}]."""
    out = []
    for rel in INPUT_FILES:
        src = os.path.join(repo_root, *rel.split("/"))
        if not os.path.isfile(src):
            continue
        data = _read_bytes(src)
        target = os.path.join(staging, INPUTS_DIR, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _write_bytes(target, data)
        out.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest()})
    return out


def freeze(results_root: str = RESULTS_ROOT, frozen_dir: str = FROZEN_DIR, now=None, repo_root: str = None) -> dict:
    """Freeze into a staging directory and swap it in (no render); returns the freeze record. build() stages,
    renders into the staging directory and only then swaps (WR-03)."""
    staging, record = stage_freeze(results_root, frozen_dir, now, repo_root=repo_root)
    try:
        _swap_in(staging, frozen_dir)
    except BaseException:
        _remove(staging)
        raise
    return record


def stage_freeze(results_root: str = RESULTS_ROOT, frozen_dir: str = FROZEN_DIR, now=None,
                 repo_root: str = None) -> tuple:
    """Copy the eleven sources into the staging sibling of frozen_dir and write its freeze.json; returns
    (staging path, freeze record). The frozen directory itself is not touched; on any error the staging
    directory is removed.

    `now` (timezone-aware datetime) is the snapshot clock. The data state is "snapshot" before FREEZE_UTC; at or
    after it, it is "final" only when data_state() says the frozen hosted states are terminal and show the freeze
    (CR-03), else "snapshot" with a data_state_reason. Each source's hosted_state.json is copied before its
    raw.jsonl: a terminal state is written only after the loop's child for that model has exited, so a terminal
    state copied first guarantees the raw.jsonl copied after it holds every row. config/models.yaml and the two
    model manifests are copied from repo_root into inputs/ and hashed in freeze.json (WR-04)."""
    repo_root = repo_root or REPO_ROOT
    snapshot = _utc(now or datetime.datetime.now(datetime.timezone.utc))
    check_frozen_target(results_root, frozen_dir)
    specs = source_specs(results_root)
    for name, origin, label, _ in specs:
        if not os.path.isfile(os.path.join(origin, "raw.jsonl")):
            raise DeliverablesError(f"source directory {posix(origin)} is missing or has no raw.jsonl")

    frozen_abs, staging, _ = _staging_paths(frozen_dir)
    os.makedirs(os.path.dirname(frozen_abs), exist_ok=True)
    _remove(staging)
    try:
        sources, hosted_states = [], {}
        for name, origin, label, group in specs:
            target = os.path.join(staging, SOURCES_DIR, name)
            os.makedirs(target)
            copied = []
            for filename in sorted(COPY_FILES, key=lambda f: f != HOSTED_STATE_FILE):    # the state first
                src = os.path.join(origin, filename)
                if not os.path.isfile(src):
                    continue
                data = _read_bytes(src)
                if filename == "raw.jsonl":
                    data = data[:data.rfind(b"\n") + 1]      # complete lines only
                    raw_lines, raw_sha = data.count(b"\n"), hashlib.sha256(data).hexdigest()
                if filename == HOSTED_STATE_FILE:
                    hosted_states[name] = _read_state_bytes(data)
                _write_bytes(os.path.join(target, filename), data)
                copied.append(filename)
            copied.sort(key=COPY_FILES.index)
            sources.append({"name": name, "origin": label, "group": group, "files": copied,
                            "raw_lines": raw_lines, "raw_sha256": raw_sha})
        freeze_at = parse_utc(FREEZE_UTC)
        state, reason = data_state(snapshot, freeze_at,
                                   {m: hosted_states.get(model_source(m)) for m in HOSTED_MODELS})
        record = {
            "schema": SCHEMA,
            "snapshot_utc": utc_seconds(snapshot),
            "snapshot_display": utc_display(snapshot),
            "freeze_utc": utc_seconds(freeze_at),
            "freeze_display": utc_display(freeze_at),
            "data_state": state,
            "code_commit": _git_head(),
            "sources": sources,
            "inputs": _copy_inputs(repo_root, staging),
        }
        if reason:
            record["data_state_reason"] = reason
        _write_atomic(os.path.join(staging, FREEZE_FILE), _json_bytes(record))
    except BaseException:
        _remove(staging)
        raise
    return staging, record


def stage_reuse(frozen_dir: str = FROZEN_DIR, repo_root: str = None) -> tuple:
    """(staging path, freeze record, adopted): the verified frozen sources, inputs and freeze.json copied byte for
    byte into the staging sibling, for a --reuse-frozen render that must not touch the frozen directory until it
    succeeds.

    A freeze written before WR-04 has no inputs: they are adopted once from repo_root into the staging copy and
    added to its freeze.json (adopted=True; the sources and their hashes are unchanged), so every later render
    of that directory reads only the frozen copies."""
    record = read_freeze(frozen_dir)
    _, staging, _ = _staging_paths(frozen_dir)
    _remove(staging)
    adopted = "inputs" not in record
    try:
        os.makedirs(staging)
        shutil.copytree(os.path.join(frozen_dir, SOURCES_DIR), os.path.join(staging, SOURCES_DIR))
        if adopted:
            record["inputs"] = _copy_inputs(repo_root or REPO_ROOT, staging)
            _write_atomic(os.path.join(staging, FREEZE_FILE), _json_bytes(record))
        else:
            if os.path.isdir(os.path.join(frozen_dir, INPUTS_DIR)):
                shutil.copytree(os.path.join(frozen_dir, INPUTS_DIR), os.path.join(staging, INPUTS_DIR))
            shutil.copyfile(os.path.join(frozen_dir, FREEZE_FILE), os.path.join(staging, FREEZE_FILE))
        record = read_freeze(staging)
    except BaseException:
        _remove(staging)
        raise
    return staging, record, adopted


def read_freeze(frozen_dir: str) -> dict:
    path = os.path.join(frozen_dir, FREEZE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError) as e:
        raise DeliverablesError(f"cannot read {posix(path)} ({type(e).__name__}); run build without --reuse-frozen")
    if not isinstance(record, dict) or record.get("schema") != SCHEMA or not isinstance(record.get("sources"), list):
        raise DeliverablesError(f"{posix(path)} is not a schema {SCHEMA} freeze record")
    for s in record["sources"]:
        raw = os.path.join(source_dir(frozen_dir, s["name"]), "raw.jsonl")
        if not os.path.isfile(raw) or hashlib.sha256(_read_bytes(raw)).hexdigest() != s["raw_sha256"]:
            raise DeliverablesError(f"frozen copy {posix(raw)} is missing or does not match freeze.json")
    for item in record.get("inputs") or []:
        path = os.path.join(frozen_dir, INPUTS_DIR, *item["path"].split("/"))
        if not os.path.isfile(path) or hashlib.sha256(_read_bytes(path)).hexdigest() != item["sha256"]:
            raise DeliverablesError(f"frozen input {posix(path)} is missing or does not match freeze.json")
    return record


# ── Render ──

def preamble_lines(record: dict) -> list:
    """The data line under the report title (D-01: the snapshot time and the data state on every deliverable);
    the same sentence opens the generated status block."""
    return [data_line(record)]


def load_frozen_rows(paths: list) -> list:
    rows = []
    for path in paths:
        rows += load_rows_counted(os.path.join(path, "raw.jsonl"))[0]
    return rows


def _manifest_expected(paths: list, rows: list) -> dict:
    """{model: planned episodes} by generate_report's rule: the largest manifest value over the directories the
    model's rows came from (a model without one gets the report default)."""
    expected = {}
    for path in paths:
        manifest = read_manifest(path) or {}
        planned = manifest.get("expected_episodes_per_model")
        if type(planned) is not int or planned <= 0:
            continue
        for model in dict.fromkeys(r["model"] for r in load_rows_counted(os.path.join(path, "raw.jsonl"))[0]):
            expected[model] = max(expected.get(model, 0), planned)
    return expected or {m: DEFAULT_EXPECTED_EPISODES for m in dict.fromkeys(r["model"] for r in rows)}


def _validate_models(rows: list):
    models = set(r["model"] for r in rows)
    known = set(MODEL_ORDER) | {CONTROL_MODEL}
    unknown = sorted(models - known)
    if unknown:
        raise DeliverablesError(f"frozen rows hold models outside MODEL_ORDER and the control: {', '.join(unknown)}")
    missing = [m for m in MODEL_ORDER + (CONTROL_MODEL,) if m not in models]
    if missing:
        raise DeliverablesError(f"no frozen rows for: {', '.join(missing)}")
    for r in rows:
        is_control = (r.get("provider") == CONTROL_PROVIDER)
        if is_control != (r["model"] == CONTROL_MODEL):
            raise DeliverablesError(f"model {r['model']} has provider {r.get('provider')!r}; only the control "
                                    f"may be {CONTROL_PROVIDER!r}")


def model_label(s: dict) -> str:
    """'complete', or 'partial, D/E episodes' (with '(error-dominated)' for that status)."""
    if s["status"] == "complete":
        return "complete"
    label = f"partial, {s['episodes_done']}/{s['episodes_expected']} episodes"
    return label + " (error-dominated)" if s["status"] == "error-dominated" else label


def _prop(k: int, n: int) -> dict:
    f = prop_fields("p", k, n)
    return {key: f[f"p_{key}"] for key in PROP_KEYS}


def _metrics(s: dict) -> dict:
    """Every PREFIX_k of a summary row that has a PREFIX_n, as {k, n, pct, lo, hi}."""
    out = {}
    for key in s:
        if key.endswith("_k") and f"{key[:-2]}_n" in s:
            prefix = key[:-2]
            out[prefix] = {part: s.get(f"{prefix}_{part}") for part in PROP_KEYS}
    return out


def _layers(entries: list) -> dict:
    """{group: {"n": n, layer: {k, n, pct, lo, hi, actions}}} from layer_block_summary entries."""
    out = {}
    for e in entries:
        item = {"n": e["n"]}
        for layer in LAYERS:
            item[layer] = dict(_prop(e["layers"][layer]["episodes"], e["n"]), actions=e["layers"][layer]["actions"])
        out[e["group"]] = item
    return out


def _hosted_state(frozen_dir: str, model: str) -> dict:
    path = os.path.join(source_dir(frozen_dir, model_source(model)), "hosted_state.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    launches = [x for x in data.get("launches") or [] if isinstance(x, dict)]
    return {"state": data.get("state") or "pending", "launches": len(launches),
            "quota_stops": sum(1 for x in launches if x.get("quota_stop"))}


def model_group(model: str) -> str:
    if model == CONTROL_MODEL:
        return "control"
    return "local" if model in LOCAL_MODELS else "hosted"


def build_numbers(record: dict, rows: list, summary: list, frozen_dir: str) -> dict:
    """numbers.json from the frozen rows and the report's own summary rows (D-03)."""
    by_model = {s["model"]: s for s in summary}
    llm_rows = [r for r in rows if r["model"] != CONTROL_MODEL]
    layer_entries = layer_block_summary(llm_rows, {m: s["status"] for m, s in by_model.items()})
    breakdown = kind_breakdown(rows)
    models = []
    for model in MODEL_ORDER + (CONTROL_MODEL,):
        s = by_model[model]
        model_rows = [r for r in rows if r["model"] == model]
        finished = Counter(r.get("kind") for r in model_rows if classify_outcome(r) != "llm_error")
        entry = {
            "model": model,
            "group": model_group(model),
            "family": s["family"],
            "provider": s["provider"],
            "status": s["status"],
            "complete": s["status"] == "complete",
            "label": model_label(s),
            "episodes_done": s["episodes_done"],
            "episodes_expected": s["episodes_expected"],
            "pairs_complete": s["pairs_complete"],
            "infra_errors": s["infra_errors"],
            "attack_pairs": s["attack_pairs"],
            "format_failures": {arm: s[f"outcomes_{arm}_format_failure"] for arm in ("unprotected", "protected")},
            "metrics": _metrics(s),
            "by_kind": {b["kind"]: {"episodes": b["episodes"], **{p: v for p, v in _metrics(b).items()}}
                        for b in breakdown if b["model"] == model},
            "finished_by_kind": {kind: finished.get(kind, 0) for kind in KINDS},
            "context_overflow_episodes": sum(1 for r in model_rows
                                             if classify_llm_error(r.get("llm_error")) == "context_overflow"),
            "layers": _layers([e for e in layer_entries if e["model"] == model]) if model != CONTROL_MODEL else None,
        }
        if model in HOSTED_MODELS:
            entry["hosted_state"] = _hosted_state(frozen_dir, model)
        models.append(entry)
    llm = [by_model[m] for m in MODEL_ORDER]
    pooled = {
        "label": POOLED_LABEL,
        "blocked_given_attempt": _prop(sum(s["blocked_given_attempt_k"] for s in llm),
                                       sum(s["blocked_given_attempt_n"] for s in llm)),
        "layers": _layers([e for e in layer_entries if e["status"] == "pooled"]),
    }
    return {
        "schema": SCHEMA,
        "snapshot_utc": record["snapshot_utc"],
        "snapshot_display": record["snapshot_display"],
        "freeze_utc": record["freeze_utc"],
        "freeze_display": record["freeze_display"],
        "data_state": record["data_state"],
        "sources": [{key: s[key] for key in ("name", "origin", "group", "raw_lines", "raw_sha256")}
                    for s in record["sources"]],
        "models": models,
        "pooled": pooled,
        "control": control_summary(rows, [CONTROL_MODEL]),
    }


def render(frozen_dir: str = FROZEN_DIR) -> dict:
    """Regenerate the report, the three charts and numbers.json from the frozen directory; returns numbers.

    The numbers are computed first (the threats section quotes them), then the report is written through
    generate_report; its own summary must equal the one the numbers came from, or the build fails."""
    record = read_freeze(frozen_dir)
    paths = [source_dir(frozen_dir, s["name"]) for s in record["sources"] if s["group"] in ("current", "control")]
    rows = load_frozen_rows(paths)
    _validate_models(rows)
    summary = summarize(rows, expected=_manifest_expected(paths, rows))
    numbers = build_numbers(record, rows, summary, frozen_dir)
    llm_rows = [r for r in rows if r["model"] != CONTROL_MODEL]
    legacy = sections.legacy_summary(frozen_dir) if any(s["group"] == "legacy" for s in record["sources"]) else []
    inputs = os.path.join(frozen_dir, INPUTS_DIR)          # the frozen inputs only, never the working tree (WR-04)
    numbers["confounds"] = sections.confound_rows(frozen_dir, llm_rows, repo_root=inputs)
    numbers["legacy"] = legacy
    numbers["local_training"] = sections.local_training(inputs)
    numbers["threats"] = sections.threat_items(numbers, legacy)
    links = charts.chart_links()
    numbers["charts"] = [{"file": path, "caption": caption} for path, caption in links]
    report_summary = generate_report(frozen_dir, charts=False, sources=paths, preamble=preamble_lines(record),
                                     confounds=numbers["confounds"], legacy=legacy, chart_links=links,
                                     threats=numbers["threats"])
    if report_summary != summary:
        raise DeliverablesError("the report's summary differs from the summary numbers.json was built from")
    charts.write_phase5_charts(numbers, os.path.join(frozen_dir, "charts"))
    _write_atomic(os.path.join(frozen_dir, NUMBERS_FILE), _json_bytes(numbers))
    return numbers


# ── Build ──

def apply_docs(repo_root: str, frozen_dir: str, docs, numbers: dict) -> list:
    """Rewrite the generated blocks of the selected present target docs; returns their BLOCKS lines."""
    lines = []
    try:
        selected = select_docs(docs)
    except ValueError as e:
        raise DeliverablesError(str(e))
    for doc in selected:
        path = os.path.join(repo_root, *doc.split("/"))
        if not os.path.isfile(path):
            continue
        try:
            count, changed = apply_blocks(path, numbers, frozen_dir, display=doc)
        except BlockError as e:
            raise DeliverablesError(str(e))
        lines.append(f"BLOCKS {doc} {count} {'updated' if changed else 'unchanged'}")
    return lines


def _display(path: str, repo_root: str) -> str:
    """A path as printed: relative to the repository root when inside it, forward slashes."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(repo_root))
    return posix(path if rel.startswith("..") or os.path.isabs(rel) else rel)


def deck_step(numbers: dict, frozen_dir: str, deck_out: str, template: str, repo_root: str,
              allow_blank_template: bool = False) -> list:
    """Render and check the Review-3 deck; returns the TEMPLATE and DECK lines. A deck drawn on the blank fallback
    because the template could not be opened (WR-05), unless allow_blank_template, or a check problem raises
    DeliverablesError with the TEMPLATE and FAIL deck lines in the error's `lines`; a missing python-pptx raises
    DeckUnavailable."""
    display = _display(deck_out, repo_root)
    try:
        how = slides.render_deck(slides.deck_spec(numbers), deck_out, frozen_dir, template)
        template_line = f"TEMPLATE {how}"
        if slides.is_blank_fallback(how) and not allow_blank_template:
            error = DeliverablesError("the deck was drawn without the Review template (see the FAIL deck line)")
            error.lines = [template_line, f"FAIL deck: {slides.blank_fallback_problem(display)}"]
            raise error
        problems, counts = slides.check_deck(deck_out, numbers, frozen_dir)
    except slides.PptxUnavailable:
        line = slides.missing_line(sys.executable)
        if os.path.exists(deck_out):
            line += f"; {display} was not rebuilt and is stale"
        raise DeckUnavailable(line)
    if problems:
        error = DeliverablesError("the deck check failed (see the FAIL deck lines)")
        error.lines = [template_line] + [f"FAIL deck: {p}" for p in problems]
        raise error
    return [template_line, slides.deck_line(display, counts)]


def build(results_root: str = RESULTS_ROOT, frozen_dir: str = FROZEN_DIR, reuse: bool = False, now=None,
          say=None, repo_root: str = REPO_ROOT, docs=None, deck_out: str = None, no_deck: bool = False,
          template: str = None, allow_blank_template: bool = False) -> list:
    """freeze (unless reuse) and render, then (when `docs` is a list of target docs) apply their blocks and run
    the check on them, then (when `deck_out` is given and not no_deck) render and check the deck; returns the
    printed lines (each also passed to `say`). docs=None skips the doc step and deck_out=None the deck step;
    no_deck prints 'DECK skipped (--no-deck)'."""
    lines = []

    def emit(line: str):
        lines.append(line)
        if say:
            say(line)

    check_frozen_target(results_root, frozen_dir)
    adopted = False
    if reuse:
        staging, record, adopted = stage_reuse(frozen_dir, repo_root)
        word = "REUSE"
    else:
        (staging, record), word = stage_freeze(results_root, frozen_dir, now, repo_root=repo_root), "FREEZE"
    try:
        emit(f"{word} {record['snapshot_display']} state {record['data_state']} sources {len(record['sources'])}")
        if adopted:
            emit(f"INPUTS adopted {len(record['inputs'])} from the working tree (the freeze predates WR-04)")
        if record.get("data_state_reason"):
            emit(f"NOT FINAL {record['data_state_reason']}")
        for s in record["sources"]:
            emit(f"SOURCE {s['name']} lines {s['raw_lines']} sha256 {s['raw_sha256'][:12]}")
        render(staging)                            # every output is written into the staging directory (WR-03)
        _swap_in(staging, frozen_dir)              # the previous frozen directory survives until this point
    except BaseException:
        _remove(staging)
        raise
    numbers = load_numbers(frozen_dir)
    models = sum(1 for m in numbers["models"] if m["group"] != "control")
    controls = sum(1 for m in numbers["models"] if m["group"] == "control")
    emit(f"REPORT {posix(os.path.join(frozen_dir, 'summary.md'))} models {models} control {controls} "
         f"legacy {len(numbers['legacy'])}")
    emit(f"CHARTS {len(numbers['charts'])}")
    emit(f"NUMBERS {posix(os.path.join(frozen_dir, NUMBERS_FILE))}")
    if docs is not None:
        numbers = load_numbers(frozen_dir)          # exactly what blocks and check read later
        for line in apply_docs(repo_root, frozen_dir, docs, numbers):
            emit(line)
        check_lines, code = run_check(repo_root, frozen_dir, docs, numbers=numbers)
        for line in check_lines:
            emit(line)
        if code:
            raise DeliverablesError("the deliverables check failed (see the FAIL check lines)")
    if no_deck:
        emit("DECK skipped (--no-deck)")
    elif deck_out is not None:
        numbers = load_numbers(frozen_dir)
        template = template or os.path.join(repo_root, slides.DEFAULT_TEMPLATE)
        try:
            deck_lines = deck_step(numbers, frozen_dir, deck_out, template, repo_root,
                                   allow_blank_template=allow_blank_template)
        except DeckUnavailable:
            raise
        except DeliverablesError as e:
            for line in getattr(e, "lines", []):
                emit(line)
            raise
        for line in deck_lines:
            emit(line)
    emit("BUILD ok")
    return lines
