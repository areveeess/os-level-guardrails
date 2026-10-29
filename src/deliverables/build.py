"""
Deliverables build — freeze the Phase 4 data, regenerate the report and numbers.json
====================================================================================
freeze  Copies every source directory into a sibling staging directory, then swaps it
        in place of the frozen directory. Sources are opened only in binary read mode
        and raw.jsonl is copied up to and including its last newline byte, so a line
        the Groq loop is still writing is never copied half-way and nothing is ever
        written into results/phase4-* or results/final-20261011 (T-05-01). On any
        error the staging directory is removed and the old frozen directory is kept.
render  Regenerates summary.md, summary.csv and layer_blocks.csv through
        src/benchmark/report.py from the frozen current-engine copies only (the legacy
        copies never reach the headline, D-02), then writes numbers.json, the one
        source of numbers for every document (D-03). numbers.json is a pure function of
        the frozen directory: nothing in it depends on the clock or git state at render
        time, so re-rendering a frozen directory rewrites it byte-identically.
build   freeze (unless reuse) then render; returns the printed lines.
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter

from src.benchmark.outcomes import classify_llm_error, classify_outcome
from src.benchmark.report import (CONTROL_PROVIDER, LAYERS, POOLED_LABEL, control_summary, generate_report,
                                  kind_breakdown, layer_block_summary, load_rows_counted)
from src.benchmark.stats import prop_fields
from src.deliverables import (CONTROL_MODEL, COPY_FILES, CURRENT_SOURCES, FREEZE_UTC, FROZEN_DIR,
                              HOSTED_MODELS, LEGACY_SOURCES, LOCAL_MODELS, MODEL_ORDER, RESULTS_ROOT)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = 1
FREEZE_FILE = "freeze.json"
NUMBERS_FILE = "numbers.json"
SOURCES_DIR = "sources"
CONTROL_SOURCE = "phase4-control"
KINDS = ("dh", "ds", "benign")
PROP_KEYS = ("k", "n", "pct", "lo", "hi")


class DeliverablesError(Exception):
    """Any failure of the deliverables build; the CLI prints it as one 'FAIL build:' line."""


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


def _under(path: str, root: str) -> bool:
    p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
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


def freeze(results_root: str = RESULTS_ROOT, frozen_dir: str = FROZEN_DIR, now=None) -> dict:
    """Copy the eleven sources into frozen_dir/sources and write freeze.json; returns the freeze record.

    `now` (timezone-aware datetime) is the snapshot clock; the data state is "snapshot" before FREEZE_UTC and
    "final" at or after it."""
    snapshot = _utc(now or datetime.datetime.now(datetime.timezone.utc))
    specs = source_specs(results_root)
    for name, origin, label, _ in specs:
        if _under(frozen_dir, origin) or _under(origin, frozen_dir):
            raise DeliverablesError(f"the frozen directory {posix(frozen_dir)} overlaps the source {label}")
        if not os.path.isfile(os.path.join(origin, "raw.jsonl")):
            raise DeliverablesError(f"source directory {posix(origin)} is missing or has no raw.jsonl")

    frozen_abs = os.path.abspath(frozen_dir)
    parent, base = os.path.dirname(frozen_abs), os.path.basename(frozen_abs)
    staging = os.path.join(parent, f".{base}.staging")
    aside = os.path.join(parent, f".{base}.old")
    os.makedirs(parent, exist_ok=True)
    _remove(staging)
    try:
        sources = []
        for name, origin, label, group in specs:
            target = os.path.join(staging, SOURCES_DIR, name)
            os.makedirs(target)
            copied = []
            for filename in COPY_FILES:
                src = os.path.join(origin, filename)
                if not os.path.isfile(src):
                    continue
                data = _read_bytes(src)
                if filename == "raw.jsonl":
                    data = data[:data.rfind(b"\n") + 1]      # complete lines only
                    raw_lines, raw_sha = data.count(b"\n"), hashlib.sha256(data).hexdigest()
                _write_bytes(os.path.join(target, filename), data)
                copied.append(filename)
            sources.append({"name": name, "origin": label, "group": group, "files": copied,
                            "raw_lines": raw_lines, "raw_sha256": raw_sha})
        freeze_at = parse_utc(FREEZE_UTC)
        record = {
            "schema": SCHEMA,
            "snapshot_utc": utc_seconds(snapshot),
            "snapshot_display": utc_display(snapshot),
            "freeze_utc": utc_seconds(freeze_at),
            "freeze_display": utc_display(freeze_at),
            "data_state": "final" if snapshot >= freeze_at else "snapshot",
            "code_commit": _git_head(),
            "sources": sources,
        }
        _write_atomic(os.path.join(staging, FREEZE_FILE), _json_bytes(record))
        # swap: the old frozen directory is moved aside, the staging directory takes its place
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
    except BaseException:
        _remove(staging)
        raise
    _remove(aside)
    return record


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
    return record


# ── Render ──

def preamble_lines(record: dict) -> list:
    """The data line under the report title (D-01: the snapshot time and the data state on every deliverable)."""
    if record["data_state"] == "snapshot":
        return [f"Data: snapshot of {record['snapshot_display']} (UTC); the D-09 freeze is "
                f"{record['freeze_display']}. Partial rows are labelled with their episodes done and are not final."]
    return [f"Data: final, copied at {record['snapshot_display']} (UTC), after the D-09 freeze of "
            f"{record['freeze_display']}. A row still labelled partial stopped at the freeze or on its quota."]


def load_frozen_rows(paths: list) -> list:
    rows = []
    for path in paths:
        rows += load_rows_counted(os.path.join(path, "raw.jsonl"))[0]
    return rows


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
    """Regenerate the report and numbers.json from the frozen directory; returns numbers."""
    record = read_freeze(frozen_dir)
    paths = [source_dir(frozen_dir, s["name"]) for s in record["sources"] if s["group"] in ("current", "control")]
    rows = load_frozen_rows(paths)
    _validate_models(rows)
    summary = generate_report(frozen_dir, charts=False, sources=paths, preamble=preamble_lines(record))
    numbers = build_numbers(record, rows, summary, frozen_dir)
    _write_atomic(os.path.join(frozen_dir, NUMBERS_FILE), _json_bytes(numbers))
    return numbers


# ── Build ──

def build(results_root: str = RESULTS_ROOT, frozen_dir: str = FROZEN_DIR, reuse: bool = False, now=None,
          say=None) -> list:
    """freeze (unless reuse) and render; returns the printed lines (each also passed to `say`)."""
    lines = []

    def emit(line: str):
        lines.append(line)
        if say:
            say(line)

    if reuse:
        record, word = read_freeze(frozen_dir), "REUSE"
    else:
        record, word = freeze(results_root, frozen_dir, now), "FREEZE"
    emit(f"{word} {record['snapshot_display']} state {record['data_state']} sources {len(record['sources'])}")
    for s in record["sources"]:
        emit(f"SOURCE {s['name']} lines {s['raw_lines']} sha256 {s['raw_sha256'][:12]}")
    numbers = render(frozen_dir)
    models = sum(1 for m in numbers["models"] if m["group"] != "control")
    controls = sum(1 for m in numbers["models"] if m["group"] == "control")
    legacy = sum(1 for s in record["sources"] if s["group"] == "legacy")
    emit(f"REPORT {posix(os.path.join(frozen_dir, 'summary.md'))} models {models} control {controls} legacy {legacy}")
    emit(f"NUMBERS {posix(os.path.join(frozen_dir, NUMBERS_FILE))}")
    emit("BUILD ok")
    return lines
