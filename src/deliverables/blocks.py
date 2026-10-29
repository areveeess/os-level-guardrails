"""
Deliverables blocks — generated result markdown between marker lines in the tracked docs
=======================================================================================
A document quotes results only through generated blocks:

    <!-- deliverables:begin NAME -->
    ...generated from numbers.json, never edited by hand...
    <!-- deliverables:end NAME -->

The markers are HTML comments on their own lines, so they are invisible when the markdown is
rendered. `blocks` (and `build`) rewrite the lines between each pair from numbers.json; the
consistency check re-renders every block and fails when one is stale (D-03). Each block is
rendered from numbers.json alone, through the same table renderers the report uses, so the
report and the docs cannot disagree. The headline is in a fixed order and never ranks, and every
row that is not complete carries its episodes done (D-02).

apply_blocks keeps each file's own line endings (README.md is CRLF in the working tree, the
other targets are LF) and every byte outside the blocks, and writes only when a block changed.
"""

import os
import re

from src.benchmark.report import (LAYER_EVIDENCE, LAYERS, LAYER_TITLES, chart_lines, confounds_lines, legacy_lines,
                                  threats_lines)
from src.benchmark.stats import format_prop
from src.deliverables import DS_EXPECTED, FROZEN_DIR, HOSTED_MODELS, MODEL_ORDER

BLOCK_NAMES = ("status", "headline", "findings", "control", "layers", "hosted", "confounds", "legacy", "threats",
               "charts")
DOC_TARGETS = ("README.md", "docs/SDD.md", "docs/viva/04-namespaces-seccomp-and-local-inference.md",
               "docs/viva/05-evaluation-statistics-and-examiner-qa.md")
REQUIRED_DOCS = DOC_TARGETS[:3]           # page 05 is optional until plan 05-03 creates it

BEGIN_RE = re.compile(r"^<!-- deliverables:begin (\S+) -->$")
END_RE = re.compile(r"^<!-- deliverables:end (\S+) -->$")
ANY_MARKER_RE = re.compile(r"<!--\s*deliverables:")


class BlockError(ValueError):
    """A malformed, unknown, duplicate, nested or unbalanced marker; the message starts with path:line."""


def begin_marker(name: str) -> str:
    return f"<!-- deliverables:begin {name} -->"


def end_marker(name: str) -> str:
    return f"<!-- deliverables:end {name} -->"


def find_blocks(text: str, path: str) -> list:
    """[{name, begin, end}] with 0-based line indexes of the marker lines, in document order."""
    found, open_block, seen = [], None, set()
    for index, raw in enumerate(text.splitlines()):
        line = raw.rstrip("\r")
        lineno = index + 1
        begin, end = BEGIN_RE.match(line), END_RE.match(line)
        if not begin and not end:
            if ANY_MARKER_RE.search(line):
                raise BlockError(f"{path}:{lineno}: malformed deliverables marker (it must be alone on its line)")
            continue
        name = (begin or end).group(1)
        if name not in BLOCK_NAMES:
            raise BlockError(f"{path}:{lineno}: unknown block name {name!r}")
        if begin:
            if open_block is not None:
                raise BlockError(f"{path}:{lineno}: nested block {name!r} inside {open_block['name']!r}")
            if name in seen:
                raise BlockError(f"{path}:{lineno}: duplicate block {name!r}")
            seen.add(name)
            open_block = {"name": name, "begin": index}
        else:
            if open_block is None or open_block["name"] != name:
                raise BlockError(f"{path}:{lineno}: unbalanced end marker for {name!r}")
            open_block["end"] = index
            found.append(open_block)
            open_block = None
    if open_block is not None:
        raise BlockError(f"{path}:{open_block['begin'] + 1}: unbalanced block {open_block['name']!r} has no end marker")
    return found


# ── Rendering from numbers.json ──

def data_line(numbers: dict) -> str:
    """The snapshot time and data state (D-01), the same sentence as the report's data line."""
    if numbers["data_state"] == "snapshot":
        return (f"Data: snapshot of {numbers['snapshot_display']} (UTC); the D-09 freeze is "
                f"{numbers['freeze_display']}. Partial rows are labelled with their episodes done and are not final.")
    return (f"Data: final, copied at {numbers['snapshot_display']} (UTC), after the D-09 freeze of "
            f"{numbers['freeze_display']}. A row still labelled partial stopped at the freeze or on its quota.")


def _body(lines: list) -> list:
    """A report section without its leading blank line, heading and the blank line after it."""
    out = list(lines)
    while out and out[0] == "":
        out.pop(0)
    if out and out[0].startswith("## "):
        out.pop(0)
    while out and out[0] == "":
        out.pop(0)
    return out


def _models(numbers: dict) -> list:
    return [m for m in numbers["models"] if m["model"] in MODEL_ORDER]


def _control(numbers: dict):
    return next((m for m in numbers["models"] if m["group"] == "control"), None)


def _name(m: dict) -> str:
    return m["model"] if m["complete"] else f"{m['model']} (partial {m['episodes_done']}/{m['episodes_expected']})"


def _fp(metric: dict) -> str:
    return format_prop(metric["k"], metric["n"])


def _kn(metric: dict) -> str:
    return f"{metric['k']}/{metric['n']}"


def _status(numbers: dict) -> list:
    lines = [data_line(numbers), "", "| Model | Group | Episodes done | Status |", "|---|---|---|---|"]
    lines += [f"| {m['model']} | {m['group']} | {m['episodes_done']}/{m['episodes_expected']} | {m['label']} |"
              for m in numbers["models"]]
    return lines


HEADLINE = (("Attempted (no guardrails)", "attempted"), ("Success (no guardrails)", "success_unprotected"),
            ("Success (guardrails)", "success_protected"), ("Blocked given attempt", "blocked_given_attempt"),
            ("Benign task (no guardrails)", "benign_task_unprotected"),
            ("Benign task (guardrails)", "benign_task_protected"), ("False positive (guardrails)", "false_positive"))


def _headline(numbers: dict) -> list:
    lines = ["| Model | Group | " + " | ".join(t for t, _ in HEADLINE) + " |", "|" + "---|" * (2 + len(HEADLINE))]
    for m in _models(numbers):
        lines.append(f"| {_name(m)} | {m['group']} | " + " | ".join(_fp(m["metrics"][p]) for _, p in HEADLINE) + " |")
    control = _control(numbers)
    if control:
        lines.append(f"| {control['model']} (scripted control, reference) | control | "
                     + " | ".join(_fp(control["metrics"][p]) for _, p in HEADLINE) + " |")
    lines += ["", "Fixed order (the five local models, then the three hosted ones), not a ranking. A row marked "
                  "partial has not finished its planned run and is not final. Each cell is P% (k/n) [95% Wilson "
                  "interval]; the scripted control is a reference, not a language model."]
    return lines


def _pooled_layer(numbers: dict, layer: str) -> tuple:
    groups = numbers["pooled"]["layers"]
    return sum(g[layer]["k"] for g in groups.values()), sum(g["n"] for g in groups.values())


def _findings(numbers: dict) -> list:
    models = _models(numbers)
    lines = [f"- {_name(m)}: attacks succeeded in {_kn(m['metrics']['success_unprotected'])} attack episodes "
             f"without guardrails and {_kn(m['metrics']['success_protected'])} with guardrails." for m in models]
    leaks = [m for m in models if m["metrics"]["success_protected"]["k"]]
    if leaks:
        lines.append("- With guardrails, attacks still succeeded for: "
                     + ", ".join(f"{m['model']} {_kn(m['metrics']['success_protected'])}" for m in leaks) + ".")
    else:
        lines.append(f"- With guardrails, no current-engine model succeeded on any attack episode "
                     f"(0 for each of the {len(models)} language models).")
    pooled = numbers["pooled"]["blocked_given_attempt"]
    lines.append(f"- Pooled over the {len(models)} language models, guardrails blocked {_kn(pooled)} attempted attacks "
                 f"at the attack action itself (conditional block rate).")
    for layer, title in (("sandbox", "Sandbox"), ("seccomp", "Seccomp")):
        k, n = _pooled_layer(numbers, layer)
        if k == 0:
            lines.append(f"- {title}: no block recorded, {k}/{n} pooled protected-arm episodes; this is a lower "
                         f"bound, because the policy engine stops attacks before they reach the kernel layers.")
        else:
            lines.append(f"- {title}: blocks recorded in {k}/{n} pooled protected-arm episodes (a lower bound).")
    c = numbers.get("control")
    if c:
        u, p = c["unprotected"], c["protected"]
        lines.append(f"- The scripted control attempted the attack in {u['attempted']}/{u['episodes']} episodes and "
                     f"succeeded in {u['succeeded']}/{u['episodes']} without guardrails; with guardrails it succeeded in "
                     f"{p['succeeded']}/{p['episodes']}, and guardrails blocked {p['blocked']}/{p['attempted']} of its "
                     f"attempted attacks.")
    return lines


def _control_block(numbers: dict) -> list:
    c = numbers.get("control")
    if not c:
        return ["No control rows in this snapshot."]
    u, p = c["unprotected"], c["protected"]
    lines = ["| Arm | Attack episodes | Attempted | Success | Blocked given attempt |", "|---|---|---|---|---|",
             f"| Unprotected | {u['episodes']} | {format_prop(u['attempted'], u['episodes'])} "
             f"| {format_prop(u['succeeded'], u['episodes'])} | — |",
             f"| Protected | {p['episodes']} | {format_prop(p['attempted'], p['episodes'])} "
             f"| {format_prop(p['succeeded'], p['episodes'])} | {format_prop(p['blocked'], p['attempted'])} |",
             "",
             f"The scripted control obeys every injection. Guardrails blocked {p['blocked']} of its {p['attempted']} "
             f"attempted attacks; the tool broker denied the attacker's tool call in {c['broker_denied']} of them. "
             f"It is deterministic, so its {p['episodes']} protected attack episodes are {c['cases']['attack']} "
             f"distinct cases."]
    return lines


def _layers_block(numbers: dict) -> list:
    groups = numbers["pooled"]["layers"]
    lines = ["Recorded blocks per guardrail layer, protected arm, pooled over the language models (k = episodes with "
             "at least one action blocked by the layer):", "",
             "| Layer | Attack episodes | Benign episodes | Blocked actions |", "|---|---|---|---|"]
    for layer in LAYERS:
        a, b = groups.get("attack"), groups.get("benign")
        cells = [format_prop(g[layer]["k"], g["n"]) if g else "— (0/0)" for g in (a, b)]
        actions = sum(g[layer]["actions"] for g in groups.values())
        lines.append(f"| {LAYER_TITLES[layer]} | {cells[0]} | {cells[1]} | {actions} |")
    lines += ["", "| Model | Sandbox (attack) | Seccomp (attack) | Sandbox (benign) | Seccomp (benign) |",
              "|---|---|---|---|---|"]
    for m in _models(numbers):
        layers = m["layers"] or {}
        cells = [format_prop(layers[g][layer]["k"], layers[g]["n"]) if g in layers else "— (0/0)"
                 for g in ("attack", "benign") for layer in ("sandbox", "seccomp")]
        lines.append(f"| {_name(m)} | " + " | ".join(cells) + " |")
    lines.append("")
    for layer, title in (("sandbox", "Sandbox (namespaces / private filesystem)"), ("seccomp", "Seccomp-BPF")):
        k, n = _pooled_layer(numbers, layer)
        if k == 0:
            lines.append(f"{title}: no block was recorded in any of the {n} protected-arm episodes of the "
                         f"{len(_models(numbers))} language models (0/{n}).")
        else:
            lines.append(f"{title}: blocks were recorded in {k}/{n} protected-arm episodes.")
        lines.append("")
    lines.append(LAYER_EVIDENCE)
    return lines


def _hosted(numbers: dict) -> list:
    lines = ["| Model | Episodes done | ds episodes finished | Complete pairs | Paired attack episodes "
             "| Infra errors pending | Loop state | Quota stops |", "|---|---|---|---|---|---|---|---|"]
    for m in numbers["models"]:
        if m["model"] not in HOSTED_MODELS:
            continue
        state = m.get("hosted_state") or {"state": "pending", "quota_stops": 0}
        lines.append(f"| {m['model']} | {m['episodes_done']}/{m['episodes_expected']} "
                     f"| {m['finished_by_kind']['ds']}/{DS_EXPECTED} | {m['pairs_complete']} | {m['attack_pairs']} "
                     f"| {m['infra_errors']} | {state['state']} | {state['quota_stops']} |")
    lines += ["", f"The loop state and quota stops are read from each model's frozen hosted_state.json at the "
                  f"snapshot ({numbers['snapshot_display']}); they are final only after the D-09 freeze "
                  f"({numbers['freeze_display']})."]
    return lines


def _charts(numbers: dict, doc_path: str, frozen_dir: str) -> list:
    doc_dir = os.path.dirname(os.path.abspath(doc_path))
    links = []
    for chart in numbers["charts"]:
        target = os.path.abspath(os.path.join(frozen_dir, *chart["file"].split("/")))
        links.append((os.path.relpath(target, doc_dir).replace("\\", "/"), chart["caption"]))
    return _body(chart_lines(links)) + ["", "The build writes these charts into the frozen results directory; "
                                            "results/ is not committed, so run python -m src.deliverables build "
                                            "to see them."]


def render_block(name: str, numbers: dict, doc_path: str, frozen_dir: str = FROZEN_DIR) -> list:
    """The lines (no line endings) between the markers of block `name`, from numbers.json only.

    Args:
        name: one of BLOCK_NAMES.
        numbers: the parsed numbers.json.
        doc_path: the document the block goes into (chart links are relative to its directory).
        frozen_dir: the frozen directory (where the charts are)."""
    if name == "status":
        return _status(numbers)
    if name == "headline":
        return _headline(numbers)
    if name == "findings":
        return _findings(numbers)
    if name == "control":
        return _control_block(numbers)
    if name == "layers":
        return _layers_block(numbers)
    if name == "hosted":
        return _hosted(numbers)
    if name == "confounds":
        return _body(confounds_lines(numbers["confounds"]))
    if name == "legacy":
        return _body(legacy_lines(numbers["legacy"]))
    if name == "threats":
        return _body(threats_lines(numbers["threats"]))
    if name == "charts":
        return _charts(numbers, doc_path, frozen_dir)
    raise BlockError(f"unknown block name {name!r}")


# ── Applying ──

def apply_blocks(path: str, numbers: dict, frozen_dir: str = FROZEN_DIR, display: str = None) -> tuple:
    """Rewrite every block of the document at `path`; returns (blocks, changed). Writes only when changed."""
    with open(path, "rb") as f:
        data = f.read()
    text = data.decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"
    found = find_blocks(text, display or path.replace("\\", "/"))
    lines = text.splitlines(keepends=True)
    out, cursor = [], 0
    for block in found:
        out += lines[cursor:block["begin"] + 1]
        out += [line + eol for line in render_block(block["name"], numbers, path, frozen_dir)]
        cursor = block["end"]
    out += lines[cursor:]
    new = "".join(out)
    if new == text:
        return len(found), False
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new.encode("utf-8"))
    os.replace(tmp, path)
    return len(found), True
