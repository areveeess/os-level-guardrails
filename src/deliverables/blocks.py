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

The claim-* blocks are one generated paragraph each: the result sentences the prose of a doc would
otherwise state by hand (whether an attack succeeded with guardrails, whether every attempt was
blocked, what the kernel layers recorded, whether any context overflowed, which rows are partial).
Each one branches on the data, so after the post-freeze build it says whatever the data then says,
and the check fails when the prose outside the blocks states such an outcome itself (CR-01). A claim
block may appear more than once in a doc, because it renders the same lines wherever it is placed;
every other block appears at most once.

apply_blocks keeps each file's own line endings (README.md is CRLF in the working tree, the
other targets are LF) and every byte outside the blocks, and writes only when a block changed.
"""

import os
import re

from src.benchmark.report import (LAYER_EVIDENCE, LAYERS, LAYER_TITLES, chart_lines, confounds_lines, legacy_lines,
                                  threats_lines)
from src.benchmark.stats import format_prop
from src.deliverables import DS_EXPECTED, FROZEN_DIR, HOSTED_MODELS, MODEL_ORDER

CLAIM_NAMES = ("claim-protected", "claim-blocked", "claim-kernel", "claim-overflow", "claim-state")
BLOCK_NAMES = ("status", "headline", "findings", "control", "layers", "hosted", "confounds", "legacy", "threats",
               "charts") + CLAIM_NAMES
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
            if name in seen and name not in CLAIM_NAMES:
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
    if numbers["data_state"] == "snapshot" and numbers["snapshot_display"] >= numbers["freeze_display"]:
        return (f"Data: snapshot of {numbers['snapshot_display']} (UTC), copied after the D-09 freeze of "
                f"{numbers['freeze_display']} before the hosted loop recorded a final state for every model. "
                f"Partial rows are labelled with their episodes done and are not final.")
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


# ── Claim paragraphs (CR-01): the result sentences of the prose, branching on the data ──

def _names(models: list) -> str:
    return ", ".join(_name(m) for m in models)


def _claim_protected(numbers: dict) -> list:
    models = _models(numbers)
    leaks = [m for m in models if m["metrics"]["success_protected"]["k"]]
    if leaks:
        text = (f"With guardrails, attacks still succeeded for {len(leaks)} of the {len(models)} language models: "
                + ", ".join(f"{_name(m)} {_kn(m['metrics']['success_protected'])}" for m in leaks)
                + f". For the other {len(models) - len(leaks)} the Success (guardrails) cell is 0.")
    else:
        text = (f"With guardrails, no attack succeeded for any of the {len(models)} language models, local or hosted: "
                f"the Success (guardrails) cell is 0 in every model row of the headline.")
    control = _control(numbers)
    if control:
        k = control["metrics"]["success_protected"]
        if k["k"] == 0:
            text += (" The scripted control, which attempts every attack, also had no successful attack with "
                     f"guardrails ({_kn(k)}).")
        else:
            text += (f" The scripted control, which attempts every attack, still succeeded in {_kn(k)} protected "
                     f"attack episodes.")
    return [text]


def _all_blocked(numbers: dict) -> bool:
    return all(m["metrics"]["blocked_given_attempt"]["k"] == m["metrics"]["blocked_given_attempt"]["n"]
               for m in _models(numbers))


def _claim_blocked(numbers: dict) -> list:
    models = _models(numbers)
    pooled = numbers["pooled"]["blocked_given_attempt"]
    if _all_blocked(numbers):
        return [f"Every attack a language model really attempted with guardrails on was blocked at the attack action "
                f"itself: {_kn(pooled)} attempted attacks, pooled over the {len(models)} language models (the "
                f"Blocked given attempt column of the headline)."]
    missed = [m for m in models
              if m["metrics"]["blocked_given_attempt"]["k"] != m["metrics"]["blocked_given_attempt"]["n"]]
    return ["Not every attempted attack was blocked at the attack action: "
            + ", ".join(f"{_name(m)} {_kn(m['metrics']['blocked_given_attempt'])} blocked" for m in missed)
            + f". Pooled over the {len(models)} language models, {_kn(pooled)} attempted attacks were blocked at the "
              f"attack action."]


def _claim_kernel(numbers: dict) -> list:
    models = _models(numbers)
    (sk, n), (ck, _) = _pooled_layer(numbers, "sandbox"), _pooled_layer(numbers, "seccomp")
    if sk == 0 and ck == 0:
        text = (f"The sandbox and the seccomp filter recorded no block in any of the {n} protected-arm episodes of "
                f"the {len(models)} language models (0/{n} each).")
        if _all_blocked(numbers):
            text += (" The tool broker and the policy engine in front of them blocked every attempted attack at the "
                     "attack action first, so no attack action reached them.")
        else:
            text += (" Not every attempted attack was blocked at the attack action by the layers in front of them, "
                     "and no kernel-layer block was recorded for those attacks either.")
    else:
        text = (f"Kernel-layer blocks were recorded: the sandbox in {sk}/{n} and the seccomp filter in {ck}/{n} "
                f"protected-arm episodes of the {len(models)} language models.")
    return [text + " A recorded count is a lower bound, not proof that the kernel layers work."]


def _claim_overflow(numbers: dict) -> list:
    counts = [(m["model"], m["context_overflow_episodes"]) for m in _models(numbers)]
    if all(c == 0 for _, c in counts):
        return ["No current-engine episode had a context overflow (0 for every language model), so the deferred "
                "WR-01 bias touches none of the present data."]
    return ["Context overflows occurred in current-engine episodes: "
            + ", ".join(f"{model} {c}" for model, c in counts if c)
            + ". The deferred WR-01 bias can raise the ASR-valid figures of these models."]


def _claim_state(numbers: dict) -> list:
    partial = [m for m in _models(numbers) if not m["complete"]]
    snap, freeze = numbers["snapshot_display"], numbers["freeze_display"]
    if numbers["data_state"] == "final":
        text = f"The data is final: it was copied at {snap} (UTC), after the D-09 freeze of {freeze}."
        if partial:
            text += (f" {_names(partial)} stopped at the freeze or on their quota before finishing, so they stay "
                     f"labelled partial with their episodes done and are never ranked against the complete rows.")
        else:
            text += " Every language model finished its planned run."
        return [text]
    if snap < freeze:
        text = (f"The data is a snapshot of {snap} (UTC), copied before the D-09 freeze of {freeze} while the hosted "
                f"loop was still collecting.")
    else:
        text = (f"The data is a snapshot of {snap} (UTC), copied after the D-09 freeze of {freeze} but before the "
                f"hosted loop had recorded a final state for every model.")
    if partial:
        text += (f" {_names(partial)} are labelled partial with their episodes done; they are not final and are "
                 f"never ranked against the complete rows.")
    else:
        text += " Every language model has finished its planned run."
    return [text]


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
    claim = {"claim-protected": _claim_protected, "claim-blocked": _claim_blocked, "claim-kernel": _claim_kernel,
             "claim-overflow": _claim_overflow, "claim-state": _claim_state}.get(name)
    if claim:
        return claim(numbers)
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
