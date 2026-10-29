"""
Deliverables consistency check — every quoted number must come from numbers.json (D-03)
======================================================================================
A document that quotes a number by hand drifts silently when the data changes (the Groq
re-run keeps adding episodes until the freeze). This check makes the drift loud:

  stale block    a generated block that no longer equals render_block(name, numbers, doc)
  fraction       a k/n on a line naming a model (its id or an alias, case-insensitive, whole word)
                 that is not one of that model's generated fractions; a line naming two models
                 accepts either model's; a line saying "pooled" accepts the pooled fractions
  timestamp      an ISO timestamp (minute precision) other than the snapshot or the freeze
  percentage     'P% (k/n' whose P is not k/n rounded to P's decimals
  outcome        a paragraph outside the blocks that states a headline outcome in its own words
                 (OUTCOME_PATTERNS: "no attack succeeded", "zero in every row", "every attempted attack
                 was blocked", "recorded no block", "no episode had a context overflow", "were still
                 running when" and the like). Such a sentence would stay true-looking after the data
                 changes, so it must come from a claim-* block, which branches on the data (CR-01).
                 The patterns are a fixed list of the forms the docs used, not a language model: a new
                 wording of an outcome is caught only once it is added here.
  key-shaped     text matching the fixed key regex, in a checked doc (inside blocks too) or in any
                 text file under the frozen directory; reported as path:line only, never the text

Lines inside blocks are exempt from the fraction, timestamp, percentage and outcome rules: the
stale-block rule covers them. A paragraph is a run of plain lines outside the blocks; a blank
line, a heading, a list item, a table row, a code fence and a block marker each start a new one. SECRET_RE is a copy of src.benchmark.run_checks.SECRET_RE (pinned
by a test), not an import: run_checks imports the provider registry, which loads API keys.
"""

import json
import os
import re

from src.deliverables import CONTROL_MODEL, DS_EXPECTED, FROZEN_DIR, HOSTED_MODELS
from src.deliverables.blocks import DOC_TARGETS, REQUIRED_DOCS, BlockError, find_blocks, render_block

SECRET_RE = re.compile(r"gsk_[A-Za-z0-9]{20,}")

MODEL_ALIASES = {
    "qwen2.5-0.5b-agentlora": ("qwen2.5-0.5b-agentlora", "Qwen LoRA", "agentlora"),
    "scratch-gpt": ("scratch-gpt", "ScratchGPT"),
    "qwen2.5-0.5b-base": ("qwen2.5-0.5b-base", "Qwen base"),
    "ollama-qwen2.5-3b": ("ollama-qwen2.5-3b", "qwen2.5:3b", "qwen2.5-3b"),
    "ollama-llama3.2-3b": ("ollama-llama3.2-3b", "llama3.2:3b", "llama3.2-3b"),
    "groq-gpt-oss-120b": ("groq-gpt-oss-120b", "gpt-oss-120b"),
    "groq-gpt-oss-20b": ("groq-gpt-oss-20b", "gpt-oss-20b"),
    "groq-qwen3.8-27b": ("groq-qwen3.8-27b", "qwen3.8-27b"),
    CONTROL_MODEL: (CONTROL_MODEL, "scripted control", "control agent"),
    "gemini-flash-lite": ("gemini-flash-lite", "Gemini"),
    "gemma-4-31b": ("gemma-4-31b", "Gemma"),
}

# A fraction token is digits/digits not touching another slash, a word character or a decimal
FRACTION_RE = re.compile(r"(?<![\w/.])(\d+)/(\d+)(?![\w/]|\.\d)")
TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})")
PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.(\d+))?)% \((\d+)/(\d+)")
POOLED_RE = re.compile(r"\bpooled\b", re.IGNORECASE)
_ALIAS_RES = {model: [re.compile(rf"(?<![\w.-]){re.escape(a)}(?![\w-]|\.\w)", re.IGNORECASE) for a in aliases]
              for model, aliases in MODEL_ALIASES.items()}

# Headline outcomes stated as prose (CR-01); matched case-insensitively over a whole paragraph, so a sentence
# wrapped across lines is still found
OUTCOME_PATTERNS = (
    r"no attack (?:\w+ )?succeeded",
    r"not one attack succeeded",
    r"attacks? (?:still )?never succeeded",
    r"succeeded zero times",
    r"(?:zero|0) in every (?:model )?row",
    r"(?:is|are|was|were) (?:zero|0) for every",
    r"every attempted attack (?:was|were) (?:blocked|stopped)",
    r"every attack a model really (?:tried|attempted) was (?:blocked|stopped)",
    r"stopped every (?:attempted )?attack",
    r"stopped every attempt",
    r"recorded (?:no|zero|0) blocks?\b",
    r"recorded no (?:sandbox|seccomp) block",
    r"no block at all",
    r"no (?:current(?:-engine)? |phase 4 )?episode had a context overflow",
    r"(?:were|was) still running when",
    r"they missed nothing",
)
OUTCOME_RE = re.compile(r"\b(?:" + "|".join(OUTCOME_PATTERNS) + r")", re.IGNORECASE)
_PARAGRAPH_START_RE = re.compile(r"^\s*(?:#|[-*+] |\d+\. |\||```)")


def _add_metrics(out: set, metrics: dict):
    for metric in (metrics or {}).values():
        if isinstance(metric, dict) and "k" in metric and "n" in metric:
            out.add(f"{metric['k']}/{metric['n']}")


def _flat_fractions(entry: dict) -> set:
    """Every PREFIX_k/PREFIX_n of a flat summary-shaped dict (the legacy entries)."""
    return {f"{entry[key]}/{entry[key[:-2] + '_n']}" for key in entry
            if key.endswith("_k") and key[:-2] + "_n" in entry}


def allowed_fractions(numbers: dict) -> tuple:
    """({model: set of 'k/n' strings}, pooled set) from numbers.json."""
    per_model = {}
    for m in numbers["models"]:
        s = {f"{m['episodes_done']}/{m['episodes_expected']}"}
        _add_metrics(s, m["metrics"])
        for kind in (m.get("by_kind") or {}).values():
            _add_metrics(s, {k: v for k, v in kind.items() if isinstance(v, dict)})
        for group in (m.get("layers") or {}).values():
            _add_metrics(s, {k: v for k, v in group.items() if isinstance(v, dict)})
        if m["model"] in HOSTED_MODELS:
            s.add(f"{m['finished_by_kind']['ds']}/{DS_EXPECTED}")
        per_model[m["model"]] = s
    c = numbers.get("control")
    if c:
        u, p = c["unprotected"], c["protected"]
        s = per_model.setdefault(CONTROL_MODEL, set())
        s |= {f"{u['attempted']}/{u['episodes']}", f"{u['succeeded']}/{u['episodes']}",
              f"{p['attempted']}/{p['episodes']}", f"{p['succeeded']}/{p['episodes']}",
              f"{p['blocked']}/{p['attempted']}", f"{c['cases']['blocked']}/{c['cases']['attempted']}"}
        s |= {f"{k['blocked']}/{k['attempted']}" for k in c["by_kind"].values()}
        s |= {f"{item['episodes']}/{p['attempted']}" for item in c["layers"].values()}
    for entry in numbers.get("legacy") or []:
        s = per_model.setdefault(entry["model"], set())
        s |= _flat_fractions(entry)
        s.add(f"{entry['episodes_done']}/{entry['episodes_expected']}")
    pooled = set()
    _add_metrics(pooled, {"blocked_given_attempt": numbers["pooled"]["blocked_given_attempt"]})
    groups = numbers["pooled"]["layers"]
    for group in groups.values():
        _add_metrics(pooled, {k: v for k, v in group.items() if isinstance(v, dict)})
    if groups:
        n = sum(g["n"] for g in groups.values())
        for layer in next(iter(groups.values())):
            if layer != "n":
                pooled.add(f"{sum(g[layer]['k'] for g in groups.values())}/{n}")
    return per_model, pooled


def models_on_line(line: str) -> list:
    return [model for model, patterns in _ALIAS_RES.items() if any(p.search(line) for p in patterns)]


def _minutes(display: str) -> str:
    return display[:16]


def _line_problems(line: str, numbers: dict, allowed: tuple) -> list:
    """Fraction, timestamp and percentage problems of one plain line (reasons, never the line's text)."""
    per_model, pooled = allowed
    problems = []
    named = models_on_line(line)
    says_pooled = bool(POOLED_RE.search(line))
    if named or says_pooled:
        ok = set(pooled) if says_pooled else set()
        for model in named:
            ok |= per_model.get(model, set())
        for m in FRACTION_RE.finditer(line):
            token = f"{m.group(1)}/{m.group(2)}"
            if token not in ok:
                who = ", ".join(named + (["pooled"] if says_pooled else []))
                problems.append(f"fraction {token} is not a generated value for {who}")
    known = {_minutes(numbers["snapshot_display"]), _minutes(numbers["freeze_display"])}
    for m in TIMESTAMP_RE.finditer(line):
        if m.group(1) not in known:
            problems.append(f"timestamp {m.group(1)} is neither the snapshot nor the freeze")
    for m in PERCENT_RE.finditer(line):
        pct, decimals, k, n = float(m.group(1)), len(m.group(2) or ""), int(m.group(3)), int(m.group(4))
        if n == 0 or abs(pct - round(100.0 * k / n, decimals)) > 1e-9:
            problems.append(f"percentage {m.group(1)}% does not match {k}/{n}")
    return list(dict.fromkeys(problems))          # a token repeated on one line is reported once


def _key_problem(line: str) -> list:
    return ["key-shaped text"] if SECRET_RE.search(line) else []


def scan_text(lines: list, numbers: dict) -> list:
    """[(line number, reason)] for plain lines (1-based): fraction, timestamp, percentage and key rules."""
    allowed = allowed_fractions(numbers)
    out = []
    for lineno, line in enumerate(lines, 1):
        for reason in _key_problem(line) + _line_problems(line, numbers, allowed):
            out.append((lineno, reason))
    return out


def paragraphs(lines: list, inside: set) -> list:
    """[[line index, ...]] of the plain paragraphs: runs of non-blank lines outside the blocks, split at
    headings, list items, table rows and code fences."""
    out, current = [], []
    for index, line in enumerate(lines):
        if index in inside or not line.strip():
            if current:
                out.append(current)
            current = []
            continue
        if current and _PARAGRAPH_START_RE.match(line):
            out.append(current)
            current = []
        current.append(index)
    if current:
        out.append(current)
    return out


def outcome_problems(lines: list, para: list) -> list:
    """[(line index, reason)] for headline outcomes stated in one paragraph (CR-01), never echoing the text."""
    joined, starts = "", []
    for index in para:
        starts.append((len(joined), index))
        joined += lines[index].strip() + " "
    out = []
    for m in OUTCOME_RE.finditer(joined):
        index = [i for offset, i in starts if offset <= m.start()][-1]
        out.append((index, "outcome stated outside a generated block (use a claim-* block, CR-01)"))
    return out


def check_doc(path: str, display: str, numbers: dict, frozen_dir: str) -> tuple:
    """(block count, ['display:line: reason']) for one document."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    try:
        found = find_blocks(text, display)
    except BlockError as e:
        return 0, [str(e)]
    lines = [line.rstrip("\r") for line in text.splitlines()]
    inside, problems = set(), []
    for block in found:
        inside.update(range(block["begin"], block["end"] + 1))
        if lines[block["begin"] + 1:block["end"]] != render_block(block["name"], numbers, path, frozen_dir):
            problems.append(f"{display}:{block['begin'] + 1}: block {block['name']} is stale "
                            f"(run python -m src.deliverables blocks)")
    allowed = allowed_fractions(numbers)
    reasons = {}
    for index, line in enumerate(lines):
        reasons[index] = _key_problem(line)
        if index not in inside:
            reasons[index] += _line_problems(line, numbers, allowed)
    for para in paragraphs(lines, inside):
        for index, reason in outcome_problems(lines, para):
            reasons[index].append(reason)
    for index in range(len(lines)):
        problems += [f"{display}:{index + 1}: {reason}" for reason in dict.fromkeys(reasons[index])]
    return len(found), problems


def scan_frozen(frozen_dir: str, display: str) -> tuple:
    """(text files scanned, ['display/rel:line: key-shaped text']) over every file under the frozen directory.
    Files with a NUL byte in their first 8 KiB are skipped; symlinks are not followed."""
    scanned, problems = 0, []
    for dirpath, dirnames, names in os.walk(frozen_dir):
        dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            scanned += 1
            rel = os.path.relpath(path, frozen_dir).replace("\\", "/")
            for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if SECRET_RE.search(line):
                    problems.append(f"{display}/{rel}:{lineno}: key-shaped text")
    return scanned, problems


def load_numbers(frozen_dir: str) -> dict:
    with open(os.path.join(frozen_dir, "numbers.json"), encoding="utf-8") as f:
        return json.load(f)


def normalize_doc(doc: str) -> str:
    doc = doc.replace("\\", "/")
    return doc[2:] if doc.startswith("./") else doc


def select_docs(docs) -> list:
    """The selected target docs (all when docs is None); raises ValueError for a doc outside DOC_TARGETS."""
    if docs is None:
        return list(DOC_TARGETS)
    out = []
    for doc in docs:
        doc = normalize_doc(doc)
        if doc not in DOC_TARGETS:
            raise ValueError(f"{doc} is not a deliverables target doc ({', '.join(DOC_TARGETS)})")
        if doc not in out:
            out.append(doc)
    return out


def run_check(repo_root: str, frozen_dir: str = FROZEN_DIR, docs=None, numbers: dict = None) -> tuple:
    """(printed lines, exit code) of `check`: one DOC / SKIP / FAIL line per selected doc, the SECRETS line or its
    FAIL lines, then 'deliverables-check-ok docs N' when nothing failed."""
    numbers = numbers if numbers is not None else load_numbers(frozen_dir)
    lines, failed, ok_docs = [], False, 0
    for doc in select_docs(docs):
        path = os.path.join(repo_root, *doc.split("/"))
        if not os.path.isfile(path):
            if doc in REQUIRED_DOCS:
                lines.append(f"FAIL check: {doc}:0: required document is missing")
                failed = True
            else:
                lines.append(f"SKIP {doc} absent")
            continue
        count, problems = check_doc(path, doc, numbers, frozen_dir)
        if problems:
            lines += [f"FAIL check: {p}" for p in problems]
            failed = True
        else:
            lines.append(f"DOC {doc} blocks {count} ok")
            ok_docs += 1
    display = frozen_dir.replace("\\", "/").rstrip("/")
    scanned, problems = scan_frozen(frozen_dir, display)
    if problems:
        lines += [f"FAIL check: {p}" for p in problems]
        failed = True
    else:
        lines.append(f"SECRETS {display} files {scanned} ok")
    if failed:
        return lines, 1
    lines.append(f"deliverables-check-ok docs {ok_docs}")
    return lines, 0
