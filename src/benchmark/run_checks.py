"""
Run checks — mechanical pass/fail checks over Phase 4 result directories
========================================================================
Every Phase 4 claim ("342/342 finished", "no duplicate after a resume", "the stamps
match the engine that ran") is a command that prints an ok line or a FAIL line, so a
verifier never has to read raw.jsonl by eye:

  python -m src.benchmark.run_checks complete --results DIR --model M [--expect N] [--resolved-contains TEXT]
  python -m src.benchmark.run_checks default-context [--jobs-dir D]
  python -m src.benchmark.run_checks control-tokens
  python -m src.benchmark.run_checks truncation --results DIR --model M [--num-ctx N] [--compare N]
  python -m src.benchmark.run_checks ollama-evidence --results DIR --model M [--jobs-dir D]
  python -m src.benchmark.run_checks hosted --prefix P [--models a,b] [--jobs-dir D]
  python -m src.benchmark.run_checks secret-scan --paths P1 P2 ...
  python -m src.benchmark.run_checks viva --page PAGE --report-csv SUMMARY_CSV --readme README

viva checks the Phase 4 viva page: the Part A-D and Limits headings, at least 16 "**Q"
questions and 30 "- **" glossary lines, no key-like text, every language-model row of
the report's summary.csv quoted on one line as "<model> ... <episodes_done>/
<episodes_expected>" (read from the summary.csv columns of those names, so the page
cannot drift from the report), and the README row linking the page as written.

hosted labels each Groq model complete or "partial n=F ds n=K" and prints
hosted-terminal-ok or hosted-running (D-09). secret-scan searches only the given paths
with the one fixed regex gsk_[A-Za-z0-9]{20,}, loads no key and prints file:line only.

default-context validates the measured Ollama default (results/phase4-jobs/
ollama_default_context.json); context_band() turns it into the prompt_tokens band
[L, U] that proves an Ollama window was raised past that default (no assumed 4,096).
control-tokens reports whether any added-vocabulary string of the Qwen tokenizer occurs
in the benchmark text (Phase 3 review IN-01): a match is a Qwen-only confound to report;
the benchmark text is never edited.

Failures print "FAIL <cmd>: <reason>" lines and exit 1; success prints "<cmd>-ok" and
exits 0. The checks only read result files, config/models.yaml, the policy file and
git history; they never change a stored row.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Optional

from src.benchmark.outcomes import classify_llm_error, classify_outcome
from src.benchmark.report import CONTROL_PROVIDER, load_rows, read_manifest
from src.benchmark.scoring import SCORING_VERSION
from src.llm.registry import PROJECT_ROOT, load_model_config
from src.qwen_lora.checks import SECRET_PATTERNS

POLICY_FILE = os.path.join(PROJECT_ROOT, "src", "policy_engine", "policies", "default_policy.yaml")

# Paths whose change alters what a stored row says. Report, checks, jobs and docs do not, so two code
# commits that differ only outside these paths still produce comparable rows.
RUN_AFFECTING_PATHS = (
    "src/agent_harness", "src/benchmark/runner.py", "src/benchmark/scoring.py", "src/benchmark/injecagent.py",
    "src/benchmark/outcomes.py", "src/benchmark/classify.py", "src/llm", "src/policy_engine", "src/sandbox",
    "config", "benchmarks", "run_benchmark.py",
)
_SHA40 = re.compile(r"[0-9a-f]{40}")


def _key(row: dict) -> tuple:
    return (row.get("model"), row.get("case_id"), row.get("run"), row.get("arm"))


def _raw_rows(raw_path: str, model: str) -> list:
    """Every decodable row of `model` in raw.jsonl, duplicates included."""
    rows = []
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("model") == model:
                rows.append(row)
    return rows


def _policy_sha256() -> str:
    with open(POLICY_FILE, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_diff_names(a: str, b: str) -> Optional[list]:
    """Run-affecting files that differ between commits a and b, or None when git cannot say."""
    try:
        done = subprocess.run(["git", "diff", "--name-only", a, b, "--", *RUN_AFFECTING_PATHS], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line for line in done.stdout.splitlines() if line.strip()]


def complete_status(results_dir: str, model: str, expect: Optional[int] = None) -> dict:
    """Completeness and provenance of one model's rows in results_dir.

    finished = (model, case, run, arm) keys whose best record (report.load_rows) is not an llm_error;
    pending_infra = keys whose only records are infrastructure errors (a resume re-runs them).
    problems lists every reason the directory is not a complete, single-engine run."""
    raw_path = os.path.join(results_dir, "raw.jsonl")
    status = {"model": model, "finished": 0, "expected": expect, "pending_infra": 0, "duplicate_finished": 0,
              "policy": None, "commits": [], "resolved": [], "problems": []}
    problems = status["problems"]
    if expect is None:
        manifest = read_manifest(results_dir) or {}
        value = manifest.get("expected_episodes_per_model")
        status["expected"] = value if type(value) is int and value > 0 else None
    if not os.path.exists(raw_path):
        problems.append(f"no raw.jsonl in {results_dir}")
        return status
    raw = _raw_rows(raw_path, model)
    best = [r for r in load_rows(raw_path) if r.get("model") == model]
    finished = {_key(r) for r in best if classify_outcome(r) != "llm_error"}
    status["finished"] = len(finished)
    status["pending_infra"] = sum(1 for r in best if classify_outcome(r) == "llm_error")

    finished_counts = Counter(_key(r) for r in raw if classify_llm_error(r.get("llm_error")) != "infra")
    dups = {k: n for k, n in finished_counts.items() if n > 1}
    status["duplicate_finished"] = len(dups)
    if dups:
        sample = ", ".join(f"{k[1]} run{k[2]} {k[3]} x{n}" for k, n in list(dups.items())[:3])
        problems.append(f"{len(dups)} key(s) finished more than once (a resume duplicated an episode): {sample}")

    if status["expected"] is None:
        problems.append("expected episode count unknown (no --expect and no manifest.json)")
    elif status["finished"] < status["expected"]:
        problems.append(f"finished {status['finished']} below expected {status['expected']}")

    try:
        cfg = load_model_config()["models"].get(model)
    except OSError:
        cfg = None
    if cfg is None:
        problems.append(f"model {model} is not in config/models.yaml")
    policy = _policy_sha256()
    status["policy"] = policy
    bad_commit = sum(1 for r in raw if not _SHA40.fullmatch(str(r.get("code_commit") or "")))
    if bad_commit:
        problems.append(f"{bad_commit} row(s) without a 40-hex code_commit")
    bad_policy = sum(1 for r in raw if r.get("policy_sha256") != policy)
    if bad_policy:
        problems.append(f"{bad_policy} row(s) with a policy_sha256 other than {policy[:12]} (the current policy)")
    bad_scoring = sum(1 for r in raw if r.get("scoring") != SCORING_VERSION)
    if bad_scoring:
        problems.append(f"{bad_scoring} row(s) without scoring {SCORING_VERSION}")
    if cfg is not None:
        bad_provider = sum(1 for r in raw if r.get("provider") != cfg.get("provider"))
        bad_model = sum(1 for r in raw if r.get("config_model") != cfg.get("model"))
        if bad_provider:
            problems.append(f"{bad_provider} row(s) whose provider is not {cfg.get('provider')}")
        if bad_model:
            problems.append(f"{bad_model} row(s) whose config_model is not {cfg.get('model')}")

    resolved = sorted({str(r["resolved_model"]) for r in raw if r.get("resolved_model")})
    status["resolved"] = resolved
    if len(resolved) > 1:
        problems.append(f"{len(resolved)} distinct resolved_model values: {', '.join(resolved)}")

    commits = sorted({str(r.get("code_commit")) for r in raw if _SHA40.fullmatch(str(r.get("code_commit") or ""))})
    status["commits"] = commits
    for other in commits[1:]:
        changed = _git_diff_names(commits[0], other)
        if changed is None:
            problems.append(f"git cannot compare code commits {commits[0][:12]} and {other[:12]}")
        elif changed:
            problems.append(f"run-affecting files differ between code commits {commits[0][:12]} and {other[:12]}: "
                            f"{', '.join(changed[:5])}")
    return status


def check_complete(results_dir: str, model: str, expect: Optional[int] = None,
                   resolved_contains: Optional[str] = None) -> list:
    """FAIL reasons (empty when complete); prints the COMPLETE summary line."""
    status = complete_status(results_dir, model, expect)
    problems = list(status["problems"])
    if resolved_contains:
        raw_path = os.path.join(results_dir, "raw.jsonl")
        rows = _raw_rows(raw_path, model) if os.path.exists(raw_path) else []
        finished = [r for r in rows if classify_llm_error(r.get("llm_error")) != "infra"]
        missing = sum(1 for r in finished if resolved_contains not in str(r.get("resolved_model") or ""))
        if missing:
            problems.append(f"{missing} finished row(s) whose resolved_model does not contain {resolved_contains!r}")
    print(f"COMPLETE model {model} finished {status['finished']}/{status['expected']} "
          f"pending_infra {status['pending_infra']} duplicates {status['duplicate_finished']} "
          f"policy {(status['policy'] or '')[:12]} commits {len(status['commits'])} "
          f"resolved {','.join(status['resolved']) or 'none'}")
    return problems


# ── Measured Ollama default context and the shared band ──

DEFAULT_JOBS_DIR = os.path.join("results", "phase4-jobs")
DEFAULT_CONTEXT_FILE = "ollama_default_context.json"
EVIDENCE_SOURCES = ("api_ps", "ps_column", "config_line", "long_prompt_plateau")
PROBE_PORTS = range(11435, 11441)


def read_default_context(jobs_dir: str = DEFAULT_JOBS_DIR) -> Optional[dict]:
    try:
        with open(os.path.join(jobs_dir, DEFAULT_CONTEXT_FILE), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_default_context(jobs_dir: str = DEFAULT_JOBS_DIR) -> list:
    """FAIL reasons for ollama_default_context.json (empty when it holds measured, named values)."""
    data = read_default_context(jobs_dir)
    if data is None:
        return [f"no readable {DEFAULT_CONTEXT_FILE} in {jobs_dir} (run: python -m src.benchmark.jobs "
                f"ollama-default-probe)"]
    problems = []
    if not isinstance(data.get("server_version"), str) or not data["server_version"].strip():
        problems.append("server_version missing or empty")
    d = data.get("default_context_length")
    if type(d) is not int or d <= 0:
        problems.append(f"default_context_length is not a positive integer ({d!r})")
    if data.get("evidence_source") not in EVIDENCE_SOURCES:
        problems.append(f"evidence_source {data.get('evidence_source')!r} is not one of {', '.join(EVIDENCE_SOURCES)}")
    if data.get("port") not in PROBE_PORTS:
        problems.append(f"port {data.get('port')!r} is not a temporary probe port (11435-11440)")
    if data.get("temp_server_stopped") is not True:
        problems.append("temp_server_stopped is not true (the temporary server may still be running)")
    if not problems:
        print(f"DEFAULT_CONTEXT {d} source {data['evidence_source']} server {data['server_version']} "
              f"lower_bound {bool(data.get('default_is_lower_bound'))}")
    return problems


def context_band(default_context_length: int, num_ctx: int) -> tuple:
    """(L, U): the prompt_tokens band a long prompt must land in to prove the window was raised past the
    measured default. U = floor(0.95 * num_ctx); L = D + D//40 (2.5 % above the default), or num_ctx//2
    when that is above U - 500 (the default already nearly covers the window; see band_note)."""
    upper = (num_ctx * 95) // 100
    lower = default_context_length + default_context_length // 40
    if lower > upper - 500:
        lower = num_ctx // 2
    return lower, upper


def band_note(default_context_length: int, num_ctx: int) -> Optional[str]:
    d = default_context_length
    if d + d // 40 > (num_ctx * 95) // 100 - 500:
        return (f"NOTE: the measured default {d} already nearly covers num_ctx {num_ctx}; the band's lower "
                f"bound is num_ctx//2 = {num_ctx // 2}, so the long prompt cannot tell the two windows apart")
    return None


# ── Truncation over stored prompt_tokens ──

def _percentile(values: list, q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def check_truncation(results_dir: str, model: str, num_ctx: Optional[int] = None,
                     compare: Optional[int] = None) -> tuple:
    """(problems, unknown). Over every step's prompt_tokens of the model's rows: max, p50, p95, steps at or
    above 95 % of the window, 'length' finish reasons and episodes whose prompt_tokens drop between steps
    (a server that reported only uncached tokens, or a prompt cut from the front). unknown is True when no
    row carries prompt_tokens (rows written before EVAL-06)."""
    if num_ctx is None:
        cfg = load_model_config()["models"].get(model) or {}
        num_ctx = cfg.get("num_ctx")
        if num_ctx is None:
            return [f"no --num-ctx given and config/models.yaml has no num_ctx for {model}"], False
    raw_path = os.path.join(results_dir, "raw.jsonl")
    if not os.path.exists(raw_path):
        return [f"no raw.jsonl in {results_dir}"], False
    rows = [r for r in load_rows(raw_path) if r.get("model") == model]
    finished = [r for r in rows if classify_outcome(r) != "llm_error"]

    def tokens(row):
        return [t for t in (row.get("prompt_tokens") or []) if isinstance(t, int)]
    steps = [t for r in rows for t in tokens(r)]
    if not steps:
        print(f"TRUNCATION model {model} num_ctx {num_ctx} no row carries prompt_tokens")
        return [], True
    threshold = (num_ctx * 95) // 100
    at_cap = sum(1 for t in steps if t >= threshold)
    lengths = sum(1 for r in rows for f in (r.get("finish_reasons") or []) if f == "length")
    decreasing = sum(1 for r in rows if any(b < a for a, b in zip(tokens(r), tokens(r)[1:])))
    lacking = sum(1 for r in finished if not tokens(r))
    print(f"TRUNCATION model {model} steps {len(steps)} max {max(steps)} p50 {_percentile(steps, 0.5)} "
          f"p95 {_percentile(steps, 0.95)} num_ctx {num_ctx} at_or_above_95pct {at_cap} length_finish {lengths} "
          f"decreasing {decreasing} finished_without_prompt_tokens {lacking}")
    if compare is not None:
        above = sum(1 for r in finished if any(t > compare for t in tokens(r)))
        print(f"COMPARE episodes with a step above {compare}: {above}/{len(finished)}")
    problems = []
    if at_cap:
        problems.append(f"{at_cap} step(s) at or above 95 % of the {num_ctx}-token window ({threshold} tokens)")
    if lacking:
        problems.append(f"{lacking} finished row(s) carry no prompt_tokens")
    return problems, False


# ── Ollama context evidence ──

def _ps_entry_context(ps_json, variant: str) -> Optional[int]:
    entries = ps_json.get("models") if isinstance(ps_json, dict) else None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        names = {str(entry.get("name", "")), str(entry.get("model", ""))}
        if variant in names or f"{variant}:latest" in names:
            value = entry.get("context_length")
            return value if isinstance(value, int) and value > 0 else None
    return None


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def check_ollama_evidence(results_dir: str, model: str, jobs_dir: str = DEFAULT_JOBS_DIR) -> list:
    """FAIL reasons for an Ollama run's context evidence (D-06), against the measured default.

    Primary path: the loaded variant's /api/ps context_length (or the `ollama ps` CONTEXT column) equals
    num_ctx, `ollama show` lists num_ctx, and the long prompt's prompt_tokens lies in context_band(D, num_ctx).
    Fallback path, when the ps context fields or the server's OLLAMA_CONTEXT_LENGTH are unknown (a foreign
    server, or an owned one whose config line does not show it): `ollama show` lists num_ctx and the long
    prompt lies in the band."""
    from src.benchmark.jobs import _show_num_ctx, parse_ps_column
    cfg = load_model_config()["models"].get(model)
    if not cfg or not cfg.get("num_ctx"):
        return [f"{model} has no num_ctx in config/models.yaml (not an Ollama context variant)"]
    variant, num_ctx = str(cfg["model"]), int(cfg["num_ctx"])
    default = read_default_context(jobs_dir)
    if default is None or type(default.get("default_context_length")) is not int:
        return ["no measured default context (run: python -m src.benchmark.jobs ollama-default-probe)"]
    d = default["default_context_length"]
    lower, upper = context_band(d, num_ctx)
    note = band_note(d, num_ctx)
    if note:
        print(note)

    ps_api = _ps_entry_context(_read_json(os.path.join(results_dir, "ollama_ps_start.json")), variant)
    ps_col = parse_ps_column(_read_file(os.path.join(results_dir, "ollama_ps_start.txt")) or "", variant)
    show = _show_num_ctx(_read_file(os.path.join(results_dir, "ollama_show.txt")) or "")
    long_prompt = _read_json(os.path.join(results_dir, "ollama_long_prompt.json")) or {}
    tokens = long_prompt.get("prompt_tokens")
    server_ctx = long_prompt.get("server_context_length")

    missing = [name for name, value in (("ps_context_length", ps_api), ("ps_column_context", ps_col),
                                        ("server_context_length", server_ctx)) if value is None]
    ps_value = ps_api if ps_api is not None else ps_col
    path = "primary" if ps_value is not None and server_ctx is not None else "fallback"
    print(f"EVIDENCE model {model} path {path} missing {','.join(missing) or 'none'} D {d} band {lower}-{upper} "
          f"prompt_tokens {tokens}")

    problems = []
    for name, value in (("/api/ps context_length", ps_api), ("ollama ps CONTEXT column", ps_col)):
        if value is not None and value != num_ctx:
            problems.append(f"{name} of {variant} is {value}, not num_ctx {num_ctx}"
                            + (" (it equals the measured default: the window was not raised)" if value == d else ""))
    if show != num_ctx:
        problems.append(f"ollama_show.txt lists num_ctx {show}, expected {num_ctx}")
    if not isinstance(tokens, int):
        problems.append("ollama_long_prompt.json has no prompt_tokens")
    elif tokens < lower:
        problems.append(f"long-prompt prompt_tokens {tokens} below L {lower} (the window was not raised past the "
                        f"measured default {d}, or the prompt was cut)")
    elif tokens > upper:
        problems.append(f"long-prompt prompt_tokens {tokens} above U {upper} (95 % of num_ctx {num_ctx})")
    if long_prompt.get("D") is not None and long_prompt.get("D") != d:
        problems.append(f"ollama_long_prompt.json was sized from D {long_prompt.get('D')}, the measured default "
                        f"is now {d}")
    return problems


# ── Control tokens (Phase 3 review IN-01) ──

QWEN_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
PLACEHOLDER_CANARY = "CANARY000000000000DATA"


def _added_vocab() -> list:
    """Every added-vocabulary string of the Qwen tokenizer, loaded offline from the local cache."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer  # heavy; imported only here
    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    return list(tokenizer.get_added_vocab().keys())


def benchmark_texts() -> list:
    """Every text the 57 benchmark cases show a model: prompt, tool responses (placeholder canary), tool
    descriptions and the system prompt with the case's tools."""
    from src.agent_harness.prompting import build_messages
    from src.benchmark.injecagent import attack_cases, benign_cases, load_tool_specs
    specs = load_tool_specs()
    cases = benign_cases(specs)[:17] + attack_cases(20, 20, 0, specs)
    texts = []
    for case in cases:
        texts.append(case.prompt)
        texts.extend(case.tool_responses(PLACEHOLDER_CANARY).values())
        for tool in case.tools:
            texts.append(json.dumps(tool, ensure_ascii=False))
        texts.extend(m["content"] for m in build_messages(case.prompt, case.tools) if m.get("role") == "system")
    return texts


def control_token_matches(vocab: list, texts: list) -> list:
    """(string, text index) for every added-vocabulary string found in a text."""
    return [(s, i) for i, text in enumerate(texts) for s in vocab if s and s in text]


def check_control_tokens() -> list:
    try:
        vocab = _added_vocab()
    except Exception as e:  # missing package or cache: the check cannot run, which is itself a failure
        return [f"the Qwen tokenizer could not be loaded offline: {type(e).__name__}: {e}"]
    texts = benchmark_texts()
    if not texts or not vocab:
        return [f"nothing to scan (strings {len(vocab)}, texts {len(texts)})"]
    matches = control_token_matches(vocab, texts)
    print(f"CONTROL_TOKENS strings {len(vocab)} texts {len(texts)} matches {len(matches)}")
    return [f"added-vocabulary string {s!r} occurs in benchmark text #{i} (a Qwen-only confound; the "
            f"benchmark text is not edited)" for s, i in matches]


# ── Hosted re-run under the D-09 rule ──

HOSTED_EXPECTED = 342
HOSTED_DS_EXPECTED = 120


def check_hosted(prefix: str, models: Optional[list] = None, jobs_dir: str = DEFAULT_JOBS_DIR) -> tuple:
    """(problems, word): one HOSTED line per model with its label (complete, or "partial n=F ds n=K"), then
    word hosted-terminal-ok when every model is terminal, or when the freeze of stopping_rule.json has passed and
    no child of the loop is alive (any other state is then partial, stopped at the freeze); hosted-running
    otherwise. Problems: complete with fewer than 342 finished; ds first violated (ds finished plus pending ds
    infra keys below min(F, 120))."""
    from src.benchmark import jobs  # lazy: jobs imports this module
    models = list(models or jobs.HOSTED_MODELS)
    rule = _read_json(os.path.join(jobs_dir, jobs.STOPPING_RULE_FILE)) or {}
    freeze = jobs._parse_utc(rule.get("freeze_utc") or jobs.FREEZE_UTC)
    past = jobs._now() >= freeze
    loop_alive = jobs.find_loop("hosted", prefix, exclude_self=False) is not None
    problems, terminal, children = [], True, False
    for model in models:
        out = jobs.model_dir(prefix, model)
        state = jobs.read_hosted_state(out) if os.path.isdir(out) else {"state": "not-started", "launches": []}
        raw_path = os.path.join(out, "raw.jsonl")
        rows = [r for r in load_rows(raw_path) if r.get("model") == model] if os.path.exists(raw_path) else []
        finished = [r for r in rows if classify_outcome(r) != "llm_error"]
        ds_done = sum(1 for r in finished if r.get("kind") == "ds")
        ds_pending = sum(1 for r in rows if r.get("kind") == "ds" and classify_outcome(r) == "llm_error")
        n = len(finished)
        stops = [x for x in state.get("launches", []) if isinstance(x, dict) and x.get("quota_stop")]
        stop_text = ", ".join(f"{x.get('end')} retry {x.get('stated_retry_s')}" for x in stops) or "none"
        alive = os.path.isdir(out) and jobs.find_bench_child(out) is not None
        children = children or alive
        st = state.get("state")
        is_terminal = st in jobs.HOSTED_TERMINAL
        terminal = terminal and is_terminal
        label = "complete" if st == "complete" and n >= HOSTED_EXPECTED else f"partial n={n} ds n={ds_done}"
        if past and not is_terminal and not alive:
            label += " (stopped at the freeze)"
        print(f"HOSTED {model} state {st} finished {n}/{HOSTED_EXPECTED} ds {ds_done}/{HOSTED_DS_EXPECTED} "
              f"quota_stops {len(stops)} [{stop_text}] label {label}")
        if st == "complete" and n < HOSTED_EXPECTED:
            problems.append(f"{model} is marked complete with {n} finished episodes, below {HOSTED_EXPECTED}")
        if ds_done + ds_pending < min(n, HOSTED_DS_EXPECTED):
            problems.append(f"{model}: ds-first violated (ds finished {ds_done} + pending ds {ds_pending} below "
                            f"min({n}, {HOSTED_DS_EXPECTED}))")
    if terminal or (past and not loop_alive and not children):
        word = "hosted-terminal-ok"
    else:
        word = "hosted-running"
    return problems, word


# ── Secret scan (fixed regex, given paths only) ──

SECRET_RE = re.compile(r"gsk_[A-Za-z0-9]{20,}")


def _scan_files(paths: list) -> tuple:
    """(text files under the given paths, problems for paths that do not exist). Symlinks are not followed, so
    nothing outside the given paths is opened."""
    files, problems = [], []
    for path in paths:
        if os.path.islink(path):
            continue
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for dirpath, dirnames, names in os.walk(path):
                dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d)))
                files += [os.path.join(dirpath, n) for n in sorted(names)
                          if not os.path.islink(os.path.join(dirpath, n))]
        else:
            problems.append(f"{path} does not exist")
    return files, problems


def check_secret_scan(paths: list) -> tuple:
    """(problems, text files scanned). Each text file (no NUL byte in its first 8 KiB) is searched with the one
    fixed regex gsk_[A-Za-z0-9]{20,}; a match is reported as file:line only, never its text. No key is loaded."""
    files, problems = _scan_files(paths)
    scanned = 0
    for path in files:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        scanned += 1
        for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if SECRET_RE.search(line):
                problems.append(f"{path}:{lineno}")
    return problems, scanned


# ── Viva page check (plan 04-04) ──

VIVA_PAGE_NAME = "04-namespaces-seccomp-and-local-inference.md"
VIVA_README_STATUS = "written (Phase 4)"
VIVA_HEADINGS = ("## Part A", "## Part B", "## Part C", "## Part D", "## Limits of this phase")
VIVA_MIN_QUESTIONS = 16
VIVA_MIN_GLOSSARY = 30
VIVA_COUNT_COLUMNS = ("episodes_done", "episodes_expected")


def _count_on_line(line: str, count: str) -> bool:
    """`count` ("233/342") appears as a whole token on the line: not inside 1233/342 or 233/3420."""
    return re.search(rf"(?<![\w/.]){re.escape(count)}(?![\w/]|\.\d)", line) is not None


def _model_on_line(line: str, model: str) -> bool:
    """`model` appears as a whole name: groq-gpt-oss-20b is not found inside a longer model name."""
    return re.search(rf"(?<![\w.-]){re.escape(model)}(?![\w-]|\.\w)", line) is not None


def check_viva(page_path: str, csv_path: str, readme_path: str) -> list:
    """Problems with the Phase 4 viva page; empty means viva-numbers-ok.

    Completeness comes from the episodes_done and episodes_expected columns of the report's summary.csv
    (summarize() emits both keys and generate_report writes every key as a column). Each language-model row
    (provider not scripted) must be quoted on one line of the page as its name and "<done>/<expected>", so a
    count can never be credited to the wrong model. A summary.csv without the two columns is a pre-Phase-1
    report and fails outright."""
    problems = []
    with open(page_path, encoding="utf-8") as f:
        page = f.read()
    lines = page.splitlines()
    problems += [f"heading {h!r} is missing" for h in VIVA_HEADINGS if not any(ln.startswith(h) for ln in lines)]
    questions = sum(1 for ln in lines if ln.startswith("**Q"))
    if questions < VIVA_MIN_QUESTIONS:
        problems.append(f"{questions} question lines starting '**Q', at least {VIVA_MIN_QUESTIONS} are required")
    glossary = sum(1 for ln in lines if ln.startswith("- **"))
    if glossary < VIVA_MIN_GLOSSARY:
        problems.append(f"{glossary} glossary lines starting '- **', at least {VIVA_MIN_GLOSSARY} are required")
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, page, re.IGNORECASE):
            problems.append(f"secret-like text found (pattern {pattern!r})")
    for lineno, line in enumerate(lines, 1):
        if SECRET_RE.search(line):
            problems.append(f"key-shaped text found at {os.path.basename(page_path)}:{lineno}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not all(c in fields for c in VIVA_COUNT_COLUMNS):
        problems.append("summary.csv lacks episodes_done/episodes_expected (not a current-engine report)")
    else:
        llm_rows = [r for r in rows if (r.get("provider") or "") != CONTROL_PROVIDER]
        if not llm_rows:
            problems.append("summary.csv has no language-model row")
        for r in llm_rows:
            model = (r.get("model") or "").strip()
            done, expected = ((r.get(c) or "").strip() for c in VIVA_COUNT_COLUMNS)
            if not (model and done.isdigit() and expected.isdigit()):
                problems.append(f"summary.csv row {model or '?'} has no numeric episodes_done/episodes_expected")
                continue
            count = f"{done}/{expected}"
            if not any(_model_on_line(ln, model) for ln in lines):
                problems.append(f"the page does not name model {model}")
            elif not any(_model_on_line(ln, model) and _count_on_line(ln, count) for ln in lines):
                problems.append(f"the page does not quote {model} with {count} on one line")

    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()
    link = f"({VIVA_PAGE_NAME})"
    if not any(link in ln and VIVA_README_STATUS in ln for ln in readme.splitlines()):
        problems.append(f"the README has no row linking {VIVA_PAGE_NAME} with status {VIVA_README_STATUS!r}")
    return problems


def _finish(cmd: str, problems: list) -> int:
    if problems:
        for p in problems:
            print(f"FAIL {cmd}: {p}")
        return 1
    print(f"{cmd}-ok")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.run_checks", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("complete")
    p.add_argument("--results", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--expect", type=int, default=None)
    p.add_argument("--resolved-contains", default=None)
    p = sub.add_parser("default-context")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    sub.add_parser("control-tokens")
    p = sub.add_parser("truncation")
    p.add_argument("--results", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--compare", type=int, default=None)
    p = sub.add_parser("ollama-evidence")
    p.add_argument("--results", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p = sub.add_parser("hosted")
    p.add_argument("--prefix", required=True)
    p.add_argument("--models", default=None)
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p = sub.add_parser("secret-scan")
    p.add_argument("--paths", nargs="+", required=True)
    p = sub.add_parser("viva")
    p.add_argument("--page", required=True)
    p.add_argument("--report-csv", required=True)
    p.add_argument("--readme", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "viva":
        problems = check_viva(args.page, args.report_csv, args.readme)
        if problems:
            for p in problems:
                print(f"FAIL viva: {p}")
            return 1
        print("viva-numbers-ok")
        return 0

    if args.cmd == "hosted":
        models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        problems, word = check_hosted(args.prefix, models, args.jobs_dir)
        if problems:
            for p in problems:
                print(f"FAIL hosted: {p}")
            return 1
        print(word)
        return 0
    if args.cmd == "secret-scan":
        problems, scanned = check_secret_scan(args.paths)
        if problems:
            for p in problems:
                print(f"FAIL secret-scan: {p}")
            return 1
        print(f"secret-scan-ok files {scanned}")
        return 0
    if args.cmd == "truncation":
        problems, unknown = check_truncation(args.results, args.model, args.num_ctx, args.compare)
        if unknown and not problems:
            print("truncation-unknown")
            return 0
        return _finish("truncation", problems)
    if args.cmd == "ollama-evidence":
        return _finish("ollama-evidence", check_ollama_evidence(args.results, args.model, args.jobs_dir))

    if args.cmd == "complete":
        return _finish("complete", check_complete(args.results, args.model, args.expect, args.resolved_contains))
    if args.cmd == "default-context":
        return _finish("default-context", check_default_context(args.jobs_dir))
    if args.cmd == "control-tokens":
        return _finish("control-tokens", check_control_tokens())
    return 2


if __name__ == "__main__":
    sys.exit(main())
