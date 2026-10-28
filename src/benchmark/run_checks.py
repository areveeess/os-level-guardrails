"""
Run checks — mechanical pass/fail checks over Phase 4 result directories
========================================================================
Every Phase 4 claim ("342/342 finished", "no duplicate after a resume", "the stamps
match the engine that ran") is a command that prints an ok line or a FAIL line, so a
verifier never has to read raw.jsonl by eye:

  python -m src.benchmark.run_checks complete --results DIR --model M [--expect N] [--resolved-contains TEXT]
  python -m src.benchmark.run_checks default-context [--jobs-dir D]
  python -m src.benchmark.run_checks control-tokens

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
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Optional

from src.benchmark.outcomes import classify_llm_error, classify_outcome
from src.benchmark.report import load_rows, read_manifest
from src.benchmark.scoring import SCORING_VERSION
from src.llm.registry import PROJECT_ROOT, load_model_config

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
    args = parser.parse_args(argv)

    if args.cmd == "complete":
        return _finish("complete", check_complete(args.results, args.model, args.expect, args.resolved_contains))
    if args.cmd == "default-context":
        return _finish("default-context", check_default_context(args.jobs_dir))
    if args.cmd == "control-tokens":
        return _finish("control-tokens", check_control_tokens())
    return 2


if __name__ == "__main__":
    sys.exit(main())
