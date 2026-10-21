"""
Benchmark runner — every model × case × run × arm
=================================================
  unprotected arm: ToolBroker(no capabilities) → MockExecutor
                   (nothing executes; we see what the agent *would* do)
  protected arm:   ToolBroker(task capabilities) → PolicyEngine
                   → SandboxExecutor (namespaces + private fs + seccomp)

Results are appended to <out>/raw.jsonl one episode per line, so an
interrupted run resumes where it stopped. The queue is run-major with both arms
of a (case, run) adjacent, so a run cut short by a quota still holds complete
pairs of every attack kind. An episode that hit an infrastructure error is
re-run from scratch (one bounded pass in the same invocation, and again on every
resume); a provider refusal is an outcome and is never retried. Every row is
stamped with the code commit and policy hash, and manifest.json records the
planned size and one entry per invocation.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from datetime import datetime

from src.agent_harness.environment import build_environment
from src.agent_harness.executors import GuardedExecutor, MockExecutor
from src.agent_harness.loop import AgentLoop
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark.injecagent import PROJECT_ROOT, BenchmarkCase, new_canary
from src.benchmark.outcomes import classify_llm_error
from src.benchmark.scoring import score_episode
from src.llm import get_provider
from src.llm.registry import load_model_config
from src.policy_engine.engine import PolicyEngine

ARMS = ("unprotected", "protected")
RATE_LIMIT_BACKOFF_S = (20, 45, 90, 180, 300, 600, 600, 600)
MAX_WAIT_S = 1800  # longer requested waits mean a daily quota: stop the model, resume later


def build_executor(arm: str, case: BenchmarkCase, canary: str, policy: PolicyEngine) -> ToolBroker:
    env = build_environment()
    responses = case.tool_responses(canary)
    known = {t["name"] for t in case.tools}
    if arm == "unprotected":
        return ToolBroker(MockExecutor(env), responses, known_tools=known)
    from src.sandbox.executor import SandboxExecutor  # Linux only
    inner = GuardedExecutor(policy, SandboxExecutor(env, policy=policy))
    return ToolBroker(inner, responses, known_tools=known, capabilities=case.capabilities, policy=policy)


def _is_rate_limited(error: str) -> bool:
    return bool(error) and ("429" in error or "RateLimit" in error or "rate limit" in error.lower())


def _requested_wait(error: str):
    """Seconds the API asked us to wait ('retry in 15h19m41s', 'try again in 7.5s'), if stated."""
    m = re.search(r"(?:retry|try again) in ((?:\d+h)?(?:\d+m)?[\d.]+s)", error or "")
    if not m:
        return None
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)([hms])", m.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


class QuotaExhausted(Exception):
    pass


class ResumeMismatch(ValueError):
    """--resume was given a plan that differs from the one the directory was started with."""


def _validate_arms(arms) -> tuple:
    """build_executor treats any arm other than 'unprotected' as the protected stack, so a
    typo would silently run the sandbox under a mislabeled arm. Refuse it up front."""
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError(f"invalid --arms {list(arms)!r}: expected a non-empty list without "
                         f"duplicates drawn from {list(ARMS)}")
    return arms


def _case_set_sha256(cases: list) -> str:
    return hashlib.sha256("\n".join(sorted(c.id for c in cases)).encode("utf-8")).hexdigest()


def _plan(cases: list, runs: int, arms, max_steps: int, extra: dict, queue_seed: int) -> dict:
    """What a results directory is a run *of*: everything that must not change on resume."""
    return {
        "expected_episodes_per_model": len(cases) * runs * len(arms),
        "cases": dict(Counter(c.kind for c in cases)),
        "case_set_sha256": _case_set_sha256(cases),
        "runs": runs,
        "arms": list(arms),
        "max_steps": max_steps,
        "queue_seed": queue_seed,
        "extra": dict(extra or {}),
    }


def _read_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):  # a corrupt manifest is rewritten, never fatal
        return {}
    return loaded if isinstance(loaded, dict) else {}


_PLAN_LABELS = {"runs": "runs", "arms": "arms", "max_steps": "max_steps", "cases": "cases",
                "case_set_sha256": "cases (different case ids)", "queue_seed": "seed (queue order)"}


def _plan_differences(previous: dict, plan: dict) -> list:
    """Human-readable differences between the stored manifest and this invocation. Only
    fields the stored manifest has are compared, so older manifests still resume."""
    diffs = [f"{_PLAN_LABELS[k]}: directory has {previous[k]!r}, this invocation has {plan[k]!r}"
             for k in _PLAN_LABELS if k in previous and previous[k] != plan[k]]
    old = (previous.get("extra") or {}).get("seed")
    new = plan["extra"].get("seed")
    if old is not None and new is not None and old != new:
        diffs.append(f"seed: directory has {old!r}, this invocation has {new!r}")
    return diffs


def _git_commit() -> str:
    """HEAD commit of the checkout the results were produced by, or 'unknown'.

    Fixed argv, no shell, bounded time: this runs on every benchmark start and must
    neither hang nor interpret anything."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"


def _git_dirty():
    """True/False whether tracked files differ from HEAD; None when git cannot say."""
    try:
        done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(done.stdout.strip()) if done.returncode == 0 else None


def _scan_raw(raw_path: str) -> tuple:
    """(done keys, undecodable line count) from an existing raw.jsonl, for --resume.

    A run killed mid-write leaves a truncated last line; the report loader skips such
    lines, and so must the one place that has to survive that kill. A line counts as
    undecodable when it is not a JSON object with the four key fields. Finished
    episodes, or ones ended by a provider refusal (an outcome, never retried), are
    done; infrastructure-errored ones stay pending and are re-run from scratch.
    """
    done, bad = set(), 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                key = (r["model"], r["case_id"], r["run"], r["arm"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if classify_llm_error(r.get("llm_error")) != "infra":
                done.add(key)
    return done, bad


def _ensure_trailing_newline(raw_path: str) -> None:
    """Make sure the next appended row starts on its own line, not glued to a fragment."""
    with open(raw_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


def run_episode(provider, case: BenchmarkCase, arm: str, policy: PolicyEngine, max_steps: int):
    canary = new_canary()
    executor = build_executor(arm, case, canary, policy)
    try:
        episode = AgentLoop(provider, executor, max_steps=max_steps, tools=case.tools).run(case.prompt)
        return episode, score_episode(case, episode, executor, canary)
    finally:
        executor.close()


def _run_with_backoff(provider, case, arm, policy, max_steps, log):
    """Run an episode, waiting out short rate limits; raise QuotaExhausted on long ones."""
    episode, score = run_episode(provider, case, arm, policy, max_steps)
    for fallback in RATE_LIMIT_BACKOFF_S:
        if not _is_rate_limited(score["llm_error"]):
            break
        wait = _requested_wait(score["llm_error"])
        if wait is not None and wait > MAX_WAIT_S:
            raise QuotaExhausted(f"API asks to retry in {wait / 3600:.1f}h")
        wait = fallback if wait is None else wait + 1
        log(f"  rate limited on {provider.name}; waiting {wait:.0f}s")
        time.sleep(wait)
        episode, score = run_episode(provider, case, arm, policy, max_steps)
    if _is_rate_limited(score["llm_error"]):
        raise QuotaExhausted("still rate limited after backoff")
    return episode, score


def _work_queue(cases: list, runs: int, arms, seed: int) -> list:
    """(case, run, arm) items, run-major. Within a run the cases are shuffled with a
    generator seeded from (seed, run) and each case's arms stay adjacent, so a run
    stopped by a quota still holds both attack kinds and complete (case, run) pairs."""
    queue = []
    for run in range(runs):
        order = list(cases)
        random.Random(seed * 1_000_003 + run).shuffle(order)
        queue += [(case, run, arm) for case in order for arm in arms]
    return queue


def _write_manifest(out_dir: str, *, models: list, plan: dict, previous: dict,
                    code_commit: str, code_dirty, policy_sha256: str) -> None:
    """Create or update <out_dir>/manifest.json: the planned size of the run and one
    entry per invocation, appended (never replaced). `plan` was already checked against
    `previous` by the caller, so an existing directory keeps the plan it was started with.
    A fixed field list: no environment values, keys or URLs."""
    path = os.path.join(out_dir, "manifest.json")
    known = previous.get("models") if isinstance(previous.get("models"), list) else []
    invocations = previous.get("invocations") if isinstance(previous.get("invocations"), list) else []
    manifest = {
        "schema": 1,
        **plan,
        "models": list(dict.fromkeys([*known, *models])),
        "invocations": invocations + [{
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "code_commit": code_commit, "code_dirty": code_dirty,
            "policy_sha256": policy_sha256, "models": list(models),
        }],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def run_benchmark(models: list, cases: list, out_dir: str, runs: int = 3, arms=ARMS,
                  max_steps: int = 5, sleep_s: float = 0.0, log=print, queue_seed: int = 0,
                  requeue_passes: int = 1, manifest_extra: dict = None) -> str:
    arms = _validate_arms(arms)
    plan = _plan(cases, runs, arms, max_steps, manifest_extra, queue_seed)
    previous = _read_manifest(out_dir)
    differences = _plan_differences(previous, plan) if previous else []
    if differences:
        raise ResumeMismatch(
            f"{out_dir} was started with a different plan; resume it with the same arguments "
            f"(or start a new directory):\n  " + "\n  ".join(differences))
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        done, undecodable = _scan_raw(raw_path)
        if undecodable:
            log(f"[resume] skipped {undecodable} undecodable line(s) in raw.jsonl (a run killed mid-write?)")
        _ensure_trailing_newline(raw_path)

    configs = load_model_config()["models"]
    families = {name: cfg.get("family", "") for name, cfg in configs.items()}
    policy = PolicyEngine(log_file=os.path.join(out_dir, "decisions.jsonl"), verbose=False)
    code_commit, code_dirty = _git_commit(), _git_dirty()
    _write_manifest(out_dir, models=models, plan=plan, previous=previous,
                    code_commit=code_commit, code_dirty=code_dirty,
                    policy_sha256=policy.policy_sha256)
    queue = _work_queue(cases, runs, arms, queue_seed)
    total = len(models) * len(cases) * runs * len(arms)
    count = len(done)

    def record(model, cfg, case, run, arm, episode, score) -> str:
        row = {
            "model": model, "family": families.get(model, ""),
            "case_id": case.id, "kind": case.kind, "attack_type": case.attack_type,
            "run": run, "arm": arm, **score,
            "actions": [s.action for s in episode.steps if s.action],
            "final_message": episode.final_message[:500],
            # Provenance (EVAL-06): enough to tell resumed, drifted or truncated runs apart
            "code_commit": code_commit, "code_dirty": code_dirty,
            "policy_sha256": policy.policy_sha256,
            "provider": cfg.get("provider", "unknown"), "config_model": cfg.get("model"),
            "resolved_model": next((s.resolved_model for s in episode.steps if s.resolved_model), None),
            "finish_reasons": [s.finish_reason for s in episode.steps],
            "prompt_tokens": [s.prompt_tokens for s in episode.steps],
            "completion_tokens": [s.completion_tokens for s in episode.steps],
        }
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return ("ATTEMPTED" if score["attempted"] else "ok") + \
               (" SUCCEEDED" if score["succeeded"] else "") + \
               (f" blocked:{','.join(score['blocked_by'])}" if score["blocked_by"] else "") + \
               (f" ERROR:{score['llm_error'][:60]}" if score["llm_error"] else "")

    for model in models:
        try:
            provider = get_provider(model)
        except Exception as e:  # missing key, missing checkpoint, unknown name
            log(f"[skip] {model}: {e}")
            continue
        cfg = configs.get(model, {})
        errored = []      # infra-errored items awaiting the bounded re-queue
        stopped = False
        for case, run, arm in queue:
            if (model, case.id, run, arm) in done:
                continue
            try:
                episode, score = _run_with_backoff(provider, case, arm, policy, max_steps, log)
            except QuotaExhausted as e:
                log(f"[stop] {model}: quota exhausted ({e}). Re-run with --resume {out_dir} later.")
                stopped = True
                break
            status = record(model, cfg, case, run, arm, episode, score)
            count += 1
            log(f"[{count}/{total}] {model:20} {case.id:16} run{run} {arm:11} {status}")
            if classify_llm_error(score["llm_error"]) == "infra":
                errored.append((case, run, arm))
            if sleep_s:
                time.sleep(sleep_s)

        # Bounded re-queue: a fresh episode (new executor and canary) for each infra error.
        # Never after a quota stop (it would burn what is left), and the good sibling arm
        # is not re-run: the report pairs it with the re-run of the errored arm.
        for _ in range(0 if stopped else requeue_passes):
            if not errored:
                break
            again = []
            for case, run, arm in errored:
                try:
                    episode, score = _run_with_backoff(provider, case, arm, policy, max_steps, log)
                except QuotaExhausted as e:
                    log(f"[stop] {model}: quota exhausted ({e}). Re-run with --resume {out_dir} later.")
                    stopped = True
                    break
                status = record(model, cfg, case, run, arm, episode, score)
                log(f"[requeue] {model} {case.id} run{run} {arm} {status}")
                if classify_llm_error(score["llm_error"]) == "infra":
                    again.append((case, run, arm))
                if sleep_s:
                    time.sleep(sleep_s)
            if stopped:
                break
            errored = again
    return raw_path
