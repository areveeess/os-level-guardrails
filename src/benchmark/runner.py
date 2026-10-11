"""
Benchmark runner — every model × case × run × arm
=================================================
  unprotected arm: ToolBroker(no capabilities) → MockExecutor
                   (nothing executes; we see what the agent *would* do)
  protected arm:   ToolBroker(task capabilities) → PolicyEngine
                   → SandboxExecutor (namespaces + private fs + seccomp)

Results are appended to <out>/raw.jsonl one episode per line, so an
interrupted run resumes where it stopped.
"""

import itertools
import json
import os
import re
import time

from src.agent_harness.environment import build_environment
from src.agent_harness.executors import GuardedExecutor, MockExecutor
from src.agent_harness.loop import AgentLoop
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark.injecagent import BenchmarkCase, new_canary
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


def run_benchmark(models: list, cases: list, out_dir: str, runs: int = 3, arms=ARMS,
                  max_steps: int = 5, sleep_s: float = 0.0, log=print) -> str:
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("llm_error"):  # episodes that hit an API error are retried
                    done.add((r["model"], r["case_id"], r["run"], r["arm"]))

    families = {name: cfg.get("family", "") for name, cfg in load_model_config()["models"].items()}
    policy = PolicyEngine(log_file=os.path.join(out_dir, "decisions.jsonl"), verbose=False)
    total = len(models) * len(cases) * runs * len(arms)
    count = len(done)

    for model in models:
        try:
            provider = get_provider(model)
        except Exception as e:  # missing key, missing checkpoint, unknown name
            log(f"[skip] {model}: {e}")
            continue
        for case, run, arm in itertools.product(cases, range(runs), arms):
            key = (model, case.id, run, arm)
            if key in done:
                continue
            try:
                episode, score = _run_with_backoff(provider, case, arm, policy, max_steps, log)
            except QuotaExhausted as e:
                log(f"[stop] {model}: quota exhausted ({e}). Re-run with --resume {out_dir} later.")
                break

            row = {
                "model": model, "family": families.get(model, ""),
                "case_id": case.id, "kind": case.kind, "attack_type": case.attack_type,
                "run": run, "arm": arm, **score,
                "actions": [s.action for s in episode.steps if s.action],
                "final_message": episode.final_message[:500],
            }
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            count += 1
            status = ("ATTEMPTED" if score["attempted"] else "ok") + \
                     (" SUCCEEDED" if score["succeeded"] else "") + \
                     (f" blocked:{','.join(score['blocked_by'])}" if score["blocked_by"] else "") + \
                     (f" ERROR:{score['llm_error'][:60]}" if score["llm_error"] else "")
            log(f"[{count}/{total}] {model:20} {case.id:16} run{run} {arm:11} {status}")
            if sleep_s:
                time.sleep(sleep_s)
    return raw_path
