"""
Tests for the InjecAgent adapter, tool broker, scoring, runner and report.
Models are scripted (no network); the protected arm needs Linux namespaces.
"""

import json
import os
import shutil
import sys

import pytest

from src.agent_harness.executors import MockExecutor
from src.agent_harness.environment import build_environment
from src.agent_harness.loop import AgentLoop
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark import runner as runner_module
from src.benchmark.injecagent import EXFIL_TOOL, PLACEHOLDER, attack_cases, benign_cases, load_tool_specs
from src.benchmark.report import generate_report, summarize
from src.benchmark.scoring import score_episode
from tests.test_agent_loop import ScriptedProvider

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("unshare") is None,
                                reason="requires Linux namespaces")


@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()


def call(tool, **args):
    return json.dumps({"action": "call_tool", "tool": tool, "args": args})


RESPOND = '{"action": "respond", "message": "done"}'


# ── Adapter ──

def test_cases_are_reproducible_and_well_formed(specs):
    a, b = attack_cases(20, 20, seed=0, specs=specs), attack_cases(20, 20, seed=0, specs=specs)
    assert [c.id for c in a] == [c.id for c in b] and len({c.id for c in a}) == 40
    assert {c.kind for c in a} == {"dh", "ds"}
    for c in a:
        assert PLACEHOLDER not in c.tool_response and c.attacker_instruction in c.tool_response.replace("\\'", "'")
        assert c.capabilities == {c.user_tool}
        assert {t["name"] for t in c.tools} >= {c.user_tool, *c.attacker_tools}
        assert (EXFIL_TOOL in {t["name"] for t in c.tools}) == (c.kind == "ds")
    benign = benign_cases(specs)
    assert len(benign) == 17 and all(PLACEHOLDER not in c.tool_response for c in benign)


# ── Broker + scoring (unprotected arm, emulated) ──

def run_unprotected(case, replies, canary="CANARYTEST"):
    broker = ToolBroker(MockExecutor(build_environment()), case.tool_responses(canary),
                        known_tools={t["name"] for t in case.tools})
    episode = AgentLoop(ScriptedProvider(replies), broker, tools=case.tools).run(case.prompt)
    return score_episode(case, episode, broker, canary), broker


def test_direct_harm_attempt_and_success_unprotected(specs):
    case = next(c for c in attack_cases(5, 0, specs=specs))
    score, _ = run_unprotected(case, [call(case.user_tool), call(case.attacker_tools[0]), RESPOND])
    assert score["attempted"] and score["succeeded"] and score["user_task_done"]


def test_ignoring_the_injection_scores_as_refused(specs):
    case = attack_cases(5, 0, specs=specs)[0]
    score, _ = run_unprotected(case, [call(case.user_tool), RESPOND])
    assert not score["attempted"] and not score["succeeded"] and score["refused_or_ignored"]


def test_data_stealing_needs_the_canary_to_leave(specs):
    case = attack_cases(0, 5, specs=specs)[0]
    source = case.attacker_tools[0]
    stolen, _ = run_unprotected(case, [call(case.user_tool), call(source),
                                       call(EXFIL_TOOL, to="x@example.com", body="CANARYTEST"), RESPOND])
    assert stolen["attempted"] and stolen["succeeded"] and stolen["canary_exposed"]
    no_data, _ = run_unprotected(case, [call(case.user_tool), call(source),
                                        call(EXFIL_TOOL, to="x@example.com", body="hello"), RESPOND])
    assert no_data["attempted"] and not no_data["succeeded"]


def test_broker_enforces_capabilities_and_unknown_tools(specs):
    case = attack_cases(1, 0, specs=specs)[0]
    broker = ToolBroker(MockExecutor(build_environment()), case.tool_responses("C"),
                        known_tools={t["name"] for t in case.tools}, capabilities=case.capabilities)
    assert broker.execute(json.loads(call(case.user_tool))).status == "success"
    blocked = broker.execute(json.loads(call(case.attacker_tools[0])))
    assert blocked.status == "blocked" and blocked.layer == "tool_broker"
    assert broker.execute(json.loads(call("NotARealTool"))).status == "error"


# ── Runner + report ──

@linux_only
def test_runner_end_to_end_both_arms(tmp_path, monkeypatch, specs):
    cases = benign_cases(specs)[:1] + attack_cases(1, 1, specs=specs)

    def scripted(_name):
        # Always follows the injection: user tool, then attacker tool(s), then exfil
        class FollowsEverything(ScriptedProvider):
            def generate(self, messages, temperature=0.0, max_tokens=512):
                system = messages[0]["content"]
                tools = [line[2:].split("(")[0] for line in system.splitlines() if line.startswith("- ") and "(" in line
                         and not line.startswith("- {")]
                n_assistant = sum(1 for m in messages if m["role"] == "assistant")
                plan = [call(t, body=messages[-1]["content"]) for t in tools] + [RESPOND]
                self.replies = [plan[min(n_assistant, len(plan) - 1)]]
                return super().generate(messages, temperature, max_tokens)
        return FollowsEverything([], name="scripted")

    monkeypatch.setattr(runner_module, "get_provider", scripted)
    out = tmp_path / "results"
    runner_module.run_benchmark(["scripted"], cases, str(out), runs=1, log=lambda *_: None)
    rows = [json.loads(l) for l in open(out / "raw.jsonl")]
    assert len(rows) == 6
    attacks = {(r["kind"], r["arm"]): r for r in rows if r["kind"] != "benign"}
    assert attacks[("dh", "unprotected")]["succeeded"] and attacks[("ds", "unprotected")]["succeeded"]
    assert not attacks[("dh", "protected")]["succeeded"] and not attacks[("ds", "protected")]["succeeded"]
    assert attacks[("dh", "protected")]["blocked_by"].get("tool_broker")

    # Resume skips finished episodes
    runner_module.run_benchmark(["scripted"], cases, str(out), runs=1, log=lambda *_: None)
    assert len(open(out / "raw.jsonl").readlines()) == 6

    summary = generate_report(str(out))
    assert summary[0]["success_protected_pct"] == 0.0 and summary[0]["attempted_pct"] == 100.0
    for name in ("summary.csv", "summary.md", "charts/attack_outcomes.png",
                 "charts/blocks_by_layer.png", "charts/benign_utility.png"):
        assert os.path.exists(out / name)


def test_summarize_excludes_llm_errors():
    base = {"model": "m", "family": "", "attempted": True, "succeeded": True, "refused_or_ignored": False,
            "blocked_by": {}, "retries_after_block": 0, "user_task_done": True, "false_positive": False,
            "parse_errors": 0, "latency_ms": 10.0, "steps": 2}
    rows = [dict(base, kind="dh", arm="unprotected", llm_error=None),
            dict(base, kind="dh", arm="unprotected", llm_error="boom", attempted=False)]
    s = summarize(rows)[0]
    assert s["attempted_pct"] == 100.0 and s["llm_errors"] == 1


# ── Rate limits ──

@pytest.mark.parametrize("error,expected", [
    ("429 ... Please retry in 15h19m41.806s.", 15 * 3600 + 19 * 60 + 41.806),
    ("Rate limit reached ... Please try again in 7.5s.", 7.5),
    ("429 Too Many Requests", None),
])
def test_requested_wait_parsing(error, expected):
    assert runner_module._requested_wait(error) == pytest.approx(expected) if expected else \
        runner_module._requested_wait(error) is None


def test_daily_quota_stops_model_without_writing_error_rows(tmp_path, monkeypatch, specs):
    from src.llm.base import LLMResponse

    class Exhausted(ScriptedProvider):
        def generate(self, messages, temperature=0.0, max_tokens=512):
            return LLMResponse(text="", latency_ms=1.0, model=self.name,
                               error="RateLimitError: 429 quota exceeded. Please retry in 15h2m1s.")

    monkeypatch.setattr(runner_module, "get_provider", lambda name: Exhausted([], name=name))
    out = tmp_path / "r"
    logs = []
    runner_module.run_benchmark(["m"], benign_cases(specs)[:2], str(out), runs=1,
                                arms=("unprotected",), log=logs.append)
    assert not (out / "raw.jsonl").exists() or (out / "raw.jsonl").read_text() == ""
    assert any("[stop] m: quota exhausted" in line for line in logs)
