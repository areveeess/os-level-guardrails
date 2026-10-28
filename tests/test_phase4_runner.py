"""
Tests for the runner fixes a multi-hour Phase 4 run needs (plan 04-05 Task 2):
- IN-02: a respond action whose message is null, a number or an object no longer aborts a run;
- a context overflow of a local model is a finished episode (format_failure, or attacked when
  the attack was already attempted), never re-queued and counted as done by the report;
- --priority-kinds ds puts every data-stealing episode first without changing the case set.
Providers are scripted (no network); the protected arm needs Linux namespaces.
"""

import json
import shutil
import sys

import pytest

import run_benchmark as cli
from src.benchmark import runner as runner_module
from src.benchmark.injecagent import attack_cases, benign_cases, load_tool_specs
from src.benchmark.outcomes import classify_llm_error, classify_outcome
from src.benchmark.report import generate_report
from src.benchmark.runner import ARMS, _work_queue
from src.llm.base import LLMProvider, LLMResponse

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("unshare") is None,
                                reason="requires Linux namespaces")

OVERFLOW = ("ContextOverflowError: context_overflow: the prompt leaves no room in the configured context "
            "window for a reply")


@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()


class ScriptedProvider(LLMProvider):
    """Replies with `reply` on every call, or returns `error` on every call."""

    def __init__(self, reply=None, error=None, name="m"):
        self.name, self.reply, self.error = name, reply, error
        self.calls = 0

    def generate(self, messages, temperature=0.0, max_tokens=512):
        self.calls += 1
        if self.error:
            return LLMResponse(text="", latency_ms=1.0, model=self.name, error=self.error,
                               finish_reason="context_overflow")
        return LLMResponse(text=self.reply, latency_ms=1.0, model=self.name)


def use(monkeypatch, provider):
    monkeypatch.setattr(runner_module, "get_provider", lambda _name: provider)


def rows(out):
    return [json.loads(line) for line in open(out / "raw.jsonl", encoding="utf-8") if line.strip()]


# ── IN-02: non-text respond messages ──

@pytest.mark.parametrize("message,expected", [(None, ""), ({"a": 1}, '{"a": 1}'), (7, "7"), ("done", "done")])
def test_non_text_respond_message_never_aborts_a_run(tmp_path, monkeypatch, specs, message, expected):
    use(monkeypatch, ScriptedProvider(reply=json.dumps({"action": "respond", "message": message})))
    out = tmp_path / "r"
    runner_module.run_benchmark(["m"], benign_cases(specs)[:2], str(out), runs=1, arms=("unprotected",),
                                log=lambda *_: None)
    written = rows(out)
    assert len(written) == 2
    assert all(isinstance(r["final_message"], str) and r["final_message"] == expected for r in written)


def test_agent_loop_stores_final_message_as_text():
    from src.agent_harness.executors import MockExecutor
    from src.agent_harness.environment import build_environment
    from src.agent_harness.loop import AgentLoop
    provider = ScriptedProvider(reply='{"action": "respond", "message": {"k": "ü"}}')
    episode = AgentLoop(provider, MockExecutor(build_environment()), max_steps=2).run("hi")
    assert episode.final_message == '{"k": "ü"}' and episode.stop_reason == "responded"


# ── Context overflow: a finished outcome, never retried ──

def test_classify_llm_error_has_a_context_overflow_class():
    assert classify_llm_error(OVERFLOW) == "context_overflow"
    assert classify_llm_error("ValueError: context_overflow: the prompt is 9000 characters") == "context_overflow"
    assert classify_llm_error("InternalServerError: Error code: 500") == "infra"
    assert classify_llm_error("content_filter: response blocked by the safety system") == "provider_refusal"
    assert classify_llm_error(None) is None and classify_llm_error("") is None


def test_classify_outcome_of_a_context_overflow():
    attack = {"kind": "ds", "llm_error": OVERFLOW, "steps": 1, "actions": []}
    assert classify_outcome({**attack, "attempted": False}) == "format_failure"
    assert classify_outcome({**attack, "attempted": True, "actions": [{"action": "call_tool"}]}) == "attacked"
    assert classify_outcome({"kind": "benign", "llm_error": OVERFLOW, "steps": 0, "actions": []}) == "format_failure"
    # a context overflow that also names a refusal word is still an overflow, not a refusal
    assert classify_outcome({"kind": "benign", "llm_error": OVERFLOW + " (safety margin)", "steps": 0,
                             "actions": []}) == "format_failure"


def test_overflow_is_finished_once_and_never_requeued(tmp_path, monkeypatch, specs):
    provider = ScriptedProvider(error=OVERFLOW)
    use(monkeypatch, provider)
    out = tmp_path / "r"
    logs = []
    cases = benign_cases(specs)[:2]
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=logs.append)
    first = rows(out)
    assert len(first) == 2 and len({(r["case_id"], r["run"], r["arm"]) for r in first}) == 2
    assert not [line for line in logs if "[requeue]" in line]
    runner_module.run_benchmark(["m"], cases, str(out), runs=1, arms=("unprotected",), log=logs.append)
    assert len(rows(out)) == 2                     # the resume counts them as done
    assert all(classify_outcome(r) == "format_failure" for r in first)


@linux_only
def test_report_counts_overflow_episodes_as_done(tmp_path, monkeypatch, specs):
    use(monkeypatch, ScriptedProvider(error=OVERFLOW))
    out = tmp_path / "r"
    runner_module.run_benchmark(["m"], benign_cases(specs)[:1], str(out), runs=1, log=lambda *_: None)
    summary = generate_report(str(out), charts=False, expected=2)
    assert summary[0]["episodes_done"] == 2 and summary[0]["status"] == "complete"


# ── ds-first queue (D-02) ──

def _ids(queue):
    return [(c.id, r, a) for c, r, a in queue]


def test_default_queue_is_unchanged_by_the_priority_parameter(specs):
    cases = benign_cases(specs) + attack_cases(20, 20, seed=0, specs=specs)
    assert _ids(_work_queue(cases, 3, ARMS, 0, priority_kinds=())) == _ids(_work_queue(cases, 3, ARMS, 0))


def test_ds_first_queue_keeps_pairs_run_major_and_the_same_items(specs):
    cases = benign_cases(specs) + attack_cases(20, 20, seed=0, specs=specs)
    default = _work_queue(cases, 3, ARMS, 0)
    queue = _work_queue(cases, 3, ARMS, 0, priority_kinds=("ds",))
    assert len(queue) == 342 and sorted(_ids(queue)) == sorted(_ids(default))
    kinds = [c.kind for c, _, _ in queue]
    assert kinds[:120] == ["ds"] * 120 and "ds" not in kinds[120:]
    for block in (queue[:120], queue[120:]):
        runs = [r for _, r, _ in block]
        assert runs == sorted(runs)                                    # run-major inside each block
        for i in range(0, len(block), 2):                              # both arms of a (case, run) adjacent
            (c1, r1, a1), (c2, r2, a2) = block[i], block[i + 1]
            assert c1.id == c2.id and r1 == r2 and (a1, a2) == ARMS
    # a stable partition: the relative order inside each block is the default order
    assert _ids(queue[:120]) == [x for x in _ids(default) if x[0].startswith("ds")]
    assert _ids(queue[120:]) == [x for x in _ids(default) if not x[0].startswith("ds")]


# ── run_benchmark.py --priority-kinds ──

def _cli(monkeypatch, out, *extra):
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--models", "m", "--runs", "1", "--dh", "1", "--ds", "1",
                                      "--benign", "1", "--arms", "unprotected", "--resume", str(out), *extra])
    cli.main()


def test_cli_records_priority_kinds_and_resumes_with_another_order(tmp_path, monkeypatch, specs):
    use(monkeypatch, ScriptedProvider(reply='{"action": "respond", "message": "done"}'))
    monkeypatch.setattr(cli, "generate_report", lambda *a, **k: [])
    out = tmp_path / "r"
    out.mkdir()
    _cli(monkeypatch, out, "--priority-kinds", "ds")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["priority_kinds"] == ["ds"]
    assert len(rows(out)) == 3 and rows(out)[0]["kind"] == "ds"
    _cli(monkeypatch, out)                                   # a different order resumes the same directory
    assert len(rows(out)) == 3
    with pytest.raises(SystemExit) as exc:                   # a different seed is still refused
        _cli(monkeypatch, out, "--seed", "1")
    assert exc.value.code == 2


def test_cli_rejects_an_unknown_priority_kind(tmp_path, monkeypatch, capsys):
    def must_not_run(*_a, **_k):
        raise AssertionError("the benchmark must not start with invalid arguments")
    monkeypatch.setattr(cli, "run_benchmark", must_not_run)
    out = tmp_path / "r"
    out.mkdir()
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, out, "--priority-kinds", "ds,exfil")
    assert exc.value.code == 2
    assert "priority-kinds" in capsys.readouterr().err
