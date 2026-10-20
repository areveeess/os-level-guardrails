"""
Tests for the benchmark work queue and run bookkeeping (EVAL-05 runner side, EVAL-06):
a paired, interleaved, seeded queue; a bounded re-queue of infrastructure-errored
episodes; provider refusals kept as outcomes and never retried; and a manifest.json
per results directory that the report uses for its completeness denominator.
Providers are scripted (no network); the protected arm needs Linux namespaces.
"""

import json
import shutil
import sys

import pytest

from src.benchmark import runner as runner_module
from src.benchmark.injecagent import attack_cases, benign_cases, load_tool_specs
from src.benchmark.report import generate_report
from src.benchmark.runner import ARMS, _work_queue
from src.llm.base import LLMProvider, LLMResponse

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("unshare") is None,
                                reason="requires Linux namespaces")

RESPOND = '{"action": "respond", "message": "done"}'
INFRA = "InternalServerError: Error code: 500"
REFUSAL = "content_filter: response blocked by the safety system"


@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()


class FlakyProvider(LLMProvider):
    """Responds immediately; `error_for(n)` returns an error string for the n-th episode
    (1-based, counted at the first call of an episode) or None."""

    def __init__(self, error_for=lambda n: None, name="m", reply=RESPOND):
        self.name = name
        self.error_for = error_for
        self.reply = reply
        self.episodes = 0

    def generate(self, messages, temperature=0.0, max_tokens=512):
        if not any(m["role"] == "assistant" for m in messages):
            self.episodes += 1
        error = self.error_for(self.episodes)
        if error:
            return LLMResponse(text="", latency_ms=1.0, model=self.name, error=error)
        return LLMResponse(text=self.reply, latency_ms=1.0, model=self.name)


def use(monkeypatch, provider):
    monkeypatch.setattr(runner_module, "get_provider", lambda _name: provider)


def lines(out):
    return [json.loads(line) for line in open(out / "raw.jsonl", encoding="utf-8") if line.strip()]


# ── Work queue ──

def real_cases(specs):
    return benign_cases(specs) + attack_cases(20, 20, seed=0, specs=specs)


def test_work_queue_is_run_major_paired_and_deterministic(specs):
    cases = real_cases(specs)
    queue = _work_queue(cases, 3, ARMS, 0)
    assert len(cases) == 57 and len(queue) == 342
    assert [(c.id, r, a) for c, r, a in queue] == [(c.id, r, a) for c, r, a in _work_queue(cases, 3, ARMS, 0)]
    runs = [r for _, r, _ in queue]
    assert runs == sorted(runs)                       # every run-0 item precedes every run-1 item
    for i in range(0, len(queue), 2):                 # both arms of a (case, run) are adjacent
        (c1, r1, a1), (c2, r2, a2) = queue[i], queue[i + 1]
        assert c1.id == c2.id and r1 == r2 and (a1, a2) == ARMS
    assert sorted((c.id, r, a) for c, r, a in queue) == sorted(
        (c.id, r, a) for c in cases for r in range(3) for a in ARMS)


def test_work_queue_prefix_holds_both_attack_kinds(specs):
    queue = _work_queue(real_cases(specs), 3, ARMS, 0)
    kinds = {c.kind for c, _, _ in queue[:56]}
    assert {"dh", "ds"} <= kinds


def test_work_queue_seed_changes_the_order_and_runs_shuffle_independently(specs):
    cases = real_cases(specs)
    a, b = _work_queue(cases, 2, ARMS, 0), _work_queue(cases, 2, ARMS, 1)
    assert [c.id for c, _, _ in a] != [c.id for c, _, _ in b]
    n = len(cases) * len(ARMS)
    assert [c.id for c, _, _ in a[:n]] != [c.id for c, _, _ in a[n:]]


# ── Re-queue of infrastructure errors ──

def test_infra_error_is_requeued_and_rerun_in_the_same_invocation(tmp_path, monkeypatch, specs):
    use(monkeypatch, FlakyProvider(lambda n: INFRA if n == 1 else None))
    out, logs = tmp_path / "r", []
    runner_module.run_benchmark(["m"], benign_cases(specs)[:1], str(out), runs=1, arms=("unprotected",),
                                log=logs.append)
    rows = lines(out)
    assert [bool(r["llm_error"]) for r in rows] == [True, False]
    assert any("[requeue]" in line for line in logs)
    summary = generate_report(str(out), charts=False, expected=1)[0]
    assert summary["infra_errors"] == 0 and summary["pairs_dropped"] == 0 and summary["status"] == "complete"


def test_requeue_is_bounded_by_requeue_passes(tmp_path, monkeypatch, specs):
    case = benign_cases(specs)[:1]
    use(monkeypatch, FlakyProvider(lambda n: INFRA))
    one = tmp_path / "one"
    runner_module.run_benchmark(["m"], case, str(one), runs=1, arms=("unprotected",), log=lambda *_: None)
    assert len(lines(one)) == 2                       # the attempt and exactly one re-queue pass
    use(monkeypatch, FlakyProvider(lambda n: INFRA))
    none = tmp_path / "none"
    runner_module.run_benchmark(["m"], case, str(none), runs=1, arms=("unprotected",), log=lambda *_: None,
                                requeue_passes=0)
    assert len(lines(none)) == 1
    use(monkeypatch, FlakyProvider(lambda n: INFRA))
    runner_module.run_benchmark(["m"], case, str(one), runs=1, arms=("unprotected",), log=lambda *_: None)
    assert len(lines(one)) == 4                       # a resume retries the key again


@linux_only
def test_requeued_protected_arm_is_compared_like_for_like(tmp_path, monkeypatch, specs):
    case = attack_cases(1, 0, specs=specs)
    # Episode 1 is the unprotected arm, episode 2 the protected arm (infra error), 3 its re-run
    use(monkeypatch, FlakyProvider(lambda n: INFRA if n == 2 else None))
    out = tmp_path / "r"
    runner_module.run_benchmark(["m"], case, str(out), runs=1, log=lambda *_: None)
    assert len(lines(out)) == 3
    s = generate_report(str(out), charts=False, expected=2)[0]
    assert s["pairs_dropped"] == 0 and s["infra_errors"] == 0
    assert s["attack_pairs"] == 1 and s["success_unprotected_n"] == 1 and s["success_protected_n"] == 1


def test_provider_refusal_is_recorded_once_and_never_retried(tmp_path, monkeypatch, specs):
    use(monkeypatch, FlakyProvider(lambda n: REFUSAL))
    out, logs = tmp_path / "r", []
    case = benign_cases(specs)[:1]
    runner_module.run_benchmark(["m"], case, str(out), runs=1, arms=("unprotected",), log=logs.append)
    assert len(lines(out)) == 1 and not any("[requeue]" in line for line in logs)
    runner_module.run_benchmark(["m"], case, str(out), runs=1, arms=("unprotected",), log=logs.append)
    assert len(lines(out)) == 1


def test_no_requeue_pass_after_quota_exhaustion(tmp_path, monkeypatch, specs):
    quota = "RateLimitError: 429 quota exceeded. Please retry in 15h2m1s."
    use(monkeypatch, FlakyProvider(lambda n: INFRA if n == 1 else quota))
    out, logs = tmp_path / "r", []
    runner_module.run_benchmark(["m"], benign_cases(specs)[:2], str(out), runs=1, arms=("unprotected",),
                                log=logs.append)
    assert len(lines(out)) == 1                       # the infra error; quota writes nothing
    assert any("[stop] m: quota exhausted" in line for line in logs)
    assert not any("[requeue]" in line for line in logs)


# ── Manifest ──

def manifest_cases(specs):
    return benign_cases(specs)[:1] + attack_cases(1, 1, specs=specs)


def test_manifest_records_plan_and_one_invocation_per_call(tmp_path, monkeypatch, specs):
    use(monkeypatch, FlakyProvider())
    out, cases = tmp_path / "r", manifest_cases(specs)
    extra = {"seed": 7, "dh": 1}
    for _ in range(2):
        runner_module.run_benchmark(["m"], cases, str(out), runs=2, arms=("unprotected",), max_steps=4,
                                    log=lambda *_: None, manifest_extra=extra)
    manifest = json.load(open(out / "manifest.json", encoding="utf-8"))
    assert manifest["schema"] == 1
    assert manifest["expected_episodes_per_model"] == len(cases) * 2 * 1
    assert manifest["cases"] == {"benign": 1, "dh": 1, "ds": 1}
    assert (manifest["runs"], manifest["arms"], manifest["max_steps"], manifest["models"]) == \
           (2, ["unprotected"], 4, ["m"])
    assert manifest["extra"] == extra
    assert len(manifest["invocations"]) == 2
    for inv in manifest["invocations"]:
        assert set(inv) == {"started_at", "code_commit", "code_dirty", "policy_sha256", "models"}
        assert inv["models"] == ["m"]


def test_corrupt_manifest_is_rewritten(tmp_path, monkeypatch, specs):
    use(monkeypatch, FlakyProvider())
    out = tmp_path / "r"
    out.mkdir()
    (out / "manifest.json").write_text("{not json", encoding="utf-8")
    runner_module.run_benchmark(["m"], benign_cases(specs)[:1], str(out), runs=1, arms=("unprotected",),
                                log=lambda *_: None)
    assert len(json.load(open(out / "manifest.json", encoding="utf-8"))["invocations"]) == 1


def test_secrets_never_reach_raw_jsonl_or_manifest(tmp_path, monkeypatch, specs):
    secret = "sk-test-not-a-real-key"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    use(monkeypatch, FlakyProvider())
    out = tmp_path / "r"
    runner_module.run_benchmark(["m"], manifest_cases(specs), str(out), runs=1, arms=("unprotected",),
                                log=lambda *_: None, manifest_extra={"seed": 0})
    for name in ("raw.jsonl", "manifest.json"):
        assert secret not in (out / name).read_text(encoding="utf-8")


# ── Report uses the manifest ──

def run_six(tmp_path, monkeypatch, specs, name="r", model="m"):
    use(monkeypatch, FlakyProvider(name=model))
    out = tmp_path / name
    runner_module.run_benchmark([model], manifest_cases(specs), str(out), runs=2, arms=("unprotected",),
                                log=lambda *_: None)
    return out


def test_report_completeness_follows_the_manifest(tmp_path, monkeypatch, specs):
    out = run_six(tmp_path, monkeypatch, specs)
    s = generate_report(str(out), charts=False)[0]
    assert (s["status"], s["episodes_done"], s["episodes_expected"]) == ("complete", 6, 6)
    assert "Invocations" in (out / "summary.md").read_text(encoding="utf-8")


def test_explicit_expected_overrides_the_manifest(tmp_path, monkeypatch, specs):
    out = run_six(tmp_path, monkeypatch, specs)
    s = generate_report(str(out), charts=False, expected=10)[0]
    assert (s["status"], s["episodes_expected"]) == ("partial", 10)


def test_missing_manifest_falls_back_to_the_default(tmp_path, monkeypatch, specs):
    out = run_six(tmp_path, monkeypatch, specs)
    (out / "manifest.json").unlink()
    s = generate_report(str(out), charts=False)[0]
    assert s["episodes_expected"] == 342 and s["status"] == "partial"


def test_merged_sources_use_each_directorys_own_manifest(tmp_path, monkeypatch, specs):
    a = run_six(tmp_path, monkeypatch, specs, name="a", model="m1")
    use(monkeypatch, FlakyProvider(name="m2"))
    b = tmp_path / "b"
    runner_module.run_benchmark(["m2"], manifest_cases(specs), str(b), runs=1, arms=("unprotected",),
                                log=lambda *_: None)
    summary = {s["model"]: s for s in generate_report(str(tmp_path / "merged"), charts=False,
                                                       sources=[str(a), str(b)])}
    assert summary["m1"]["episodes_expected"] == 6 and summary["m1"]["status"] == "complete"
    assert summary["m2"]["episodes_expected"] == 3 and summary["m2"]["status"] == "complete"
