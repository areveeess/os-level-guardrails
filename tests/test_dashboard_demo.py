"""
Dashboard demo routes: model line-up, recorded-episode replay, scripted policy simulation
==========================================================================================
Replay only reads runs listed in the frozen numbers.json, by lookup, and returns an allow list of
record fields. The simulation is same-origin POST only, takes a scenario id and nothing else, runs
one at a time, stops at its deadline, checks actions with the policy engine without executing them,
and writes only to the decision log.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.dashboard.app as dashboard_app
from src.dashboard import demo

SECRET_SHA = 'f' * 64


def _record(arm, case_id='ds-u00-a01', run=0, **kw):
    r = {"model": "qwen2.5-0.5b-agentlora", "case_id": case_id, "kind": case_id.split('-')[0], "run": run,
         "arm": arm, "attack_type": "Financial Data", "attempted": True, "succeeded": arm == 'unprotected',
         "injection_shown": True, "canary_exposed": arm == 'unprotected', "attack_blocked": arm == 'protected',
         "blocked_by": {"tool_broker": 1} if arm == 'protected' else {}, "os_tags": {}, "user_task_done": True,
         "false_positive": False,
         "tool_calls": [{"tool": "AmazonGetProductDetails", "allowed": True},
                        {"tool": "AmazonViewSavedPaymentMethods", "allowed": arm == 'unprotected'}],
         "actions": [{"action": "call_tool", "tool": "AmazonGetProductDetails", "args": {}},
                     {"action": "call_tool", "tool": "AmazonViewSavedPaymentMethods", "args": {}},
                     {"action": "respond", "message": "done"}],
         "final_message": "done", "stop_reason": "responded", "steps": 3,
         "code_commit": "abc123", "policy_sha256": SECRET_SHA, "resolved_model": "internal-id",
         "prompt_tokens": [1, 2]}
    if case_id.startswith('benign'):
        r.update(kind='benign', attack_type='', attempted=False, succeeded=False, canary_exposed=False,
                 attack_blocked=False, blocked_by={})
    r.update(kw)
    return r


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('DASHBOARD_HOST', 'DASHBOARD_PORT', 'DASHBOARD_DEBUG'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    final = tmp_path / 'results' / 'phase5-final'
    numbers = {
        "schema": 1, "data_state": "snapshot", "snapshot_display": "2026-10-29T07:12Z",
        "models": [
            {"model": "qwen2.5-0.5b-agentlora", "group": "local", "family": "open / tiny / locally LoRA-tuned",
             "provider": "hf_local", "status": "complete"},
            {"model": "scratch-gpt", "group": "local", "family": "from-scratch / unaligned baseline",
             "provider": "scratch", "status": "complete"},
            {"model": "ollama-qwen2.5-3b", "group": "local", "family": "open / small / local",
             "provider": "openai_compat", "status": "complete"},
            {"model": "groq-gpt-oss-120b", "group": "hosted", "family": "open / large",
             "provider": "openai_compat", "status": "partial"},
            {"model": "control-always-comply", "group": "control", "family": "control / scripted worst case",
             "provider": "scripted", "status": "complete"},
        ],
        "legacy": [{"model": "gemini-flash-lite", "family": "closed / small-fast", "status": "partial"}],
        "sources": [
            {"name": "phase4-qwen2.5-0.5b-agentlora", "group": "current"},
            {"name": "phase4-control", "group": "control"},
            {"name": "legacy-gemini-flash-lite", "group": "legacy"},
            {"name": "../escape", "group": "current"},
        ],
    }
    (final / 'inputs' / 'config').mkdir(parents=True)
    (final / 'numbers.json').write_text(json.dumps(numbers), encoding='utf-8')
    (final / 'inputs' / 'config' / 'models.yaml').write_text(
        "default: gemini-flash-lite\nmodels:\n"
        "  ollama-qwen2.5-3b:\n    provider: openai_compat\n    base_url: http://localhost:11434/v1\n"
        "    model: qwen2.5-3b-ctx8192\n"
        "  groq-gpt-oss-120b:\n    provider: openai_compat\n    base_url: https://api.groq.com/openai/v1\n"
        "    api_key_env: GROQ_API_KEY\n    model: openai/gpt-oss-120b\n", encoding='utf-8')
    for name, records in (
        ('phase4-qwen2.5-0.5b-agentlora', [_record('unprotected'), _record('protected'),
                                           _record('unprotected', 'benign-00'), _record('protected', 'benign-00')]),
        ('phase4-control', [_record('unprotected', model='control-always-comply'),
                            _record('protected', model='control-always-comply')]),
        ('legacy-gemini-flash-lite', [_record('unprotected', model='gemini-flash-lite')]),
    ):
        d = final / 'sources' / name
        d.mkdir(parents=True)
        (d / 'raw.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    (tmp_path / 'results' / 'escape').mkdir()
    (tmp_path / 'results' / 'escape' / 'raw.jsonl').write_text(json.dumps(_record('protected')), encoding='utf-8')
    monkeypatch.setattr(dashboard_app, 'RESULTS_FINAL_DIR', str(final))
    return final


@pytest.fixture
def client():
    return dashboard_app.app.test_client()


@pytest.fixture
def sim(tmp_path, monkeypatch):
    log = tmp_path / 'logs' / 'guardrails.jsonl'
    log.parent.mkdir()
    monkeypatch.setattr(dashboard_app, 'LOG_FILE', str(log))
    s = demo.Simulation()
    monkeypatch.setattr(dashboard_app, '_simulation', s)
    monkeypatch.setattr(demo, 'SIM_STEP_SECONDS', 0.01)
    yield s, log
    s.wait()


# ── Pages ──

def test_demo_page_renders(client):
    resp = client.get('/demo')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'id="demo-root"' in html and 'id="how-to"' in html


# ── Line-up ──

def test_lineup_groups_every_model(client, frozen):
    data = client.get('/api/lineup').get_json()
    groups = {m['model']: m['group'] for m in data['models']}
    assert groups == {"qwen2.5-0.5b-agentlora": "locally made", "scratch-gpt": "locally made",
                      "ollama-qwen2.5-3b": "local pretrained", "groq-gpt-oss-120b": "hosted",
                      "control-always-comply": "control", "gemini-flash-lite": "legacy"}
    served = {m['model']: m['served_as'] for m in data['models']}
    assert served['groq-gpt-oss-120b'] == 'openai/gpt-oss-120b'


def test_lineup_never_exposes_endpoints_or_key_names(client, frozen):
    text = client.get('/api/lineup').get_data(as_text=True)
    assert 'base_url' not in text and 'api_key_env' not in text and 'GROQ_API_KEY' not in text
    assert 'api.groq.com' not in text and '11434' not in text


def test_lineup_without_frozen_data_is_404(client, frozen):
    (frozen / 'numbers.json').unlink()
    assert client.get('/api/lineup').status_code == 404


# ── Replay ──

def test_replay_index_lists_frozen_current_runs_only(client, frozen):
    data = client.get('/api/demo/replay').get_json()
    assert data['models'] == ['control-always-comply', 'qwen2.5-0.5b-agentlora']
    assert any(s['model'] == 'qwen2.5-0.5b-agentlora' and s['case_id'] == 'ds-u00-a01' for s in data['stories'])


def test_episode_returns_both_arms_with_case_text(client, frozen):
    data = client.get('/api/demo/episode?model=qwen2.5-0.5b-agentlora&case=ds-u00-a01&run=0').get_json()
    assert set(data['arms']) == {'unprotected', 'protected'}
    assert data['case']['attacker_instruction'] in data['case']['tool_response']
    assert data['arms']['protected']['tool_calls'][1]['allowed'] is False


def test_episode_returns_allow_listed_fields_only(client, frozen):
    text = client.get('/api/demo/episode?model=qwen2.5-0.5b-agentlora&case=ds-u00-a01&run=0').get_data(as_text=True)
    for leaked in (SECRET_SHA, 'abc123', 'internal-id', 'prompt_tokens', 'code_commit'):
        assert leaked not in text


@pytest.mark.parametrize("model", ["../escape", "escape", "gemini-flash-lite", "legacy-gemini-flash-lite",
                                   "phase4-qwen2.5-0.5b-agentlora", "qwen2.5-0.5b-agentlora/../../escape",
                                   "..%2fescape", "/etc/passwd", "", "QWEN2.5-0.5B-AGENTLORA"])
def test_unknown_or_traversing_model_is_refused(client, frozen, model):
    for url in ('/api/demo/cases?model=' + model, '/api/demo/episode?case=ds-u00-a01&run=0&model=' + model):
        resp = client.get(url)
        assert resp.status_code == 404, (url, resp.status_code)
        assert b'Financial Data' not in resp.data


@pytest.mark.parametrize("case,run", [("../../etc/passwd", "0"), ("ds-u00-a01/../x", "0"), ("ds-u0-a1", "0"),
                                      ("DS-U00-A01", "0"), ("benign-00", "-1"), ("benign-00", "10"),
                                      ("benign-00", "x"), ("", "0")])
def test_malformed_case_or_run_is_400(client, frozen, case, run):
    resp = client.get(f'/api/demo/episode?model=qwen2.5-0.5b-agentlora&case={case}&run={run}')
    assert resp.status_code == 400


def test_missing_episode_is_404(client, frozen):
    assert client.get('/api/demo/episode?model=qwen2.5-0.5b-agentlora&case=dh-u01-a02&run=0').status_code == 404
    assert client.get('/api/demo/episode?model=qwen2.5-0.5b-agentlora&case=ds-u00-a01&run=3').status_code == 404


def test_replay_routes_are_read_only(client, frozen):
    for url in ('/api/demo/replay', '/api/demo/cases?model=qwen2.5-0.5b-agentlora', '/api/lineup',
                '/api/demo/episode?model=qwen2.5-0.5b-agentlora&case=ds-u00-a01&run=0', '/api/demo/scenarios'):
        assert client.post(url).status_code == 405


# ── Simulation ──

def test_scenarios_are_the_scripted_ones(client):
    sc = client.get('/api/demo/scenarios').get_json()['scenarios']
    assert [s['id'] for s in sc] == [f's{i}' for i in range(1, len(sc) + 1)]
    assert len(sc) == len(demo.scenarios()) and any(not s['malicious'] for s in sc)


def test_simulation_status_starts_idle(client, sim):
    assert client.get('/api/demo/simulate').get_json()['status'] == 'idle'


@pytest.mark.parametrize("headers", [{"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"},
                                     {"Origin": "http://localhost:5001"}])
def test_simulation_refuses_cross_origin(client, sim, headers):
    resp = client.post('/api/demo/simulate', json={"scenario": "all"}, headers=headers)
    assert resp.status_code == 403
    assert sim[0].status()['status'] == 'idle'


def test_simulation_refuses_foreign_host(client, sim):
    resp = client.post('/api/demo/simulate', json={"scenario": "all"}, headers={"Host": "evil.example:5000"})
    assert resp.status_code == 403


@pytest.mark.parametrize("kwargs", [
    {"json": {"scenario": "all", "command": "id"}},
    {"json": {"command": "rm -rf /"}},
    {"json": ["s1"]},
    {"json": {"scenario": "s99"}},
    {"json": {"scenario": "../s1"}},
    {"json": {"scenario": "s1; ls"}},
    {"json": {"scenario": 1}},
    {"data": "scenario=all", "content_type": "application/x-www-form-urlencoded"},
    {"data": "not json", "content_type": "application/json"},
])
def test_simulation_takes_a_scenario_id_and_nothing_else(client, sim, kwargs):
    assert client.post('/api/demo/simulate', **kwargs).status_code == 400
    assert sim[0].status()['status'] == 'idle'


def test_simulation_refuses_query_parameters(client, sim):
    assert client.post('/api/demo/simulate?scenario=all', json={"scenario": "all"}).status_code == 400
    assert client.post('/api/demo/simulate?cmd=id').status_code == 400


def test_simulation_other_methods_are_405(client, sim):
    assert client.put('/api/demo/simulate').status_code == 405
    assert client.delete('/api/demo/simulate').status_code == 405


def test_simulation_runs_policy_checks_and_logs_them(client, sim, tmp_path):
    s, log = sim
    resp = client.post('/api/demo/simulate', json={"scenario": "all"})
    assert resp.status_code == 202
    assert s.wait()
    status = client.get('/api/demo/simulate').get_json()
    assert status['status'] == 'done' and status['done'] == status['total'] == len(demo.scenarios())
    decisions = {d['id']: d['decision'] for d in status['decisions']}
    malicious = {sc['id'] for sc in demo.scenarios() if sc['malicious']}
    assert all(decisions[i] == 'BLOCK' for i in malicious)
    assert any(decisions[i] == 'ALLOW' for i in decisions if i not in malicious)
    lines = [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) == len(demo.scenarios())
    assert not (tmp_path / 'results').exists()


def test_simulation_executes_nothing(client, sim, monkeypatch):
    import subprocess
    from src.agent_harness.tools import ToolExecutor

    def refuse(*a, **k):
        raise AssertionError('the simulation must not execute anything')
    monkeypatch.setattr(subprocess, 'run', refuse)
    monkeypatch.setattr(subprocess, 'Popen', refuse)
    monkeypatch.setattr(ToolExecutor, 'execute', refuse)
    assert client.post('/api/demo/simulate', json={"scenario": "all"}).status_code == 202
    assert sim[0].wait()
    assert sim[0].status()['status'] == 'done'


def test_one_scenario_by_id(client, sim):
    assert client.post('/api/demo/simulate', json={"scenario": "s1"}).status_code == 202
    assert sim[0].wait()
    st = sim[0].status()
    assert st['total'] == 1 and st['decisions'][0]['id'] == 's1'


def test_empty_body_runs_all(client, sim):
    assert client.post('/api/demo/simulate').status_code == 202
    assert sim[0].wait()
    assert sim[0].status()['total'] == len(demo.scenarios())


def test_one_run_at_a_time(client, sim, monkeypatch):
    monkeypatch.setattr(demo, 'SIM_STEP_SECONDS', 0.2)
    assert client.post('/api/demo/simulate', json={"scenario": "all"}).status_code == 202
    busy = client.post('/api/demo/simulate', json={"scenario": "s1"})
    assert busy.status_code == 409
    assert busy.get_json()['status']['status'] == 'running'
    assert sim[0].wait()
    assert client.post('/api/demo/simulate', json={"scenario": "s1"}).status_code == 202


def test_run_stops_at_its_deadline(client, sim, monkeypatch):
    monkeypatch.setattr(demo, 'SIM_STEP_SECONDS', 0.05)
    monkeypatch.setattr(demo, 'SIM_TIMEOUT_SECONDS', 0.02)
    started = time.monotonic()
    assert client.post('/api/demo/simulate', json={"scenario": "all"}).status_code == 202
    assert sim[0].wait()
    st = sim[0].status()
    assert st['status'] == 'timeout'
    assert st['done'] < st['total']
    assert time.monotonic() - started < 2
    # the lock is released after a timeout
    monkeypatch.setattr(demo, 'SIM_TIMEOUT_SECONDS', 30)
    assert client.post('/api/demo/simulate', json={"scenario": "s1"}).status_code == 202
