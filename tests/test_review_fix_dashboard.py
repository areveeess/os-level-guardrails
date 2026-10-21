"""
Regression tests for 01-REVIEW.md WR-09: dashboard hardening beside the loopback default.

  (a) DASHBOARD_DEBUG together with a non-loopback DASHBOARD_HOST recreates the exposure
      SEC-02 removed, so startup is refused;
  (b) POST /api/clear must be same-origin, and foreign Host headers (DNS rebinding) are
      rejected while the server is bound to loopback;
  (c) a bad DASHBOARD_PORT is a clear startup error, not an unhandled ValueError.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.dashboard.app as dashboard_app


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('DASHBOARD_HOST', 'DASHBOARD_PORT', 'DASHBOARD_DEBUG'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    log = tmp_path / 'logs' / 'guardrails.jsonl'
    log.parent.mkdir()
    log.write_text(json.dumps({"action": "run_shell", "target": "ls", "decision": "ALLOW",
                               "reason": "r", "layer": "policy_engine"}) + "\n", encoding='utf-8')
    monkeypatch.setattr(dashboard_app, 'LOG_FILE', str(log))
    monkeypatch.setattr(dashboard_app, '_log_cache',
                        {"mtime": 0, "logs": [], "stats": {"total": 0, "allowed": 0, "blocked": 0,
                                                            "flagged": 0, "layers": {}}})
    c = dashboard_app.app.test_client()
    c.log_path = log
    return c


# ── (a) debugger only on loopback ──

@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "example.com", "10.0.0.5"])
def test_debug_with_a_non_loopback_host_is_refused(monkeypatch, host):
    monkeypatch.setenv('DASHBOARD_DEBUG', '1')
    monkeypatch.setenv('DASHBOARD_HOST', host)
    with pytest.raises(SystemExit) as exc:
        dashboard_app._run_settings()
    assert "DASHBOARD_DEBUG" in str(exc.value) and "loopback" in str(exc.value)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_debug_on_loopback_is_still_allowed(monkeypatch, host):
    monkeypatch.setenv('DASHBOARD_DEBUG', '1')
    monkeypatch.setenv('DASHBOARD_HOST', host)
    assert dashboard_app._run_settings() == {"host": host, "port": 5000, "debug": True}


def test_debug_default_host_is_loopback(monkeypatch):
    monkeypatch.setenv('DASHBOARD_DEBUG', 'yes')
    assert dashboard_app._run_settings()["host"] == "127.0.0.1"


def test_non_loopback_without_debug_is_still_allowed(monkeypatch):
    monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
    assert dashboard_app._run_settings() == {"host": "0.0.0.0", "port": 5000, "debug": False}


def test_main_does_not_start_the_app_when_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard_app.app, 'run', lambda **kw: calls.append(kw))
    monkeypatch.setenv('DASHBOARD_DEBUG', '1')
    monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
    with pytest.raises(SystemExit):
        dashboard_app.main()
    assert calls == []


# ── (c) port validation ──

@pytest.mark.parametrize("value", ["abc", "0", "-1", "65536", "99999", "80.5", "5000x"])
def test_bad_port_is_a_clear_startup_error(monkeypatch, value):
    monkeypatch.setenv('DASHBOARD_PORT', value)
    with pytest.raises(SystemExit) as exc:
        dashboard_app._run_settings()
    assert "DASHBOARD_PORT" in str(exc.value) and value in str(exc.value)


@pytest.mark.parametrize("value,port", [("1", 1), ("8080", 8080), ("65535", 65535)])
def test_valid_ports_are_accepted(monkeypatch, value, port):
    monkeypatch.setenv('DASHBOARD_PORT', value)
    assert dashboard_app._run_settings()["port"] == port


# ── (b) POST /api/clear is same-origin only ──

@pytest.mark.parametrize("headers", [
    {"Origin": "http://evil.example"},
    {"Origin": "http://evil.example:5000"},
    {"Origin": "http://localhost.evil.example"},
    {"Origin": "http://127.0.0.1:5000"},                 # a different origin than Host: localhost
    {"Origin": "null"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
    {"Sec-Fetch-Site": "cross-site", "Origin": "http://localhost"},
])
def test_cross_origin_clear_is_refused_and_the_log_survives(client, headers):
    before = client.log_path.read_text(encoding='utf-8')
    resp = client.post('/api/clear', headers=headers)
    assert resp.status_code == 403
    assert client.log_path.read_text(encoding='utf-8') == before


@pytest.mark.parametrize("headers", [
    {},                                                   # curl, scripts, the existing test client
    {"Origin": "http://localhost"},
    {"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost"},
    {"Sec-Fetch-Site": "none"},
])
def test_same_origin_clear_still_works(client, headers):
    resp = client.post('/api/clear', headers=headers)
    assert resp.status_code == 200 and resp.get_json()["status"] == "cleared"
    assert client.log_path.read_text(encoding='utf-8') == ""


def test_origin_must_match_the_host_header_including_port(client):
    ok = client.post('/api/clear', headers={"Host": "127.0.0.1:5000", "Origin": "http://127.0.0.1:5000"})
    assert ok.status_code == 200
    bad = client.post('/api/clear', headers={"Host": "127.0.0.1:5000", "Origin": "http://127.0.0.1:6000"})
    assert bad.status_code == 403


# ── (b) foreign Host headers (DNS rebinding) ──

@pytest.mark.parametrize("host", ["evil.example", "evil.example:5000", "localhost.evil.example",
                                  "192.168.1.5:5000", "0.0.0.0:5000", "127.0.0.1.evil.example"])
@pytest.mark.parametrize("path", ["/api/logs", "/api/stats", "/api/info", "/"])
def test_foreign_host_is_rejected_while_bound_to_loopback(client, host, path):
    assert client.get(path, headers={"Host": host}).status_code == 403


@pytest.mark.parametrize("host", ["localhost", "localhost:5000", "127.0.0.1", "127.0.0.1:5000",
                                  "[::1]:5000", "LOCALHOST:8080"])
def test_loopback_hosts_are_served(client, host):
    assert client.get('/api/logs', headers={"Host": host}).status_code == 200


def test_post_with_a_foreign_host_is_rejected(client):
    before = client.log_path.read_text(encoding='utf-8')
    assert client.post('/api/clear', headers={"Host": "evil.example"}).status_code == 403
    assert client.log_path.read_text(encoding='utf-8') == before


def test_host_check_is_off_when_the_operator_bound_a_non_loopback_address(client, monkeypatch):
    monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
    assert client.get('/api/logs', headers={"Host": "192.168.1.5:5000"}).status_code == 200
    # ... but cross-site writes are still refused
    assert client.post('/api/clear', headers={"Host": "192.168.1.5:5000",
                                              "Origin": "http://evil.example"}).status_code == 403
