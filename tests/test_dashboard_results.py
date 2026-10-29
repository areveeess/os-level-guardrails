"""
Dashboard results view: read-only routes over results/phase5-final/
=====================================================================
/api/results returns numbers.json unchanged, and /api/results/files/<name> serves only an allow
list of frozen-result files. Nothing outside results/phase5-final/ (or inside it, but not on the
list, such as raw episodes with canary values) can be read, and no route writes.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.dashboard.app as dashboard_app

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('DASHBOARD_HOST', 'DASHBOARD_PORT', 'DASHBOARD_DEBUG'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A fake results/ tree: phase5-final with its files, plus things that must stay unreachable."""
    results = tmp_path / 'results'
    final = results / 'phase5-final'
    (final / 'charts').mkdir(parents=True)
    (final / 'sources' / 'run').mkdir(parents=True)
    (final / 'inputs').mkdir()
    numbers = {"schema": 1, "data_state": "snapshot", "snapshot_display": "2026-10-29T07:12Z",
               "freeze_display": "2026-10-29T15:30Z", "models": [{"model": "m", "status": "complete"}]}
    (final / 'numbers.json').write_text(json.dumps(numbers), encoding='utf-8')
    (final / 'freeze.json').write_text('{"schema": 1}', encoding='utf-8')
    (final / 'summary.md').write_text('# Benchmark summary\n', encoding='utf-8')
    (final / 'charts' / 'asr_ci.png').write_bytes(PNG)
    (final / 'sources' / 'run' / 'raw.jsonl').write_text('{"canary": "SECRET-CANARY"}\n', encoding='utf-8')
    (final / 'inputs' / 'models.yaml').write_text('key: value\n', encoding='utf-8')
    (results / 'other.json').write_text('{"outside": true}', encoding='utf-8')
    (tmp_path / '.env').write_text('API_KEY=not-a-real-key\n', encoding='utf-8')
    monkeypatch.setattr(dashboard_app, 'RESULTS_FINAL_DIR', str(final))
    return {"dir": final, "numbers": numbers}


@pytest.fixture
def client():
    return dashboard_app.app.test_client()


# ── /api/results ──

def test_api_results_returns_numbers_json_unchanged(client, frozen):
    resp = client.get('/api/results')
    assert resp.status_code == 200
    assert resp.get_json() == frozen["numbers"]


def test_api_results_missing_numbers_is_a_json_404(client, frozen):
    (frozen["dir"] / 'numbers.json').unlink()
    resp = client.get('/api/results')
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "no frozen results"


def test_api_results_unreadable_numbers_is_a_json_500(client, frozen):
    (frozen["dir"] / 'numbers.json').write_text('{not json', encoding='utf-8')
    resp = client.get('/api/results')
    assert resp.status_code == 500
    assert "unreadable" in resp.get_json()["error"]


def test_api_results_is_read_only(client, frozen):
    for method in (client.post, client.put, client.delete):
        assert method('/api/results').status_code == 405
        assert method('/api/results/files/numbers.json').status_code == 405
    assert json.loads((frozen["dir"] / 'numbers.json').read_text(encoding='utf-8')) == frozen["numbers"]


# ── /api/results/files/<name> ──

@pytest.mark.parametrize("name", ["numbers.json", "freeze.json", "summary.md", "charts/asr_ci.png"])
def test_allow_listed_files_are_served(client, frozen, name):
    resp = client.get('/api/results/files/' + name)
    assert resp.status_code == 200
    assert resp.data == (frozen["dir"] / name).read_bytes()
    assert resp.headers['X-Content-Type-Options'] == 'nosniff'


def test_chart_is_served_as_png(client, frozen):
    resp = client.get('/api/results/files/charts/asr_ci.png')
    assert resp.mimetype == 'image/png'


@pytest.mark.parametrize("name", [
    "../other.json",                      # sibling of phase5-final
    "../../.env",                         # secrets above results/
    "..%2f..%2f.env",                     # encoded separators
    "%2e%2e/%2e%2e/.env",                 # encoded dots
    "charts/../numbers.json/../../other.json",
    "charts/../../other.json",
    "charts/..%2f..%2fother.json",
    "sources/run/raw.jsonl",              # raw episodes carry canary values
    "inputs/models.yaml",
    "charts/missing.png",                 # on the pattern, but absent
    "charts/asr_ci.png.bak",
    "charts/sub/asr_ci.png",
    "charts\\asr_ci.png",
    "/etc/passwd",
    "NUMBERS.JSON",
    "",
])
def test_everything_else_is_refused(client, frozen, name):
    # Werkzeug answers a doubled slash with a redirect to the merged path; follow it.
    resp = client.get('/api/results/files/' + name, follow_redirects=True)
    assert resp.status_code == 404, name
    assert b'SECRET-CANARY' not in resp.data
    assert b'not-a-real-key' not in resp.data
    assert b'outside' not in resp.data


def test_absolute_path_segment_is_refused(client, frozen):
    # A double slash makes the <path:name> value start with '/'
    resp = client.get('/api/results/files//' + str(frozen["dir"] / 'numbers.json').lstrip('/'),
                      follow_redirects=True)
    assert resp.status_code == 404
    assert b'snapshot_display' not in resp.data


# ── /results page ──

def test_results_page_renders(client, frozen):
    resp = client.get('/results')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'id="results-root"' in html
    assert '/api/results' in html


def test_results_routes_keep_the_dns_rebinding_guard(client, frozen):
    for path in ('/results', '/api/results', '/api/results/files/numbers.json'):
        assert client.get(path, headers={'Host': 'evil.example:5000'}).status_code == 403
