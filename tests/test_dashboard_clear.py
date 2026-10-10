"""
Test Dashboard & Clear Log API
==============================
Verifies the frontend template has no hardcoded rows,
and tests the /api/clear endpoint and mtime caching behavior.
"""

import os
import sys
import json
import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import src.dashboard.app as dashboard_app
from src.dashboard.app import app


def test_dashboard_clear_flow(tmp_path, monkeypatch):
    # Point the dashboard at a throwaway log so the real logs/guardrails.jsonl is untouched
    LOG_FILE = str(tmp_path / 'logs' / 'guardrails.jsonl')
    monkeypatch.setattr(dashboard_app, 'LOG_FILE', LOG_FILE)
    monkeypatch.setattr(dashboard_app, '_log_cache', {"mtime": 0, "logs": [], "stats": {}})
    client = app.test_client()

    print("=" * 60)
    print("Running Dashboard & Clear Log Verification Tests")
    print("=" * 60)

    # 1. Test index page rendering
    resp = client.get('/')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    html = resp.get_data(as_text=True)
    assert '<tbody id="tbody"></tbody>' in html, "tbody must be empty (no hardcoded event logs in HTML)"
    assert 'id="btn-clear"' in html, "Clear button must be present in DOM"
    assert 'id="empty"' in html, "Empty state placeholder must be present"
    print("  [PASS] 1. Index page renders cleanly with no hardcoded event rows")

    # 2. Test initial clear state
    # Ensure log file is clean first
    client.post('/api/clear')
    logs = client.get('/api/logs').get_json()
    stats = client.get('/api/stats').get_json()
    assert logs == [], f"Expected empty logs array, got {logs}"
    assert stats['total'] == 0, f"Expected total 0, got {stats['total']}"
    assert stats['allowed'] == 0
    assert stats['blocked'] == 0
    assert stats['flagged'] == 0
    print("  [PASS] 2. Initial state returns 0 events and empty logs array")

    # 3. Simulate events being logged
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    sample_events = [
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "run_shell",
            "target": "cat /etc/shadow",
            "decision": "BLOCK",
            "reason": "Blocked access to sensitive password hashes",
            "layer": "policy_engine"
        },
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "run_shell",
            "target": "ls /tmp",
            "decision": "ALLOW",
            "reason": "Safe directory listing allowed",
            "layer": "policy_engine"
        },
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "write_file",
            "target": "/etc/hosts",
            "decision": "FLAG",
            "reason": "Suspicious modification flagged",
            "layer": "policy_engine"
        }
    ]

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        for ev in sample_events:
            f.write(json.dumps(ev) + '\n')

    stats_after = client.get('/api/stats').get_json()
    logs_after = client.get('/api/logs').get_json()
    assert stats_after['total'] == 3, f"Expected 3, got {stats_after['total']}"
    assert stats_after['blocked'] == 1
    assert stats_after['allowed'] == 1
    assert stats_after['flagged'] == 1
    assert len(logs_after) == 3
    print("  [PASS] 3. Dashboard API successfully ingests simulated runtime events")

    # 4. Clear the logs via POST /api/clear
    clear_resp = client.post('/api/clear')
    assert clear_resp.status_code == 200
    data = clear_resp.get_json()
    assert data.get('status') == 'cleared'
    assert data.get('total') == 0

    # Verify state after clear
    stats_cleared = client.get('/api/stats').get_json()
    logs_cleared = client.get('/api/logs').get_json()
    assert stats_cleared['total'] == 0, f"Expected 0 after clear, got {stats_cleared['total']}"
    assert stats_cleared['allowed'] == 0
    assert stats_cleared['blocked'] == 0
    assert stats_cleared['flagged'] == 0
    assert logs_cleared == []
    assert os.path.exists(LOG_FILE)
    assert os.path.getsize(LOG_FILE) == 0
    print("  [PASS] 4. POST /api/clear successfully truncates file and resets stats/logs")

    # 5. GET /api/clear must be rejected (a link or prefetch must not wipe logs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"action": "run_shell", "target": "ls", "decision": "ALLOW", "reason": "test"}) + "\n")
    clear_get_resp = client.get('/api/clear')
    assert clear_get_resp.status_code == 405
    assert os.path.getsize(LOG_FILE) > 0
    print("  [PASS] 5. GET /api/clear is rejected and logs are preserved")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v', '-s']))
