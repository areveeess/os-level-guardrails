"""
Results-view wording for final vs snapshot data
================================================
Once numbers.json is final, runs that did not finish are "incomplete" with where they stopped (at
the freeze, by quota), never "partial, not final". The wording lives in
src/dashboard/static/labels.js and is tested in tests/test_dashboard_labels.mjs; this file runs
that suite under pytest when node is available, and checks that results.js does not hard-code it.
"""

import os
import shutil
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.dashboard import demo

STATIC = os.path.join(PROJECT_ROOT, 'src', 'dashboard', 'static')


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_label_suite_passes_under_node():
    result = subprocess.run(['node', '--test', os.path.join('tests', 'test_dashboard_labels.mjs')],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_results_view_does_not_hard_code_not_final():
    with open(os.path.join(STATIC, 'results.js'), encoding='utf-8') as f:
        source = f.read()
    # Only the snapshot banner may say "not final"; row tags and headings come from labels.js.
    assert source.count('not final') == 1
    assert "'Snapshot, not final. '" in source
    assert "'partial, not final'" not in source
    assert 'Partial runs, not final' not in source


def test_results_page_loads_labels_before_results():
    with open(os.path.join(PROJECT_ROOT, 'src', 'dashboard', 'templates', 'results.html'), encoding='utf-8') as f:
        html = f.read()
    assert 0 <= html.index("filename='labels.js'") < html.index("filename='results.js'")


def test_lineup_carries_completion_and_stop_state():
    numbers = {"models": [
        {"model": "groq-gpt-oss-120b", "group": "hosted", "complete": False, "status": "partial",
         "hosted_state": {"launches": 9, "quota_stops": 6, "state": "stopped_at_freeze"}},
        {"model": "scratch-gpt", "group": "local", "family": "from-scratch", "complete": True, "status": "complete"},
    ]}
    rows = {r['model']: r for r in demo.lineup(numbers, None)}
    assert rows['groq-gpt-oss-120b']['complete'] is False
    assert rows['groq-gpt-oss-120b']['hosted_state'] == {"state": "stopped_at_freeze"}   # counters not exposed
    assert rows['scratch-gpt']['hosted_state'] is None
