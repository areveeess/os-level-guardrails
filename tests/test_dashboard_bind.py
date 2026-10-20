"""
Test Dashboard Bind Defaults (SEC-02)
=====================================
The dashboard used to start with the Werkzeug debugger on every network
interface, which is remote code execution for anyone who can reach the port.
These tests pin the safe defaults: loopback only, debugger off, and both
changeable only through explicit environment variables.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.dashboard.app as dashboard_app


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('DASHBOARD_HOST', 'DASHBOARD_PORT', 'DASHBOARD_DEBUG'):
        monkeypatch.delenv(name, raising=False)


# ── Defaults ──

def test_defaults_are_loopback_and_debug_off():
    assert dashboard_app._run_settings() == {"host": "127.0.0.1", "port": 5000, "debug": False}


# ── Environment overrides ──

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_debug_only_on_explicit_opt_in(monkeypatch, value):
    monkeypatch.setenv('DASHBOARD_DEBUG', value)
    assert dashboard_app._run_settings()["debug"] is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off", "debug"])
def test_other_debug_values_stay_off(monkeypatch, value):
    monkeypatch.setenv('DASHBOARD_DEBUG', value)
    assert dashboard_app._run_settings()["debug"] is False


def test_host_and_port_overrides(monkeypatch):
    monkeypatch.setenv('DASHBOARD_HOST', '0.0.0.0')
    monkeypatch.setenv('DASHBOARD_PORT', '8080')
    settings = dashboard_app._run_settings()
    assert settings["host"] == '0.0.0.0'
    assert settings["port"] == 8080
    assert settings["debug"] is False


# ── main() ──

def test_main_runs_app_once_with_run_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard_app.app, 'run', lambda **kwargs: calls.append(kwargs))
    dashboard_app.main()
    assert calls == [dashboard_app._run_settings()]
