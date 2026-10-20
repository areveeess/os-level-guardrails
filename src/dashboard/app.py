"""
Dashboard — Real-Time Log/Alert Visualization
================================================
A Flask web application that displays the policy engine's decisions
in real-time. Shows what the AI agent tried to do and whether each
action was Allowed, Blocked, or Flagged.

Usage:
  python -m src.dashboard.app
  Then open http://127.0.0.1:5000 in your browser

  Safe by default: the server listens on 127.0.0.1 only, with Flask debug off.
  DASHBOARD_HOST=<addr>  listen on another interface (only on a trusted network)
  DASHBOARD_PORT=<port>  listen on another port
  DASHBOARD_DEBUG=1      enable the Werkzeug debugger. WARNING: the debugger allows
                         arbitrary code execution for anyone who can reach the port.
"""

import os
import sys
import json
from flask import Flask, render_template, jsonify

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'guardrails.jsonl')

app = Flask(__name__)


# Cache to avoid re-reading and re-parsing unchanged log file on every 2-second poll
_log_cache = {
    "mtime": 0,
    "logs": [],
    "stats": {"total": 0, "allowed": 0, "blocked": 0, "flagged": 0, "layers": {}}
}


def _get_model_name():
    """Extract model name from .env or default."""
    env_file = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_MODEL='):
                        return line.split('=', 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return "gemini-2.5-flash"


def _read_logs_cached():
    """Read logs with mtime caching to eliminate unnecessary file I/O and JSON parsing."""
    global _log_cache
    if not os.path.exists(LOG_FILE):
        return [], {"total": 0, "allowed": 0, "blocked": 0, "flagged": 0, "layers": {"policy_engine": 0, "sandbox": 0, "seccomp": 0}}

    try:
        mtime = os.path.getmtime(LOG_FILE)
    except OSError:
        mtime = 0

    if mtime == _log_cache["mtime"] and _log_cache["mtime"] > 0:
        return _log_cache["logs"], _log_cache["stats"]

    logs = []
    layers = {"policy_engine": 0, "sandbox": 0, "seccomp": 0}
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        # Ensure layer is set
                        if "layer" not in entry:
                            entry["layer"] = "policy_engine"
                        layer_name = entry["layer"]
                        layers[layer_name] = layers.get(layer_name, 0) + 1
                        logs.append(entry)
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass

    # Reverse for newest-first display
    reversed_logs = list(reversed(logs))
    stats = {
        "total": len(logs),
        "allowed": sum(1 for l in logs if l.get("decision") == "ALLOW"),
        "blocked": sum(1 for l in logs if l.get("decision") == "BLOCK"),
        "flagged": sum(1 for l in logs if l.get("decision") == "FLAG"),
        "layers": layers
    }

    _log_cache = {
        "mtime": mtime,
        "logs": reversed_logs,
        "stats": stats
    }
    return reversed_logs, stats


@app.route('/')
def index():
    """Serve the main dashboard page with dynamic system info."""
    info = {
        "model": _get_model_name(),
        "platform": sys.platform,
        "is_linux": sys.platform.startswith('linux')
    }
    return render_template('index.html', info=info)


@app.route('/api/info')
def get_info():
    """Return runtime metadata and active protection layers."""
    return jsonify({
        "model": _get_model_name(),
        "platform": sys.platform,
        "is_linux": sys.platform.startswith('linux'),
        "layers": ["Access-Matrix Policy Engine", "Linux Namespaces", "Seccomp-BPF"]
    })


@app.route('/api/logs')
def get_logs():
    """API endpoint: return all log entries as JSON array (cached by mtime)."""
    logs, _ = _read_logs_cached()
    return jsonify(logs)


@app.route('/api/stats')
def get_stats():
    """API endpoint: return summary statistics including layer counts."""
    _, stats = _read_logs_cached()
    return jsonify(stats)


@app.route('/api/clear', methods=['POST'])
def clear_logs():
    """API endpoint: clear all logs (for starting fresh demo runs)."""
    global _log_cache
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.truncate(0)
    except Exception as e:
        print(f"Error clearing log file: {e}")

    _log_cache = {
        "mtime": 0,
        "logs": [],
        "stats": {"total": 0, "allowed": 0, "blocked": 0, "flagged": 0, "layers": {"policy_engine": 0, "sandbox": 0, "seccomp": 0}}
    }
    return jsonify({"status": "cleared", "total": 0})


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 5000


def _run_settings():
    """Server settings: loopback only and debugger off unless explicitly overridden."""
    return {
        'host': os.environ.get('DASHBOARD_HOST') or DEFAULT_HOST,
        'port': int(os.environ.get('DASHBOARD_PORT') or DEFAULT_PORT),
        'debug': os.environ.get('DASHBOARD_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'on'),
    }


def main():
    settings = _run_settings()
    print(f"\n{'='*50}")
    print(f"  OS-Level Guardrails — Dashboard")
    print(f"{'='*50}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Open:     http://{settings['host']}:{settings['port']}")
    if settings['debug']:
        print(f"  WARNING:  Werkzeug debugger enabled (DASHBOARD_DEBUG) - arbitrary code execution")
    print(f"{'='*50}\n")
    app.run(**settings)


if __name__ == '__main__':
    main()
