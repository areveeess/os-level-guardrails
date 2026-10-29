"""
Dashboard — Real-Time Log/Alert Visualization
================================================
A Flask web application that displays the policy engine's decisions
in real-time. Shows what the AI agent tried to do and whether each
action was Allowed, Blocked, or Flagged.

Usage:
  python -m src.dashboard.app
  Then open http://127.0.0.1:5000 in your browser
  Live monitor at /, frozen Phase 5 comparison at /results (read-only, results/phase5-final/),
  demo at /demo (recorded-episode replay and the scripted policy simulation)

  Safe by default: the server listens on 127.0.0.1 only, with Flask debug off.
  DASHBOARD_HOST=<addr>  listen on another interface (only on a trusted network)
  DASHBOARD_PORT=<port>  listen on another port
  DASHBOARD_DEBUG=1      enable the Werkzeug debugger. WARNING: the debugger allows
                         arbitrary code execution for anyone who can reach the port, so it
                         is refused unless DASHBOARD_HOST is a loopback address.
"""

import os
import sys
import json
import ipaddress
import re
import yaml
from urllib.parse import urlsplit
from flask import Flask, render_template, jsonify
from flask import request  # WR-09: Host / Origin checks
from flask import abort, send_from_directory
from src.dashboard import demo

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'guardrails.jsonl')

# Frozen Phase 5 comparison. Read-only: only the files named here are ever served, so raw
# episodes (canary values), sources/ and inputs/ stay unreachable.
RESULTS_FINAL_DIR = os.path.join(PROJECT_ROOT, 'results', 'phase5-final')
RESULTS_FILES = frozenset({'numbers.json', 'freeze.json', 'summary.md', 'summary.csv', 'layer_blocks.csv'})
RESULTS_CHART = re.compile(r'charts/[A-Za-z0-9_-]+\.png')

app = Flask(__name__)

# One scripted policy simulation at a time (see src/dashboard/demo.py)
_simulation = demo.Simulation()


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


@app.route('/results')
def results_page():
    """Serve the Phase 5 comparison view; its data comes from /api/results."""
    info = {
        "model": _get_model_name(),
        "platform": sys.platform,
        "is_linux": sys.platform.startswith('linux')
    }
    return render_template('results.html', info=info)


@app.route('/api/results')
def get_results():
    """Return results/phase5-final/numbers.json unchanged. The dashboard only reads the
    frozen data; rebuilding it is `src.deliverables build`, never a web request."""
    path = os.path.join(RESULTS_FINAL_DIR, 'numbers.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "no frozen results", "path": "results/phase5-final/numbers.json"}), 404
    except (OSError, ValueError):
        return jsonify({"error": "results/phase5-final/numbers.json is unreadable"}), 500
    return jsonify(data)


@app.route('/api/results/files/<path:name>')
def get_results_file(name):
    """Serve one file of the frozen results by allow list. '..', absolute paths, backslashes
    and anything outside the list (raw.jsonl, sources/, inputs/) never match, so a request
    cannot leave results/phase5-final/ or read the episode data."""
    if name not in RESULTS_FILES and not RESULTS_CHART.fullmatch(name):
        abort(404)
    response = send_from_directory(RESULTS_FINAL_DIR, name, max_age=0)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.route('/demo')
def demo_page():
    """Serve the demo page: recorded-episode replay and the scripted policy simulation."""
    info = {
        "model": _get_model_name(),
        "platform": sys.platform,
        "is_linux": sys.platform.startswith('linux')
    }
    return render_template('demo.html', info=info)


def _frozen_numbers():
    with open(os.path.join(RESULTS_FINAL_DIR, 'numbers.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def _numbers_or_error():
    """(numbers, None) or (None, error response) for the frozen numbers.json."""
    try:
        return _frozen_numbers(), None
    except FileNotFoundError:
        return None, (jsonify({"error": "no frozen results", "path": "results/phase5-final/numbers.json"}), 404)
    except (OSError, ValueError):
        return None, (jsonify({"error": "results/phase5-final/numbers.json is unreadable"}), 500)


@app.route('/api/lineup')
def get_lineup():
    """Every model of the study, grouped (locally made, local pretrained, hosted, legacy, control).
    The live decision log records no model, so this is the only place the dashboard names models."""
    numbers, error = _numbers_or_error()
    if error:
        return error
    models_yaml = None
    try:
        with open(os.path.join(RESULTS_FINAL_DIR, 'inputs', 'config', 'models.yaml'), 'r', encoding='utf-8') as f:
            models_yaml = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        pass
    return jsonify({"models": demo.lineup(numbers, models_yaml), "data_state": numbers.get('data_state'),
                    "snapshot_display": numbers.get('snapshot_display')})


@app.route('/api/demo/replay')
def get_replay_index():
    """The replayable models (frozen runs only) and the pre-picked story episodes."""
    numbers, error = _numbers_or_error()
    if error:
        return error
    sources = demo.replay_sources(numbers, RESULTS_FINAL_DIR)
    return jsonify({"models": sorted(sources), "stories": demo.stories(sources)})


def _replay_path(numbers):
    """The raw.jsonl of the requested model, looked up (never joined) from the frozen source list."""
    model = request.args.get('model', '')
    return demo.replay_sources(numbers, RESULTS_FINAL_DIR).get(model)


@app.route('/api/demo/cases')
def get_replay_cases():
    numbers, error = _numbers_or_error()
    if error:
        return error
    path = _replay_path(numbers)
    if path is None:
        return jsonify({"error": "unknown model"}), 404
    return jsonify({"cases": demo.case_list(demo.load_records(path))})


@app.route('/api/demo/episode')
def get_replay_episode():
    """Both arms of one recorded episode plus the case text. Query: model, case, run (0-9)."""
    case_id = request.args.get('case', '')
    run_text = request.args.get('run', '0')
    if not demo.CASE_ID.fullmatch(case_id) or not run_text.isdigit() or int(run_text) > demo.MAX_RUN:
        return jsonify({"error": "bad case or run"}), 400
    numbers, error = _numbers_or_error()
    if error:
        return error
    path = _replay_path(numbers)
    if path is None:
        return jsonify({"error": "unknown model"}), 404
    arms = demo.episode(path, case_id, int(run_text))
    case = demo.case_text(case_id)
    if not arms or case is None:
        return jsonify({"error": "no such episode"}), 404
    return jsonify({"model": request.args.get('model'), "case": case, "run": int(run_text), "arms": arms})


@app.route('/api/demo/scenarios')
def get_scenarios():
    """The scripted scenarios of run_attack_simulation.py (read-only list)."""
    return jsonify({"scenarios": [{k: s[k] for k in ('id', 'description', 'action_type', 'target', 'malicious')}
                                  for s in demo.scenarios()]})


@app.route('/api/demo/simulate', methods=['GET', 'POST'])
def simulate():
    """GET: the current run. POST {"scenario": "all" | "s1".."s8"}: start one run.

    Same-origin only, like /api/clear. The body may name a scenario id and nothing else; the actions
    come from run_attack_simulation.py on the server and are only checked by the policy engine,
    never executed. One run at a time (409 while busy); a run stops at its deadline."""
    if request.method == 'GET':
        return jsonify(_simulation.status())
    if not _same_origin(request):
        return jsonify({"error": "cross-origin request refused"}), 403
    if request.query_string:
        return jsonify({"error": "no query parameters accepted"}), 400
    body = {}
    if request.get_data():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {'scenario'}:
            return jsonify({"error": "body must be {\"scenario\": <id>}"}), 400
    scenario = body.get('scenario', 'all')
    if not isinstance(scenario, str) or not demo.SCENARIO_ID.fullmatch(scenario):
        return jsonify({"error": "unknown scenario"}), 400
    try:
        started = _simulation.start(scenario, LOG_FILE)
    except KeyError:
        return jsonify({"error": "unknown scenario"}), 400
    if not started:
        return jsonify({"error": "a simulation is already running", "status": _simulation.status()}), 409
    return jsonify(_simulation.status()), 202


def _is_loopback(host: str) -> bool:
    """True for localhost and loopback IP literals (127.0.0.0/8, ::1)."""
    host = host.strip().strip('[]')
    if host.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _allowed_hostnames():
    """Host header names accepted while the server is bound to loopback, or None when the
    operator explicitly bound another interface (those Host values cannot be enumerated)."""
    host = os.environ.get('DASHBOARD_HOST') or DEFAULT_HOST
    if not _is_loopback(host):
        return None
    return {'localhost', '127.0.0.1', '::1', host.strip().strip('[]').lower()}


@app.before_request
def _reject_foreign_host():
    """DNS rebinding: a page on evil.example whose name resolves to 127.0.0.1 can read the
    API as 'same origin'. Its Host header still says evil.example, so refuse it."""
    allowed = _allowed_hostnames()
    if allowed is None:
        return None
    try:
        hostname = urlsplit('//' + request.host).hostname
    except ValueError:
        hostname = None
    if hostname is None or hostname.lower() not in allowed:
        return jsonify({"error": "unexpected Host header"}), 403
    return None


def _same_origin(req) -> bool:
    """Browsers label cross-site requests (Sec-Fetch-Site) and send Origin on POST. Refuse
    anything not from this page; clients that send neither header (curl, scripts) pass."""
    site = req.headers.get('Sec-Fetch-Site')
    if site is not None and site not in ('same-origin', 'none'):
        return False
    origin = req.headers.get('Origin')
    if origin is not None:
        return urlsplit(origin).netloc.lower() == req.host.lower()
    return True


@app.route('/api/clear', methods=['POST'])
def clear_logs():
    """API endpoint: clear all logs (for starting fresh demo runs). Same-origin only: a
    cross-site 'simple' POST from any web page could otherwise wipe the decision log."""
    global _log_cache
    if not _same_origin(request):
        return jsonify({"error": "cross-origin request refused"}), 403
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
    """Server settings: loopback only and debugger off unless explicitly overridden.

    Refuses (SystemExit with a message) a bad port, and the debugger on a non-loopback
    host: the Werkzeug debugger is remote code execution for anyone who can reach it.
    """
    host = os.environ.get('DASHBOARD_HOST') or DEFAULT_HOST
    raw_port = os.environ.get('DASHBOARD_PORT') or str(DEFAULT_PORT)
    try:
        port = int(raw_port)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        sys.exit(f"DASHBOARD_PORT must be an integer from 1 to 65535 (got {raw_port!r})")
    debug = os.environ.get('DASHBOARD_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if debug and not _is_loopback(host):
        sys.exit(f"Refusing to start: DASHBOARD_DEBUG enables the Werkzeug debugger (arbitrary code "
                 f"execution) and DASHBOARD_HOST={host!r} is not a loopback address. "
                 f"Unset DASHBOARD_DEBUG or bind to 127.0.0.1.")
    return {'host': host, 'port': port, 'debug': debug}


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
