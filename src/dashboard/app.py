"""
Dashboard — Real-Time Log/Alert Visualization
================================================
A Flask web application that displays the policy engine's decisions
in real-time. Shows what the AI agent tried to do and whether each
action was Allowed, Blocked, or Flagged.

Usage:
  python -m src.dashboard.app
  Then open http://localhost:5000 in your browser
"""

import os
import sys
import json
from flask import Flask, render_template, jsonify

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'guardrails.jsonl')

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/logs')
def get_logs():
    """API endpoint: return all log entries as JSON array.
    
    The dashboard polls this endpoint every 2 seconds to update.
    """
    logs = []
    # Only read if the log file exists
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Ignore invalid JSON lines gracefully
                        pass
    # Return most recent first so the newest logs are at the top
    logs.reverse()
    return jsonify(logs)


@app.route('/api/stats')
def get_stats():
    """API endpoint: return summary statistics."""
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    
    # Calculate totals and categories
    stats = {
        "total": len(logs),
        "allowed": sum(1 for l in logs if l.get("decision") == "ALLOW"),
        "blocked": sum(1 for l in logs if l.get("decision") == "BLOCK"),
        "flagged": sum(1 for l in logs if l.get("decision") == "FLAG"),
    }
    return jsonify(stats)


@app.route('/api/clear', methods=['POST'])
def clear_logs():
    """API endpoint: clear all logs (for starting fresh demo runs)."""
    if os.path.exists(LOG_FILE):
        # Open in write mode and close to truncate the file
        open(LOG_FILE, 'w').close()
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  OS-Level Guardrails — Dashboard")
    print(f"{'='*50}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Open:     http://localhost:5000")
    print(f"{'='*50}\n")
    # Start the Flask development server on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)
