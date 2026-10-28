"""
Benchmark jobs — restart-proof one-model runs and an ownership-safe Ollama server
================================================================================
Phase 4 runs every local model for 342 episodes. This module is the launcher those
runs go through:

  python -m src.benchmark.jobs ollama-up | ollama-down | ollama-variants
  python -m src.benchmark.jobs run --model M --out DIR [--smoke]

Why runs are foreground processes: WSL shuts its VM down when no attached client is
left, which kills detached background work (Phase 2 lost runs this way). `run` is
therefore started from a wsl.exe call that stays attached for the whole run, writes
everything to DIR/progress.log and leaves a done marker, so a lost run is visible and
is resumed by repeating the same command (run_benchmark --resume skips finished
episodes).

Why only owned processes are ever signalled: another project's job and possibly a
foreign Ollama server share this machine. An Ollama server this project starts gets a
minimal environment carrying the marker GUARDRAILS_OLLAMA_OWNER=os_level_guardrails
and its pid is written to a pidfile under the jobs directory. A signal is sent only to
a pid that is in that pidfile AND whose /proc/<pid>/environ carries the marker AND whose
command line is `ollama serve`, so a reused pid or a foreign server is never touched. A
foreign server is used as it is: the Modelfile variants (FROM <pulled tag>, PARAMETER
num_ctx 8192) raise the context window without restarting or reconfiguring it.

Done-marker codes in DIR/bench.done: 0 the benchmark finished, 5 the preflight never
passed (no child was started), 6 the Ollama server or the context variant could not be
set up, anything else the benchmark process's own exit code.

Every subprocess is an argv list (no shell), HTTP goes through urllib with short
timeouts, and the HTTP, /proc, signal, subprocess and clock helpers are module-level
functions so the tests replace them with fakes.
"""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from src.llm.registry import PROJECT_ROOT, load_model_config

DEFAULT_JOBS_DIR = os.path.join("results", "phase4-jobs")
OWNER_KEY = "GUARDRAILS_OLLAMA_OWNER"
OWNER_VALUE = "os_level_guardrails"
HOST = "127.0.0.1"
MAIN_PORT = 11434
MAIN_CONTEXT_LENGTH = 8192
MAIN_LOG, MAIN_PIDFILE = "ollama-serve.log", "ollama.pid"
SERVER_START_WAIT_S = 60
STOP_WAIT_S = 30
MINIMAL_ENV_KEYS = ("PATH", "HOME", "LANG")
DONE_PREFLIGHT, DONE_OLLAMA_SETUP = 5, 6
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")

# Phase 1 arguments (D-05) and the 6-episode smoke subset of them
FULL_ARGS = ("--runs", "3", "--dh", "20", "--ds", "20", "--benign", "17", "--seed", "0", "--max-steps", "5")
SMOKE_ARGS = ("--runs", "1", "--dh", "1", "--ds", "1", "--benign", "1", "--seed", "0", "--max-steps", "5")


class JobError(RuntimeError):
    """A setup step failed; the message is printed as a FAIL line."""


# ── Injectable helpers (tests replace these) ──

_CHILDREN = {}   # pid -> Popen of servers started by this process, so a stopped child is reaped, not a zombie


def _http_json(url: str, payload: Optional[dict] = None, timeout: float = 3.0):
    """GET (payload None) or POST JSON; the decoded reply, or None on any connection or HTTP failure."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "null")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _read_proc(pid: int, name: str) -> Optional[bytes]:
    try:
        with open(f"/proc/{pid}/{name}", "rb") as f:
            return f.read()
    except OSError:
        return None


def _send_signal(pid: int, sig) -> None:
    os.kill(pid, sig)


def _popen(argv: list, env: dict, log_path: str):
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    _CHILDREN[proc.pid] = proc
    return proc


def _run(argv: list, env: Optional[dict] = None, timeout: float = 120.0, stdout_path: Optional[str] = None):
    """(returncode, text). With stdout_path the output is appended to that file instead and text is ''."""
    try:
        if stdout_path:
            with open(stdout_path, "ab") as out:
                done = subprocess.run(argv, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout,
                                      cwd=PROJECT_ROOT)
            return done.returncode, ""
        done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except FileNotFoundError as e:
        return 127, str(e)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {argv[0]}"


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _clock() -> float:
    return time.monotonic()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pid_alive(pid: int) -> bool:
    child = _CHILDREN.get(pid)
    if child is not None:
        return child.poll() is None
    stat = _read_proc(pid, "stat")
    if stat is None:
        return False
    # field 3 of /proc/<pid>/stat is the state; Z (zombie) and X (dead) are not running
    after = stat.rsplit(b")", 1)[-1].split()
    return bool(after) and after[0] not in (b"Z", b"X")


# ── Config ──

def model_config(model: str) -> dict:
    models = load_model_config()["models"]
    if model not in models:
        raise JobError(f"unknown model {model!r} (not in config/models.yaml)")
    return models[model]


def is_ollama_model(cfg: dict) -> bool:
    return bool(cfg.get("ollama_base")) and bool(cfg.get("num_ctx"))


def ollama_models() -> dict:
    """{name: cfg} of every config entry served as an Ollama context variant."""
    return {name: cfg for name, cfg in load_model_config()["models"].items() if is_ollama_model(cfg)}


def _checked_name(value) -> str:
    """A model or tag name safe to pass as one argv element (never an option, no spaces)."""
    text = str(value)
    if not _NAME_RE.fullmatch(text):
        raise JobError(f"refusing unexpected model name {text!r}")
    return text


# ── Ollama ownership ──

def ollama_listening(port: int = MAIN_PORT) -> Optional[str]:
    """The version string of the Ollama server answering on 127.0.0.1:<port>, or None."""
    reply = _http_json(f"http://{HOST}:{port}/api/version", timeout=3.0)
    if isinstance(reply, dict) and reply.get("version"):
        return str(reply["version"])
    return None


def server_env(port: int, context_length: Optional[int] = None) -> dict:
    """The minimal environment of a server this project starts: no API keys, no parent variables beyond
    PATH, HOME and LANG, the ownership marker, one request and one loaded model at a time."""
    env = {k: os.environ[k] for k in MINIMAL_ENV_KEYS if k in os.environ}
    env.update({"OLLAMA_HOST": f"{HOST}:{port}", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
                OWNER_KEY: OWNER_VALUE})
    if context_length is not None:
        env["OLLAMA_CONTEXT_LENGTH"] = str(int(context_length))
    return env


def client_env(port: int) -> dict:
    """Environment for the `ollama` CLI talking to the server on <port>."""
    return dict(os.environ, OLLAMA_HOST=f"{HOST}:{port}")


def start_owned_server(jobs_dir: str, port: int, context_length: Optional[int] = None,
                       log_name: str = MAIN_LOG, pidfile_name: str = MAIN_PIDFILE) -> int:
    """The one place that starts `ollama serve`. Returns the pid once /api/version answers (60 s at most);
    a server that never answers is stopped again (it is ours) and JobError is raised."""
    os.makedirs(jobs_dir, exist_ok=True)
    proc = _popen(["ollama", "serve"], server_env(port, context_length), os.path.join(jobs_dir, log_name))
    with open(os.path.join(jobs_dir, pidfile_name), "w", encoding="utf-8") as f:
        f.write(f"{proc.pid}\n")
    deadline = _clock() + SERVER_START_WAIT_S
    while _clock() < deadline:
        if ollama_listening(port):
            return proc.pid
        if not _pid_alive(proc.pid):
            break
        _sleep(1)
    stop_owned(jobs_dir, pidfile_name)
    raise JobError(f"ollama serve on port {port} did not answer within {SERVER_START_WAIT_S}s "
                   f"(see {os.path.join(jobs_dir, log_name)})")


def _pidfile_pid(jobs_dir: str, pidfile_name: str) -> Optional[int]:
    try:
        with open(os.path.join(jobs_dir, pidfile_name), encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def owned_pid(jobs_dir: str, pidfile_name: str = MAIN_PIDFILE) -> Optional[int]:
    """The pid in the pidfile, only when that process is `ollama serve` AND carries the ownership marker.
    A pid reused by any other process (or one without the marker) is not ours."""
    pid = _pidfile_pid(jobs_dir, pidfile_name)
    if pid is None:
        return None
    cmdline = _read_proc(pid, "cmdline")
    environ = _read_proc(pid, "environ")
    if not cmdline or not environ:
        return None
    args = cmdline.split(b"\0")
    if not any(a.endswith(b"ollama") for a in args) or b"serve" not in args:
        return None
    marker = f"{OWNER_KEY}={OWNER_VALUE}".encode()
    return pid if marker in environ.split(b"\0") else None


def stop_owned(jobs_dir: str, pidfile_name: str = MAIN_PIDFILE) -> Optional[int]:
    """SIGTERM the owned pid, wait up to 30 s, SIGKILL that same pid only if it is still alive, remove the
    pidfile. Returns the stopped pid, or None when the pidfile names nothing this project owns (nothing is
    signalled then; a stale pidfile is removed)."""
    pid = owned_pid(jobs_dir, pidfile_name)
    path = os.path.join(jobs_dir, pidfile_name)
    if pid is None:
        if os.path.exists(path) and _pidfile_pid(jobs_dir, pidfile_name) is not None:
            os.remove(path)  # stale: the process is gone or is no longer ours
        return None
    _send_signal(pid, signal.SIGTERM)
    deadline = _clock() + STOP_WAIT_S
    while _clock() < deadline and _pid_alive(pid):
        _sleep(0.5)
    if _pid_alive(pid):
        _send_signal(pid, signal.SIGKILL)
        _sleep(0.5)
        _pid_alive(pid)  # reap a child
    _CHILDREN.pop(pid, None)
    if os.path.exists(path):
        os.remove(path)
    return pid


def ollama_up(jobs_dir: str) -> str:
    """owned-started | owned-running | foreign. Starts a server (OLLAMA_CONTEXT_LENGTH 8192, marker) only
    when nothing answers on 127.0.0.1:11434; a foreign server is used as it is."""
    if ollama_listening(MAIN_PORT):
        return "owned-running" if owned_pid(jobs_dir, MAIN_PIDFILE) is not None else "foreign"
    start_owned_server(jobs_dir, MAIN_PORT, MAIN_CONTEXT_LENGTH, MAIN_LOG, MAIN_PIDFILE)
    return "owned-started"


def ollama_down(jobs_dir: str) -> str:
    """Stop the main server only when this project owns it; otherwise send nothing."""
    pid = stop_owned(jobs_dir, MAIN_PIDFILE)
    if pid is not None:
        print(f"OLLAMA stopped pid {pid}")
        return "stopped"
    if ollama_listening(MAIN_PORT):
        print("OLLAMA not-owned: left running")
        return "not-owned"
    print("OLLAMA none")
    return "none"


# ── Context variants ──

def modelfile_text(cfg: dict) -> str:
    return f"FROM {_checked_name(cfg['ollama_base'])}\nPARAMETER num_ctx {int(cfg['num_ctx'])}\n"


def _show_num_ctx(text: str) -> Optional[int]:
    """num_ctx from `ollama show --parameters` / `ollama show` output, or None."""
    m = re.search(r"^\s*num_ctx\s+(\d+)\s*$", text or "", re.MULTILINE)
    return int(m.group(1)) if m else None


def ollama_variants(jobs_dir: str, models: Optional[list] = None, port: int = MAIN_PORT) -> list:
    """Create (idempotently) the context variant of every Ollama config entry, or only of `models`, and
    confirm `ollama show <variant> --parameters` lists the configured num_ctx. Raises JobError on failure."""
    entries = ollama_models()
    if models is not None:
        entries = {name: entries[name] for name in models if name in entries}
    folder = os.path.join(jobs_dir, "modelfiles")
    os.makedirs(folder, exist_ok=True)
    lines = []
    for name, cfg in entries.items():
        variant = _checked_name(cfg["model"])
        num_ctx = int(cfg["num_ctx"])
        path = os.path.join(folder, f"{variant.replace(':', '_').replace('/', '_')}.Modelfile")
        with open(path, "w", encoding="utf-8") as f:
            f.write(modelfile_text(cfg))
        # The base tag must already be pulled: `ollama create` would otherwise download it, and this project
        # downloads nothing here.
        base = _checked_name(cfg["ollama_base"])
        code, out = _run(["ollama", "show", base, "--modelfile"], env=client_env(port), timeout=60)
        if code != 0:
            raise JobError(f"base tag {base} is not available locally (not pulled?): {out.strip()[-200:]}")
        code, out = _run(["ollama", "create", variant, "-f", path], env=client_env(port), timeout=600)
        if code != 0:
            raise JobError(f"ollama create {variant} exited {code}: {out.strip()[-300:]}")
        code, out = _run(["ollama", "show", variant, "--parameters"], env=client_env(port), timeout=60)
        shown = _show_num_ctx(out) if code == 0 else None
        if shown != num_ctx:
            raise JobError(f"ollama show {variant} lists num_ctx {shown}, expected {num_ctx}")
        line = f"VARIANT {variant} from {cfg['ollama_base']} num_ctx {num_ctx} ok"
        print(line)
        lines.append(line)
    return lines


# ── One-model run ──

def benchmark_argv(model: str, out_dir: str, smoke: bool = False) -> list:
    return [sys.executable, "-u", os.path.join(PROJECT_ROOT, "run_benchmark.py"), "--models", _checked_name(model),
            "--resume", out_dir, *(SMOKE_ARGS if smoke else FULL_ARGS)]


def _write_done(out_dir: str, code: int) -> None:
    with open(os.path.join(out_dir, "bench.done"), "w", encoding="utf-8") as f:
        f.write(f"{code}\n")


def _append(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def run(model: str, out_dir: str, smoke: bool = False, jobs_dir: str = DEFAULT_JOBS_DIR) -> int:
    """Run one model's benchmark into out_dir (resuming whatever is there) and return the done code."""
    cfg = model_config(model)
    out_dir = os.path.abspath(out_dir)
    jobs_dir = os.path.abspath(jobs_dir)
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "bench.done")
    if os.path.exists(done_path):
        os.remove(done_path)
    progress = os.path.join(out_dir, "progress.log")
    started_server = False
    try:
        if is_ollama_model(cfg):
            try:
                state = ollama_up(jobs_dir)
                started_server = state == "owned-started"
                print(f"OLLAMA {state}")
                ollama_variants(jobs_dir, [model])
            except JobError as e:
                print(f"FAIL run: {e}")
                _append(progress, f"=== invocation {_utc_now()} model {model} ===\nFAIL ollama setup: {e}\n")
                _write_done(out_dir, DONE_OLLAMA_SETUP)
                return DONE_OLLAMA_SETUP
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\n")
        code, _ = _run(benchmark_argv(model, out_dir, smoke), timeout=7 * 24 * 3600, stdout_path=progress)
        _write_done(out_dir, code)
        print(f"BENCH {model} exit {code}")
        return code
    finally:
        if started_server:
            ollama_down(jobs_dir)


# ── CLI ──

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.jobs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ollama-up", "ollama-down", "ollama-variants"):
        p = sub.add_parser(name)
        p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p = sub.add_parser("run")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "ollama-up":
            print(f"OLLAMA {ollama_up(args.jobs_dir)}")
            return 0
        if args.cmd == "ollama-down":
            ollama_down(args.jobs_dir)
            return 0
        if args.cmd == "ollama-variants":
            ollama_variants(args.jobs_dir)
            return 0
        if args.cmd == "run":
            return run(args.model, args.out, smoke=args.smoke, jobs_dir=args.jobs_dir)
    except JobError as e:
        print(f"FAIL {args.cmd}: {e}")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
