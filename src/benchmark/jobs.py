"""
Benchmark jobs — restart-proof one-model runs and an ownership-safe Ollama server
================================================================================
Phase 4 runs every local model for 342 episodes. This module is the launcher those
runs go through:

  python -m src.benchmark.jobs ollama-up | ollama-down | ollama-variants
  python -m src.benchmark.jobs run --model M --out DIR [--smoke]
  python -m src.benchmark.jobs preflight --out DIR
  python -m src.benchmark.jobs ollama-default-probe [--model llama3.2:3b]
  python -m src.benchmark.jobs local --prefix P [--models a,b] [--smoke]
  python -m src.benchmark.jobs status --prefix P [--models a,b] [--wait S]
  python -m src.benchmark.jobs report --prefix P --out DIR
  python -m src.benchmark.jobs hosted-probe [--models a,b] [--key-check] [--prefix P]
  python -m src.benchmark.jobs hosted --prefix P [--models a,b]

`hosted` re-runs the Groq models fresh (D-08) into P-<model>, data-stealing episodes first
(D-02), under the D-09 rule only: the hard freeze plus QuotaExhausted handling, written
once to <jobs-dir>/stopping_rule.json (RULE_TEXT). `hosted-probe --key-check` prints only
"KEY <name> set|missing" from the environment the children inherit and sends nothing.

`local` runs the five D-01 models one at a time (the GPU is serial, D-04) into P-<model>,
shortest GPU runs first for the D-09 freeze: it skips a complete directory, resumes an
incomplete one (at most twice per invocation), launches nothing at or after FREEZE_UTC
and at the freeze stops only the run child it started itself (done code 7). `status`
only reads; it prints a WARN no-progress line when a live run has written nothing for
more than STALE_MINUTES (D-04: the laptop must stay awake and on mains power). `report`
merges only the final P-<model> directories.

The preflight only reads (nvidia-smi free MiB, MemAvailable) and appends a KEEP_AWAKE
line for D-04 (keeping the laptop on mains power and awake is the user's job; boot_id
and uptime are logged so a VM restart is visible). The default-context probe measures
the installed Ollama server's version and the context a plain tag gets by default on a
temporary server of our own (first free port 11435-11440, no OLLAMA_CONTEXT_LENGTH),
stopped again in a finally block; the 11434 server, if any, is only asked its version.
Sources, first present wins: /api/ps context_length, the `ollama ps` CONTEXT column, the
OLLAMA_CONTEXT_LENGTH entry of the server's startup config line; if all are missing, the
prompt_tokens plateau of one unique long prompt. Two present sources that disagree are a
FAIL, never resolved by guessing.

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

Context evidence (D-06): GPU models (scratch, hf_local, localhost servers) pass the preflight
before the child starts. An Ollama run needs the measured default first and saves
ollama_ps_start/end (.json from /api/ps, .txt from `ollama ps`), ollama_show.txt,
ollama_list.txt and ollama_long_prompt.json: one unique long prompt sized to the middle of
run_checks.context_band(D, num_ctx), the same band `run_checks ollama-evidence` checks.
Ollama 0.40.3 cuts an over-long prompt to about half the window ("truncating input
prompt"), so a window that was not raised shows up below L, not as a plateau at the cap.
The variant is unloaded (keep_alive 0) only on a server this project owns.

Done-marker codes in DIR/bench.done: 0 the benchmark finished, 5 the preflight never
passed (no child was started), 6 the Ollama server or the context variant could not be
set up, 7 stopped at (or not launched because of) the D-09 freeze, anything else the
benchmark process's own exit code.

Every subprocess is an argv list (no shell), HTTP goes through urllib with short
timeouts, and the HTTP, /proc, signal, subprocess and clock helpers are module-level
functions so the tests replace them with fakes.
"""

import argparse
import datetime
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
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
DONE_PREFLIGHT, DONE_OLLAMA_SETUP, DONE_FREEZE = 5, 6, 7

# D-09: hard data freeze, 2026-10-29 15:30 UTC (21:00 IST). Nothing is launched at or after it, and at it every
# run this project started is stopped. A constant on purpose: the freeze is never moved.
FREEZE_UTC = "2026-10-29T15:30:00+00:00"
CHILD_POLL_S = 5            # how often a run with a deadline checks its child and the clock
FREEZE_STOP_WAIT_S = 60     # SIGTERM -> SIGKILL grace for our own child at the freeze

# Serial local queue (D-01 models; GPU serial per D-04; shortest GPU runs first for the freeze, D-09)
LOCAL_ORDER = ("qwen2.5-0.5b-agentlora", "scratch-gpt", "qwen2.5-0.5b-base", "ollama-qwen2.5-3b",
               "ollama-llama3.2-3b")
MAX_LAUNCHES_PER_MODEL = 2
STALE_MINUTES = 20          # D-04 keep-awake duty: a live run without progress this long gets a WARN line
STATUS_WAIT_CAP_S = 400

# Read-only GPU/RAM preflight (the src/qwen_lora/run_train.sh pattern); environment overrides are read per call
MIN_FREE_MIB = 3584
MIN_AVAIL_RAM_MIB = 3000
PREFLIGHT_RECHECKS = 6
PREFLIGHT_WAIT_S = 300
NVIDIA_QUERY = ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits"]

# Default-context probe: a temporary server of our own on the first free port, never 11434
PROBE_PORTS = tuple(range(11435, 11441))
PROBE_PIDFILE = "ollama-probe.pid"
DEFAULT_CONTEXT_FILE = "ollama_default_context.json"
EVIDENCE_SOURCES = ("api_ps", "ps_column", "config_line", "long_prompt_plateau")
PROBE_TARGET_TOKENS = 12000
CALIBRATION_LINES = (20, 60)
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


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _port_busy(port: int) -> bool:
    """True when anything accepts a TCP connection on 127.0.0.1:<port> (the holder is never touched)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((HOST, port)) == 0


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _clock() -> float:
    return time.monotonic()


def _now() -> datetime.datetime:
    """The one wall clock (aware UTC). Every freeze, staleness and quota comparison goes through it, so the
    tests replace it with a fake clock."""
    return datetime.datetime.now(datetime.timezone.utc)


def _utc_now() -> str:
    return _now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _list_pids() -> list:
    """Every numeric /proc entry. Read-only: used to find our own loops and children, never to signal."""
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def _spawn(argv: list, stdout_path: str):
    """Start a benchmark child with its output appended to stdout_path; the Popen handle is what may later be
    signalled (its pid only)."""
    with open(stdout_path, "ab") as out:
        return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=PROJECT_ROOT)


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
    """SIGTERM the owned pid, wait up to 30 s, SIGKILL that same pid only if it is still alive AND still passes
    the ownership check, remove the pidfile. Returns the stopped pid, or None when the pidfile names nothing this
    project owns (nothing is signalled then; a stale pidfile is removed).

    Ownership is checked again right before the SIGKILL (WR-02): when another process started the server
    (`jobs ollama-up` exits, a later `jobs ollama-down` stops it), the pid is not in _CHILDREN, so nothing holds
    it; the server can exit on SIGTERM and the pid can be reused within the wait. The pidfile is removed only
    after that check, because owned_pid reads it."""
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
    if _pid_alive(pid) and owned_pid(jobs_dir, pidfile_name) == pid:
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


# ── Read-only preflight (D-04) ──

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else default


def preflight(out_dir: str) -> bool:
    """Read the free GPU memory (nvidia-smi) and MemAvailable, append the reading and one KEEP_AWAKE line to
    out_dir/preflight.log, and say whether both are above MIN_FREE_MIB / MIN_AVAIL_RAM_MIB (environment
    overrides of the same names). It only reads: it never stops anything and changes no power setting; a
    busy GPU means waiting or a human decision."""
    os.makedirs(out_dir, exist_ok=True)
    min_free, min_ram = _env_int("MIN_FREE_MIB", MIN_FREE_MIB), _env_int("MIN_AVAIL_RAM_MIB", MIN_AVAIL_RAM_MIB)
    code, text = _run(NVIDIA_QUERY, timeout=30)
    reading = (text or "").strip().splitlines()[0].strip() if code == 0 and (text or "").strip() else ""
    parts = [p.strip() for p in reading.split(",")]
    free = int(parts[2]) if len(parts) >= 4 and parts[2].isdigit() else None
    meminfo = _read_text("/proc/meminfo") or ""
    m = re.search(r"^MemAvailable:\s+(\d+)\s*kB", meminfo, re.MULTILINE)
    avail = int(m.group(1)) // 1024 if m else 0
    boot_id = (_read_text("/proc/sys/kernel/random/boot_id") or "unknown").strip()
    uptime_text = (_read_text("/proc/uptime") or "").split()
    uptime = int(float(uptime_text[0])) if uptime_text else -1
    if free is None:
        result = f"not-ok no usable nvidia-smi reading, {min_free} MiB required"
    elif free < min_free:
        result = f"not-ok {free} MiB GPU free, {min_free} MiB required"
    elif avail < min_ram:
        result = f"not-ok {avail} MiB RAM available, {min_ram} MiB required"
    else:
        result = "ok"
    _append(os.path.join(out_dir, "preflight.log"),
            f"=== preflight {_utc_now()} ===\n"
            f"nvidia-smi: {reading or '<no reading>'}\n"
            f"free_mib {free if free is not None else 'unknown'}\n"
            f"avail_ram_mib {avail}\n"
            f"result {result}\n"
            f"KEEP_AWAKE D-04: keep the laptop on mains power and awake (user responsibility); "
            f"boot_id {boot_id} uptime_s {uptime}\n")
    print(f"PREFLIGHT {result} (free_mib {free}, avail_ram_mib {avail})")
    return result == "ok"


def preflight_with_rechecks(out_dir: str) -> bool:
    """The preflight, re-checked PREFLIGHT_RECHECKS more times PREFLIGHT_WAIT_S apart; it only waits."""
    attempts = PREFLIGHT_RECHECKS + 1
    for attempt in range(1, attempts + 1):
        if preflight(out_dir):
            return True
        if attempt < attempts:
            print(f"preflight attempt {attempt}/{attempts} failed; checking again in {PREFLIGHT_WAIT_S} s")
            _sleep(PREFLIGHT_WAIT_S)
    return False


# ── Long prompts and context readings ──

def long_prompt_text(lines: int, token: str) -> str:
    """A unique prompt (random token on the first line, so no prompt cache can shorten it) and numbered filler."""
    body = [f"{i:05d} filler line for the context probe; nothing to do here." for i in range(1, lines + 1)]
    return "\n".join([f"probe {token}", *body])


def long_prompt_tokens(port: int, model: str, lines: int) -> Optional[int]:
    """usage.prompt_tokens the server reports for one unique long prompt (max_tokens 1), or None."""
    reply = _http_json(f"http://{HOST}:{port}/v1/chat/completions",
                       {"model": model, "max_tokens": 1, "temperature": 0,
                        "messages": [{"role": "user", "content": long_prompt_text(lines, secrets.token_hex(8))}]},
                       timeout=900)
    usage = reply.get("usage") if isinstance(reply, dict) else None
    value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return value if isinstance(value, int) else None


def calibrate_lines(port: int, model: str) -> Optional[tuple]:
    """(base tokens, tokens per filler line) from two short prompts well inside any default window."""
    small, large = CALIBRATION_LINES
    a, b = long_prompt_tokens(port, model, small), long_prompt_tokens(port, model, large)
    if a is None or b is None or b <= a:
        return None
    per_line = (b - a) / (large - small)
    return a - per_line * small, per_line


def parse_ps_column(text: str, model: str) -> Optional[int]:
    """The CONTEXT column of `ollama ps` for `model`, or None when the column or the row is absent."""
    rows = [line for line in (text or "").splitlines() if line.strip()]
    if not rows:
        return None
    header = re.split(r"\s{2,}", rows[0].strip())
    if "CONTEXT" not in header:
        return None
    idx = header.index("CONTEXT")
    for line in rows[1:]:
        cells = re.split(r"\s{2,}", line.strip())
        if cells and cells[0] in (model, f"{model}:latest") and len(cells) > idx:
            return int(cells[idx]) if cells[idx].isdigit() else None
    return None


def parse_config_context(log_text: str) -> Optional[int]:
    """OLLAMA_CONTEXT_LENGTH from the last "server config" line of a server log. 0 (an automatic default) and
    an absent entry both give None: neither is a context length."""
    lines = [line for line in (log_text or "").splitlines() if "server config" in line]
    if not lines:
        return None
    m = re.search(r"OLLAMA_CONTEXT_LENGTH:(\d+)", lines[-1])
    return int(m.group(1)) if m and int(m.group(1)) > 0 else None


def _ps_entry(ps_json, model: str) -> Optional[dict]:
    entries = ps_json.get("models") if isinstance(ps_json, dict) else None
    for entry in entries or []:
        if isinstance(entry, dict) and model in (entry.get("name"), entry.get("model"),
                                                 str(entry.get("name", "")).removesuffix(":latest")):
            return entry
    return None


def _ps_context(port: int, model: str) -> tuple:
    """(/api/ps context_length, `ollama ps` CONTEXT column) of the loaded model; either may be None."""
    entry = _ps_entry(_http_json(f"http://{HOST}:{port}/api/ps", timeout=10), model)
    api = entry.get("context_length") if entry else None
    _, text = _run(["ollama", "ps"], env=client_env(port), timeout=30)
    return (api if isinstance(api, int) and api > 0 else None), parse_ps_column(text, model)


def _version_from(text: str) -> Optional[str]:
    m = re.search(r"client version is (\S+)", text or "") or re.search(r"version is (\S+)", text or "")
    return m.group(1) if m else None


def _load_model(port: int, model: str) -> None:
    _http_json(f"http://{HOST}:{port}/api/generate",
               {"model": model, "prompt": "ok", "stream": False, "options": {"num_predict": 1}}, timeout=600)


# ── Default-context probe ──

def _measure_default(jobs_dir: str, port: int, model: str, log_name: str) -> dict:
    """Every reading of the temporary server; the decision on the default is made by the caller."""
    m = {"server_version": ollama_listening(port), "client_version": None, "config_context_length": None,
         "ps_context_length": None, "ps_column_context": None, "show_num_ctx": None, "long_prompt_tokens": None,
         "long_prompt_sent_lines": None, "long_prompt_sent_tokens": None}
    _, text = _run(["ollama", "--version"], env=client_env(port), timeout=30)
    m["client_version"] = _version_from(text)
    m["config_context_length"] = parse_config_context(_read_text(os.path.join(jobs_dir, log_name)))
    _load_model(port, model)
    m["ps_context_length"], m["ps_column_context"] = _ps_context(port, model)
    if None in (m["ps_context_length"], m["ps_column_context"], m["config_context_length"]):
        _, shown = _run(["ollama", "show", model, "--parameters"], env=client_env(port), timeout=60)
        m["show_num_ctx"] = _show_num_ctx(shown)
        cal = calibrate_lines(port, model)
        if cal is not None:
            base, per_line = cal
            lines = max(1, int((PROBE_TARGET_TOKENS - base) / per_line))
            m["long_prompt_sent_lines"] = lines
            m["long_prompt_sent_tokens"] = int(base + per_line * lines)
            m["long_prompt_tokens"] = long_prompt_tokens(port, model, lines)
    return m


def ollama_default_probe(jobs_dir: str = DEFAULT_JOBS_DIR, model: str = "llama3.2:3b") -> int:
    """Measure the installed Ollama server's version and the context `model` gets by default, on a temporary
    server this project starts (first free port 11435-11440, no OLLAMA_CONTEXT_LENGTH, ownership marker) and
    stops again, also when a step fails. The 11434 server, if any, is only asked for its version. Writes
    <jobs_dir>/ollama_default_context.json. Returns 0, 1 (FAIL) or 5 (preflight not ok)."""
    model = _checked_name(model)
    os.makedirs(jobs_dir, exist_ok=True)
    if not preflight(jobs_dir):
        print("FAIL ollama-default-probe: preflight not ok (see preflight.log); nothing was started")
        return DONE_PREFLIGHT
    main_version = ollama_listening(MAIN_PORT)
    main_state = "none" if not main_version else (
        "owned" if owned_pid(jobs_dir, MAIN_PIDFILE) is not None else "foreign")
    stale = stop_owned(jobs_dir, PROBE_PIDFILE)        # only a crashed earlier probe of ours, by marker
    if stale is not None:
        print(f"TEMP_SERVER stale probe server stopped pid {stale}")
    port = next((p for p in PROBE_PORTS if not _port_busy(p)), None)
    if port is None:
        print(f"FAIL ollama-default-probe: no free port in {PROBE_PORTS[0]}-{PROBE_PORTS[-1]} (holders left alone)")
        return 1
    log_name = f"ollama-probe-{port}.log"
    pid = start_owned_server(jobs_dir, port, None, log_name, PROBE_PIDFILE)
    stopped = None
    try:
        m = _measure_default(jobs_dir, port, model, log_name)
    finally:
        stopped = stop_owned(jobs_dir, PROBE_PIDFILE)
        print(f"TEMP_SERVER stopped pid {stopped}")

    present = [(s, v) for s, v in (("api_ps", m["ps_context_length"]), ("ps_column", m["ps_column_context"]),
                                   ("config_line", m["config_context_length"])) if v is not None]
    default, source, lower_bound, problem = None, None, False, None
    if len({v for _, v in present}) > 1:
        problem = "sources disagree: " + ", ".join(f"{s} {v}" for s, v in present)
    elif present:
        source, default = present[0]
    elif m["long_prompt_tokens"] is not None:
        source, default = "long_prompt_plateau", m["long_prompt_tokens"]
        lower_bound = m["long_prompt_tokens"] >= 0.97 * (m["long_prompt_sent_tokens"] or 0)
    else:
        problem = "no source reported a context length (api_ps, ps_column, config_line, long prompt all missing)"
    data = {"server_version": m["server_version"], "client_version": m["client_version"],
            "port_11434": {"version": main_version, "state": main_state}, "model": model, "port": port,
            "default_context_length": default, "default_is_lower_bound": lower_bound,
            "evidence_source": source if problem is None else "none",
            "ps_context_length": m["ps_context_length"], "ps_column_context": m["ps_column_context"],
            "config_context_length": m["config_context_length"], "show_num_ctx": m["show_num_ctx"],
            "long_prompt_tokens": m["long_prompt_tokens"], "long_prompt_sent_lines": m["long_prompt_sent_lines"],
            "temp_server_stopped": stopped == pid and not _pid_alive(pid), "measured_at": _utc_now()}
    if problem is not None:
        data["finding"] = problem
    tmp = os.path.join(jobs_dir, DEFAULT_CONTEXT_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, os.path.join(jobs_dir, DEFAULT_CONTEXT_FILE))
    print(f"SERVER_VERSION {data['server_version']} CLIENT_VERSION {data['client_version']}")
    if problem is not None:
        print(f"FAIL ollama-default-probe: {problem}")
        return 1
    print(f"DEFAULT_CONTEXT {model} {default} source {source}"
          + (" (lower bound: the long prompt was not cut)" if lower_bound else ""))
    return 0


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


def needs_gpu_preflight(cfg: dict) -> bool:
    """Models that run on this machine's GPU: scratch, hf_local and OpenAI-compatible servers on localhost.
    Scripted and hosted models skip the preflight."""
    provider = cfg.get("provider")
    if provider in ("scratch", "hf_local"):
        return True
    if provider == "openai_compat":
        host = urllib.parse.urlparse(str(cfg.get("base_url") or "")).hostname
        return host in ("localhost", "127.0.0.1")
    return False


TOKENS_PER_LINE_GUESS = 16   # the first long-prompt size is a guess; up to two corrections use the measurement


def size_long_prompt(port: int, model: str, lower: int, upper: int) -> dict:
    """Send one unique long prompt aimed at the middle of [lower, upper], correcting the line count at most
    twice from the reported prompt_tokens. A window that was not raised cuts the prompt (Ollama keeps about
    half the window), so prompt_tokens stays below `lower` and the evidence check fails, as it should."""
    target = (lower + upper) // 2
    lines = max(1, target // TOKENS_PER_LINE_GUESS)
    attempts, tokens = [], None
    for _ in range(3):
        tokens = long_prompt_tokens(port, model, lines)
        attempts.append([lines, tokens])
        if tokens is None or lower <= tokens <= upper:
            break
        lines = max(1, int(lines * target / max(tokens, 1)))
    return {"prompt_tokens": tokens, "lines": attempts[-1][0], "attempts": attempts}


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def capture_ps(out_dir: str, variant: str, suffix: str, port: int = MAIN_PORT) -> None:
    """Warm the variant up (one token), then save /api/ps as JSON and `ollama ps` as text."""
    _load_model(port, variant)
    _write_json(os.path.join(out_dir, f"ollama_ps_{suffix}.json"), _http_json(f"http://{HOST}:{port}/api/ps",
                                                                               timeout=10))
    _, text = _run(["ollama", "ps"], env=client_env(port), timeout=30)
    _write_text(os.path.join(out_dir, f"ollama_ps_{suffix}.txt"), text)


def capture_start_evidence(out_dir: str, jobs_dir: str, cfg: dict, state: str, default_context: dict,
                           port: int = MAIN_PORT) -> dict:
    """ps (start), `ollama show <variant>`, `ollama list` and the long-prompt probe sized from the measured
    default: the same context_band the ollama-evidence check uses."""
    from src.benchmark.run_checks import band_note, context_band  # lazy: run_checks imports this module
    variant, num_ctx = _checked_name(cfg["model"]), int(cfg["num_ctx"])
    capture_ps(out_dir, variant, "start", port)
    _, text = _run(["ollama", "show", variant], env=client_env(port), timeout=60)
    _write_text(os.path.join(out_dir, "ollama_show.txt"), text)
    _, text = _run(["ollama", "list"], env=client_env(port), timeout=60)
    _write_text(os.path.join(out_dir, "ollama_list.txt"), text)
    d = int(default_context["default_context_length"])
    lower, upper = context_band(d, num_ctx)
    note = band_note(d, num_ctx)
    if note:
        print(note)
    sized = size_long_prompt(port, variant, lower, upper)
    server_ctx = None
    if state.startswith("owned"):
        server_ctx = parse_config_context(_read_text(os.path.join(jobs_dir, MAIN_LOG)))
    data = {"model": variant, "num_ctx": num_ctx, "D": d, "L": lower, "U": upper, **sized,
            "server_state": state, "server_context_length": server_ctx, "note": note, "measured_at": _utc_now()}
    _write_json(os.path.join(out_dir, "ollama_long_prompt.json"), data)
    print(f"LONG_PROMPT {variant} prompt_tokens {sized['prompt_tokens']} band {lower}-{upper} D {d} "
          f"attempts {len(sized['attempts'])}")
    return data


def _parse_utc(value) -> datetime.datetime:
    """An aware UTC datetime from an ISO string ('Z' or an offset) or a datetime."""
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        dt = datetime.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def freeze_time() -> datetime.datetime:
    return _parse_utc(FREEZE_UTC)


def past_freeze() -> bool:
    return _now() >= freeze_time()


def _fmt_utc(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stop_own_child(proc, wait_s: float = FREEZE_STOP_WAIT_S) -> None:
    """SIGTERM the pid of a Popen this process created, wait up to wait_s, then SIGKILL that same pid. Nothing
    else is ever signalled: the handle is the proof of ownership."""
    if proc.poll() is not None:
        return
    _send_signal(proc.pid, signal.SIGTERM)
    waited = 0.0
    while waited < wait_s and proc.poll() is None:
        _sleep(1)
        waited += 1
    if proc.poll() is None:
        _send_signal(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass


def run_child_until(argv: list, progress: str, deadline) -> int:
    """Run the benchmark child, polling it every CHILD_POLL_S; at the deadline stop only that child, log the
    freeze line and return DONE_FREEZE (7). The clock is checked before the spawn too (WR-03): nothing is
    launched at or after the deadline, whatever the caller did before."""
    if _now() >= deadline:
        _append(progress, f"FREEZE not launched {_utc_now()} (D-09)\n")
        return DONE_FREEZE
    proc = _spawn(argv, progress)
    while True:
        code = proc.poll()
        if code is not None:
            return code
        if _now() >= deadline:
            stop_own_child(proc)
            _append(progress, f"FREEZE stopped at {_utc_now()} (D-09)\n")
            return DONE_FREEZE
        _sleep(CHILD_POLL_S)


def run(model: str, out_dir: str, smoke: bool = False, jobs_dir: str = DEFAULT_JOBS_DIR, deadline=None) -> int:
    """Run one model's benchmark into out_dir (resuming whatever is there) and return the done code.

    GPU models pass the read-only preflight first (done code 5 when it never passes). Ollama models need the
    measured default context (jobs ollama-default-probe) and get start and end context evidence. With a
    deadline (FREEZE_UTC for the Phase 4 queue) the child is polled and, at the deadline, only that child is
    stopped and bench.done is 7; nothing is launched when the deadline has already passed. The deadline is
    checked again after the preflight (it can wait 6 x 300 s) and after the Ollama setup (server start,
    variant, long-prompt probes), so a launch that starts shortly before the freeze cannot start a child
    after it (WR-03); a server this run started is still stopped by the finally block."""
    cfg = model_config(model)
    deadline = _parse_utc(deadline) if deadline is not None else None
    out_dir = os.path.abspath(out_dir)
    jobs_dir = os.path.abspath(jobs_dir)
    ollama = is_ollama_model(cfg)
    default_context = None
    if ollama:
        default_context = _read_default_context(jobs_dir)
        if default_context is None:
            print("FAIL run: measure the default first (jobs ollama-default-probe)")
            return 1
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "bench.done")
    if os.path.exists(done_path):
        os.remove(done_path)
    progress = os.path.join(out_dir, "progress.log")

    def freeze_reached(stage: str) -> bool:
        """True (and the freeze line and done code 7 written) when the deadline has passed."""
        if deadline is None or _now() < deadline:
            return False
        print(f"FREEZE {model}: not launched (D-09{stage})")
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\nFREEZE not launched (D-09{stage})\n")
        _write_done(out_dir, DONE_FREEZE)
        return True

    if freeze_reached(""):
        return DONE_FREEZE
    if needs_gpu_preflight(cfg) and not preflight_with_rechecks(out_dir):
        print("preflight never passed; not starting the benchmark")
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\npreflight never passed\n")
        _write_done(out_dir, DONE_PREFLIGHT)
        return DONE_PREFLIGHT
    if freeze_reached(", reached during the preflight"):
        return DONE_FREEZE
    started_server, state = False, None
    try:
        if ollama:
            try:
                state = ollama_up(jobs_dir)
                started_server = state == "owned-started"
                print(f"OLLAMA {state}")
                ollama_variants(jobs_dir, [model])
                capture_start_evidence(out_dir, jobs_dir, cfg, state, default_context)
            except JobError as e:
                print(f"FAIL run: {e}")
                _append(progress, f"=== invocation {_utc_now()} model {model} ===\nFAIL ollama setup: {e}\n")
                _write_done(out_dir, DONE_OLLAMA_SETUP)
                return DONE_OLLAMA_SETUP
            if freeze_reached(", reached during the Ollama setup"):
                return DONE_FREEZE
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\n")
        if deadline is None:
            code, _ = _run(benchmark_argv(model, out_dir, smoke), timeout=7 * 24 * 3600, stdout_path=progress)
        else:
            code = run_child_until(benchmark_argv(model, out_dir, smoke), progress, deadline)
        _write_done(out_dir, code)
        print(f"BENCH {model} exit {code}" + (" (stopped at the freeze, D-09)" if code == DONE_FREEZE
                                               and deadline is not None else ""))
        if ollama and not (deadline is not None and code == DONE_FREEZE):
            capture_ps(out_dir, _checked_name(cfg["model"]), "end")
        return code
    finally:
        if ollama and state and state.startswith("owned"):
            # unload the variant (keep_alive 0) only on a server this project owns
            _http_json(f"http://{HOST}:{MAIN_PORT}/api/generate", {"model": cfg["model"], "keep_alive": 0},
                       timeout=60)
        if started_server:
            ollama_down(jobs_dir)


def _read_default_context(jobs_dir: str) -> Optional[dict]:
    path = os.path.join(jobs_dir, DEFAULT_CONTEXT_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or type(data.get("default_context_length")) is not int:
        return None
    return data


# ── Process discovery (read-only /proc scans) ──

def _cmdline(pid: int) -> list:
    raw = _read_proc(pid, "cmdline")
    if not raw:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _ppid(pid: int) -> Optional[int]:
    stat = _read_proc(pid, "stat")
    if not stat:
        return None
    after = stat.rsplit(b")", 1)[-1].split()
    return int(after[1]) if len(after) > 1 and after[1].isdigit() else None


def _ancestors(pid: int) -> set:
    """pid and every ancestor (a `timeout` or shell wrapper of this process is not another instance)."""
    seen = set()
    while pid and pid not in seen:
        seen.add(pid)
        pid = _ppid(pid)
    return seen


def _arg_value(argv: list, flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


def _is_python(argv: list) -> bool:
    return bool(argv) and "python" in os.path.basename(argv[0])


def find_loop(cmd: str, prefix: str, exclude_self: bool = True) -> Optional[int]:
    """The pid of a running `python -m src.benchmark.jobs <cmd> --prefix <prefix>` other than this process and
    its wrappers, or None. Only reads /proc."""
    skip = _ancestors(os.getpid()) if exclude_self else set()
    for pid in _list_pids():
        if pid in skip:
            continue
        argv = _cmdline(pid)
        if not _is_python(argv) or "src.benchmark.jobs" not in argv:
            continue
        i = argv.index("src.benchmark.jobs")
        if i + 1 < len(argv) and argv[i + 1] == cmd and _same_path(_arg_value(argv, "--prefix"), prefix):
            return pid
    return None


def find_bench_child(out_dir: str) -> Optional[int]:
    """The pid of a running run_benchmark.py whose --resume is out_dir, or None. Only reads /proc."""
    for pid in _list_pids():
        argv = _cmdline(pid)
        if (_is_python(argv) and any(a.endswith("run_benchmark.py") for a in argv)
                and _same_path(_arg_value(argv, "--resume"), out_dir)):
            return pid
    return None


# ── Serial local queue (D-04, D-09) ──

def model_dir(prefix: str, model: str) -> str:
    return f"{prefix.rstrip('/')}-{model}"


def _jobs_log(jobs_dir: str, kind: str, prefix: str) -> str:
    return os.path.join(jobs_dir, f"{kind}-{os.path.basename(prefix.rstrip('/'))}.log")


def model_status(out_dir: str, model: str) -> dict:
    """run_checks.complete_status of one directory (expected from its manifest); the tests replace this."""
    from src.benchmark.run_checks import complete_status  # lazy: run_checks imports this module
    return complete_status(out_dir, model)


def is_complete(out_dir: str, model: str) -> bool:
    """Every expected episode finished and no infra-errored key left (the skip rule of the queues). A provenance
    problem does not reopen a finished model: relaunching cannot fix it, so it is reported (provenance_problems,
    PROVENANCE lines) and still fails the strict `run_checks complete` (WR-04)."""
    from src.benchmark.run_checks import progress_complete  # lazy: run_checks imports this module
    if not os.path.isdir(out_dir):
        return False
    return progress_complete(model_status(out_dir, model))


def provenance_problems(out_dir: str, model: str) -> list:
    """The problems of complete_status a relaunch cannot fix (stamps, policy, resolved model, commit drift,
    duplicates); empty when there is no directory."""
    if not os.path.isdir(out_dir):
        return []
    return list(model_status(out_dir, model).get("provenance") or [])


def local_queue(prefix: str, models: Optional[list] = None, smoke: bool = False,
                jobs_dir: str = DEFAULT_JOBS_DIR) -> int:
    """Run the local models one at a time (GPU serial, D-04) into <prefix>-<model>, in LOCAL_ORDER unless
    `models` is given. A complete directory is skipped; an incomplete one is resumed, at most
    MAX_LAUNCHES_PER_MODEL times per invocation and never again after a freeze stop (code 7); nothing is
    launched at or after FREEZE_UTC (D-09). The Ollama server is brought up once before the first Ollama
    launch and, only if this invocation started it, stopped after the last model."""
    models = list(models or LOCAL_ORDER)
    os.makedirs(jobs_dir, exist_ok=True)
    log_path = _jobs_log(jobs_dir, "local", prefix)

    def say(line: str) -> None:
        print(line, flush=True)
        _append(log_path, line + "\n")

    other = find_loop("local", prefix)
    if other is not None:
        say(f"LOCAL already running pid {other}")
        return 0
    say(f"=== local {_utc_now()} prefix {prefix} models {','.join(models)} smoke {smoke} freeze {FREEZE_UTC} ===")
    freeze = freeze_time()
    complete, ollama_state = 0, None
    try:
        for model in models:
            cfg = model_config(model)
            out = model_dir(prefix, model)
            if is_complete(out, model):
                say(f"[local] skip {model}: complete")
                for problem in provenance_problems(out, model):
                    say(f"PROVENANCE FAIL {model}: {problem}")
                complete += 1
                continue
            launches = 0
            while launches < MAX_LAUNCHES_PER_MODEL:
                if _now() >= freeze:
                    say(f"[local] freeze {model}: not launched (D-09)")
                    break
                if is_ollama_model(cfg) and ollama_state is None:
                    try:
                        ollama_state = ollama_up(jobs_dir)
                    except JobError as e:
                        ollama_state = "failed"
                        say(f"[local] FAIL ollama-up: {e}")
                    say(f"[local] ollama {ollama_state}")
                launches += 1
                say(f"[local] launch {model} {launches}/{MAX_LAUNCHES_PER_MODEL} -> {out}")
                code = run(model, out, smoke=smoke, jobs_dir=jobs_dir, deadline=freeze)
                say(f"[local] {model} done {code}")
                if is_complete(out, model):
                    say(f"[local] complete {model}")
                    for problem in provenance_problems(out, model):
                        say(f"PROVENANCE FAIL {model}: {problem}")
                    complete += 1
                    break
                if code == DONE_FREEZE:
                    say(f"[local] freeze {model}: stopped at the freeze, not relaunched (D-09)")
                    break
            else:
                say(f"[local] incomplete {model}: {MAX_LAUNCHES_PER_MODEL} launches used, moving on")
    finally:
        if ollama_state == "owned-started":
            say(f"[local] ollama-down {ollama_down(jobs_dir)}")
    say(f"LOCAL done: {complete}/{len(models)} complete")
    return 0


# ── Read-only status (D-04 keep-awake warning) ──

def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _last_line(path: str) -> str:
    text = _read_text(path) or ""
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else "-"


def _done_code(out_dir: str) -> Optional[str]:
    text = _read_text(os.path.join(out_dir, "bench.done"))
    return text.strip() if text is not None and text.strip() else None


def _read_json_file(path: str):
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _done_snapshot(prefix: str, models: list) -> dict:
    return {m: (_mtime(os.path.join(model_dir(prefix, m), "bench.done")),
                _done_code(model_dir(prefix, m))) for m in models}


def _status_models(prefix: str) -> list:
    """Every config model with a <prefix>-<model> directory, else the local queue's models."""
    names = list(load_model_config()["models"])
    ordered = list(LOCAL_ORDER) + [n for n in names if n not in LOCAL_ORDER]
    present = [m for m in ordered if os.path.isdir(model_dir(prefix, m))]
    return present or list(LOCAL_ORDER)


def status(prefix: str, models: Optional[list] = None, wait_s: float = 0) -> str:
    """Print one line per model, the keep-awake WARN / quota WAIT lines, the time to the freeze and one STATUS
    line; return the STATUS word. Only reads (files and /proc): it never starts, changes or signals anything.
    all-complete is about progress (every model finished, no infra key left); a provenance problem of a model
    is printed as its own "PROVENANCE FAIL <model>: <reason>" line (WR-04)."""
    from src.benchmark.run_checks import progress_complete  # lazy: run_checks imports this module
    models = list(models or _status_models(prefix))
    wait_s = max(0.0, min(float(wait_s or 0), STATUS_WAIT_CAP_S))
    if wait_s:
        before = _done_snapshot(prefix, models)
        waited = 0.0
        while waited < wait_s and _done_snapshot(prefix, models) == before:
            _sleep(5)
            waited += 5
    now = _now()
    loops = {kind: find_loop(kind, prefix, exclude_self=False) for kind in ("local", "hosted")}
    any_alive, all_complete, any_data = False, True, False
    for model in models:
        out = model_dir(prefix, model)
        exists = os.path.isdir(out)
        st = model_status(out, model) if exists else {"finished": 0, "expected": None, "pending_infra": 0,
                                                      "problems": ["no directory"]}
        child = find_bench_child(out) if exists else None
        alive = child is not None
        any_alive = any_alive or alive
        done = _done_code(out) if exists else None
        complete = exists and progress_complete(st)       # WR-04: progress only; provenance printed below
        all_complete = all_complete and complete
        any_data = any_data or (exists and (os.path.exists(os.path.join(out, "raw.jsonl")) or done is not None))
        expected = st.get("expected") if st.get("expected") is not None else "-"
        print(f"MODEL {model} finished {st.get('finished', 0)}/{expected} pending_infra {st.get('pending_infra', 0)} "
              f"done {done or '-'} alive {'yes' if alive else 'no'}")
        for problem in st.get("provenance") or []:
            print(f"PROVENANCE FAIL {model}: {problem}")
        if exists:
            print(f"  last: {_last_line(os.path.join(out, 'progress.log'))[:200]}")
        hosted = _read_json_file(os.path.join(out, "hosted_state.json")) if exists else None
        next_launch = hosted.get("next_launch_utc") if isinstance(hosted, dict) else None
        if isinstance(hosted, dict) and hosted.get("state") == "waiting_quota" and next_launch \
                and _parse_utc(next_launch) > now:
            print(f"WAIT {model} quota until {_fmt_utc(_parse_utc(next_launch))}")
        elif alive:
            stamps = [t for t in (_mtime(os.path.join(out, "raw.jsonl")), _mtime(os.path.join(out, "progress.log")))
                      if t is not None]
            if stamps:
                idle_min = int((now.timestamp() - max(stamps)) // 60)
                if idle_min > STALE_MINUTES:
                    print(f"WARN no-progress {model} {idle_min} min: check that the laptop is awake and on mains "
                          f"power (D-04)")
    for kind, pid in loops.items():
        if pid is not None:
            any_alive = True
            print(f"LOOP {kind} alive pid {pid}")
    left = freeze_time() - now
    if left.total_seconds() > 0:
        minutes = int(left.total_seconds() // 60)
        print(f"FREEZE in {minutes // 60}h{minutes % 60}m")
    else:
        print("FREEZE passed")
    word = "all-complete" if all_complete else "running" if any_alive else "incomplete" if any_data else "idle"
    print(f"STATUS {word}")
    return word


# ── Merged report ──

def report_sources(prefix: str, out_dir: Optional[str] = None) -> list:
    """The final Phase 4 directories that exist: <prefix>-<every benchmark model> and <prefix>-control. Never a
    tracer, smoke, jobs or report directory: only these exact names are considered, so <prefix>-tracer-*,
    <prefix>-smoke-*, <prefix>-jobs and the report directory itself can never be merged."""
    from src.llm.registry import list_models
    names = list(list_models(benchmark_only=True)) + ["control"]
    sources = []
    for name in names:
        path = model_dir(prefix, name)
        if out_dir and _same_path(path, out_dir):
            continue
        if os.path.isdir(path) and path not in sources:
            sources.append(path)
    return sources


def report(prefix: str, out_dir: str) -> int:
    from src.benchmark.report import generate_report  # lazy: matplotlib is only needed here
    sources = report_sources(prefix, out_dir)
    if not sources:
        print(f"FAIL report: no Phase 4 directory {prefix}-<model> exists")
        return 1
    for path in sources:
        print(f"MERGE {path}")
    generate_report(out_dir, sources=sources)
    print(f"REPORT {out_dir} sources {len(sources)}")
    return 0


# ── Hosted re-run under the D-09 stopping rule (D-02, D-08) ──

HOSTED_MODELS = ("groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.8-27b")
# complete_with_problems (WR-04): every episode finished, but complete_status names a provenance problem (policy
# drift, run-affecting commit drift, a second resolved model, ...). Relaunching cannot fix that, so it is terminal.
HOSTED_COMPLETE = ("complete", "complete_with_problems")
HOSTED_TERMINAL = HOSTED_COMPLETE + ("stopped_by_quota", "stopped_at_freeze", "stalled", "unavailable")
HOSTED_EXPECTED = 342
QUOTA_MARGIN_S = 120            # after a stated retry time, so the quota has really reset
QUOTA_DEFAULT_WAIT_S = 3600     # a quota stop that states no retry time
RELAUNCH_WAIT_S = 600           # any other exit of an incomplete model
STALL_LAUNCHES = 3              # consecutive launches with zero new finished episodes and no quota stop
HOSTED_POLL_S = 10
STOPPING_RULE_FILE = "stopping_rule.json"
HOSTED_STATE_FILE = "hosted_state.json"
SUPERSEDED_DIR = os.path.join("results", "final-20261011")   # D-08: October rows, never resumed or written
RULE_TEXT = ("D-09: hard freeze at freeze_utc (no launch at or after it; at it SIGTERM to this loop's own children "
             "only, then SIGKILL of the same pids after 60 s; unfinished models are stopped_at_freeze). "
             "QuotaExhausted: next launch = stop time + stated retry time + 120 s, or + 60 min when no retry time "
             "is stated; a next launch at or after the freeze ends the model as stopped_by_quota. Any other exit "
             "relaunches after 10 min (not at or after the freeze); 3 consecutive launches with no new finished "
             "episode and no quota stop end the model as stalled. Incomplete models are reported as partial with "
             "their n.")
_QUOTA_STOP_RE = re.compile(r"\[stop\] (\S+): quota exhausted \((.*)\)\. Re-run with --resume")
_RETRY_HOURS_RE = re.compile(r"API asks to retry in ([\d.]+)h")
_AUTH_RE = re.compile(r"\b401\b|\b403\b|invalid[_ ]api[_ ]key|unauthori[sz]ed|authentication|permission[_ ]denied",
                      re.IGNORECASE)
_UNAVAILABLE_RE = re.compile(r"\b404\b|model[_ ]not[_ ]found|does not exist|decommissioned|not found|no such model",
                             re.IGNORECASE)


def _under(path: str, root: str) -> bool:
    p, r = os.path.normpath(os.path.abspath(path)), os.path.normpath(os.path.abspath(root))
    return p == r or p.startswith(r + os.sep)


def hosted_argv(model: str, out_dir: str) -> list:
    """The Phase 1 arguments (D-05), data-stealing episodes first (D-02), resuming the model's own directory."""
    return [sys.executable, "-u", os.path.join(PROJECT_ROOT, "run_benchmark.py"), "--models", _checked_name(model),
            "--resume", out_dir, *FULL_ARGS, "--priority-kinds", "ds"]


def read_hosted_state(out_dir: str) -> dict:
    data = _read_json_file(os.path.join(out_dir, HOSTED_STATE_FILE))
    if not isinstance(data, dict):
        data = {}
    data.setdefault("launches", [])
    data.setdefault("state", "pending")
    data.setdefault("next_launch_utc", None)
    return data


def write_hosted_state(out_dir: str, state: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, HOSTED_STATE_FILE)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(path + ".tmp", path)


def _git_head() -> str:
    code, text = _run(["git", "rev-parse", "HEAD"], timeout=30)
    sha = (text or "").strip()
    return sha if code == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"


def ensure_stopping_rule(jobs_dir: str, models: list, dirs: dict) -> dict:
    """Write <jobs_dir>/stopping_rule.json once (at the first hosted launch) and reuse it unchanged afterwards. A
    stored freeze that differs from FREEZE_UTC is refused: the freeze is never moved (D-09)."""
    path = os.path.join(jobs_dir, STOPPING_RULE_FILE)
    existing = _read_json_file(path)
    if isinstance(existing, dict):
        stored = existing.get("freeze_utc")
        try:
            same = stored is not None and _parse_utc(stored) == freeze_time()
        except ValueError:
            same = False
        if not same:
            raise JobError(f"{path} holds freeze_utc {stored!r} but FREEZE_UTC is {FREEZE_UTC}; the freeze is never "
                           f"moved (D-09), so this loop does not start")
        return existing
    rule = {"written_utc": _utc_now(), "freeze_utc": FREEZE_UTC, "rule_sha": _git_head(), "models": list(models),
            "directories": [dirs[m] for m in models], "superseded": SUPERSEDED_DIR.replace(os.sep, "/"),
            "rule": RULE_TEXT}
    os.makedirs(jobs_dir, exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rule, f, indent=2)
    os.replace(path + ".tmp", path)
    return rule


def parse_quota_stop(text: str) -> tuple:
    """(quota stopped?, stated retry seconds or None) from the output of one child invocation."""
    m = _QUOTA_STOP_RE.search(text or "")
    if not m:
        return False, None
    hours = _RETRY_HOURS_RE.search(m.group(2))
    return True, (float(hours.group(1)) * 3600 if hours else None)


def _finished(out_dir: str, model: str) -> int:
    if not os.path.exists(os.path.join(out_dir, "raw.jsonl")):
        return 0
    return int(model_status(out_dir, model).get("finished") or 0)


def _read_from(path: str, offset: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def hosted_loop(prefix: str, models: Optional[list] = None, jobs_dir: str = DEFAULT_JOBS_DIR) -> int:
    """Re-run the Groq models fresh (D-08) into <prefix>-<model>, data-stealing first (D-02), one own
    run_benchmark child per model in parallel (separate per-model quotas), under the D-09 rule only (RULE_TEXT).
    Signals go only to the Popen handles this process created. Never prints an environment value."""
    models = list(models or HOSTED_MODELS)
    for model in models:
        model_config(model)
    dirs = {m: os.path.abspath(model_dir(prefix, m)) for m in models}
    if _under(prefix, SUPERSEDED_DIR) or any(_under(d, SUPERSEDED_DIR) for d in dirs.values()):
        print(f"FAIL hosted: {prefix} resolves under {SUPERSEDED_DIR}; the October rows are superseded and are never "
              f"resumed or written (D-08)")
        return 1
    os.makedirs(jobs_dir, exist_ok=True)
    log_path = _jobs_log(jobs_dir, "hosted", prefix)

    def say(line: str) -> None:
        print(line, flush=True)
        _append(log_path, line + "\n")

    other = find_loop("hosted", prefix)
    if other is not None:
        say(f"HOSTED already running pid {other}")
        return 0
    freeze = freeze_time()
    if _now() >= freeze:
        say(f"HOSTED freeze passed ({FREEZE_UTC}): nothing launched (D-09)")
        return 0
    try:
        ensure_stopping_rule(jobs_dir, models, dirs)
    except JobError as e:
        say(f"FAIL hosted: {e}")
        return 1
    say(f"=== hosted {_utc_now()} prefix {prefix} models {','.join(models)} freeze {FREEZE_UTC} ===")
    states = {m: read_hosted_state(dirs[m]) for m in models}
    procs = {}   # model -> (Popen, launch record, progress offset, finished before)

    for model, st in states.items():
        if st["state"] == "running":       # the previous loop ended without recording this launch's end
            if st["launches"] and st["launches"][-1].get("end") is None:
                st["launches"][-1].update(end=_utc_now(), stop_reason="loop restarted; launch end not recorded")
            st["state"] = "pending"
            write_hosted_state(dirs[model], st)

    def launch(model: str) -> None:
        out = dirs[model]
        os.makedirs(out, exist_ok=True)
        progress = os.path.join(out, "progress.log")
        _append(progress, f"=== invocation {_utc_now()} model {model} (hosted, freeze {FREEZE_UTC}) ===\n")
        before = _finished(out, model)
        offset = _size(progress)
        proc = _spawn(hosted_argv(model, out), progress)
        record = {"start": _utc_now(), "end": None, "exit": None, "pid": proc.pid, "finished_before": before,
                  "new_finished": None, "quota_stop": None, "stated_retry_s": None, "stop_reason": None}
        st = states[model]
        st["launches"].append(record)
        st["state"], st["next_launch_utc"] = "running", None
        write_hosted_state(out, st)
        procs[model] = (proc, record, offset, before)
        say(f"[hosted] launch {model} #{len(st['launches'])} pid {proc.pid}")

    def completed_state(model: str) -> str:
        """complete, or complete_with_problems (each provenance problem logged as a PROVENANCE FAIL line); WR-04."""
        problems = provenance_problems(dirs[model], model)
        for problem in problems:
            say(f"PROVENANCE FAIL {model}: {problem}")
        return "complete_with_problems" if problems else "complete"

    def finish(model: str, code: int) -> None:
        proc, record, offset, before = procs.pop(model)
        out, st, now = dirs[model], states[model], _now()
        text = _read_from(os.path.join(out, "progress.log"), offset)
        quota, retry_s = parse_quota_stop(text)
        after = _finished(out, model)
        record.update(end=_utc_now(), exit=code, new_finished=after - before, quota_stop=quota,
                      stated_retry_s=retry_s)
        if is_complete(out, model):
            st["state"] = record["stop_reason"] = completed_state(model)
        elif quota:
            next_launch = now + datetime.timedelta(seconds=(retry_s + QUOTA_MARGIN_S) if retry_s is not None
                                                   else QUOTA_DEFAULT_WAIT_S)
            if next_launch >= freeze:
                st["state"] = "stopped_by_quota"
                record["stop_reason"] = (f"quota stop; the next launch {_fmt_utc(next_launch)} is at or after the "
                                         f"freeze (D-09)")
            else:
                st["state"], st["next_launch_utc"] = "waiting_quota", _fmt_utc(next_launch)
                record["stop_reason"] = f"quota stop; waiting until {_fmt_utc(next_launch)}"
        else:
            streak = 0
            for item in reversed(st["launches"]):
                if item.get("quota_stop") or item.get("new_finished") is None or item.get("new_finished") > 0:
                    break
                streak += 1
            next_launch = now + datetime.timedelta(seconds=RELAUNCH_WAIT_S)
            if streak >= STALL_LAUNCHES:
                st["state"], record["stop_reason"] = "stalled", f"{streak} launches without a new finished episode"
            elif next_launch >= freeze:
                st["state"], record["stop_reason"] = "stopped_at_freeze", "a relaunch would fall at the freeze"
            else:
                st["state"], st["next_launch_utc"] = "waiting_retry", _fmt_utc(next_launch)
                record["stop_reason"] = f"exit {code}; relaunch at {_fmt_utc(next_launch)}"
        write_hosted_state(out, st)
        say(f"[hosted] {model} exit {code} new {record['new_finished']} quota {'yes' if quota else 'no'} "
            f"state {st['state']}")

    while True:
        now = _now()
        if now >= freeze:
            for model in list(procs):
                proc, record = procs[model][0], procs[model][1]
                stop_own_child(proc)
                record.update(end=_utc_now(), exit=proc.poll(), stop_reason="stopped at the freeze (D-09)")
                _append(os.path.join(dirs[model], "progress.log"), f"FREEZE stopped at {_utc_now()} (D-09)\n")
                procs.pop(model)
            for model, st in states.items():
                if st["state"] not in HOSTED_TERMINAL:
                    st["state"], st["next_launch_utc"] = "stopped_at_freeze", None
                    write_hosted_state(dirs[model], st)
            say(f"HOSTED freeze reached {_utc_now()}: own children stopped, nothing more launched (D-09)")
            break
        for model in models:
            st = states[model]
            if model in procs:
                code = procs[model][0].poll()
                if code is not None:
                    finish(model, code)
                continue
            if st["state"] in HOSTED_TERMINAL:
                continue
            due = st.get("next_launch_utc")
            if due and _parse_utc(due) > now:
                continue
            if find_bench_child(dirs[model]) is not None:
                continue          # a run this loop did not start is already on the directory: never a duplicate
            if is_complete(dirs[model], model):
                st["state"] = completed_state(model)
                write_hosted_state(dirs[model], st)
                say(f"[hosted] skip {model}: {st['state']}")
                continue
            launch(model)
        if not procs and all(states[m]["state"] in HOSTED_TERMINAL for m in models):
            break
        if procs:
            _sleep(HOSTED_POLL_S)
        else:
            dues = [_parse_utc(s["next_launch_utc"]) for s in states.values()
                    if s["state"] not in HOSTED_TERMINAL and s.get("next_launch_utc")]
            wake = min(dues + [freeze])
            _sleep(max(1.0, min(60.0, (wake - _now()).total_seconds())))
    for model in models:
        st = states[model]
        say(f"[hosted] {model} state {st['state']} finished {_finished(dirs[model], model)}/{HOSTED_EXPECTED}")
    say(f"HOSTED done: {sum(1 for m in models if states[m]['state'] in HOSTED_COMPLETE)}/{len(models)} complete")
    return 0


def _probe_class(error: Optional[str]) -> str:
    from src.benchmark.runner import _is_rate_limited
    if not error:
        return "ok"
    if _is_rate_limited(error):
        return "rate_limited"
    if _AUTH_RE.search(error):
        return "auth"
    if _UNAVAILABLE_RE.search(error):
        return "unavailable"
    return "error"


def hosted_probe(models: Optional[list] = None, key_check: bool = False, prefix: Optional[str] = None) -> int:
    """KEY lines first, then (without --key-check) one tiny request per model, printing only a class.

    Key presence is an environment test on the variable named by api_key_env, the in-process form of a shell
    [ -n ] test on the environment the hosted children inherit. The registry's usual start-up (load_model_config
    in src/llm/registry.py, whose module start-up loads .env through python-dotenv) runs first, so this sees the
    same environment the children see. This function opens no .env itself and prints set or missing, never a
    value. With a prefix, an unavailable model is recorded in <prefix>-<model>/hosted_state.json and the hosted
    loop never launches it; no substitute model is chosen (D-02)."""
    configs = load_model_config()["models"]           # the provider registry's existing start-up (W1)
    models = list(models or HOSTED_MODELS)
    for model in models:
        if model not in configs:
            raise JobError(f"unknown model {model!r} (not in config/models.yaml)")
    names = list(dict.fromkeys(str(configs[m].get("api_key_env")) for m in models if configs[m].get("api_key_env")))
    for name in names:
        print(f"KEY {name} {'set' if os.environ.get(name) else 'missing'}")
    if key_check:
        return 0
    from src.llm.registry import get_provider
    blocked = False
    for model in models:
        try:
            provider = get_provider(model)
            response = provider.generate([{"role": "user", "content": "Reply with the word OK."}], max_tokens=8)
            cls = _probe_class(response.error)
        except Exception as e:   # a missing key or a broken entry: report the class only, never the message
            cls = "auth" if "key" in type(e).__name__.lower() or "key" in str(e).lower() else "error"
        print(f"PROBE {model} {cls}")
        blocked = blocked or cls == "auth"
        if cls == "unavailable" and prefix:
            out = os.path.abspath(model_dir(prefix, model))
            if not (_under(prefix, SUPERSEDED_DIR) or _under(out, SUPERSEDED_DIR)):
                st = read_hosted_state(out)
                st["state"], st["next_launch_utc"] = "unavailable", None
                write_hosted_state(out, st)
    return 1 if blocked else 0


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
    p = sub.add_parser("preflight")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ollama-default-probe")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--model", default="llama3.2:3b")
    p = sub.add_parser("local")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--prefix", required=True)
    p.add_argument("--models", default=None)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--prefix", required=True)
    p.add_argument("--models", default=None)
    p.add_argument("--wait", type=float, default=0)
    p = sub.add_parser("report")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("hosted-probe")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--models", default=None)
    p.add_argument("--key-check", action="store_true")
    p.add_argument("--prefix", default=None)
    p = sub.add_parser("hosted")
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--prefix", required=True)
    p.add_argument("--models", default=None)
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if getattr(args, "models", None) else None

    try:
        if args.cmd == "hosted-probe":
            return hosted_probe(models, key_check=args.key_check, prefix=args.prefix)
        if args.cmd == "hosted":
            return hosted_loop(args.prefix, models, jobs_dir=args.jobs_dir)
        if args.cmd == "local":
            return local_queue(args.prefix, models, smoke=args.smoke, jobs_dir=args.jobs_dir)
        if args.cmd == "status":
            status(args.prefix, models, wait_s=args.wait)
            return 0
        if args.cmd == "report":
            return report(args.prefix, args.out)
        if args.cmd == "preflight":
            return 0 if preflight(args.out) else 1
        if args.cmd == "ollama-default-probe":
            return ollama_default_probe(args.jobs_dir, args.model)
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
