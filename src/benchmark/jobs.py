"""
Benchmark jobs — restart-proof one-model runs and an ownership-safe Ollama server
================================================================================
Phase 4 runs every local model for 342 episodes. This module is the launcher those
runs go through:

  python -m src.benchmark.jobs ollama-up | ollama-down | ollama-variants
  python -m src.benchmark.jobs run --model M --out DIR [--smoke]
  python -m src.benchmark.jobs preflight --out DIR
  python -m src.benchmark.jobs ollama-default-probe [--model llama3.2:3b]

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
DONE_PREFLIGHT, DONE_OLLAMA_SETUP = 5, 6

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


def run(model: str, out_dir: str, smoke: bool = False, jobs_dir: str = DEFAULT_JOBS_DIR) -> int:
    """Run one model's benchmark into out_dir (resuming whatever is there) and return the done code.

    GPU models pass the read-only preflight first (done code 5 when it never passes). Ollama models need the
    measured default context (jobs ollama-default-probe) and get start and end context evidence."""
    cfg = model_config(model)
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
    if needs_gpu_preflight(cfg) and not preflight_with_rechecks(out_dir):
        print("preflight never passed; not starting the benchmark")
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\npreflight never passed\n")
        _write_done(out_dir, DONE_PREFLIGHT)
        return DONE_PREFLIGHT
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
        _append(progress, f"=== invocation {_utc_now()} model {model} ===\n")
        code, _ = _run(benchmark_argv(model, out_dir, smoke), timeout=7 * 24 * 3600, stdout_path=progress)
        _write_done(out_dir, code)
        print(f"BENCH {model} exit {code}")
        if ollama:
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
    args = parser.parse_args(argv)

    try:
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
