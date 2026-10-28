"""
Tests for src/benchmark/jobs.py: Ollama server ownership, context variants and the
one-model launcher. Everything is faked (HTTP, /proc, signals, subprocesses, clock):
no real server, no GPU, no benchmark child.
"""

import json
import os
import signal

import pytest

from src.benchmark import jobs
from src.llm.registry import list_models, load_model_config

MARKER = f"{jobs.OWNER_KEY}={jobs.OWNER_VALUE}"


class FakeWorld:
    """A fake machine: HTTP servers by port, processes by pid, recorded signals and subprocess calls."""

    def __init__(self):
        self.listening = {}          # port -> version
        self.procs = {}              # pid -> {"cmdline": bytes, "environ": bytes, "alive": bool}
        self.signals = []            # (pid, sig)
        self.popens = []             # (argv, env, log_path)
        self.runs = []               # (argv, env, stdout_path)
        self.run_results = {}        # first argv words tuple -> (code, text)
        self.http_posts = []         # (url, payload)
        self.http_replies = {}       # url suffix -> reply
        self.now = 0.0
        self.next_pid = 4000
        self.ports = {}              # pid -> port of servers started through the fake popen
        self.term_kills = True       # SIGTERM ends the process
        self.child_code = 0

    def add_proc(self, pid, cmdline=(b"/usr/local/bin/ollama", b"serve"), environ=(b"PATH=/usr/bin",), alive=True):
        self.procs[pid] = {"cmdline": b"\0".join(cmdline) + b"\0", "environ": b"\0".join(environ) + b"\0",
                           "alive": alive}

    # injected helpers
    def http_json(self, url, payload=None, timeout=3.0):
        if payload is not None:
            self.http_posts.append((url, payload))
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.listening:
            return None
        path = "/" + url.split("/", 3)[3]
        if path == "/api/version":
            return {"version": self.listening[port]}
        for suffix, reply in self.http_replies.items():
            if url.endswith(suffix):
                return reply(url, payload) if callable(reply) else reply
        return {}

    def read_proc(self, pid, name):
        p = self.procs.get(pid)
        if p is None or (not p["alive"] and name != "stat"):
            return None
        if name == "stat":
            return f"{pid} (ollama) {'S' if p['alive'] else 'Z'} 1".encode() if p["alive"] else None
        return p.get(name)

    def send_signal(self, pid, sig):
        self.signals.append((pid, sig))
        if pid in self.procs and (sig == signal.SIGKILL or self.term_kills):
            self.procs[pid]["alive"] = False
            self.listening.pop(self.ports.get(pid), None)

    def popen(self, argv, env, log_path):
        pid = self.next_pid
        self.next_pid += 1
        self.popens.append((argv, env, log_path))
        environ = tuple(f"{k}={v}".encode() for k, v in env.items())
        self.add_proc(pid, cmdline=tuple(a.encode() for a in argv), environ=environ)
        port = int(env["OLLAMA_HOST"].rsplit(":", 1)[1])
        self.listening[port] = "0.99.0"
        self.ports[pid] = port

        class P:
            pass
        proc = P()
        proc.pid = pid
        return proc

    def run(self, argv, env=None, timeout=120.0, stdout_path=None):
        self.runs.append((list(argv), env, stdout_path))
        if stdout_path:
            with open(stdout_path, "a", encoding="utf-8") as f:
                f.write("child output\n")
            return self.child_code, ""
        for n in (3, 2, 1):
            key = tuple(argv[:n])
            if key in self.run_results:
                result = self.run_results[key]
                return result(argv) if callable(result) else result
        if argv[:2] == ["ollama", "show"] and "--parameters" in argv:
            return 0, "num_ctx                        8192\nstop    \"<|im_end|>\"\n"
        return 0, ""

    def sleep(self, seconds):
        self.now += seconds

    def clock(self):
        return self.now


@pytest.fixture
def world(monkeypatch):
    w = FakeWorld()
    monkeypatch.setattr(jobs, "_http_json", w.http_json)
    monkeypatch.setattr(jobs, "_read_proc", w.read_proc)
    monkeypatch.setattr(jobs, "_send_signal", w.send_signal)
    monkeypatch.setattr(jobs, "_popen", w.popen)
    monkeypatch.setattr(jobs, "_run", w.run)
    monkeypatch.setattr(jobs, "_sleep", w.sleep)
    monkeypatch.setattr(jobs, "_clock", w.clock)
    monkeypatch.setattr(jobs, "_CHILDREN", {})
    return w


def _write_pid(jobs_dir, pid, name="ollama.pid"):
    os.makedirs(jobs_dir, exist_ok=True)
    with open(os.path.join(jobs_dir, name), "w", encoding="utf-8") as f:
        f.write(f"{pid}\n")


# ── Config ──

def test_config_has_both_ollama_variants_and_no_descoped_models():
    models = load_model_config()["models"]
    bench = list_models(benchmark_only=True)
    for name, base, variant in (("ollama-qwen2.5-3b", "qwen2.5:3b", "qwen2.5-3b-ctx8192"),
                                ("ollama-llama3.2-3b", "llama3.2:3b", "llama3.2-3b-ctx8192")):
        assert name in bench
        cfg = models[name]
        assert cfg["num_ctx"] == 8192 and cfg["ollama_base"] == base and cfg["model"] == variant
        assert cfg["timeout"] == 180 and cfg["benchmark"] is True
    assert not [n for n in models if "phi3.5" in n or "gemma3" in n]
    assert not [c for c in models.values() if "phi3.5" in str(c.get("model")) or "gemma3" in str(c.get("model"))]


def test_modelfile_text_for_both_config_entries():
    entries = jobs.ollama_models()
    assert set(entries) == {"ollama-qwen2.5-3b", "ollama-llama3.2-3b"}
    assert jobs.modelfile_text(entries["ollama-qwen2.5-3b"]) == "FROM qwen2.5:3b\nPARAMETER num_ctx 8192\n"
    assert jobs.modelfile_text(entries["ollama-llama3.2-3b"]) == "FROM llama3.2:3b\nPARAMETER num_ctx 8192\n"


# ── Ownership ──

def test_nothing_listening_starts_one_marked_server_with_minimal_env(world, tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_LIKE_VAR", "x")
    jobs_dir = str(tmp_path / "jobs")
    assert jobs.ollama_up(jobs_dir) == "owned-started"
    assert len(world.popens) == 1
    argv, env, log_path = world.popens[0]
    assert argv == ["ollama", "serve"]
    assert env[jobs.OWNER_KEY] == jobs.OWNER_VALUE
    assert env["OLLAMA_CONTEXT_LENGTH"] == "8192"
    assert env["OLLAMA_NUM_PARALLEL"] == "1" and env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    allowed = set(jobs.MINIMAL_ENV_KEYS) | {"OLLAMA_HOST", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
                                            "OLLAMA_CONTEXT_LENGTH", jobs.OWNER_KEY}
    assert set(env) <= allowed and "SOME_SECRET_LIKE_VAR" not in env
    assert log_path.endswith("ollama-serve.log")
    pid = int(open(os.path.join(jobs_dir, "ollama.pid")).read())
    assert jobs.owned_pid(jobs_dir) == pid
    assert world.signals == []


def test_start_without_context_length_sets_none(world, tmp_path):
    jobs.start_owned_server(str(tmp_path), 11436, None, "probe.log", "probe.pid")
    env = world.popens[0][1]
    assert "OLLAMA_CONTEXT_LENGTH" not in env and env["OLLAMA_HOST"] == "127.0.0.1:11436"


def test_owned_running_is_reused_without_starting(world, tmp_path):
    jobs_dir = str(tmp_path)
    world.listening[11434] = "0.99.0"
    world.add_proc(77, environ=(b"PATH=/usr/bin", MARKER.encode()))
    _write_pid(jobs_dir, 77)
    assert jobs.ollama_up(jobs_dir) == "owned-running"
    assert world.popens == [] and world.signals == []


def test_foreign_server_without_pidfile_gets_no_signal(world, tmp_path, capsys):
    jobs_dir = str(tmp_path)
    world.listening[11434] = "0.99.0"
    world.add_proc(55)  # a foreign `ollama serve`, no marker
    assert jobs.ollama_up(jobs_dir) == "foreign"
    assert jobs.ollama_down(jobs_dir) == "not-owned"
    assert world.signals == [] and world.popens == []
    assert "OLLAMA not-owned: left running" in capsys.readouterr().out


def test_reused_pid_without_marker_is_foreign_and_never_signalled(world, tmp_path):
    jobs_dir = str(tmp_path)
    world.listening[11434] = "0.99.0"
    world.add_proc(91)                    # the pidfile's pid now belongs to a process without the marker
    _write_pid(jobs_dir, 91)
    assert jobs.owned_pid(jobs_dir) is None
    assert jobs.ollama_up(jobs_dir) == "foreign"
    assert jobs.ollama_down(jobs_dir) == "not-owned"
    assert world.signals == []


def test_marker_on_a_non_ollama_process_is_not_owned(world, tmp_path):
    world.add_proc(92, cmdline=(b"python3", b"train.py"), environ=(MARKER.encode(),))
    _write_pid(str(tmp_path), 92)
    assert jobs.owned_pid(str(tmp_path)) is None
    jobs.ollama_down(str(tmp_path))
    assert world.signals == []


def test_ollama_down_signals_exactly_the_owned_pid(world, tmp_path, capsys):
    jobs_dir = str(tmp_path)
    world.listening[11434] = "0.99.0"
    world.add_proc(55)                                         # foreign bystander
    world.add_proc(77, environ=(MARKER.encode(),))
    _write_pid(jobs_dir, 77)
    assert jobs.ollama_down(jobs_dir) == "stopped"
    assert world.signals == [(77, signal.SIGTERM)]
    assert not os.path.exists(os.path.join(jobs_dir, "ollama.pid"))
    assert "OLLAMA stopped pid 77" in capsys.readouterr().out


def test_ollama_down_escalates_to_sigkill_only_for_the_owned_pid(world, tmp_path):
    world.term_kills = False
    world.add_proc(77, environ=(MARKER.encode(),))
    _write_pid(str(tmp_path), 77)
    jobs.ollama_down(str(tmp_path))
    assert world.signals == [(77, signal.SIGTERM), (77, signal.SIGKILL)]


def test_nothing_running_prints_none(world, tmp_path, capsys):
    assert jobs.ollama_down(str(tmp_path)) == "none"
    assert world.signals == [] and "OLLAMA none" in capsys.readouterr().out


# ── Variants ──

def test_variants_write_modelfiles_create_and_confirm(world, tmp_path, capsys):
    jobs.ollama_variants(str(tmp_path))
    creates = [r[0] for r in world.runs if r[0][:2] == ["ollama", "create"]]
    assert sorted(c[2] for c in creates) == ["llama3.2-3b-ctx8192", "qwen2.5-3b-ctx8192"]
    text = open(tmp_path / "modelfiles" / "qwen2.5-3b-ctx8192.Modelfile").read()
    assert text == "FROM qwen2.5:3b\nPARAMETER num_ctx 8192\n"
    assert "VARIANT qwen2.5-3b-ctx8192 from qwen2.5:3b num_ctx 8192 ok" in capsys.readouterr().out


def test_variant_with_wrong_num_ctx_fails(world, tmp_path):
    world.run_results[("ollama", "show", "qwen2.5-3b-ctx8192")] = (0, "num_ctx 4096\n")
    with pytest.raises(jobs.JobError, match="num_ctx 4096"):
        jobs.ollama_variants(str(tmp_path), ["ollama-qwen2.5-3b"])


def test_variant_refuses_a_base_that_is_not_pulled(world, tmp_path):
    world.run_results[("ollama", "show", "qwen2.5:3b")] = (1, "Error: model 'qwen2.5:3b' not found")
    with pytest.raises(jobs.JobError, match="not available locally"):
        jobs.ollama_variants(str(tmp_path), ["ollama-qwen2.5-3b"])
    assert not [r for r in world.runs if r[0][:2] == ["ollama", "create"]]


# ── run ──

def _tail(argv):
    i = argv.index("--models")
    return argv[i:]


@pytest.mark.parametrize("smoke,expected", [
    (False, ["--runs", "3", "--dh", "20", "--ds", "20", "--benign", "17", "--seed", "0", "--max-steps", "5"]),
    (True, ["--runs", "1", "--dh", "1", "--ds", "1", "--benign", "1", "--seed", "0", "--max-steps", "5"]),
])
def test_run_passes_exact_argv_and_writes_exit_code(world, tmp_path, smoke, expected):
    world.child_code = 3
    out = tmp_path / "out"
    code = jobs.run("control-always-comply", str(out), smoke=smoke, jobs_dir=str(tmp_path / "jobs"))
    assert code == 3
    child = [r for r in world.runs if r[2]]
    assert len(child) == 1
    argv = child[0][0]
    assert argv[1] == "-u" and argv[2].endswith("run_benchmark.py")
    assert _tail(argv) == ["--models", "control-always-comply", "--resume", str(out)] + expected
    assert open(out / "bench.done").read().strip() == "3"
    log = open(out / "progress.log").read().splitlines()
    assert log[0].startswith("=== invocation ") and log[0].endswith(" model control-always-comply ===")
    assert world.popens == []   # not an Ollama model: no server


def test_run_ollama_model_starts_and_stops_its_own_server(world, tmp_path):
    jobs_dir = str(tmp_path / "jobs")
    code = jobs.run("ollama-qwen2.5-3b", str(tmp_path / "out"), smoke=True, jobs_dir=jobs_dir)
    assert code == 0
    assert len(world.popens) == 1
    assert world.signals == [(4000, signal.SIGTERM)]       # the pid the fake popen gave the started server
    assert any(r[0][:3] == ["ollama", "create", "qwen2.5-3b-ctx8192"] for r in world.runs)


def test_run_on_foreign_server_never_signals_it(world, tmp_path):
    world.listening[11434] = "0.99.0"
    world.add_proc(55)
    code = jobs.run("ollama-qwen2.5-3b", str(tmp_path / "out"), smoke=True, jobs_dir=str(tmp_path / "jobs"))
    assert code == 0 and world.signals == [] and world.popens == []


def test_run_removes_a_stale_done_marker_and_rejects_unknown_models(world, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "bench.done").write_text("0\n")
    with pytest.raises(jobs.JobError):
        jobs.run("no-such-model", str(out), jobs_dir=str(tmp_path / "jobs"))
    jobs.run("control-always-comply", str(out), jobs_dir=str(tmp_path / "jobs"))
    assert (out / "bench.done").read_text().strip() == "0"


# ── Preflight (read-only, D-04 keep-awake line) ──

NVIDIA_OK = "NVIDIA GeForce RTX 3060 Laptop GPU, 931, 5064, 6144\n"
MEMINFO_OK = "MemTotal:       16000000 kB\nMemAvailable:    9349884 kB\n"


@pytest.fixture
def machine(world, monkeypatch):
    """The world plus fake /proc text files and a fake port table."""
    world.files = {"/proc/meminfo": MEMINFO_OK, "/proc/sys/kernel/random/boot_id": "51d2-boot\n",
                   "/proc/uptime": "15212.87 243013.53\n"}
    world.busy_ports = set()
    world.run_results[("nvidia-smi",)] = (0, NVIDIA_OK)
    monkeypatch.setattr(jobs, "_read_text", lambda path: world.files.get(path))
    monkeypatch.setattr(jobs, "_port_busy", lambda port: port in world.busy_ports or port in world.listening)
    return world


def test_preflight_ok_logs_reading_and_keep_awake(machine, tmp_path):
    assert jobs.preflight(str(tmp_path)) is True
    log = (tmp_path / "preflight.log").read_text()
    assert "free_mib 5064" in log and "avail_ram_mib 9130" in log and "result ok" in log
    assert ("KEEP_AWAKE D-04: keep the laptop on mains power and awake (user responsibility); "
            "boot_id 51d2-boot uptime_s 15212") in log
    assert machine.signals == [] and machine.popens == []
    assert all(r[0][0] == "nvidia-smi" for r in machine.runs)   # it only reads


def test_preflight_fails_on_low_gpu_or_ram_and_honours_overrides(machine, tmp_path, monkeypatch):
    machine.run_results[("nvidia-smi",)] = (0, "GPU, 3000, 3000, 6144\n")
    assert jobs.preflight(str(tmp_path)) is False
    monkeypatch.setenv("MIN_FREE_MIB", "2048")
    assert jobs.preflight(str(tmp_path)) is True
    machine.files["/proc/meminfo"] = "MemAvailable:    1024000 kB\n"
    assert jobs.preflight(str(tmp_path)) is False
    log = (tmp_path / "preflight.log").read_text()
    assert log.count("KEEP_AWAKE D-04") == 3 and "result not-ok" in log
    assert machine.signals == []


def test_preflight_without_nvidia_reading_is_not_ok(machine, tmp_path):
    machine.run_results[("nvidia-smi",)] = (127, "not found")
    assert jobs.preflight(str(tmp_path)) is False


def test_preflight_with_rechecks_waits_then_gives_up(machine, tmp_path, monkeypatch):
    machine.run_results[("nvidia-smi",)] = (0, "GPU, 6000, 100, 6144\n")
    monkeypatch.setattr(jobs, "PREFLIGHT_RECHECKS", 2)
    monkeypatch.setattr(jobs, "PREFLIGHT_WAIT_S", 300)
    assert jobs.preflight_with_rechecks(str(tmp_path)) is False
    assert machine.now == 600 and (tmp_path / "preflight.log").read_text().count("=== preflight") == 3


# ── Default-context probe (temporary server on 11435-11440) ──

PS_TEXT = ("NAME           ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
           "llama3.2:3b    a80c4f17acd5    3.1 GB    100% GPU     4096       4 minutes from now\n")
PS_TEXT_NO_CONTEXT = ("NAME           ID              SIZE      PROCESSOR    UNTIL\n"
                      "llama3.2:3b    a80c4f17acd5    3.1 GB    100% GPU     4 minutes from now\n")
CONFIG_LINE = ('time=2026-10-28T18:00:00Z level=INFO source=routes.go msg="server config" '
               'env="map[OLLAMA_CONTEXT_LENGTH:4096 OLLAMA_HOST:http://127.0.0.1:11435 OLLAMA_NUM_PARALLEL:1]"\n')


def _chat_reply(cap):
    def reply(url, payload):
        lines = payload["messages"][-1]["content"].count("\n") + 1
        return {"usage": {"prompt_tokens": min(30 + 12 * lines, cap)}}
    return reply


@pytest.fixture
def probe_world(machine, monkeypatch):
    w = machine
    w.serve_log = CONFIG_LINE
    w.ps_json = {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b", "context_length": 4096}]}
    w.http_replies["/api/ps"] = lambda url, payload: w.ps_json
    w.http_replies["/api/generate"] = {"done": True}
    w.http_replies["/v1/chat/completions"] = _chat_reply(4096)
    w.run_results[("ollama", "ps")] = (0, PS_TEXT)
    w.run_results[("ollama", "--version")] = (0, "ollama version is 0.99.0\n")
    w.run_results[("ollama", "show", "llama3.2:3b")] = (0, "stop \"<|eot_id|>\"\n")
    original_popen = w.popen

    def popen(argv, env, log_path):
        proc = original_popen(argv, env, log_path)
        w.files[log_path] = w.serve_log
        return proc
    monkeypatch.setattr(jobs, "_popen", popen)
    return w


def _probe_json(jobs_dir):
    with open(os.path.join(jobs_dir, "ollama_default_context.json"), encoding="utf-8") as f:
        return json.load(f)


def test_probe_uses_first_free_port_api_ps_and_stops_its_server(probe_world, tmp_path, capsys):
    w = probe_world
    w.listening[11434] = "0.98.0"                # a server already on 11434 (foreign): read only
    w.add_proc(55)
    w.busy_ports.add(11435)                      # held by something else: never touched
    jobs_dir = str(tmp_path)
    assert jobs.ollama_default_probe(jobs_dir, "llama3.2:3b") == 0
    argv, env, log_path = w.popens[0]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11436" and "OLLAMA_CONTEXT_LENGTH" not in env
    assert env[jobs.OWNER_KEY] == jobs.OWNER_VALUE and log_path.endswith("ollama-probe-11436.log")
    assert w.signals == [(4000, signal.SIGTERM)]           # only the temporary server
    data = _probe_json(jobs_dir)
    assert data["port"] == 11436 and data["server_version"] == "0.99.0" and data["client_version"] == "0.99.0"
    assert data["port_11434"] == {"version": "0.98.0", "state": "foreign"}
    assert data["default_context_length"] == 4096 and data["evidence_source"] == "api_ps"
    assert data["ps_column_context"] == 4096 and data["config_context_length"] == 4096
    assert data["temp_server_stopped"] is True and data["default_is_lower_bound"] is False
    assert data["show_num_ctx"] is None and data["long_prompt_tokens"] is None   # no fallback needed
    out = capsys.readouterr().out
    assert "TEMP_SERVER stopped pid 4000" in out and "DEFAULT_CONTEXT llama3.2:3b 4096 source api_ps" in out
    assert "SERVER_VERSION 0.99.0 CLIENT_VERSION 0.99.0" in out
    assert not os.path.exists(os.path.join(jobs_dir, "ollama-probe.pid"))


def test_probe_falls_back_to_ps_column_and_records_fallback_evidence(probe_world, tmp_path):
    del probe_world.ps_json["models"][0]["context_length"]
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    data = _probe_json(str(tmp_path))
    assert data["evidence_source"] == "ps_column" and data["default_context_length"] == 4096
    assert data["ps_context_length"] is None
    assert data["long_prompt_tokens"] == 4096 and data["long_prompt_sent_lines"] > 0   # fallback recorded


def test_probe_uses_config_line_then_long_prompt_plateau(probe_world, tmp_path):
    w = probe_world
    del w.ps_json["models"][0]["context_length"]
    w.run_results[("ollama", "ps")] = (0, PS_TEXT_NO_CONTEXT)
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    assert _probe_json(str(tmp_path))["evidence_source"] == "config_line"

    w.serve_log = 'msg="server config" env="map[OLLAMA_HOST:http://127.0.0.1:11435]"\n'
    w.listening.clear()
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    data = _probe_json(str(tmp_path))
    assert data["evidence_source"] == "long_prompt_plateau"
    assert data["default_context_length"] == 4096 and data["default_is_lower_bound"] is False
    assert data["config_context_length"] is None and data["ps_column_context"] is None


def test_probe_plateau_above_sent_size_is_a_lower_bound(probe_world, tmp_path):
    w = probe_world
    del w.ps_json["models"][0]["context_length"]
    w.run_results[("ollama", "ps")] = (0, PS_TEXT_NO_CONTEXT)
    w.serve_log = ""
    w.http_replies["/v1/chat/completions"] = _chat_reply(10 ** 9)   # nothing is cut
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    data = _probe_json(str(tmp_path))
    assert data["evidence_source"] == "long_prompt_plateau" and data["default_is_lower_bound"] is True


def test_probe_sources_that_disagree_fail_and_still_stop(probe_world, tmp_path, capsys):
    probe_world.run_results[("ollama", "ps")] = (0, PS_TEXT.replace("4096 ", "8192 "))
    assert jobs.ollama_default_probe(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "FAIL ollama-default-probe: sources disagree" in out and "8192" in out and "4096" in out
    assert probe_world.signals == [(4000, signal.SIGTERM)]


def test_probe_stops_its_server_when_a_step_raises(probe_world, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe step failed")
    monkeypatch.setattr(jobs, "_ps_context", boom)
    with pytest.raises(RuntimeError):
        jobs.ollama_default_probe(str(tmp_path))
    assert probe_world.signals == [(4000, signal.SIGTERM)]
    assert not os.path.exists(tmp_path / "ollama-probe.pid")


def test_probe_stops_only_an_owned_stale_probe_server(probe_world, tmp_path):
    w = probe_world
    w.add_proc(66)                                   # stale pidfile, process without the marker
    _write_pid(str(tmp_path), 66, "ollama-probe.pid")
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    assert (66, signal.SIGTERM) not in w.signals
    w.signals.clear()
    w.listening.clear()
    w.add_proc(67, environ=(MARKER.encode(),))       # stale pidfile of our own crashed probe
    _write_pid(str(tmp_path), 67, "ollama-probe.pid")
    assert jobs.ollama_default_probe(str(tmp_path)) == 0
    assert w.signals[0] == (67, signal.SIGTERM)


def test_probe_with_no_free_port_fails_without_touching_holders(probe_world, tmp_path, capsys):
    probe_world.busy_ports.update(range(11435, 11441))
    assert jobs.ollama_default_probe(str(tmp_path)) == 1
    assert "FAIL ollama-default-probe" in capsys.readouterr().out
    assert probe_world.signals == [] and probe_world.popens == []


def test_probe_refuses_when_preflight_fails(probe_world, tmp_path):
    probe_world.run_results[("nvidia-smi",)] = (0, "GPU, 6000, 100, 6144\n")
    assert jobs.ollama_default_probe(str(tmp_path)) == 5
    assert probe_world.popens == []
