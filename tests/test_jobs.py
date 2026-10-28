"""
Tests for src/benchmark/jobs.py: Ollama server ownership, context variants and the
one-model launcher. Everything is faked (HTTP, /proc, signals, subprocesses, clock):
no real server, no GPU, no benchmark child.
"""

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

    def popen(self, argv, env, log_path):
        pid = self.next_pid
        self.next_pid += 1
        self.popens.append((argv, env, log_path))
        environ = tuple(f"{k}={v}".encode() for k, v in env.items())
        self.add_proc(pid, cmdline=tuple(a.encode() for a in argv), environ=environ)
        port = int(env["OLLAMA_HOST"].rsplit(":", 1)[1])
        self.listening[port] = "0.99.0"

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
