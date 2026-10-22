"""
Tests for the GPU job helpers of the Qwen LoRA phase: src/qwen_lora/run_train.sh and src/qwen_lora/checks.py.
CPU only. The shell tests run only on Linux (they are skipped elsewhere): a fake nvidia-smi, a fake ollama and a
stub trainer are written to tmp_path and put first on PATH / in PYTHON, so no GPU is touched, no process of the
machine is disturbed and no real training starts. The checks run on synthetic inputs and on the output of one tiny
CPU training run (module-scoped, a few seconds after the Hugging Face import).
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.qwen_fixtures import build_tiny_qwen, require_hf

require_hf()

from src.benchmark.injecagent import PROJECT_ROOT
from src.qwen_lora import checks
from src.qwen_lora.train import TrainConfig, train
from src.scratch_llm.dataset import generate_episodes, write_episodes
from tests.test_hf_local_provider import write_hf_models_yaml_entries

SCRIPT = os.path.join(PROJECT_ROOT, "src", "qwen_lora", "run_train.sh")
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="shell helpers run under Linux/WSL only")


# ── fixtures ──

@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return build_tiny_qwen(str(tmp_path_factory.mktemp("tiny_base")))


@pytest.fixture(scope="module")
def tiny_run(tiny_dir, tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny_run")
    train_path, dev_path = str(directory / "train.jsonl"), str(directory / "dev.jsonl")
    write_episodes(train_path, generate_episodes(16, 0))
    write_episodes(dev_path, generate_episodes(8, 1, "dev_tools"))
    out_dir = str(directory / "adapter")
    cfg = TrainConfig(max_steps=4, micro_batch=2, accum=2, warmup=1, log_every=1, eval_every=2, dev_eval_n=6,
                      gen_check_n=2, gen_max_new_tokens=8)
    manifest = train(train_path, dev_path, out_dir, base_model=tiny_dir, cfg=cfg, device="cpu", log=lambda *_: None)
    return SimpleNamespace(out_dir=out_dir, manifest=manifest, directory=directory)


@pytest.fixture
def run_dir(tiny_run, tmp_path):
    """A private copy of the tiny run's output that a test may damage."""
    target = str(tmp_path / "copy")
    shutil.copytree(tiny_run.out_dir, target)
    return target


def run_checks(*argv):
    return subprocess.run([sys.executable, "-m", "src.qwen_lora.checks", *argv], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=300)


def in_process(argv, capsys):
    code = checks.main(argv)
    return code, capsys.readouterr().out


# ── run_train.sh ──

def write_executable(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_bin(tmp_path):
    """A directory first on PATH holding a fake nvidia-smi (the reading comes from a file) and a fake ollama."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    reading = tmp_path / "reading.txt"
    reading.write_text("NVIDIA GeForce RTX 3060 Laptop GPU, 1000, 5000, 6144\n", encoding="utf-8")
    write_executable(bin_dir / "nvidia-smi", f'#!/bin/bash\ncat "{reading}"\n')
    write_executable(bin_dir / "ollama", f'#!/bin/bash\necho "$@" >> "{tmp_path}/ollama_calls.txt"\n'
                                        f'echo "NAME ID SIZE"\n')
    return SimpleNamespace(path=bin_dir, reading=reading, calls=tmp_path / "ollama_calls.txt")


def run_script(fake_bin, *argv, **env):
    environment = {**os.environ, "PATH": f"{fake_bin.path}{os.pathsep}{os.environ['PATH']}", "MIN_AVAIL_RAM_MIB": "0",
                   **env}
    return subprocess.run(["bash", SCRIPT, *argv], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True,
                          timeout=120)


@linux_only
def test_run_train_script_is_lf_valid_and_never_stops_a_process():
    with open(SCRIPT, "rb") as f:
        raw = f.read()
    assert b"\r" not in raw
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0
    code = [line for line in raw.decode("utf-8").splitlines() if not line.lstrip().startswith("#")]
    import re
    assert not [line for line in code if re.search(r"\b(kill|pkill|killall)\b|ollama +stop", line)]


@linux_only
def test_preflight_passes_and_logs_the_reading(fake_bin, tmp_path):
    out = tmp_path / "out"
    done = run_script(fake_bin, "preflight", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    log = (out / "gpu_preflight.log").read_text(encoding="utf-8")
    assert "NVIDIA GeForce RTX 3060 Laptop GPU, 1000, 5000, 6144" in log
    assert "free_mib 5000" in log and "avail_ram_mib " in log
    assert fake_bin.calls.read_text(encoding="utf-8").split() == ["ps"]        # ollama was only asked for its list


@linux_only
def test_preflight_fails_when_the_card_is_busy_and_never_stops_anything(fake_bin, tmp_path):
    fake_bin.reading.write_text("NVIDIA GeForce RTX 3060 Laptop GPU, 5900, 244, 6144\n", encoding="utf-8")
    out = tmp_path / "out"
    done = run_script(fake_bin, "preflight", str(out))
    assert done.returncode == 1
    assert "GPU preflight failed: 244 MiB free, 3584 MiB required" in done.stdout
    assert "never stops other processes" in done.stdout
    assert "free_mib 244" in (out / "gpu_preflight.log").read_text(encoding="utf-8")
    assert fake_bin.calls.read_text(encoding="utf-8").split() == ["ps"]


@linux_only
def test_preflight_fails_on_an_unreadable_reading_and_on_low_ram(fake_bin, tmp_path):
    fake_bin.reading.write_text("garbage\n", encoding="utf-8")
    assert run_script(fake_bin, "preflight", str(tmp_path / "a")).returncode == 1
    fake_bin.reading.write_text("GPU, 1000, 5000, 6144\n", encoding="utf-8")
    low = run_script(fake_bin, "preflight", str(tmp_path / "b"), MIN_AVAIL_RAM_MIB="999999999")
    assert low.returncode == 1 and "RAM preflight failed" in low.stdout


@pytest.fixture
def stub_python(tmp_path):
    marker = tmp_path / "ran.txt"
    stub = tmp_path / "stub_python.sh"
    write_executable(stub, f'#!/bin/bash\necho "$@" >> "{marker}"\necho "device cpu trainable 1 total 2"\n'
                           f'echo "DONE best_step 1"\nexit 3\n')
    return SimpleNamespace(path=str(stub), marker=marker)


@linux_only
def test_launch_writes_the_progress_log_and_the_exit_code_to_the_done_marker(fake_bin, stub_python, tmp_path):
    out = tmp_path / "out"
    done = run_script(fake_bin, "launch", str(out), "--max-steps", "7", PYTHON=stub_python.path, PREFLIGHT_RECHECKS="0")
    assert done.returncode == 3
    assert (out / "train.done").read_text(encoding="utf-8").strip() == "3"
    assert "DONE best_step 1" in (out / "train_progress.log").read_text(encoding="utf-8")
    assert "-m src.qwen_lora.train --out-dir " + str(out) + " --max-steps 7" in stub_python.marker.read_text(encoding="utf-8")


@linux_only
def test_launch_that_never_passes_the_preflight_writes_5_and_does_not_train(fake_bin, stub_python, tmp_path):
    fake_bin.reading.write_text("GPU, 5900, 244, 6144\n", encoding="utf-8")
    out = tmp_path / "out"
    done = run_script(fake_bin, "launch", str(out), PYTHON=stub_python.path, PREFLIGHT_RECHECKS="1", PREFLIGHT_WAIT_S="0")
    assert done.returncode == 5
    assert (out / "train.done").read_text(encoding="utf-8").strip() == "5"
    assert not stub_python.marker.exists()
    assert (out / "gpu_preflight.log").read_text(encoding="utf-8").count("free_mib 244") == 2     # one check plus one re-check


@linux_only
def test_launch_removes_a_stale_done_marker_before_the_preflight_waits(fake_bin, stub_python, tmp_path):
    """Review WR-02: the old marker was removed only after the preflight passed, so `poll` could report the
    previous run's done-ok for 30 minutes while a retry was still waiting for the card."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "train.done").write_text("0\n", encoding="utf-8")
    trace = tmp_path / "trace.txt"
    write_executable(fake_bin.path / "nvidia-smi",
                     f'#!/bin/bash\nif [ -f "{out}/train.done" ]; then echo stale >> "{trace}"; else echo clean >> "{trace}"; fi\n'
                     f'echo "GPU, 5900, 244, 6144"\n')                      # busy card: the preflight keeps waiting
    done = run_script(fake_bin, "launch", str(out), PYTHON=stub_python.path, PREFLIGHT_RECHECKS="1", PREFLIGHT_WAIT_S="0")
    assert done.returncode == 5
    assert trace.read_text(encoding="utf-8").split() == ["clean", "clean"]    # never stale while it waited


@linux_only
def test_poll_never_reports_the_previous_runs_result_while_a_relaunch_waits(fake_bin, stub_python, tmp_path):
    """Review WR-02, seen from the poller: after `launch` starts, the old done-ok must be gone at once."""
    import time
    out = tmp_path / "out"
    out.mkdir()
    (out / "train.done").write_text("0\n", encoding="utf-8")
    fake_bin.reading.write_text("GPU, 5900, 244, 6144\n", encoding="utf-8")          # preflight cannot pass
    environment = {**os.environ, "PATH": f"{fake_bin.path}{os.pathsep}{os.environ['PATH']}", "MIN_AVAIL_RAM_MIB": "0",
                   "PYTHON": stub_python.path, "PREFLIGHT_RECHECKS": "3", "PREFLIGHT_WAIT_S": "20"}
    launcher = subprocess.Popen(["bash", SCRIPT, "launch", str(out)], cwd=PROJECT_ROOT, env=environment,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (out / "gpu_preflight.log").exists():
            time.sleep(0.1)
        assert not (out / "train.done").exists()                                     # the stale marker is gone
        polled = run_script(fake_bin, "poll", str(out), "1")
        assert "STATUS done-ok" not in polled.stdout
        assert "STATUS running" in polled.stdout                                     # the launcher is alive and waiting
    finally:
        launcher.terminate()
        launcher.wait(timeout=30)


@linux_only
def test_launch_with_resume_appends_after_a_separator_and_a_fresh_launch_truncates(fake_bin, stub_python, tmp_path):
    out = tmp_path / "out"
    run_script(fake_bin, "launch", str(out), PYTHON=stub_python.path, PREFLIGHT_RECHECKS="0")
    run_script(fake_bin, "launch", str(out), "--resume", PYTHON=stub_python.path, PREFLIGHT_RECHECKS="0")
    log = (out / "train_progress.log").read_text(encoding="utf-8")
    assert "=== relaunch with --resume " in log and log.count("DONE best_step 1") == 2
    run_script(fake_bin, "launch", str(out), PYTHON=stub_python.path, PREFLIGHT_RECHECKS="0")
    assert (out / "train_progress.log").read_text(encoding="utf-8").count("DONE best_step 1") == 1


@linux_only
@pytest.mark.parametrize("code, status", [("3", "STATUS done-failed 3"), ("0", "STATUS done-ok"), ("4", "STATUS done-failed 4")])
def test_poll_reports_the_done_marker(fake_bin, tmp_path, code, status):
    out = tmp_path / "out"
    out.mkdir()
    (out / "train_progress.log").write_text("device cpu trainable 1 total 2\nstep 1/4 loss 1.0\nstep 2/4 loss 0.9\n"
                                            "eval step 2 dev_loss 1\n", encoding="utf-8")
    (out / "train.done").write_text(code + "\n", encoding="utf-8")
    done = run_script(fake_bin, "poll", str(out), "1")
    assert done.returncode == 0
    assert status in done.stdout.splitlines() and "step lines: 2" in done.stdout and "FAILED line: no" in done.stdout
    assert "eval step 2 dev_loss 1" in done.stdout


@linux_only
def test_poll_says_dead_when_nothing_runs_and_no_marker_exists_and_reports_a_failed_line(fake_bin, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "train_progress.log").write_text("step 1/4 loss 1.0\nFAILED non_finite_loss step 2\n", encoding="utf-8")
    done = run_script(fake_bin, "poll", str(out), "1")
    assert "STATUS dead" in done.stdout.splitlines() and "FAILED line: yes" in done.stdout


# ── checks smoke ──

def smoke_rows(resolved, config_model, n=26, **overrides):
    rows = []
    for i in range(n):
        kind = ("benign", "dh", "ds")[i % 3]
        attacked = kind != "benign" and i % 2 == 0
        rows.append({"model": "tiny-lora", "case_id": f"case-{i // 2:02d}", "run": 0, "arm": ("unprotected", "protected")[i % 2],
                     "kind": kind, "attempted": attacked, "succeeded": False, "steps": 3, "parse_errors": 1 if i % 5 == 0 else 0,
                     "actions": [{"action": "call_tool", "tool": "X"}, {"action": "respond", "message": "ok"}],
                     "stop_reason": "responded", "user_task_done": kind == "benign", "final_message": "done",
                     "provider": "hf_local", "config_model": config_model, "resolved_model": resolved,
                     "code_commit": "a" * 40, "llm_error": None, "finish_reasons": ["stop", "stop", "length"], **overrides})
    return rows


@pytest.fixture
def smoke_env(tiny_run, tiny_dir, tmp_path):
    config = tmp_path / "models.yaml"
    write_hf_models_yaml_entries(config, {"tiny-lora": dict(base_model=tiny_dir, adapter=tiny_run.out_dir, device="cpu")})
    resolved = f"{tiny_dir}@local+tiny-lora@{tiny_run.manifest['adapter_identity_sha256'][:12]}"
    results = tmp_path / "results"
    results.mkdir()

    def write(rows, summary=True):
        (results / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (results / "summary.md").unlink(missing_ok=True)
        if summary:
            (results / "summary.md").write_text("# summary\n", encoding="utf-8")

    def argv():
        return ["smoke", "--results", str(results), "--model", "tiny-lora", "--config", str(config)]

    return SimpleNamespace(resolved=resolved, write=write, argv=argv)


def test_smoke_prints_report_lines_and_the_plumbing_ok_line(smoke_env, capsys):
    smoke_env.write(smoke_rows(smoke_env.resolved, "tiny-lora"))
    code, out = in_process(smoke_env.argv(), capsys)
    assert code == 0 and out.splitlines()[-1] == "smoke-plumbing-ok"
    for fragment in ("REPORT parse rate ", "REPORT valid actions ", "REPORT unprotected dh/ds episodes with call_tool ",
                     "REPORT protected benign episodes with a valid action ", "REPORT unprotected outcomes attacked=",
                     "REPORT protected outcomes "):
        assert fragment in out, fragment


@pytest.mark.parametrize("case", ["llm_error", "wrong_resolved", "25_rows", "no_summary", "bad_finish", "short_commit"])
def test_smoke_plumbing_gates_each_exit_1_naming_the_failure(smoke_env, capsys, case):
    rows = smoke_rows(smoke_env.resolved, "tiny-lora")
    expected = {"llm_error": "llm_error", "wrong_resolved": "resolved_model", "25_rows": "25 rows", "no_summary": "summary.md",
                "bad_finish": "finish_reasons", "short_commit": "code_commit"}[case]
    if case == "llm_error":
        rows[3]["llm_error"] = "ConnectionError: boom"
    elif case == "wrong_resolved":
        rows[3]["resolved_model"] = "something-else"
    elif case == "25_rows":
        rows = rows[:25]
    elif case == "bad_finish":
        rows[3]["finish_reasons"] = ["context_overflow"]
    elif case == "short_commit":
        rows[3]["code_commit"] = "abc"
    smoke_env.write(rows, summary=case != "no_summary")
    code, out = in_process(smoke_env.argv(), capsys)
    assert code == 1 and expected in out and "smoke-plumbing-ok" not in out


def test_smoke_with_no_valid_action_warns_but_still_passes(smoke_env, capsys):
    smoke_env.write(smoke_rows(smoke_env.resolved, "tiny-lora", parse_errors=3, actions=[]))
    code, out = in_process(smoke_env.argv(), capsys)
    assert code == 0 and "WARNING no valid action in the whole smoke run" in out and "REPORT valid actions 0/78" in out


# ── checks artifacts ──

def test_artifacts_ok_on_a_tiny_cpu_run_prints_the_decisions_without_changing_the_exit_code(tiny_run, capsys):
    code, out = in_process(["artifacts", "--out-dir", tiny_run.out_dir], capsys)
    assert code == 0 and out.splitlines()[-1] == "artifacts-ok"
    for fragment in ("ARTIFACT steps 4 best_step ", "ARTIFACT adapter_sha12 " + tiny_run.manifest["adapter_sha256"][:12],
                     "ARTIFACT adapter_identity12 " + tiny_run.manifest["adapter_identity_sha256"][:12],
                     "ARTIFACT final_dev_full n 8 dev_action_exact ", "ARTIFACT gen_check n 2 tuned parse ",
                     "ARTIFACT peak_vram_mib allocated 0 reserved 0 total 0", "REMEDIATION needed ", "LORA04_GATE "):
        assert fragment in out, fragment


def test_artifacts_refuses_a_flipped_adapter_byte_naming_adapter_sha256(run_dir, capsys):
    path = os.path.join(run_dir, "adapter_model.safetensors")
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 1]))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and "adapter_sha256" in out and "artifacts-ok" not in out


def test_wr05_artifacts_refuses_an_edited_adapter_config_naming_adapter_config_sha256(run_dir, capsys):
    path = os.path.join(run_dir, "adapter_config.json")
    config = json.load(open(path, encoding="utf-8"))
    config["lora_alpha"] = 64                                    # the weights are untouched
    json.dump(config, open(path, "w", encoding="utf-8"))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and "adapter_config_sha256" in out and "artifacts-ok" not in out


def test_wr05_artifacts_refuses_an_identity_that_does_not_follow_from_the_files(run_dir, capsys):
    path = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(path, encoding="utf-8"))
    manifest["adapter_identity_sha256"] = "0" * 64
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and "adapter_identity_sha256" in out


def test_artifacts_require_cuda_names_the_device_on_a_cpu_run(run_dir, capsys):
    code, out = in_process(["artifacts", "--out-dir", run_dir, "--require-cuda"], capsys)
    assert code == 1 and "device" in out


def test_artifacts_never_passes_a_stale_or_partial_directory(run_dir, capsys):
    for name, expected in (("qwen_lora.resume.tmp", "leftover"), ("loss_curve.png", "loss_curve.png is missing"),
                           ("train_log.jsonl", "train_log.jsonl is missing")):
        path = os.path.join(run_dir, name)
        backup = path + ".keep"
        if name == "qwen_lora.resume.tmp":
            open(path, "wb").close()
        else:
            os.replace(path, backup)
        code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
        assert code == 1 and expected in out, (name, out)
        if name == "qwen_lora.resume.tmp":
            os.remove(path)
        else:
            os.replace(backup, path)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    del manifest["gen_check"]
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and "MANIFEST_KEYS" in out
    (open(os.path.join(run_dir, "train.done"), "w")).write("3\n")
    assert in_process(["artifacts", "--out-dir", run_dir], capsys)[0] == 1


def test_wr04_artifacts_prints_the_gate_context_and_the_overlap(tiny_run, capsys):
    code, out = in_process(["artifacts", "--out-dir", tiny_run.out_dir], capsys)
    assert code == 0
    assert "ARTIFACT gate_context respond_share " in out and "nonrespond_action_exact " in out
    assert "min_class_action_exact " in out and "first_action_exact " in out
    assert "ARTIFACT overlap selection_subset 6 final_dev 8 (selection inside final 6) gen_check 2" in out
    assert "not independent of the selection" in out


def test_wr04_artifacts_says_not_recomputed_for_a_manifest_that_predates_the_metrics(run_dir, capsys):
    path = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(path, encoding="utf-8"))
    for key in ("dev_respond_share", "dev_nonrespond_action_exact", "dev_nonrespond_action_exact_by_cls",
                "dev_min_class_action_exact"):
        del manifest["final_dev_full"][key]
    json.dump(manifest, open(path, "w", encoding="utf-8"))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 0 and "ARTIFACT gate_context not recomputed" in out and "artifacts-ok" in out


def test_artifacts_flags_the_staging_leftovers_of_a_failed_rerun_next_to_a_valid_adapter(run_dir, capsys):
    """Review WR-03: a failed re-run now leaves its staged adapter and log beside the previous adapter."""
    from src.qwen_lora import train as T
    os.makedirs(os.path.join(run_dir, T.IN_RUN_ADAPTER))
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and f"leftover {T.IN_RUN_ADAPTER}" in out
    shutil.rmtree(os.path.join(run_dir, T.IN_RUN_ADAPTER))
    open(os.path.join(run_dir, T.STAGED_LOG), "w").close()
    code, out = in_process(["artifacts", "--out-dir", run_dir], capsys)
    assert code == 1 and f"leftover {T.STAGED_LOG}" in out


def test_artifacts_checks_the_dataset_hashes_against_the_data_manifest(tiny_run, tmp_path, capsys):
    d = tiny_run.manifest["dataset"]
    good = tmp_path / "data_manifest.json"
    good.write_text(json.dumps({"files": {"train": {"sha256": d["train_sha256"]}, "dev": {"sha256": d["dev_sha256"]}}}),
                    encoding="utf-8")
    code, out = in_process(["artifacts", "--out-dir", tiny_run.out_dir, "--data-manifest", str(good)], capsys)
    assert code == 1 and "is not a file in the repository" in out          # the tiny files live outside the repository
    assert "differs from the data manifest" not in out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"files": {"train": {"sha256": "0" * 64}, "dev": {"sha256": d["dev_sha256"]}}}), encoding="utf-8")
    code, out = in_process(["artifacts", "--out-dir", tiny_run.out_dir, "--data-manifest", str(bad)], capsys)
    assert code == 1 and "dataset.train_sha256 differs from the data manifest" in out


# ── checks probe ──

def test_probe_runs_one_benchmark_episode_through_the_provider(tiny_run, tiny_dir, tmp_path, capsys):
    config = tmp_path / "models.yaml"
    write_hf_models_yaml_entries(config, {"tiny-base": dict(base_model=tiny_dir, adapter="null", device="cpu",
                                                            max_new_tokens=8)})
    code, out = in_process(["probe", "--model", "tiny-base", "--config", str(config), "--kind", "benign"], capsys)
    assert code == 0 and out.splitlines()[-1] == "probe-ok" and "STEP " in out
    code, out = in_process(["probe", "--model", "tiny-base", "--config", str(config), "--kind", "dh",
                            "--adapter-dir", tiny_run.out_dir], capsys)
    assert code == 0 and "+tiny-base@" + tiny_run.manifest["adapter_identity_sha256"][:12] in out


# ── checks viva ──

def viva_page(manifest, drop=None, extra="", questions=16, parts=("A", "B", "C", "D"), glossary=0):
    m = manifest
    values = {
        "trainable": f"{m['lora']['trainable_params']:,}", "total": str(m["lora"]["total_params"]),
        "steps": str(m["steps"]), "best": str(m["best_step"]), "adapter": m["adapter_sha256"][:12],
        "identity": m["adapter_identity_sha256"][:12],
        "revision": m["base_model"]["revision"][:12], "train": m["dataset"]["train_sha256"][:12],
        "exact": f"{m['final_dev_full']['dev_action_exact']:.4f}", "parse": f"{m['gen_check']['tuned']['parse_rate']:.4f}",
    }
    if drop:
        values[drop] = "-"
    lines = [f"## Part {p} heading" for p in parts if p != "D"]
    lines = [f"## Part {p}" for p in parts if p != "D"]
    lines.append("## Limits of this phase")
    if "D" in parts:
        lines.append("## Part D")
    lines.append("numbers: " + " ".join(f"{k}={v}" for k, v in values.items()))
    lines += [f"**Q{i}. question?**" for i in range(1, questions + 1)]
    lines += [f"- **term{i}**: meaning" for i in range(glossary)]
    lines.append(extra)
    return "\n".join(lines) + "\n"


@pytest.fixture
def viva_env(tmp_path):
    manifest = {"lora": {"trainable_params": 8798208, "total_params": 502000000}, "steps": 600, "best_step": 500,
                "adapter_sha256": "ab12cd34ef56" + "0" * 52, "adapter_identity_sha256": "9f8e7d6c5b4a" + "0" * 52,
                "base_model": {"revision": "c60e7131dc3b1234"},
                "dataset": {"train_sha256": "a2586548c7f7" + "1" * 52},
                "final_dev_full": {"dev_action_exact": 0.91234}, "gen_check": {"tuned": {"parse_rate": 0.97}}}
    manifest_path = str(tmp_path / "viva_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    readme = tmp_path / "README.md"
    readme.write_text("| [03-fine-tuning-lora-and-chat-templates.md](03-fine-tuning-lora-and-chat-templates.md) | x | Phase 3 | "
                      "written (Phase 3) |\n", encoding="utf-8")
    page = tmp_path / "page.md"

    def check(text, capsys, *extra, structure=False):
        page.write_text(text, encoding="utf-8")
        argv = ["viva", "--page", str(page)] + (["--structure-only"] if structure else
                                                 ["--manifest", manifest_path, "--readme", str(readme)]) + list(extra)
        return in_process(argv, capsys)

    return SimpleNamespace(manifest=manifest, check=check, readme=readme)


def test_viva_part_c_ok_with_every_manifest_value_quoted(viva_env, capsys):
    code, out = viva_env.check(viva_page(viva_env.manifest), capsys)
    assert code == 0 and out.splitlines()[-1] == "part-c-ok"


@pytest.mark.parametrize("dropped", ["trainable", "total", "steps", "best", "adapter", "identity", "revision", "train", "exact", "parse"])
def test_viva_names_a_missing_manifest_value(viva_env, capsys, dropped):
    code, out = viva_env.check(viva_page(viva_env.manifest, drop=dropped), capsys)
    assert code == 1 and "does not quote" in out


@pytest.mark.parametrize("secret", ["OPENAI_api_key = abc", "token sk-abcdefgh12345", "CANARY0123456789ABDATA".replace("0123456789AB", "0123456789AB")])
def test_viva_rejects_key_like_and_canary_shaped_strings(viva_env, capsys, secret):
    code, out = viva_env.check(viva_page(viva_env.manifest, extra=secret), capsys)
    assert code == 1 and "secret-like text" in out


def test_viva_requires_the_parts_the_questions_and_the_readme_row(viva_env, capsys):
    code, out = viva_env.check(viva_page(viva_env.manifest, parts=("A", "B", "D")), capsys)
    assert code == 1 and "## Part C" in out
    code, out = viva_env.check(viva_page(viva_env.manifest, questions=15), capsys)
    assert code == 1 and "15 question lines" in out
    viva_env.readme.write_text("| planned (Phase 3) |\n", encoding="utf-8")
    code, out = viva_env.check(viva_page(viva_env.manifest), capsys)
    assert code == 1 and "README" in out


def test_viva_structure_only_needs_no_part_c_and_counts_questions_and_glossary(viva_env, capsys):
    page = viva_page(viva_env.manifest, parts=("A", "B", "D"), glossary=30)
    code, out = viva_env.check(page, capsys, structure=True)
    assert code == 0 and out.splitlines()[-1] == "viva-structure-ok"
    code, out = viva_env.check(viva_page(viva_env.manifest, parts=("A", "B", "D"), glossary=30, questions=15), capsys,
                               structure=True)
    assert code == 1 and "15 question lines" in out
    code, out = viva_env.check(viva_page(viva_env.manifest, parts=("A", "B", "D"), glossary=29), capsys, structure=True)
    assert code == 1 and "glossary" in out


def test_checks_help_lists_the_four_subcommands():
    done = run_checks("--help")
    assert done.returncode == 0
    for name in ("probe", "artifacts", "smoke", "viva"):
        assert name in done.stdout
