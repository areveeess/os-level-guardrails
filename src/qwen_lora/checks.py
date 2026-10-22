"""
Repeatable gates for the Qwen LoRA phase
========================================
Commands that replace inline `python -c` snippets (which do not survive the quoting of wsl.exe). Each
subcommand prints what it found and ends with an ok line; a failed gate prints the reason and exits 1.
Behaviour numbers are printed as REPORT lines and are never pass/fail: a smoke run is plumbing evidence only.

    python -m src.qwen_lora.checks probe     --model NAME [--adapter-dir DIR] [--kind benign|dh|ds]      -> probe-ok
    python -m src.qwen_lora.checks artifacts --out-dir DIR [--data-manifest FILE] [--require-cuda]      -> artifacts-ok
    python -m src.qwen_lora.checks smoke     --results DIR --model NAME [--expect-rows 26]               -> smoke-plumbing-ok
    python -m src.qwen_lora.checks viva      --page PAGE (--manifest FILE [--readme FILE] | --structure-only)
                                                                                -> part-c-ok | viva-structure-ok

probe      One real benchmark episode through HFLocalProvider (optionally with an adapter override).
artifacts  The training output: manifest keys, adapter hash, dataset hashes, adapter_config, train log, loss
           curve, VRAM budget, leftovers and failure markers; prints the REMEDIATION and LORA04_GATE decisions,
           which are informational here because they are taken in the GPU plan.
smoke      The plumbing of a small benchmark run (stamps, no llm_error, summary.md) plus REPORT lines.
viva       The study page: headings, questions, manifest values quoted verbatim, README row, secret scan.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

import yaml

from src.benchmark.injecagent import PROJECT_ROOT

MODELS_FILE = os.path.join(PROJECT_ROOT, "config", "models.yaml")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "manifest.json")
SECRET_PATTERNS = (r"api_key *=", r"sk-[a-z0-9]{8}", r"CANARY[0-9A-F]{12}DATA")      # as the Phase 2 viva check
README_LINK = "(03-fine-tuning-lora-and-chat-templates.md)"
README_STATUS = "written (Phase 3)"
SMOKE_FINISH_REASONS = {"stop", "length"}
ARMS = ("unprotected", "protected")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _model_config(config_path: str, model: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]
    if model not in models:
        raise SystemExit(f"unknown model {model!r}; known: {', '.join(models)}")
    return models[model]


def _expected_resolved_model(cfg: dict, model: str, adapter_dir) -> str:
    """base_model@revision[:12], plus +<model>@<adapter_sha256[:12]> when an adapter is served."""
    from src.llm.hf_local import resolve_base_model
    expected = f"{cfg['base_model']}@{resolve_base_model(cfg['base_model'])[1][:12]}"
    if adapter_dir:
        expected += f"+{model}@{_read_json(os.path.join(adapter_dir, 'manifest.json'))['adapter_sha256'][:12]}"
    return expected


def _config_adapter(cfg: dict):
    return os.path.join(PROJECT_ROOT, cfg["adapter"]) if cfg.get("adapter") else None


# ── probe ──

def cmd_probe(args) -> tuple:
    from src.benchmark.injecagent import attack_cases, benign_cases, load_tool_specs
    from src.benchmark.runner import run_episode
    from src.llm.hf_local import HFLocalProvider
    from src.policy_engine.engine import PolicyEngine
    cfg = _model_config(args.config, args.model)
    adapter = args.adapter_dir or _config_adapter(cfg)
    provider = HFLocalProvider(name=args.model, base_model=cfg["base_model"], adapter=adapter,
                               max_new_tokens=int(cfg.get("max_new_tokens", 512)),
                               max_context_tokens=int(cfg.get("max_context_tokens", 4096)),
                               merge=bool(cfg.get("merge", True)), device=cfg.get("device", "cuda"))
    specs = load_tool_specs()
    cases = benign_cases(specs) if args.kind == "benign" else \
        [c for c in attack_cases(1, 1, 0, specs) if c.kind == args.kind]
    case = cases[0]
    with tempfile.TemporaryDirectory() as tmp:
        policy = PolicyEngine(log_file=os.path.join(tmp, "decisions.jsonl"), verbose=False)
        episode, score = run_episode(provider, case, "unprotected", policy, 5)
    print(f"case {case.id} kind {case.kind} stop_reason {episode.stop_reason} steps {len(episode.steps)}")
    for step in episode.steps:
        print(f"STEP {step.index} finish={step.finish_reason} prompt_tokens={step.prompt_tokens} "
              f"parse_error={step.parse_error!r} raw={step.raw[:300]!r}")
    resolved = next((s.resolved_model for s in episode.steps if s.resolved_model), None)
    expected = _expected_resolved_model(cfg, args.model, adapter)
    print(f"resolved_model {resolved}")
    print(f"expected       {expected}")
    try:
        import torch
        print(f"cuda_memory_allocated_bytes {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}")
    except ImportError:
        pass
    problems = []
    if score["llm_error"] is not None:
        problems.append(f"llm_error: {score['llm_error']}")
    if not episode.steps:
        problems.append("the episode has no step")
    if resolved != expected:
        problems.append(f"resolved_model {resolved!r} is not the expected {expected!r}")
    return problems, "probe-ok"


# ── artifacts ──

def cmd_artifacts(args) -> tuple:
    from src.qwen_lora import train as T
    out = args.out_dir
    problems = []

    def need(condition, reason):
        if not condition:
            problems.append(reason)

    path = os.path.join(out, "manifest.json")
    if not os.path.isfile(path):
        return [f"manifest.json is missing in {out}"], ""
    m = _read_json(path)
    need(tuple(m) == T.MANIFEST_KEYS, f"manifest keys differ from MANIFEST_KEYS: "
                                      f"missing {sorted(set(T.MANIFEST_KEYS) - set(m))}, extra {sorted(set(m) - set(T.MANIFEST_KEYS))}"
                                      f"{'' if set(m) != set(T.MANIFEST_KEYS) else ', order differs'}")
    for name in ("adapter_config.json", "adapter_model.safetensors", "train_log.jsonl", "loss_curve.png"):
        need(os.path.isfile(os.path.join(out, name)), f"{name} is missing")
    for name in (T.IN_RUN_ADAPTER, T.RESUME_FILE):
        need(not os.path.exists(os.path.join(out, name)), f"leftover {name}: the run did not finish cleanly")
    weights = os.path.join(out, "adapter_model.safetensors")
    if os.path.isfile(weights) and m.get("adapter_sha256") != _sha256(weights):
        problems.append(f"adapter_sha256 {str(m.get('adapter_sha256'))[:12]} does not match adapter_model.safetensors "
                        f"{_sha256(weights)[:12]}")
    done = os.path.join(out, "train.done")
    if os.path.isfile(done):
        need(open(done, encoding="utf-8").read().strip() == "0", "train.done does not hold 0")
    progress = os.path.join(out, "train_progress.log")
    if os.path.isfile(progress):
        with open(progress, encoding="utf-8", errors="replace") as f:
            need(not any(line.startswith("FAILED") for line in f), "train_progress.log has a FAILED line")
    try:
        d, train_cfg = m["dataset"], m["config"]["train"]
        if args.data_manifest:
            files = _read_json(args.data_manifest)["files"]
            need(d["train_sha256"] == files["train"]["sha256"], "dataset.train_sha256 differs from the data manifest")
            need(d["dev_sha256"] == files["dev"]["sha256"], "dataset.dev_sha256 differs from the data manifest")
            for key in ("train", "dev"):
                on_disk = os.path.join(PROJECT_ROOT, d[f"{key}_path"])
                if not os.path.isfile(on_disk):
                    problems.append(f"dataset.{key}_path {d[f'{key}_path']} is not a file in the repository")
                else:
                    need(_sha256(on_disk) == d[f"{key}_sha256"], f"dataset.{key}_sha256 differs from the file on disk")
        config_path = os.path.join(out, "adapter_config.json")
        if os.path.isfile(config_path):
            ac = _read_json(config_path)
            leaves = {name.rsplit(".", 1)[-1] for name in ac.get("target_modules", [])}
            need(ac.get("r") == 16 and ac.get("lora_alpha") == 32, f"adapter_config r/alpha {ac.get('r')}/{ac.get('lora_alpha')} is not 16/32")
            need(leaves == set(T.LORA_TARGETS), f"adapter_config target_modules {sorted(leaves)} differ from {sorted(T.LORA_TARGETS)}")
        need(m["lora"]["target_modules"] == sorted(T.LORA_TARGETS), "manifest lora.target_modules differ from LORA_TARGETS")
        need(m["best_step"] <= m["steps"], f"best_step {m['best_step']} is above steps {m['steps']}")
        log_path = os.path.join(out, "train_log.jsonl")
        if os.path.isfile(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            need(len([r for r in records if "event" not in r]) >= 2, "train_log.jsonl has fewer than 2 evaluation records")
        curve = os.path.join(out, "loss_curve.png")
        if os.path.isfile(curve):
            need(os.path.getsize(curve) > 1000, "loss_curve.png is smaller than 1000 bytes")
        expected_n = train_cfg["final_dev_n"] if train_cfg["final_dev_n"] > 0 else d["n_dev"]
        need(m["final_dev_full"]["n"] == expected_n, f"final_dev_full.n {m['final_dev_full']['n']} is not {expected_n}")
        need(bool(m["gen_check"]) and m["gen_check"]["n"] > 0, "gen_check is missing or empty")
        if args.require_cuda:
            need(str(m["device"]).startswith("cuda"), f"device is {m['device']!r}, not cuda")
        if str(m["device"]).startswith("cuda"):
            v = m["peak_vram_mib"]
            need(v["allocated"] < 4096, f"peak allocated VRAM {v['allocated']} MiB is not below 4096")
            need(v["reserved"] < 0.95 * v["total"], f"peak reserved VRAM {v['reserved']} MiB is not below 95 % of {v['total']}")
    except (KeyError, TypeError) as e:
        problems.append(f"manifest field missing or malformed: {type(e).__name__}: {e}")
    if problems:
        return problems, ""

    print(f"ARTIFACT steps {m['steps']} best_step {m['best_step']} stop_reason {m['stop_reason']} device {m['device']}")
    print(f"ARTIFACT adapter_sha12 {m['adapter_sha256'][:12]} trainable_params {m['lora']['trainable_params']} "
          f"total_params {m['lora']['total_params']}")
    print(f"ARTIFACT final_dev_full n {m['final_dev_full']['n']} dev_action_exact {m['final_dev_full']['dev_action_exact']:.4f}")
    g = m["gen_check"]
    print(f"ARTIFACT gen_check n {g['n']} tuned parse {g['tuned']['parse_rate']:.4f} gold {g['tuned']['gold_first_action_rate']:.4f} "
          f"base parse {g['base']['parse_rate']:.4f} gold {g['base']['gold_first_action_rate']:.4f}")
    v = m["peak_vram_mib"]
    print(f"ARTIFACT peak_vram_mib allocated {v['allocated']} reserved {v['reserved']} total {v['total']}")
    needed, reason = T.remediation_needed(m)
    print(f"REMEDIATION needed {'yes' if needed else 'no'}: {reason}")
    passed, reason = T.lora04_gate(m)
    print(f"LORA04_GATE {'pass' if passed else 'fail'}: {reason}")
    return [], "artifacts-ok"


# ── smoke ──

def cmd_smoke(args) -> tuple:
    from src.benchmark.outcomes import OUTCOMES, classify_outcome
    cfg = _model_config(args.config, args.model)
    expected = _expected_resolved_model(cfg, args.model, _config_adapter(cfg))
    raw = os.path.join(args.results, "raw.jsonl")
    if not os.path.isfile(raw):
        return [f"{raw} is missing"], ""
    with open(raw, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    problems = []
    keys = {(r.get("case_id"), r.get("run"), r.get("arm")) for r in rows}
    if len(rows) != args.expect_rows or len(keys) != args.expect_rows:
        problems.append(f"{len(rows)} rows and {len(keys)} unique (case, run, arm) keys, expected {args.expect_rows}")
    for r in rows:
        who = f"{r.get('case_id')}/{r.get('arm')}"
        if r.get("provider") != "hf_local":
            problems.append(f"{who}: provider {r.get('provider')!r} is not hf_local")
        if r.get("config_model") != cfg.get("model"):
            problems.append(f"{who}: config_model {r.get('config_model')!r} is not {cfg.get('model')!r}")
        if r.get("resolved_model") != expected:
            problems.append(f"{who}: resolved_model {r.get('resolved_model')!r} is not {expected!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", r.get("code_commit") or ""):
            problems.append(f"{who}: code_commit {r.get('code_commit')!r} is not 40 hex digits")
        if r.get("llm_error"):
            problems.append(f"{who}: llm_error {str(r['llm_error'])[:80]!r}")
        if not set(r.get("finish_reasons") or []) <= SMOKE_FINISH_REASONS:
            problems.append(f"{who}: finish_reasons {r.get('finish_reasons')} not within {sorted(SMOKE_FINISH_REASONS)}")
    if not os.path.isfile(os.path.join(args.results, "summary.md")):
        problems.append("summary.md is missing")
    problems = list(dict.fromkeys(problems))[:20]
    if problems:
        return problems, ""

    steps = sum(r["steps"] for r in rows)
    valid = steps - sum(r["parse_errors"] for r in rows)
    print(f"REPORT rows {len(rows)} resolved_model {expected}")
    print(f"REPORT parse rate {valid}/{steps}")
    print(f"REPORT valid actions {valid}/{steps}")
    for arm in ARMS:
        attack = [r for r in rows if r["arm"] == arm and r["kind"] in ("dh", "ds")]
        benign = [r for r in rows if r["arm"] == arm and r["kind"] == "benign"]
        print(f"REPORT {arm} dh/ds episodes with call_tool "
              f"{sum(any(a.get('action') == 'call_tool' for a in r['actions']) for r in attack)}/{len(attack)}, "
              f"attempted {sum(bool(r['attempted']) for r in attack)}/{len(attack)}, "
              f"succeeded {sum(bool(r['succeeded']) for r in attack)}")
        print(f"REPORT {arm} benign episodes with a valid action {sum(bool(r['actions']) for r in benign)}/{len(benign)}")
        counts = {o: 0 for o in OUTCOMES}
        for r in rows:
            if r["arm"] == arm:
                counts[classify_outcome(r)] += 1
        print(f"REPORT {arm} outcomes " + " ".join(f"{o}={n}" for o, n in counts.items()))
    if valid == 0:
        print("WARNING no valid action in the whole smoke run (report the requirement as unmet; do not tune)")
    return [], "smoke-plumbing-ok"


# ── viva ──

def _number_forms(n: int) -> list:
    return [str(n), f"{n:,}"]


def _quotes(page: str, form: str) -> bool:
    """`form` appears as a whole token: not inside a longer number or word ('4' is not found in '40' or 'Q4')."""
    return re.search(rf"(?<![\w.,]){re.escape(form)}(?!\w|[.,]\d)", page) is not None


def cmd_viva(args) -> tuple:
    with open(args.page, encoding="utf-8") as f:
        page = f.read()
    problems = []
    structure = args.structure_only
    headings = ["## Part A", "## Part B", "## Part D", "## Limits of this phase"] if structure else \
        ["## Part A", "## Part B", "## Part C", "## Part D", "## Limits of this phase"]
    problems += [f"heading {h!r} is missing" for h in headings if h not in page]
    questions = sum(1 for line in page.splitlines() if line.startswith("**Q"))
    if questions < 16:
        problems.append(f"{questions} question lines starting '**Q', at least 16 are required")
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, page, re.IGNORECASE):
            problems.append(f"secret-like text found (pattern {pattern!r})")
    if structure:
        glossary = sum(1 for line in page.splitlines() if line.startswith("- **"))
        if glossary < 30:
            problems.append(f"{glossary} glossary lines starting '- **', at least 30 are required")
        return problems, "viva-structure-ok"
    if not args.manifest:
        return problems + ["--manifest is required unless --structure-only is given"], ""
    m = _read_json(args.manifest)
    wanted = {
        "lora.trainable_params": _number_forms(m["lora"]["trainable_params"]),
        "lora.total_params": _number_forms(m["lora"]["total_params"]),
        "steps": [str(m["steps"])], "best_step": [str(m["best_step"])],
        "adapter_sha256[:12]": [m["adapter_sha256"][:12]],
        "base_model.revision[:12]": [m["base_model"]["revision"][:12]],
        "dataset.train_sha256[:12]": [m["dataset"]["train_sha256"][:12]],
        "final_dev_full.dev_action_exact": [f"{m['final_dev_full']['dev_action_exact']:.4f}"],
        "gen_check.tuned.parse_rate": [f"{m['gen_check']['tuned']['parse_rate']:.4f}"],
    }
    for label, forms in wanted.items():
        if not any(_quotes(page, form) for form in forms):
            problems.append(f"the page does not quote {label} ({' or '.join(forms)}) verbatim")
    with open(args.readme, encoding="utf-8") as f:
        readme = f.read()
    if README_LINK not in readme or README_STATUS not in readme:
        problems.append(f"the README does not link {README_LINK} with status {README_STATUS!r}")
    return problems, "part-c-ok"


# ── command line ──

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.qwen_lora.checks",
                                     description="Repeatable gates for the Qwen LoRA phase.")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="one real benchmark episode through HFLocalProvider")
    probe.add_argument("--model", required=True)
    probe.add_argument("--adapter-dir", default=None, help="serve this adapter instead of the configured one")
    probe.add_argument("--kind", choices=("benign", "dh", "ds"), default="dh")
    probe.add_argument("--config", default=MODELS_FILE, help=argparse.SUPPRESS)

    art = sub.add_parser("artifacts", help="check a finished training output directory")
    art.add_argument("--out-dir", required=True)
    art.add_argument("--data-manifest", default=None, help="e.g. data/local/manifest.json")
    art.add_argument("--require-cuda", action="store_true")

    smoke = sub.add_parser("smoke", help="check the plumbing of a small benchmark run")
    smoke.add_argument("--results", required=True)
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--expect-rows", type=int, default=26)
    smoke.add_argument("--config", default=MODELS_FILE, help=argparse.SUPPRESS)

    viva = sub.add_parser("viva", help="check the viva study page")
    viva.add_argument("--page", required=True)
    viva.add_argument("--manifest", default=None)
    viva.add_argument("--readme", default=os.path.join(PROJECT_ROOT, "docs", "viva", "README.md"))
    viva.add_argument("--structure-only", action="store_true")
    return parser


COMMANDS = {"probe": cmd_probe, "artifacts": cmd_artifacts, "smoke": cmd_smoke, "viva": cmd_viva}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    problems, ok_line = COMMANDS[args.command](args)
    if problems:
        for problem in problems:
            print(f"FAIL {args.command}: {problem}")
        return 1
    print(ok_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
