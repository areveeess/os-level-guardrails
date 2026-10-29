"""
Deliverables sections — confounds, legacy rows, threats to validity, local training figures
===========================================================================================
The cross-model table compares models that differ in far more than their training: size,
quantization, where they run, context window, decoding and provenance. D-04 requires those
differences to be printed next to the results as confounds, and the limits of the evidence
as threats to validity, so the panel never reads a gap between two rows as a clean effect.

Every value here is read from a recorded source (the frozen ollama_show.txt, and the copies of
config/models.yaml and the two model manifests that the freeze puts under frozen/inputs/, WR-04)
or stated as "not recorded" / "not disclosed"; nothing is guessed. The hf_local precision and
context follow the model's config entry (device, max_context_tokens) the way the provider does.
Threat sentences are built from numbers.json, so they change with the data.

config/models.yaml is read with yaml.safe_load, never through src.llm.registry, which would
load API keys the deliverables do not need.
"""

import json
import os
import re

import yaml

from src.benchmark.report import DEFAULT_EXPECTED_EPISODES, LAYER_EVIDENCE, load_rows, provenance_summary, summarize
from src.deliverables import HOSTED_MODELS, LEGACY_SOURCES, MODEL_ORDER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mirrors src.llm.hf_local.DEFAULT_MAX_CONTEXT_TOKENS (pinned by a test that reads the source; the module is
# not imported here because importing src.llm loads keys the deliverables do not need)
HF_LOCAL_CONTEXT = 4096

NOT_RECORDED = "not recorded"
SCRATCH_MANIFEST = os.path.join("models", "scratch", "manifest.json")
LORA_MANIFEST = os.path.join("models", "qwen-agentlora", "manifest.json")
CONFIG_FILE = os.path.join("config", "models.yaml")

WR01_NONE = ("No current-engine episode had a context overflow (0 for every model), so WR-01 touches none of the "
             "current data; the bias would apply only to future runs.")


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _config(repo_root: str) -> dict:
    try:
        with open(os.path.join(repo_root, CONFIG_FILE), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def ollama_show(path: str) -> dict:
    """{parameters, quantization, num_ctx} from an `ollama show` listing; a missing value is None."""
    out = {"parameters": None, "quantization": None, "num_ctx": None}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out
    for key in out:
        m = re.search(rf"^\s*{key}\s+(\S+)", text, re.MULTILINE)
        if m:
            out[key] = m.group(1)
    return out


def _millions(n) -> str:
    return f"{n / 1e6:.1f}M" if isinstance(n, (int, float)) else NOT_RECORDED


def _hosted_size(model_id) -> str:
    m = re.search(r"-(\d+(?:\.\d+)?)b$", str(model_id or ""), re.IGNORECASE)
    return f"{m.group(1)}B (per the model id, not verified here)" if m else NOT_RECORDED


def confound_rows(frozen_dir: str, rows: list, repo_root: str = REPO_ROOT) -> list:
    """One confounds row per current-engine model in MODEL_ORDER (see CONFOUND_COLUMNS in report.py).

    Args:
        frozen_dir: the frozen directory (its sources hold each Ollama model's ollama_show.txt).
        rows: the frozen current-engine rows (for the largest prompt each model reported).
        repo_root: the directory holding config/models.yaml and models/*/manifest.json (the build passes the
            frozen inputs/ copy)."""
    config = _config(repo_root)
    scratch = _read_json(os.path.join(repo_root, SCRATCH_MANIFEST))
    lora = _read_json(os.path.join(repo_root, LORA_MANIFEST)).get("lora") or {}
    trainable, total = lora.get("trainable_params"), lora.get("total_params")
    base_params = total - trainable if isinstance(trainable, int) and isinstance(total, int) else None
    max_prompt = {p["model"]: p["max_prompt_tokens"] for p in provenance_summary(rows)}
    out = []
    for model in MODEL_ORDER:
        cfg = config.get(model) or {}
        provider = cfg.get("provider")
        row = {"model": model}
        if model.startswith("ollama-"):
            show = ollama_show(os.path.join(frozen_dir, "sources", f"phase4-{model}", "ollama_show.txt"))
            row.update(parameters=show["parameters"] or NOT_RECORDED, precision=show["quantization"] or NOT_RECORDED,
                       placement="local, Ollama server",
                       context=f"{show['num_ctx']} tokens (num_ctx)" if show["num_ctx"] else NOT_RECORDED,
                       provenance=f"pretrained instruct, pulled Ollama tag ({cfg.get('ollama_base') or NOT_RECORDED})")
        elif provider == "hf_local":
            tuned = bool(cfg.get("adapter"))
            params = _millions(base_params)
            if tuned and base_params is not None:
                params += f" + {_millions(trainable)} LoRA"
            on_gpu = str(cfg.get("device") or "cuda").startswith("cuda")      # hf_local.load_model's dtype rule
            row.update(parameters=params, precision="bf16 on the GPU" if on_gpu else "fp32 on the CPU",
                       placement="local, in-process",
                       context=f"{cfg.get('max_context_tokens') or HF_LOCAL_CONTEXT} tokens",
                       provenance="LoRA-tuned here on the Phase 2 synthetic mix" if tuned
                       else "pretrained instruct, not tuned here")
        elif provider == "scratch":
            context = ((scratch.get("config") or {}).get("model") or {}).get("context")
            row.update(parameters=_millions(scratch.get("params")), precision="fp32", placement="local, in-process",
                       context=f"{context} characters" if context else NOT_RECORDED,
                       provenance="trained from scratch here")
        elif model in HOSTED_MODELS:
            row.update(parameters=_hosted_size(cfg.get("model")), precision="not disclosed", placement="hosted (Groq)",
                       context="provider limit, not measured", provenance="hosted vendor model")
        else:
            row.update(parameters=NOT_RECORDED, precision=NOT_RECORDED, placement=NOT_RECORDED,
                       context=NOT_RECORDED, provenance=NOT_RECORDED)
        row["decoding"] = ("temperature 0 requested; server-side determinism not guaranteed" if model in HOSTED_MODELS
                           else "temperature 0 (greedy)")
        prompt = max_prompt.get(model)
        row["max_prompt"] = "not reported" if prompt is None else str(prompt)
        out.append(row)
    return out


LEGACY_FIELDS = ("model", "family", "provider", "status", "episodes_done", "episodes_expected", "attack_pairs")
LEGACY_PREFIXES = ("attempted", "success_unprotected", "success_protected", "asr_valid_unprotected",
                   "blocked_given_attempt", "benign_task_unprotected", "benign_task_protected", "false_positive")


def legacy_summary(frozen_dir: str) -> list:
    """Summary rows of the two legacy frozen sources only, expected DEFAULT_EXPECTED_EPISODES each, flattened to
    the fields the legacy table and the consistency check read."""
    out = []
    for name, _ in LEGACY_SOURCES:
        rows = load_rows(os.path.join(frozen_dir, "sources", name, "raw.jsonl"))
        for s in summarize(rows, expected=DEFAULT_EXPECTED_EPISODES):
            entry = {key: s[key] for key in LEGACY_FIELDS}
            for prefix in LEGACY_PREFIXES:
                entry.update({f"{prefix}_{part}": s[f"{prefix}_{part}"] for part in ("pct", "k", "n", "lo", "hi")})
            out.append(entry)
    return out


def _kn(metric: dict) -> str:
    return f"{metric['k']}/{metric['n']}"


def threat_items(numbers: dict, legacy: list) -> list:
    """The threats-to-validity sentences, each built from numbers.json values (no hard-coded zero)."""
    models = [m for m in numbers["models"] if m["model"] in MODEL_ORDER]
    items = []

    partial = [m for m in models if not m["complete"]]
    if partial:
        items.append("Partial runs: " + "; ".join(
            f"{m['model']} is {m['label']} with {m['attack_pairs']} paired attack episodes" for m in partial)
            + ". Their rates are not final and are not ranked against the complete rows.")
    else:
        items.append("Every current-engine model finished its planned run.")

    if legacy:
        items.append("Legacy rows: " + "; ".join(
            f"{s['model']} {s['episodes_done']}/{s['episodes_expected']} episodes, {s['attack_pairs']} paired attack "
            f"episodes" for s in legacy)
            + ". They come from the October 2026 run under an older engine and appear only in their own table; "
              "they are not compared, pooled or charted.")

    pooled = numbers["pooled"]["layers"]
    n = sum(group["n"] for group in pooled.values())
    parts = []
    for layer, title in (("sandbox", "Sandbox"), ("seccomp", "Seccomp")):
        k = sum(group[layer]["k"] for group in pooled.values())
        if k == 0:
            parts.append(f"{title}: no block recorded, 0/{n} protected-arm episodes")
        else:
            parts.append(f"{title}: blocks were recorded in {k}/{n} protected-arm episodes")
    items.append(f"Kernel layers, pooled over the {len(models)} language models: " + "; ".join(parts)
                 + ". The policy engine stops attacks before they reach the kernel layers, so a zero here is not "
                   "evidence that those layers are idle. " + LAYER_EVIDENCE)

    failing = [m for m in models if sum(m["format_failures"].values()) > 0]
    if failing:
        items.append("Format failures depress the raw success rate of weak models, because an unreadable reply is "
                     "not an attack: " + "; ".join(
                         f"{m['model']} success without guardrails {_kn(m['metrics']['success_unprotected'])} vs "
                         f"ASR-valid {_kn(m['metrics']['asr_valid_unprotected'])}, with guardrails "
                         f"{_kn(m['metrics']['success_protected'])} vs ASR-valid "
                         f"{_kn(m['metrics']['asr_valid_protected'])}" for m in failing)
                     + ". Read the ASR-valid column next to the raw one.")

    overflow = [(m["model"], m["context_overflow_episodes"]) for m in models]
    wr01 = ("Deferred review finding WR-01: a context overflow after valid steps is classed format_failure, so an "
            "overflowing episode that ignored the injection leaves the ASR-valid denominator while an attempted one "
            "stays in it, biasing ASR-valid upward; raw Success is not affected. ")
    if all(count == 0 for _, count in overflow):
        wr01 += WR01_NONE
    else:
        wr01 += "Context-overflow episodes per model: " + ", ".join(f"{m} {c}" for m, c in overflow) + "."
    wr01 += (" The fix waits until after the D-09 freeze because src/benchmark/outcomes.py is run-affecting "
             "(WINDOWS.md entry 4).")
    items.append(wr01)

    items.append("One benchmark: InjecAgent, 57 cases (20 dh, 20 ds, 17 benign) with simulated tools. The results "
                 "may not transfer to other attack sets or to real tools.")
    items.append("Decoding at temperature 0: the three runs of a case are near-duplicates, so the episode-level "
                 "Wilson intervals are optimistic; the case-level view counts each case once.")
    items.append("Confounds: the models differ in size, quantization, placement, context window and training "
                 "provenance (see Confounds between the models), so a gap between two rows is not attributable to "
                 "any one of these.")
    return items


def local_training(repo_root: str = REPO_ROOT) -> dict:
    """The local training figures of the two locally made models, copied only from the named manifest keys."""
    scratch = _read_json(os.path.join(repo_root, SCRATCH_MANIFEST))
    lora_manifest = _read_json(os.path.join(repo_root, LORA_MANIFEST))
    best = scratch.get("best") or {}
    lora = lora_manifest.get("lora") or {}
    return {
        "scratch-gpt": {
            "params": scratch.get("params"),
            "config_model": (scratch.get("config") or {}).get("model"),
            "steps": scratch.get("steps"),
            "best_step": scratch.get("best_step"),
            "dev_action_exact": best.get("dev_action_exact"),
            "dev_loss": best.get("dev_loss"),
            "stop_reason": scratch.get("stop_reason"),
        },
        "qwen2.5-0.5b-agentlora": {
            "r": lora.get("r"),
            "alpha": lora.get("alpha"),
            "trainable_params": lora.get("trainable_params"),
            "total_params": lora.get("total_params"),
            "steps": lora_manifest.get("steps"),
            "dev_action_exact": (lora_manifest.get("final_dev_full") or {}).get("dev_action_exact"),
            "base_model": (lora_manifest.get("base_model") or {}).get("id"),
        },
    }

