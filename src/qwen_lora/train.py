"""
Qwen2.5-0.5B LoRA fine-tuning on the Phase 2 episodes
=====================================================
A custom training loop (no Trainer, no TRL) that teaches the cached Qwen2.5-0.5B-Instruct the agent's
JSON action format with a LoRA adapter, inside the 6 GB of the RTX 3060 laptop. Every choice has a reason:

  D-02  Leakage gate. Both episode files are read with read_episodes and checked with assert_clean
        before step 0, so a training file that contains a benchmark tool, address or sentence stops the
        run before any weights are touched.

  D-03  One chat sequence per episode, rendered by the tokenizer's own chat template from
        chat.episode_to_chat, so a step's training context is exactly what the provider renders at
        inference (src/qwen_lora/data.py).

  D-04  Loss on the assistant action plus its <|im_end|> stop token only, chosen by character offsets
        (never by token id). The 151,936-wide LM head is applied only to the supervised positions and
        the cross-entropy is taken in fp32: 2.4 GiB instead of 8.3 GiB for the same batch. The loss is
        normalised by the number of supervised tokens of the whole accumulation window, so one
        optimizer step is a token-weighted mean over micro_batch x accum episodes. Sequences over
        MAX_LEN are an error, never truncated.

  D-05  LoRA r 16, alpha 32, dropout 0.05 on all linear layers of the frozen base (bf16 on CUDA,
        fp32 on the CPU), gradient checkpointing, AdamW at 2e-4 with 20 warmup steps and a cosine
        decay to 2e-5, gradient clip 1.0, micro-batch 4 x accumulation 4. The learning-rate schedule is
        the tested lr_at of the ScratchGPT trainer, so one schedule serves both.

  D-06  Selection and early stopping on dev_loss_action, the dev loss with the unpredictable respond
        message text excluded (the Phase 2 CR-01 lesson), never on the plain dev loss.

  D-01  The training mix is the Phase 2 50/25/25 compliance mix: it teaches the model to follow instructions
        that arrive inside tool output. The tuned model is therefore a second injection-compliant worst-case
        principal next to the un-tuned base control, and every manifest says so (D01_NOTE in run_notes).

  D-07  The finished adapter is proven through the serving path: after training, HFLocalProvider loads it
        from --out-dir with every stamp check (a run that cannot be served fails) and greedily writes the first
        action for gen_check_n dev prompts, as does the base model ("gen_check" in the manifest).

The adapter is saved with save_pretrained(save_embedding_layers=False) next to a manifest.json holding
the fields src.llm.hf_local verifies before it serves the adapter: adapter_sha256 (weights),
adapter_config_sha256 and adapter_identity_sha256 (what a result row is stamped with), base_model {id,
revision} and render {fingerprint}, plus everything the viva page and the report quote (MANIFEST_KEYS).

Outputs in --out-dir: adapter_config.json, adapter_model.safetensors, manifest.json, train_log.jsonl (one
record per evaluation) and loss_curve.png. The launcher src/qwen_lora/run_train.sh owns train_progress.log,
train.done and gpu_preflight.log in the same directory; the trainer never touches them.

Stage, then promote (review WR-03). Nothing a run produces is written to those final names while it runs. The
best adapter so far, and at the end the manifest and the curve, live in the staging directory
adapter.best.tmp, the log in train_log.new.jsonl. Only after the final dev pass and the generation check (which
serves the staged adapter through HFLocalProvider) have succeeded are the files moved into place, manifest.json
last. Until then a previous valid adapter in --out-dir is untouched, and a run that fails after training keeps its
staged adapter for inspection instead of deleting 40 minutes of GPU work.

Progress lines on stdout (the poller of the GPU plan reads these):
    device <cpu|cuda> trainable <n> total <m>
    step <s>/<max> loss <x> lr <y> tok/s <z> elapsed <t>s vram <alloc>/<reserved>MiB rss <r>MB
    eval step <s> dev_loss <x> dev_loss_action <a> dev_action_exact <e> best <b>
    final_dev_full n <n> dev_action_exact <e> dev_loss_action <a>
    gen_check tuned parse <k>/<n> gold <g>/<n> base parse <k>/<n> gold <g>/<n>
    DONE best_step <s> best_dev_loss_action <x> wall <t>s stop <max_steps|early_stop>
    RESUMED from step <s>                       (only after --resume)
    FAILED non_finite_loss step <s>             (exit code 3; also FAILED non_finite_dev_loss step <s>)
    FAILED vram_budget reserved <r> total <t>   (exit code 4; also FAILED cuda_oom step <s>)

Exit codes: 0 done, 2 bad arguments (paths outside the repository, missing files, another tool's manifest),
3 a loss stopped being finite, 4 the card's memory (the 95 % guard or a CUDA out-of-memory error); anything
else is a crash. A run that fails while training leaves no staged adapter or resume file behind; one that fails
after the last step (final dev pass, generation check) keeps them. Neither touches a previously promoted adapter.

    python -m src.qwen_lora.train [--train FILE] [--dev FILE] [--out-dir DIR] [--resume | --fresh] ...
"""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Set before torch is imported: it cut the reserved CUDA memory from 4.9 GiB to 2.5 GiB at the same batch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from src.agent_harness.prompting import OBSERVATION_TEMPLATE, SYSTEM_PROMPT, ActionParseError, build_messages, parse_action
from src.benchmark.injecagent import PROJECT_ROOT
from src.llm.hf_local import (HFLocalProvider, adapter_config_sha256, adapter_identity_sha256, resolve_base_model,
                              sha256_file)
from src.qwen_lora import chat
from src.qwen_lora import data
from src.qwen_lora.data import build_example, collate
from src.scratch_llm.dataset import assert_clean, file_sha256, held_out_sha256, read_episodes
from src.scratch_llm.format import compact_action
from src.scratch_llm.provenance import (git_commit, git_dirty, refuse_foreign_manifest, resolve_under_root,
                                        write_json_atomic)
from src.scratch_llm.train import lr_at

ALLOWED_ROOT = PROJECT_ROOT      # --train, --dev, --out-dir and a directory --base-model must stay inside this directory
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CLASSES = ("benign", "dh", "ds")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MANIFEST_KEYS = ("schema", "created_at", "code_commit", "code_dirty", "python", "torch", "cuda", "device", "packages",
                 "seed", "base_model", "tokenizer", "render", "dataset", "lora", "config", "steps", "best_step",
                 "tokens_seen", "peak_vram_mib", "selection", "best", "last_eval", "final_dev_full", "gen_check",
                 "evaluation_overlap", "adapter_sha256", "adapter_config_sha256", "adapter_identity_sha256", "wall_seconds", "stop_reason", "resumed_from_steps", "run_notes")
PACKAGES = ("transformers", "peft", "accelerate", "tokenizers", "safetensors", "huggingface_hub")
MAX_NOTE_CHARS = 200
EXIT_NON_FINITE = 3
EXIT_VRAM = 4                    # the 95 % guard and CUDA out of memory
VRAM_LIMIT_FRACTION = 0.95      # above this share of the card the WSL driver silently spills to system memory (about 7x slower)
RESUME_FILE = "qwen_lora.resume.tmp"       # everything needed to continue a killed run, rewritten after every evaluation
MANIFEST_SCHEMA = 1

D01_NOTE = ("training mix is the Phase 2 50/25/25 compliance mix: the tuned model is trained to follow injected "
            "instructions and is a second worst-case principal next to the un-tuned base row (D-01)")

# The clarified D-06 (03-CONTEXT.md, commit 5258be1), pre-registered here before the GPU run.
REMEDIATION_THRESHOLD = 0.90
REMEDIATION_RULE = (
    f"Pre-registered before the GPU run, applied at most once, from this manifest's dev metrics only (never from "
    f"benchmark or smoke outcomes): if final_dev_full.dev_action_exact < {REMEDIATION_THRESHOLD:.2f} and the run was "
    f"still improving when it ended (stop_reason max_steps and best_step >= steps - 2 x eval_every, the last two "
    f"evaluations as in Phase 2), train once more from scratch with the same data, seed and settings and max_steps "
    f"doubled; a run that converged (stop_reason early_stop) below {REMEDIATION_THRESHOLD:.2f} gets no retry and is "
    f"recorded as a finding. The 60-step research run already reached 0.899 on 200 dev episodes, so the threshold is "
    f"reachable.")

# LORA-04 "the tuned model produces valid actions", decided from dev data only, before any smoke run.
LORA04_PARSE_THRESHOLD = 0.90
LORA04_GATE_RULE = (
    f"Pre-registered before the GPU run and computed only from dev data before any smoke run: LORA-04 'produces valid "
    f"actions' passes when gen_check.tuned.parse_rate >= {LORA04_PARSE_THRESHOLD:.2f} on the gen_check_n first-turn dev "
    f"prompts of the kept, finally retained adapter (research: 20/20 first-turn replies parsed after 60 steps, 35/36 "
    f"steps parsed in real episodes). A fail blocks LORA-04 and is reported as a verification gap; it never triggers "
    f"tuning or retraining, and the smoke runs stay plumbing and behaviour evidence only.")
PAD_ID = 151643                  # <|endoftext|>: only fills attention-masked positions, never decides a mask
SELECTION_METRIC = "dev_loss_action"
SELECTION_DESCRIPTION = ("mean cross-entropy over action-level supervised dev tokens: all supervised tokens of the "
                         "300-episode stratified dev subset except the message text of respond actions, whose boundary "
                         "token BPE merges with the closing quote is excluded whole; used for the kept adapter and "
                         "early stopping")
IN_RUN_ADAPTER = "adapter.best.tmp"      # the staging directory: best adapter so far, then manifest and curve; promoted at the end
STAGED_LOG = "train_log.new.jsonl"       # the log of the running (or failed) run; becomes train_log.jsonl at promotion
PROMOTE_ORDER = ("adapter_model.safetensors", "adapter_config.json", "loss_curve.png", "train_log.jsonl",
                 "manifest.json")     # manifest.json last: it is what makes the directory a complete, verifiable adapter
# every metric evaluate() returns, in the order the manifest lists them. The last four (review WR-04) put the headline
# dev_action_exact in context: about a third of the dev spans are respond lines whose action-level tokens are boilerplate
# (the head, the closing quote, <|im_end|>), so the headline is diluted; they gate nothing.
EVAL_KEYS = ("dev_loss", "dev_loss_action", "dev_loss_respond_text", "dev_action_exact", "dev_action_exact_by_cls",
             "dev_line_exact", "dev_line_exact_by_cls", "dev_first_action_exact",
             "dev_respond_share", "dev_nonrespond_action_exact", "dev_nonrespond_action_exact_by_cls",
             "dev_min_class_action_exact")
OVERLAP_NOTE = ("the selection subset (dev_eval_n episodes: it picks the kept adapter and drives early stopping) is part "
                "of the final full-dev pass, and the generation-check prompts are drawn from the same dev file, so the "
                "final numbers are dev numbers that are not independent of the selection; no benchmark content is "
                "involved")


class NonFiniteLossError(RuntimeError):
    """The training or dev loss became NaN or infinite; the run was stopped and left nothing loadable (exit code 3)."""


class VramBudgetError(RuntimeError):
    """Reserved GPU memory passed VRAM_LIMIT_FRACTION of the card, or CUDA ran out of memory (exit code 4)."""


@dataclass
class TrainConfig:
    seed: int = 0
    max_steps: int = 600           # optimizer steps (D-05: ~41 minutes on the RTX 3060)
    micro_batch: int = 4           # episodes per forward/backward pass
    accum: int = 4                 # passes per optimizer step: effective batch micro_batch x accum = 16
    lr: float = 2e-4               # peak learning rate
    min_lr: float = 2e-5           # cosine floor, reached at the last step
    warmup: int = 20               # linear warmup steps
    grad_clip: float = 1.0         # global norm over the trainable parameters
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32           # scaling alpha / r = 2
    lora_dropout: float = 0.05
    eval_every: int = 100          # optimizer steps between dev evaluations (and one after the last step)
    log_every: int = 10            # optimizer steps between progress lines
    patience: int = 3              # evaluations without a better dev_loss_action before early stopping (D-06)
    n_train_used: int = 0          # 0: max_steps x micro_batch x accum, capped at the file size
    dev_eval_n: int = 300          # dev episodes in the in-training evaluations (stratified, fixed by the seed)
    final_dev_n: int = 0           # 0: the whole dev file in the final evaluation
    gen_check_n: int = 100         # dev prompts of the final generation check
    gen_max_new_tokens: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


# ── model ──

def _autocast(device):
    """bf16 autocast on the GPU (the frozen base is bf16, the LoRA weights fp32); nothing on the CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _model_id(base_model: str) -> str:
    """The name the adapter records for its base: a Hub id as given, a local directory by real path."""
    return os.path.realpath(base_model) if os.path.isdir(base_model) else base_model


def build_lora_model(base_model: str, cfg: TrainConfig, device: str) -> tuple:
    """(peft_model, tokenizer, revision): the frozen base from the local cache plus a fresh LoRA adapter.

    Loaded offline through transformers' own Qwen2 code (no code from a model repository is ever run). The
    gradient checkpointing is enabled before get_peft_model and enable_input_require_grads follows it, the
    order that was measured to work with PEFT; the adapter config records the configured base model, not
    the snapshot path the weights were read from, because src.llm.hf_local compares it with models.yaml.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, revision = resolve_base_model(base_model)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                  lora_dropout=cfg.lora_dropout, target_modules="all-linear",
                                                  task_type="CAUSAL_LM"))
    peft_model.peft_config["default"].base_model_name_or_path = _model_id(base_model)
    # A mismatch here means PEFT adapted something other than the seven projections of every layer.
    # (Depending on the PEFT version, "all-linear" is stored as the leaf names or as the full module names, so
    # compare the leaves; the number of adapted layers is counted on the model itself, not on the stored names.)
    resolved = set(peft_model.peft_config["default"].target_modules)
    leaves = {name.rsplit(".", 1)[-1] for name in resolved}
    n_adapted = sum(1 for m in peft_model.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B"))
    assert leaves == set(LORA_TARGETS) and n_adapted == len(LORA_TARGETS) * model.config.num_hidden_layers, \
        f"all-linear resolved to {sorted(resolved)} with {n_adapted} adapted layers, not the {list(LORA_TARGETS)} of every layer"
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    c = model.config
    expected = lora_param_count(c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads,
                                c.num_hidden_layers, cfg.lora_r)
    assert trainable == expected, f"{trainable} trainable parameters, the LoRA arithmetic gives {expected}"
    return peft_model, tokenizer, revision


def lora_param_count(hidden: int, intermediate: int, n_heads: int, n_kv_heads: int, n_layers: int, r: int) -> int:
    """Trainable LoRA parameters of the seven projections: r x (in + out) for each adapted Linear layer.

    q and o map hidden to hidden; k and v map hidden to n_kv_heads x head_dim (grouped-query attention);
    gate and up map hidden to intermediate and down maps back. For Qwen2.5-0.5B (896, 4864, 14 heads, 2 kv
    heads, 24 layers, r 16) that is 366,592 per layer and 8,798,208 in all.
    """
    kv = n_kv_heads * (hidden // n_heads)
    per_layer = (hidden + hidden) + (hidden + kv) + (hidden + kv) + (hidden + hidden) \
        + (hidden + intermediate) + (hidden + intermediate) + (intermediate + hidden)
    return r * per_layer * n_layers


def _decoder_and_head(peft_model) -> tuple:
    """(decoder, lm_head) inside the PEFT wrapper. This couples to PEFT and Qwen2 internals, hence the asserts."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
    inner = peft_model.base_model.model
    assert isinstance(inner, Qwen2ForCausalLM), f"unexpected model class {type(inner).__name__} inside the PEFT wrapper"
    assert isinstance(inner.model, Qwen2Model), f"unexpected decoder class {type(inner.model).__name__}"
    return inner.model, inner.lm_head


def supervised_logits(peft_model, input_ids, attention_mask, labels) -> tuple:
    """(logits of the supervised positions in fp32, their target ids, the boolean position mask).

    The decoder produces the hidden states; the LM head sees only the rows whose next-token label is supervised,
    so the 151,936-wide logits are never built for prompt, observation or padding positions. The model's own
    labels= loss is never used.
    """
    decoder, lm_head = _decoder_and_head(peft_model)
    hidden = decoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    targets = labels[:, 1:]
    positions = targets != -100
    return lm_head(hidden[:, :-1][positions]).float(), targets[positions], positions


def supervised_loss(peft_model, input_ids, attention_mask, labels) -> tuple:
    """(summed cross-entropy over the supervised positions, how many there are)."""
    logits, targets, _ = supervised_logits(peft_model, input_ids, attention_mask, labels)
    return torch.nn.functional.cross_entropy(logits, targets, reduction="sum"), int(targets.numel())


# ── evaluation ──

@torch.no_grad()
def evaluate(peft_model, examples: list, micro_batch: int, device, logits_fn=None) -> dict:
    """Dev loss and teacher-forced exact match, with the unpredictable respond message text kept apart.

    A span is one assistant message plus its <|im_end|>. It is exact when the argmax equals the gold token at
    every position given the gold prefix, which is what greedy decoding would produce. The dev split uses
    held-back reply wording, so the message text of a respond line is something no model can predict;
    counting it would let that text decide which adapter is kept (the Phase 2 review CR-01). The metrics
    come in two groups.

    Selection and headline (action level, respond message text excluded):
      dev_loss_action        mean cross-entropy over the action-level supervised tokens
      dev_action_exact       share of spans whose action-level tokens are all argmax-correct
      dev_action_exact_by_cls, dev_first_action_exact (span 0 of every episode)

    Context for the headline (review WR-04; they select and gate nothing). A respond span's action-level tokens are
    the fixed head, the closing quote and <|im_end|>, so respond spans (about a third of the dev spans) are nearly free
    credit and dilute dev_action_exact:
      dev_respond_share                    share of all spans that are respond lines
      dev_nonrespond_action_exact          dev_action_exact over the non-respond spans only (the tool calls)
      dev_nonrespond_action_exact_by_cls   the same per class
      dev_min_class_action_exact           the lowest of dev_action_exact_by_cls (the aggregate hides a weak class)

    Diagnostics (never used to select):
      dev_loss               mean cross-entropy over every supervised token
      dev_loss_respond_text  mean cross-entropy over the excluded respond message text only
      dev_line_exact         share of spans exact over every supervised token (respond lines practically never)
      dev_line_exact_by_cls

    `logits_fn` defaults to supervised_logits; a test passes a stand-in with the same signature.
    """
    import torch.nn.functional as F
    logits_fn = logits_fn or supervised_logits
    was_training = peft_model.training
    peft_model.eval()
    sums = {"all": [0.0, 0], "action": [0.0, 0], "text": [0.0, 0]}            # [loss sum, tokens]
    spans = {}                                                                # (example, span) -> [all ok, action ok]
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    for j in range(0, len(order), micro_batch):
        idxs = order[j:j + micro_batch]
        input_ids, attention, labels, action_labels = collate(examples, idxs, PAD_ID, device)
        with _autocast(device):
            logits, targets, positions = logits_fn(peft_model, input_ids, attention, labels)
        ce = F.cross_entropy(logits.float(), targets, reduction="none").tolist()
        correct = (logits.argmax(dim=-1) == targets).tolist()
        is_action = (action_labels[:, 1:] != -100)[positions].tolist()
        rows, cols = positions.nonzero(as_tuple=True)             # the same row-major order as positions[...] indexing
        for row, col, loss, ok, action in zip(rows.tolist(), (cols + 1).tolist(), ce, correct, is_action):
            example = idxs[row]
            flags = spans.setdefault((example, examples[example].span_of[col]), [True, True])
            flags[0] = flags[0] and ok
            if action:
                flags[1] = flags[1] and ok
            for key in ("all", "action" if action else "text"):
                sums[key][0] += loss
                sums[key][1] += 1
    peft_model.train(was_training)

    def mean(key):
        total, count = sums[key]
        return total / max(1, count)

    def share(index, cls=None, first_only=False, nonrespond_only=False):
        picked = [flags[index] for (example, span), flags in spans.items()
                  if (cls is None or examples[example].cls == cls) and (span == 0 or not first_only)
                  and not (nonrespond_only and examples[example].span_is_respond[span])]
        return sum(picked) / len(picked) if picked else 0.0

    present = [c for c in CLASSES if any(e.cls == c for e in examples)]
    by_cls = {c: share(1, c) for c in present}
    n_respond = sum(examples[example].span_is_respond[span] for (example, span) in spans)
    return {"dev_loss": mean("all"),
            "dev_loss_action": mean("action"),
            "dev_loss_respond_text": mean("text"),
            "dev_action_exact": share(1),
            "dev_action_exact_by_cls": by_cls,
            "dev_line_exact": share(0),
            "dev_line_exact_by_cls": {c: share(0, c) for c in present},
            "dev_first_action_exact": share(1, first_only=True),
            "dev_respond_share": n_respond / max(1, len(spans)),
            "dev_nonrespond_action_exact": share(1, nonrespond_only=True),
            "dev_nonrespond_action_exact_by_cls": {c: share(1, c, nonrespond_only=True) for c in present},
            "dev_min_class_action_exact": min(by_cls.values()) if by_cls else 0.0}


# ── data selection ──

def stratified_sample(keys: list, n: int, rng: random.Random) -> list:
    """Ascending indices of exactly n items that keep every stratum's share of `keys` (largest-remainder rounding)."""
    total = len(keys)
    if n >= total:
        return list(range(total))
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    names = sorted(groups)
    quotas = {name: n * len(groups[name]) // total for name in names}
    by_remainder = sorted(names, key=lambda name: (-(n * len(groups[name]) % total), name))
    for name in by_remainder[:n - sum(quotas.values())]:
        quotas[name] += 1
    chosen = []
    for name in names:
        chosen += rng.sample(groups[name], quotas[name])
    return sorted(chosen)


def epoch_stream(n_items: int, seed: int, take: int):
    """Endless lists of `take` item indices; a fresh shuffle seeded from (seed, epoch) whenever an epoch runs out."""
    epoch, pos = 0, 0
    order = list(range(n_items))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    while True:
        batch = []
        while len(batch) < take:
            if pos == len(order):
                epoch, pos = epoch + 1, 0
                order = list(range(n_items))
                random.Random(seed * 1_000_003 + epoch).shuffle(order)
            batch.append(order[pos])
            pos += 1
        yield batch


def _n_supervised(example) -> int:
    """Supervised rows of the loss: labels from position 1 on (position 0 is never predicted)."""
    return sum(1 for label in example.labels[1:] if label != -100)


# ── files ──

def _rel(path: str) -> str:
    """Repo-relative path with forward slashes; a file outside the repository is reduced to its name."""
    real, root = os.path.realpath(path), os.path.realpath(PROJECT_ROOT)
    if real == root or real.startswith(root + os.sep):
        return os.path.relpath(real, root).replace(os.sep, "/")
    return os.path.basename(real)


def _remove(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record) + "\n")


def _rss_mb() -> int:
    """Resident memory of this process in MB (Linux /proc; 0 elsewhere), for the progress line."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def read_vram_mib(device) -> dict:
    """Peak allocated and reserved CUDA memory of this process plus the card's total, in MiB; zeros off the GPU.

    One small module-level function so the guard can be tested without a GPU (a test replaces it).
    """
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    mib = 1024 * 1024
    return {"allocated": int(torch.cuda.max_memory_allocated(device) // mib),
            "reserved": int(torch.cuda.max_memory_reserved(device) // mib),
            "total": int(torch.cuda.get_device_properties(device).total_memory // mib)}


def _package_versions() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in PACKAGES}


def lora04_gate(manifest: dict) -> tuple:
    """(passed, reason): the LORA04_GATE_RULE applied to a finished run's manifest (dev data only)."""
    check = manifest.get("gen_check")
    if not check:
        return False, "gen_check is missing from the manifest, so the parse rate was never measured"
    rate, n = check["tuned"]["parse_rate"], check["n"]
    passed = rate >= LORA04_PARSE_THRESHOLD
    return passed, (f"gen_check.tuned.parse_rate {rate:.4f} {'>=' if passed else '<'} {LORA04_PARSE_THRESHOLD:.2f} "
                    f"on {n} first-turn dev prompts")


def generation_check(name: str, base_model: str, adapter, device: str, episodes: list, max_new_tokens: int) -> tuple:
    """(parsed, gold): of the first replies to the first-turn prompts of `episodes`, how many parse as an action and
    how many equal the episode's first gold action.

    The replies come from HFLocalProvider, the serving path of the benchmark: with an adapter it loads and merges it
    after every stamp check, so an adapter that cannot be served fails the run here and not in the benchmark.
    """
    provider = HFLocalProvider(name=name, base_model=base_model, adapter=adapter, max_new_tokens=max_new_tokens,
                               device=device)
    parsed = gold = 0
    try:
        for ep in episodes:
            reply = provider.generate(build_messages(ep["user"], ep["tools"]), max_tokens=max_new_tokens)
            if reply.error:
                if "OutOfMemoryError" in reply.error:       # the provider returns CUDA out of memory as a value
                    raise torch.cuda.OutOfMemoryError(reply.error)
                raise RuntimeError(f"generation check could not produce a reply for {ep['id']}: {reply.error}")
            try:
                action = parse_action(reply.text)
            except ActionParseError:
                continue
            parsed += 1
            gold += compact_action(action) == compact_action(ep["turns"][0][0])
    finally:
        del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parsed, gold


def _write_curve(path: str, history: list, trainable: int, best_step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax.plot(steps, [h["dev_loss"] for h in history], marker="o", label="dev loss (all supervised tokens)",
            linestyle="--", alpha=0.6)
    ax.plot(steps, [h[SELECTION_METRIC] for h in history], marker="o", label="dev loss, action (selection)")
    ax.axvline(best_step, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (nats per token)")
    ax.set_title(f"Qwen2.5-0.5B LoRA, {trainable / 1e6:.2f}M trainable, best step {best_step}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _note(text: str) -> str:
    if len(text) > MAX_NOTE_CHARS or not text.isascii() or "\n" in text:
        raise argparse.ArgumentTypeError(f"a run note is one line of ASCII, at most {MAX_NOTE_CHARS} characters")
    return text


def _flush_print(text: str) -> None:
    print(text, flush=True)


def _dev_pick(dev_eps: list, n: int, seed: int) -> list:
    """The dev episodes of a final check: all of them for n = 0, else a stratified (class, split) sample fixed by the seed."""
    if n <= 0 or n >= len(dev_eps):
        return list(dev_eps)
    return [dev_eps[i] for i in stratified_sample([(e["cls"], e["split"]) for e in dev_eps], n, random.Random(seed))]


def evaluation_overlap(dev_eps: list, selection_idx: list, final_eps: list, check_eps: list) -> dict:
    """How the three dev samples overlap (review WR-04): counts of the selection subset inside the final dev pass and of
    the generation-check prompts inside each. Stated in the manifest, not corrected: all three are dev data."""
    selection = {dev_eps[i]["id"] for i in selection_idx}
    final, check = {e["id"] for e in final_eps}, {e["id"] for e in check_eps}
    return {"selection_subset_n": len(selection), "final_dev_n": len(final), "final_dev_in_selection": len(final & selection),
            "gen_check_n": len(check), "gen_check_in_selection": len(check & selection),
            "gen_check_in_final_dev": len(check & final), "note": OVERLAP_NOTE}


def remediation_needed(manifest: dict) -> tuple:
    """(needed, reason): the REMEDIATION_RULE applied mechanically to a finished run's manifest (dev metrics only)."""
    value = manifest["final_dev_full"]["dev_action_exact"]
    steps, every = manifest["steps"], manifest["config"]["train"]["eval_every"]
    below = value < REMEDIATION_THRESHOLD
    reason = (f"final_dev_full.dev_action_exact {value:.4f} {'<' if below else '>='} {REMEDIATION_THRESHOLD:.2f}; "
              f"stop_reason {manifest['stop_reason']}")
    if manifest["stop_reason"] != "max_steps":
        return False, reason + " (converged, no retry; a value below the threshold is a finding)"
    improving = manifest["best_step"] >= steps - 2 * every
    reason += f", best_step {manifest['best_step']} {'>=' if improving else '<'} steps - 2 x eval_every = {steps - 2 * every}"
    return below and improving, reason


def _identity_diff(saved: dict, current: dict) -> list:
    """Dotted names (one level deep) of the identity fields that differ between a resume file and this run."""
    differs = []
    for key in sorted(set(saved) | set(current)):
        a, b = saved.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            differs += [f"{key}.{sub}" for sub in sorted(set(a) | set(b)) if a.get(sub) != b.get(sub)]
        elif a != b:
            differs.append(key)
    return differs


def _save_resume(path: str, state: dict) -> None:
    """Written next to the final name and moved into place, so a kill never leaves half a resume file."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _discard_staging(out_dir: str, keep_log: bool = False) -> None:
    """Remove the staging directory, the resume file and (unless keep_log) the staged log of an unfinished run.

    Never a promoted file (adapter, manifest, curve, train_log.jsonl) and never the launcher's own files: a previous
    valid adapter survives every failure of a later run.
    """
    names = [RESUME_FILE, RESUME_FILE + ".tmp"] + ([] if keep_log else [STAGED_LOG])
    _remove(*(os.path.join(out_dir, name) for name in names))
    shutil.rmtree(os.path.join(out_dir, IN_RUN_ADAPTER), ignore_errors=True)


def _promote_run(stage_dir: str, staged_log: str, out_dir: str) -> None:
    """Move a finished run's files from staging into out_dir over the previous run's (os.replace, manifest last).

    Every staged file is checked first, so a missing one stops the promotion before anything is replaced. A crash in
    the middle leaves new weights next to the old manifest, which the hash check of verify_adapter and of
    `checks artifacts` refuses: the directory fails closed instead of serving a mixed adapter.
    """
    sources = {"adapter_model.safetensors": os.path.join(stage_dir, "adapter_model.safetensors"),
               "adapter_config.json": os.path.join(stage_dir, "adapter_config.json"),
               "loss_curve.png": os.path.join(stage_dir, "loss_curve.png"),
               "train_log.jsonl": staged_log,
               "manifest.json": os.path.join(stage_dir, "manifest.json")}
    for name in PROMOTE_ORDER:
        if name != "loss_curve.png" and not os.path.isfile(sources[name]):
            raise RuntimeError(f"cannot promote the run: the staged {name} is missing")
    for name in PROMOTE_ORDER:
        destination = os.path.join(out_dir, name)
        if os.path.isfile(sources[name]):
            os.replace(sources[name], destination)
        else:
            _remove(destination)          # the curve failed to draw: the previous run's curve must not stay next to this adapter
    _remove(os.path.join(out_dir, "README.md"))


# ── training ──

def train(train_path: str, dev_path: str, out_dir: str, base_model: str = BASE_MODEL_ID, cfg: TrainConfig = None,
          device: str = "auto", log=print, run_notes: list = None, resume: bool = False, fresh: bool = False) -> dict:
    """Train a LoRA adapter on the episodes in `train_path`, write adapter plus manifest into `out_dir`, return the manifest.

    LeakageError is raised before anything else if either file holds benchmark content (D-02). A fresh run discards the
    staging leftovers of an earlier unfinished run (never the launcher's train_progress.log, train.done or
    gpu_preflight.log) but, when it finds the resume file of a killed run, refuses unless `fresh` is true; a directory
    that holds another tool's manifest is refused. After the last evaluation the kept adapter is evaluated once on the
    whole dev file (final_dev_full), the manifest is written into the staging directory, and the staged adapter is
    served through HFLocalProvider next to the base model (gen_check). Only then are the files promoted into
    `out_dir`, replacing a previous run's; until then that run's adapter is untouched (review WR-03).

    Restart-proof (D-06): after every evaluation of a run that continues, RESUME_FILE is written (trainable tensors,
    optimizer, RNG states, step, best, stale count, history, tokens, elapsed time, peak VRAM, identity) next to the
    in-run adapter. A killed process (WSL restart, SIGTERM, Ctrl-C) leaves both; `resume=True` continues at the next
    step and refuses, naming the field, when the data hashes, base model, training config, render fingerprint or
    package versions differ.

    Fail closed: a non-finite training or dev loss raises NonFiniteLossError, peak reserved memory above
    VRAM_LIMIT_FRACTION of the card or a CUDA out-of-memory error raises VramBudgetError, each after a FAILED line.
    A failure while training removes the staging directory and the resume file; a failure after the last step keeps
    both (and says so in a FAILED post_training line) so the trained adapter is not lost to a transient problem of
    the check. Either way no file of out_dir is replaced, so a failed run leaves nothing loadable that was not there
    before.
    """
    cfg = cfg or TrainConfig()
    if min(cfg.max_steps, cfg.micro_batch, cfg.accum, cfg.eval_every, cfg.log_every) < 1:
        raise ValueError("max_steps, micro_batch, accum, eval_every and log_every must be at least 1")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    on_cuda = torch.device(device).type == "cuda"
    started = time.time()

    train_eps, dev_eps = read_episodes(train_path), read_episodes(dev_path)
    if not train_eps or not dev_eps:
        raise ValueError("the train and dev files must each hold at least one episode")
    assert_clean(train_eps)       # a leaking file stops here, before the first step
    assert_clean(dev_eps)

    per_step = cfg.micro_batch * cfg.accum
    n_used = cfg.n_train_used or min(len(train_eps), cfg.max_steps * per_step)
    used = stratified_sample([e["cls"] for e in train_eps], n_used, random.Random(cfg.seed))
    class_counts = {c: sum(train_eps[i]["cls"] == c for i in used) for c in CLASSES}

    os.makedirs(out_dir, exist_ok=True)
    refuse_foreign_manifest(out_dir, "adapter_sha256", "Qwen LoRA training")     # before anything is deleted
    best_dir = os.path.join(out_dir, IN_RUN_ADAPTER)
    resume_path = os.path.join(out_dir, RESUME_FILE)
    log_path = os.path.join(out_dir, STAGED_LOG)
    state = None
    if resume:
        for missing in (RESUME_FILE, IN_RUN_ADAPTER):
            if not os.path.exists(os.path.join(out_dir, missing)):
                raise ValueError(f"nothing to resume in {out_dir}: {missing} is missing "
                                 f"({RESUME_FILE} and {IN_RUN_ADAPTER} must both exist)")
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
    else:
        if os.path.exists(resume_path) and not fresh:
            raise ValueError(f"{out_dir} holds the resume file of a killed run; continue it with --resume, or discard "
                             f"its state with --fresh")
        _discard_staging(out_dir)        # only the leftovers of an unfinished run: a previous valid adapter stays

    torch.manual_seed(cfg.seed)
    peft_model, tokenizer, revision = build_lora_model(base_model, cfg, device)
    assert tokenizer.convert_tokens_to_ids(chat.ENDOFTEXT) == PAD_ID, "PAD_ID is not this tokenizer's <|endoftext|>"
    identity = {"train_sha256": file_sha256(train_path), "dev_sha256": file_sha256(dev_path),
                "base_model": {"id": _model_id(base_model), "revision": revision}, "train": cfg.to_dict(),
                "render": {"fingerprint": chat.render_fingerprint(tokenizer)}, "packages": _package_versions()}
    if state is not None:
        differs = _identity_diff(state["identity"], identity)
        if differs:
            raise ValueError(f"the resume file is from a different run: {', '.join(differs)} differ")

    examples = [build_example(tokenizer, train_eps[i]) for i in used]
    dev_used = stratified_sample([(e["cls"], e["split"]) for e in dev_eps], min(cfg.dev_eval_n, len(dev_eps)),
                                 random.Random(cfg.seed + 1))
    dev_examples = [build_example(tokenizer, dev_eps[i]) for i in dev_used]
    named_trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    trainable = [p for _, p in named_trainable]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in peft_model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    log(f"device {device} trainable {n_trainable} total {n_total}")

    stream = epoch_stream(len(examples), cfg.seed, per_step)
    since_log, since_eval, tokens_seen = [], [], 0
    history, best, stale_evals, stop_reason = [], {SELECTION_METRIC: float("inf"), "step": 0}, 0, None
    peak, resumed_from, done_steps, current = {"allocated": 0, "reserved": 0}, [], 0, 0
    training_done = False
    try:
        if state is not None:
            with torch.no_grad():
                for name, param in named_trainable:
                    param.copy_(state["tensors"][name])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            if on_cuda and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"])
            history, best, stale_evals = state["history"], state["best"], state["stale_evals"]
            done_steps, tokens_seen, peak = state["step"], state["tokens_seen"], state["peak"]
            resumed_from = state["resumed_from"] + [state["step"]]
            started = time.time() - state["elapsed_s"]
            for _ in range(done_steps):                  # the epoch order is a function of the step: skip what was trained
                next(stream)
            _remove(log_path)                            # rewritten from the saved history: no duplicate or stray lines
            for record in history:
                _append_jsonl(log_path, record)
            log(f"RESUMED from step {done_steps}")
        mark = (time.time(), tokens_seen)
        peft_model.train()
        step = done_steps

        def guard_vram() -> dict:
            """Update the running peak; stop the run above VRAM_LIMIT_FRACTION of the card."""
            nonlocal peak
            vram = read_vram_mib(device)
            peak = {k: max(peak[k], vram[k]) for k in peak}
            if vram["total"] and vram["reserved"] > VRAM_LIMIT_FRACTION * vram["total"]:
                log(f"FAILED vram_budget reserved {vram['reserved']} total {vram['total']}")
                raise VramBudgetError(f"reserved {vram['reserved']} MiB of {vram['total']} MiB")
            return vram

        for s in range(done_steps, cfg.max_steps):
            current = s + 1
            lr = lr_at(s, cfg.max_steps, cfg.lr, cfg.min_lr, cfg.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = sorted(next(stream), key=lambda i: len(examples[i].input_ids))
            window_n = sum(_n_supervised(examples[i]) for i in batch)
            window_loss = 0.0
            for j in range(0, len(batch), cfg.micro_batch):
                idxs = batch[j:j + cfg.micro_batch]
                input_ids, attention, labels, _ = collate(examples, idxs, PAD_ID, device)
                with _autocast(device):
                    loss_sum, _ = supervised_loss(peft_model, input_ids, attention, labels)     # looked up on the module: tests script it
                if not bool(torch.isfinite(loss_sum)):
                    log(f"FAILED non_finite_loss step {current}")
                    _append_jsonl(log_path, {"step": current, "event": "non_finite_loss"})
                    raise NonFiniteLossError(f"training loss is not finite at step {current}")
                loss = loss_sum / window_n          # token-weighted over the whole accumulation window
                loss.backward()
                window_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tokens_seen += sum(len(examples[i].input_ids) for i in batch)
            since_log.append(window_loss)
            since_eval.append(window_loss)
            step = current
            if step % cfg.log_every == 0 or step == cfg.max_steps:
                now = time.time()
                rate = (tokens_seen - mark[1]) / max(1e-9, now - mark[0])
                mark = (now, tokens_seen)
                vram = guard_vram()
                log(f"step {step}/{cfg.max_steps} loss {sum(since_log) / len(since_log):.4f} lr {lr:.2e} "
                    f"tok/s {rate:.0f} elapsed {now - started:.0f}s vram {vram['allocated']}/{vram['reserved']}MiB "
                    f"rss {_rss_mb()}MB")
                since_log = []

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                result = evaluate(peft_model, dev_examples, cfg.micro_batch, device)      # looked up on the module: tests script it
                if not all(math.isfinite(result[k]) for k in ("dev_loss", SELECTION_METRIC)):
                    # nan < best is False, so a NaN would silently never be saved
                    log(f"FAILED non_finite_dev_loss step {step}")
                    _append_jsonl(log_path, {"step": step, "event": "non_finite_dev_loss"})
                    raise NonFiniteLossError(f"dev loss is not finite at step {step}")
                guard_vram()
                record = {"step": step, "train_loss": sum(since_eval) / len(since_eval), "lr": lr,
                          "tokens_seen": tokens_seen, "elapsed_s": round(time.time() - started, 1), **result}
                since_eval = []
                history.append(record)
                _append_jsonl(log_path, record)
                if result[SELECTION_METRIC] < best[SELECTION_METRIC]:           # never the plain or respond-text loss (CR-01)
                    best, stale_evals = {SELECTION_METRIC: result[SELECTION_METRIC], "step": step}, 0
                    peft_model.save_pretrained(best_dir, save_embedding_layers=False)
                else:
                    stale_evals += 1
                log(f"eval step {step} dev_loss {result['dev_loss']:.4f} dev_loss_action "
                    f"{result['dev_loss_action']:.4f} dev_action_exact {result['dev_action_exact']:.4f} "
                    f"best {best[SELECTION_METRIC]:.4f}")
                if stale_evals >= cfg.patience:
                    stop_reason = "early_stop"
                if stop_reason is None and step < cfg.max_steps:
                    _save_resume(resume_path, {
                        "identity": identity,
                        "tensors": {name: param.detach().cpu().clone() for name, param in named_trainable},
                        "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if on_cuda else None, "step": step, "best": best,
                        "stale_evals": stale_evals, "history": history, "tokens_seen": tokens_seen,
                        "elapsed_s": time.time() - started, "peak": peak, "resumed_from": resumed_from})
                if stop_reason:
                    break
        stop_reason = stop_reason or "max_steps"
        training_done = True

        # The kept adapter is the best evaluation, not the last step: load it back (so anything run after this sees
        # the kept weights). Its files stay in the staging directory until the generation check has passed.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(peft_model, load_file(os.path.join(best_dir, "adapter_model.safetensors")))
        adapter_sha256 = sha256_file(os.path.join(best_dir, "adapter_model.safetensors"))
        config_sha256 = adapter_config_sha256(os.path.join(best_dir, "adapter_config.json"))      # review WR-05

        # One evaluation of the kept adapter on the whole dev file (or a stratified final_dev_n of it).
        current = step
        final_eps = _dev_pick(dev_eps, cfg.final_dev_n, cfg.seed + 2)
        check_eps = _dev_pick(dev_eps, cfg.gen_check_n, cfg.seed + 3)
        final_examples = [build_example(tokenizer, ep) for ep in final_eps]
        final = evaluate(peft_model, final_examples, cfg.micro_batch, device)
        log(f"final_dev_full n {len(final_examples)} dev_action_exact {final['dev_action_exact']:.4f} "
            f"dev_loss_action {final['dev_loss_action']:.4f}")
        vram = guard_vram()

        kept = next(h for h in history if h["step"] == best["step"])
        last = history[-1]
        wall = round(time.time() - started, 1)
        model_file = os.path.join(resolve_base_model(base_model)[0], "model.safetensors")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_commit": git_commit(), "code_dirty": git_dirty(),
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": "cuda" if on_cuda else "cpu",
            "packages": identity["packages"],
            "seed": cfg.seed,
            "base_model": {"id": _model_id(base_model), "revision": revision,
                           "model_safetensors_bytes": os.path.getsize(model_file) if os.path.isfile(model_file) else 0},
            "tokenizer": {"chat_template_sha256": chat.chat_template_sha256(tokenizer),
                          "eos_id": tokenizer.convert_tokens_to_ids(chat.IM_END), "pad_id": PAD_ID},
            "render": {"fingerprint": identity["render"]["fingerprint"], "version": chat.RENDER_VERSION,
                       "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                       "observation_template": OBSERVATION_TEMPLATE, "max_len": data.MAX_LEN},
            "dataset": {"train_path": _rel(train_path), "train_sha256": identity["train_sha256"],
                        "dev_path": _rel(dev_path), "dev_sha256": identity["dev_sha256"],
                        "n_train_available": len(train_eps), "n_train_used": len(examples), "n_dev": len(dev_eps),
                        "dev_eval_n": len(dev_examples), "class_counts": class_counts,
                        "held_out_sha256": held_out_sha256()},
            "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                     "target_modules": sorted(LORA_TARGETS), "trainable_params": n_trainable, "total_params": n_total,
                     "trainable_fraction": round(n_trainable / n_total, 6)},
            "config": {"train": cfg.to_dict()},
            "steps": step, "best_step": best["step"], "tokens_seen": tokens_seen,
            "peak_vram_mib": {**peak, "total": vram["total"]},
            "selection": {"metric": SELECTION_METRIC, "direction": "min", "description": SELECTION_DESCRIPTION,
                          "remediation_rule": REMEDIATION_RULE, "lora04_gate": LORA04_GATE_RULE},
            # best = the evaluation of the adapter that is kept; last_eval = the final evaluation, possibly a later step
            "best": {"step": best["step"], "train_loss": kept["train_loss"], **{k: kept[k] for k in EVAL_KEYS}},
            "last_eval": {"step": last["step"], "train_loss": last["train_loss"], **{k: last[k] for k in EVAL_KEYS}},
            "final_dev_full": {"n": len(final_examples), **{k: final[k] for k in EVAL_KEYS}},
            "gen_check": None,
            "evaluation_overlap": evaluation_overlap(dev_eps, dev_used, final_eps, check_eps),
            "adapter_sha256": adapter_sha256,
            "adapter_config_sha256": config_sha256,
            "adapter_identity_sha256": adapter_identity_sha256(adapter_sha256, config_sha256),
            "wall_seconds": wall,
            "stop_reason": stop_reason,
            "resumed_from_steps": resumed_from,
            "run_notes": list(run_notes or []) + [D01_NOTE] + [f"resumed from step {s}" for s in resumed_from],
        }
        assert tuple(manifest) == MANIFEST_KEYS, "manifest fields drifted from MANIFEST_KEYS"
        manifest_path = os.path.join(best_dir, "manifest.json")        # staged: the directory is what HFLocalProvider verifies
        write_json_atomic(manifest_path, manifest)

        # The generation check runs through the serving path, so the training model is freed first (6 GB card).
        del peft_model, optimizer, trainable, named_trainable, examples, dev_examples, final_examples, state
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()
        tuned = generation_check("qwen-lora-gen-check", base_model, best_dir, device, check_eps, cfg.gen_max_new_tokens)
        base = generation_check("qwen-base-gen-check", base_model, None, device, check_eps, cfg.gen_max_new_tokens)
        n = len(check_eps)
        manifest["gen_check"] = {"n": n,
                                 "tuned": {"parse_rate": tuned[0] / n, "gold_first_action_rate": tuned[1] / n},
                                 "base": {"parse_rate": base[0] / n, "gold_first_action_rate": base[1] / n}}
        write_json_atomic(manifest_path, manifest)
        log(f"gen_check tuned parse {tuned[0]}/{n} gold {tuned[1]}/{n} base parse {base[0]}/{n} gold {base[1]}/{n}")
        try:
            _write_curve(os.path.join(best_dir, "loss_curve.png"), history, n_trainable, best["step"])
        except Exception as e:                   # a plotting problem must not cost a finished run its manifest
            log(f"WARNING loss curve not written: {type(e).__name__}: {e}")
        _promote_run(best_dir, log_path, out_dir)
        _remove(resume_path, resume_path + ".tmp")       # kept until the run was promoted
        shutil.rmtree(best_dir, ignore_errors=True)
        log(f"DONE best_step {best['step']} best_{SELECTION_METRIC} {best[SELECTION_METRIC]:.4f} wall {wall:.0f}s "
            f"stop {stop_reason}")
        return manifest
    except KeyboardInterrupt:
        raise                         # Ctrl-C leaves the resume file and the in-run adapter: --resume continues
    except torch.cuda.OutOfMemoryError as e:
        log(f"FAILED cuda_oom step {current}")
        _after_failure(out_dir, training_done, log)
        raise VramBudgetError(f"CUDA out of memory at step {current}") from e
    except BaseException:
        _after_failure(out_dir, training_done, log)
        raise


def _after_failure(out_dir: str, training_done: bool, log) -> None:
    """Failure policy: a training failure discards the staging; a failure after the last step keeps the trained adapter."""
    if training_done:
        log(f"FAILED post_training: the trained adapter is kept staged in {IN_RUN_ADAPTER}/ (not promoted); "
            f"a previous adapter in the output directory is untouched")
    else:
        _discard_staging(out_dir, keep_log=True)       # the staged log stays: it is the diagnostic of the failed run


# ── command line: python -m src.qwen_lora.train ──

def main(argv=None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(prog="python -m src.qwen_lora.train",
                                     description="LoRA fine-tune Qwen2.5-0.5B-Instruct on the Phase 2 agent episodes.")
    parser.add_argument("--train", default="data/local/train-seed0-n40000.jsonl")
    parser.add_argument("--dev", default="data/local/dev-seed0-n2000.jsonl")
    parser.add_argument("--out-dir", default="models/qwen-agentlora")
    parser.add_argument("--base-model", default=BASE_MODEL_ID,
                        help="a Hugging Face id from the local cache, or a directory inside the repository")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--micro-batch", type=int, default=defaults.micro_batch)
    parser.add_argument("--accum", type=int, default=defaults.accum)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--n-train-used", type=int, default=defaults.n_train_used)
    parser.add_argument("--dev-eval-n", type=int, default=defaults.dev_eval_n)
    parser.add_argument("--final-dev-n", type=int, default=defaults.final_dev_n,
                        help="dev episodes of the final evaluation; 0 = the whole dev file")
    parser.add_argument("--gen-check-n", type=int, default=defaults.gen_check_n)
    parser.add_argument("--gen-max-new-tokens", type=int, default=defaults.gen_max_new_tokens)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true",
                        help=f"continue a killed run from {RESUME_FILE} in --out-dir (refused if data, base model, "
                             f"config, render or package versions differ)")
    parser.add_argument("--fresh", action="store_true",
                        help=f"start over although {RESUME_FILE} of a killed run exists in --out-dir (its resume state is "
                             f"discarded; a previous finished adapter is kept until the new run is promoted)")
    parser.add_argument("--run-note", action="append", type=_note, default=[], metavar="TEXT",
                        help="why this run was started or relaunched; repeatable, stored in the manifest in order")
    args = parser.parse_args(argv)

    paths = {}
    for name in ("train", "dev", "out_dir"):
        try:
            paths[name] = resolve_under_root(getattr(args, name), ALLOWED_ROOT)
        except ValueError as e:
            parser.error(f"--{name.replace('_', '-')}: {e}")
    for name in ("train", "dev"):
        if not os.path.isfile(paths[name]):
            parser.error(f"--{name}: {getattr(args, name)} does not exist")
    base_model = args.base_model
    if os.path.isdir(base_model):
        try:
            base_model = resolve_under_root(base_model, ALLOWED_ROOT)
        except ValueError as e:
            parser.error(f"--base-model: {e}")
    if min(args.max_steps, args.micro_batch, args.accum, args.eval_every, args.log_every) < 1:
        parser.error("--max-steps, --micro-batch, --accum, --eval-every and --log-every must be at least 1")

    cfg = TrainConfig(seed=args.seed, max_steps=args.max_steps, micro_batch=args.micro_batch, accum=args.accum,
                      lr=args.lr, min_lr=min(defaults.min_lr, args.lr), warmup=args.warmup,
                      eval_every=args.eval_every, log_every=args.log_every, patience=args.patience,
                      n_train_used=args.n_train_used, dev_eval_n=args.dev_eval_n, final_dev_n=args.final_dev_n,
                      gen_check_n=args.gen_check_n, gen_max_new_tokens=args.gen_max_new_tokens)
    try:
        refuse_foreign_manifest(paths["out_dir"], "adapter_sha256", "Qwen LoRA training")
    except ValueError as e:
        parser.error(f"--out-dir: {e}")
    if args.resume and args.fresh:
        parser.error("--resume and --fresh exclude each other")
    if not args.resume and not args.fresh and os.path.exists(os.path.join(paths["out_dir"], RESUME_FILE)):
        parser.error(f"--out-dir holds {RESUME_FILE} of a killed run: continue it with --resume, or discard its "
                     f"state with --fresh")
    os.makedirs(paths["out_dir"], exist_ok=True)
    try:
        train(paths["train"], paths["dev"], paths["out_dir"], base_model, cfg, args.device, _flush_print,
              run_notes=args.run_note, resume=args.resume, fresh=args.fresh)
    except NonFiniteLossError:
        return EXIT_NON_FINITE        # the FAILED line was already printed by train()
    except VramBudgetError:
        return EXIT_VRAM
    return 0


if __name__ == "__main__":
    sys.exit(main())
