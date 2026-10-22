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

The adapter is saved with save_pretrained(save_embedding_layers=False) next to a manifest.json holding
the fields src.llm.hf_local verifies before it serves the adapter: adapter_sha256, base_model {id,
revision} and render {fingerprint}. The command line, the full manifest, the generation check, resume and
the failure paths are completed by the next plan.
"""

import contextlib
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Set before torch is imported: it cut the reserved CUDA memory from 4.9 GiB to 2.5 GiB at the same batch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from src.benchmark.injecagent import PROJECT_ROOT
from src.llm.hf_local import resolve_base_model, sha256_file
from src.qwen_lora import chat
from src.qwen_lora import data
from src.qwen_lora.data import build_example, collate
from src.scratch_llm.dataset import assert_clean, file_sha256, read_episodes
from src.scratch_llm.provenance import refuse_foreign_manifest, write_json_atomic
from src.scratch_llm.train import lr_at

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CLASSES = ("benign", "dh", "ds")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MANIFEST_SCHEMA = 1
PAD_ID = 151643                  # <|endoftext|>: only fills attention-masked positions, never decides a mask
SELECTION_METRIC = "dev_loss_action"
IN_RUN_ADAPTER = "adapter.best.tmp"      # the best adapter so far; only a finished run moves it into the output directory
# every metric evaluate() returns, in the order the manifest lists them
EVAL_KEYS = ("dev_loss", "dev_loss_action", "dev_loss_respond_text", "dev_action_exact", "dev_action_exact_by_cls",
             "dev_line_exact", "dev_line_exact_by_cls", "dev_first_action_exact")
STALE_FILES =("adapter_config.json", "adapter_model.safetensors", "README.md", "manifest.json", "train_log.jsonl",
               "loss_curve.png")       # never train_progress.log, train.done or gpu_preflight.log: the launcher owns them


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
    # (PEFT resolves "all-linear" to the full module names, one per layer and projection, so compare the leaves.)
    resolved = set(peft_model.peft_config["default"].target_modules)
    leaves = {name.rsplit(".", 1)[-1] for name in resolved}
    assert leaves == set(LORA_TARGETS) and len(resolved) == len(LORA_TARGETS) * model.config.num_hidden_layers, \
        f"all-linear resolved to {sorted(resolved)}, not the {list(LORA_TARGETS)} of every layer"
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

    def share(index, cls=None, first_only=False):
        picked = [flags[index] for (example, span), flags in spans.items()
                  if (cls is None or examples[example].cls == cls) and (span == 0 or not first_only)]
        return sum(picked) / len(picked) if picked else 0.0

    present = [c for c in CLASSES if any(e.cls == c for e in examples)]
    return {"dev_loss": mean("all"),
            "dev_loss_action": mean("action"),
            "dev_loss_respond_text": mean("text"),
            "dev_action_exact": share(1),
            "dev_action_exact_by_cls": {c: share(1, c) for c in present},
            "dev_line_exact": share(0),
            "dev_line_exact_by_cls": {c: share(0, c) for c in present},
            "dev_first_action_exact": share(1, first_only=True)}


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


# ── training ──

def train(train_path: str, dev_path: str, out_dir: str, base_model: str = BASE_MODEL_ID, cfg: TrainConfig = None,
          device: str = "auto", log=print, run_notes: list = None) -> dict:
    """Train a LoRA adapter on the episodes in `train_path`, write adapter plus manifest into `out_dir`, return the manifest.

    LeakageError is raised before anything else if either file holds benchmark content (D-02). Stale files
    of an earlier run in `out_dir` are removed first; a directory that holds another tool's manifest is refused.
    """
    cfg = cfg or TrainConfig()
    if min(cfg.max_steps, cfg.micro_batch, cfg.accum) < 1:
        raise ValueError("max_steps, micro_batch and accum must be at least 1")
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

    os.makedirs(out_dir, exist_ok=True)
    refuse_foreign_manifest(out_dir, "adapter_sha256", "Qwen LoRA training")     # before anything is deleted
    _remove(*(os.path.join(out_dir, name) for name in STALE_FILES))
    best_dir = os.path.join(out_dir, IN_RUN_ADAPTER)
    shutil.rmtree(best_dir, ignore_errors=True)

    torch.manual_seed(cfg.seed)
    peft_model, tokenizer, revision = build_lora_model(base_model, cfg, device)
    assert tokenizer.convert_tokens_to_ids(chat.ENDOFTEXT) == PAD_ID, "PAD_ID is not this tokenizer's <|endoftext|>"
    examples = [build_example(tokenizer, train_eps[i]) for i in used]
    dev_used = stratified_sample([(e["cls"], e["split"]) for e in dev_eps], min(cfg.dev_eval_n, len(dev_eps)),
                                 random.Random(cfg.seed + 1))
    dev_examples = [build_example(tokenizer, dev_eps[i]) for i in dev_used]
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    log(f"device {device} trainable_params {sum(p.numel() for p in trainable)} train_episodes {len(examples)} "
        f"dev_episodes {len(dev_examples)}")

    stream = epoch_stream(len(examples), cfg.seed, per_step)
    peft_model.train()
    since_log, since_eval, tokens_seen, mark = [], [], 0, (time.time(), 0)
    history, best, stale_evals, stop_reason = [], {SELECTION_METRIC: float("inf"), "step": 0}, 0, None
    log_path = os.path.join(out_dir, "train_log.jsonl")
    for s in range(cfg.max_steps):
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
                loss_sum, _ = supervised_loss(peft_model, input_ids, attention, labels)
            loss = loss_sum / window_n          # token-weighted over the whole accumulation window
            loss.backward()
            window_loss += float(loss.item())
        if not math.isfinite(window_loss):
            raise RuntimeError(f"training loss is not finite at step {s + 1}")
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += sum(len(examples[i].input_ids) for i in batch)
        since_log.append(window_loss)
        since_eval.append(window_loss)
        step = s + 1
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            now = time.time()
            rate = (tokens_seen - mark[1]) / max(1e-9, now - mark[0])
            mark = (now, tokens_seen)
            log(f"step {step}/{cfg.max_steps} loss {sum(since_log) / len(since_log):.4f} lr {lr:.2e} "
                f"tok/s {rate:.0f} elapsed {now - started:.0f}s")
            since_log = []

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            result = evaluate(peft_model, dev_examples, cfg.micro_batch, device)      # looked up on the module: tests script it
            for metric in ("dev_loss", SELECTION_METRIC):
                if not math.isfinite(result[metric]):
                    # nan < best is False, so a NaN would silently never be saved
                    raise RuntimeError(f"{metric} is not finite at step {step}")
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
            log(f"eval step {step} dev_loss {result['dev_loss']:.4f} dev_loss_action {result['dev_loss_action']:.4f} "
                f"dev_action_exact {result['dev_action_exact']:.4f} best {best[SELECTION_METRIC]:.4f}")
            if stale_evals >= cfg.patience:
                stop_reason = "early_stop"
                break
    stop_reason = stop_reason or "max_steps"

    # The kept adapter is the best evaluation, not the last step: load it back (so anything run after this sees
    # the kept weights) and move its files into place.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(peft_model, load_file(os.path.join(best_dir, "adapter_model.safetensors")))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        os.replace(os.path.join(best_dir, name), os.path.join(out_dir, name))
    shutil.rmtree(best_dir, ignore_errors=True)
    adapter_sha256 = sha256_file(os.path.join(out_dir, "adapter_model.safetensors"))
    kept = next(h for h in history if h["step"] == best["step"])
    last = history[-1]
    wall = round(time.time() - started, 1)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "torch": torch.__version__, "device": "cuda" if on_cuda else "cpu",
        "seed": cfg.seed,
        "base_model": {"id": _model_id(base_model), "revision": revision},
        "render": {"fingerprint": chat.render_fingerprint(tokenizer), "version": chat.RENDER_VERSION},
        "dataset": {"train_path": _rel(train_path), "train_sha256": file_sha256(train_path),
                    "dev_path": _rel(dev_path), "dev_sha256": file_sha256(dev_path),
                    "n_train_available": len(train_eps), "n_train_used": len(examples)},
        "config": {"train": cfg.to_dict(), "max_len": data.MAX_LEN},
        "steps": step, "best_step": best["step"], "tokens_seen": tokens_seen,
        "selection": {"metric": SELECTION_METRIC, "direction": "min"},
        # best = the evaluation of the adapter that is kept; last_eval = the final evaluation, possibly a later step
        "best": {"step": best["step"], "train_loss": kept["train_loss"], **{k: kept[k] for k in EVAL_KEYS}},
        "last_eval": {"step": last["step"], "train_loss": last["train_loss"], **{k: last[k] for k in EVAL_KEYS}},
        "adapter_sha256": adapter_sha256,
        "wall_seconds": wall,
        "stop_reason": stop_reason,
        "run_notes": list(run_notes or []),
    }
    write_json_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    log(f"DONE best_step {best['step']} best_{SELECTION_METRIC} {best[SELECTION_METRIC]:.4f} wall {wall:.0f}s "
        f"stop {stop_reason}")
    return manifest
