"""
ScratchGPT training - next-character prediction on the synthetic agent episodes
================================================================================
The model reads a compact transcript (format.py) one character at a time and is trained to
predict the next character. That is the whole task: after "A:" it has to write a JSON action,
character by character, ending with a newline.

Only the agent's own action lines carry loss (decision D-06). The T:, U: and O: lines are
context the model reads and never text it writes, so their characters are masked out of the
loss; the supervised characters are the payload of each A: line plus its newline. The A: prefix
itself is supplied by the prompt at inference, so it is not supervised either.

    optimiser   AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices only
    schedule    linear warmup, then cosine decay from the peak to a floor
    precision   bf16 autocast on the GPU (weights stay fp32), plain fp32 on the CPU
    selection   the checkpoint with the lowest loss on the dev split is kept; the benchmark
                is never read here (the leakage gate runs over train and dev before step 0)

Outputs in --out-dir: scratch_gpt.pt (tensors and primitives only, so it loads with
torch.load(weights_only=True)), manifest.json, train_log.jsonl and loss_curve.png.
"""

import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch

from src.benchmark.injecagent import PROJECT_ROOT
from src.scratch_llm.dataset import assert_clean, file_sha256, held_out_sha256, read_episodes, training_text
from src.scratch_llm.format import CONTEXT_CHARS, encode, supervised_mask
from src.scratch_llm.inference import CHECKPOINT_KEYS, current_stamps
from src.scratch_llm.model import GPTConfig, ScratchGPT
from src.scratch_llm.provenance import git_commit, git_dirty, write_json_atomic

MANIFEST_KEYS = ("schema", "created_at", "code_commit", "code_dirty", "python", "torch", "cuda", "device", "seed",
                 "dataset", "held_out_sha256", "format_version", "vocab_sha256", "render_fingerprint",
                 "context_chars", "gen_reserve", "config", "params", "steps", "best_step", "tokens_seen", "final",
                 "checkpoint_sha256", "wall_seconds", "stop_reason", "run_notes")
MANIFEST_SCHEMA = 1


@dataclass
class TrainConfig:
    seed: int = 0
    max_steps: int = 7000
    eval_every: int = 250
    log_every: int = 50
    batch_tokens: int = 32768
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    patience: int = 8
    target_loss: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def lr_at(step: int, total: int, peak: float, floor: float, warmup: int) -> float:
    """Learning rate for a 0-based step: linear warmup to `peak`, then cosine decay to `floor` at the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total - 1 - warmup)
    progress = min(1.0, (step - warmup) / span)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


# ── data ──

def encode_episode(ep: dict) -> tuple:
    """(x, y, m) uint8 tensors: inputs, next-character targets and the loss mask, already shifted by one."""
    text = training_text(ep)
    ids, mask = encode(text), supervised_mask(text)
    return (torch.tensor(ids[:-1], dtype=torch.uint8), torch.tensor(ids[1:], dtype=torch.uint8),
            torch.tensor(mask[1:], dtype=torch.uint8))


def collate(encoded: list, idxs: list, device) -> tuple:
    """Stack the chosen sequences, right-padded with id 0 and mask 0 (padding never gets loss)."""
    longest = max(len(encoded[i][0]) for i in idxs)
    x = torch.zeros(len(idxs), longest, dtype=torch.long)
    y = torch.zeros(len(idxs), longest, dtype=torch.long)
    m = torch.zeros(len(idxs), longest, dtype=torch.long)
    for row, i in enumerate(idxs):
        n = len(encoded[i][0])
        x[row, :n], y[row, :n], m[row, :n] = encoded[i][0].long(), encoded[i][1].long(), encoded[i][2].long()
    return x.to(device), y.to(device), m.to(device)


def simple_batches(encoded: list, batch_tokens: int, seed: int, epoch: int):
    """Shuffled batches of a fixed number of sequences, sized from the longest sequence."""
    longest = max(len(e[0]) for e in encoded)
    size = max(1, batch_tokens // longest)
    order = list(range(len(encoded)))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    for start in range(0, len(order), size):
        yield order[start:start + size]


# ── evaluation ──

@torch.no_grad()
def evaluate(model, dev_encoded: list, batch_tokens: int, device) -> dict:
    """Mask-weighted mean cross-entropy over every supervised dev character."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    longest = max(len(e[0]) for e in dev_encoded)
    size = max(1, batch_tokens // longest)
    for start in range(0, len(dev_encoded), size):
        x, y, m = collate(dev_encoded, list(range(start, min(start + size, len(dev_encoded)))), device)
        _, loss = model(x, y, m)
        n = int(m.sum().item())
        total, count = total + float(loss.item()) * n, count + n
    model.train(was_training)
    return {"dev_loss": total / max(1, count)}


# ── files ──

def _rel(path: str) -> str:
    """Repo-relative path with forward slashes; a file outside the repository is reduced to its name."""
    real, root = os.path.realpath(path), os.path.realpath(PROJECT_ROOT)
    if real == root or real.startswith(root + os.sep):
        return os.path.relpath(real, root).replace(os.sep, "/")
    return os.path.basename(real)


def _save_checkpoint(path: str, model, model_cfg: GPTConfig, meta: dict) -> None:
    ckpt = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": model_cfg.to_dict(), **current_stamps(), "train_meta": meta}
    assert tuple(ckpt) == CHECKPOINT_KEYS, "train.py and inference.py disagree on the checkpoint keys"
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record) + "\n")


def _write_curve(path: str, history: list, params: int, best_step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["step"] for h in history], [h["train_loss"] for h in history], label="train loss")
    ax.plot([h["step"] for h in history], [h["dev_loss"] for h in history], label="dev loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss on A: characters (nats)")
    ax.set_title(f"ScratchGPT {params / 1e6:.2f}M parameters, best step {best_step}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── training ──

def train(train_path: str, dev_path: str, out_dir: str, model_cfg: GPTConfig = None, train_cfg: TrainConfig = None,
          device: str = "auto", log=print, run_notes: list = None) -> dict:
    """Train ScratchGPT on the episodes in `train_path`, select on `dev_path`, write the artifacts, return the manifest."""
    model_cfg = model_cfg or GPTConfig()
    train_cfg = train_cfg or TrainConfig()
    if model_cfg.context != CONTEXT_CHARS:
        raise ValueError(f"model context {model_cfg.context} must equal format.CONTEXT_CHARS {CONTEXT_CHARS}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()

    train_eps, dev_eps = read_episodes(train_path), read_episodes(dev_path)
    assert_clean(train_eps)       # a leaking file stops here, before the first step
    assert_clean(dev_eps)
    train_enc = [encode_episode(e) for e in train_eps]
    dev_enc = [encode_episode(e) for e in dev_eps]

    torch.manual_seed(train_cfg.seed)
    model = ScratchGPT(model_cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2))
    params = model.num_params()
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "scratch_gpt.pt")
    log_path = os.path.join(out_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    history, best = [], {"dev_loss": float("inf"), "step": 0}
    tokens_seen, step, epoch, recent = 0, 0, 0, []
    model.train()
    while step < train_cfg.max_steps:
        for idxs in simple_batches(train_enc, train_cfg.batch_tokens, train_cfg.seed, epoch):
            if step >= train_cfg.max_steps:
                break
            lr = lr_at(step, train_cfg.max_steps, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y, m = collate(train_enc, idxs, device)
            _, loss = model(x, y, m)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            step += 1
            tokens_seen += int(x.numel())
            recent.append(float(loss.item()))
            if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
                result = evaluate(model, dev_enc, train_cfg.batch_tokens, device)
                record = {"step": step, "train_loss": sum(recent) / len(recent), **result, "lr": lr,
                          "tokens_seen": tokens_seen, "elapsed_s": round(time.time() - started, 1)}
                recent = []
                history.append(record)
                _append_jsonl(log_path, record)
                if result["dev_loss"] < best["dev_loss"]:
                    best = {"dev_loss": result["dev_loss"], "step": step}
                    _save_checkpoint(ckpt_path, model, model_cfg,
                                     {"seed": train_cfg.seed, "step": step, "dev_loss": result["dev_loss"],
                                      "tokens_seen": tokens_seen})
        epoch += 1

    _write_curve(os.path.join(out_dir, "loss_curve.png"), history, params, best["step"])
    last = history[-1]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": device, "seed": train_cfg.seed,
        "dataset": {"train_path": _rel(train_path), "train_sha256": file_sha256(train_path),
                    "dev_path": _rel(dev_path), "dev_sha256": file_sha256(dev_path),
                    "n_train": len(train_eps), "n_dev": len(dev_eps),
                    "class_counts": {c: sum(e["cls"] == c for e in train_eps) for c in ("benign", "dh", "ds")},
                    "synthetic_fraction": round(sum(bool(e["meta"]["synthetic"]) for e in train_eps) / len(train_eps), 6)},
        "held_out_sha256": held_out_sha256(),
        **current_stamps(),
        "config": {"model": model_cfg.to_dict(), "train": train_cfg.to_dict()},
        "params": params, "steps": step, "best_step": best["step"], "tokens_seen": tokens_seen,
        "final": {"train_loss": last["train_loss"], "dev_loss": last["dev_loss"]},
        "checkpoint_sha256": file_sha256(ckpt_path),
        "wall_seconds": round(time.time() - started, 1),
        "stop_reason": "max_steps",
        "run_notes": list(run_notes or []),
    }
    assert tuple(manifest) == MANIFEST_KEYS, "manifest fields drifted from MANIFEST_KEYS"
    write_json_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest
