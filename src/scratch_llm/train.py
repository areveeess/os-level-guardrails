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
    batches     sequences of similar length are grouped under a token budget, right-padded;
                padding plus the causal mask means a real position never attends to padding
    selection   the checkpoint with the lowest loss on the dev split is kept; the benchmark
                is never read here (the leakage gate runs over train and dev before step 0)
    safety      a non-finite loss stops the run at once (exit code 3, no final manifest), so a
                diverged run can never be mistaken for a finished one

Outputs in --out-dir: scratch_gpt.pt (tensors and primitives only, so it loads with
torch.load(weights_only=True)), manifest.json, train_log.jsonl and loss_curve.png.

Progress lines on stdout (a background run is polled for these):
    device <cpu|cuda> params <n>
    step <s>/<max> loss <x> lr <y> tok/s <z> elapsed <t>s
    eval step <s> dev_loss <x> dev_line_exact <y> best <z>
    DONE best_step <s> best_dev_loss <x> wall <t>s stop <max_steps|early_stop|target_loss>
    FAILED non_finite_loss step <s>          (exit code 3)

    python -m src.scratch_llm.train [--train FILE] [--dev FILE] [--out-dir DIR] ...
"""

import argparse
import contextlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch

from src.benchmark.injecagent import PROJECT_ROOT
from src.scratch_llm.dataset import assert_clean, file_sha256, held_out_sha256, read_episodes, training_text
from src.scratch_llm.format import CONTEXT_CHARS, encode, supervised_mask
from src.scratch_llm.inference import CHECKPOINT_KEYS, current_stamps
from src.scratch_llm.model import GPTConfig, ScratchGPT
from src.scratch_llm.provenance import git_commit, git_dirty, resolve_under_root, write_json_atomic

ALLOWED_ROOT = PROJECT_ROOT      # --train, --dev and --out-dir must stay inside this directory (tests point it at a tmp dir)
EXIT_NON_FINITE = 3
MAX_NOTE_CHARS = 200

MANIFEST_KEYS = ("schema", "created_at", "code_commit", "code_dirty", "python", "torch", "cuda", "device", "seed",
                 "dataset", "held_out_sha256", "format_version", "vocab_sha256", "render_fingerprint",
                 "context_chars", "gen_reserve", "config", "params", "steps", "best_step", "tokens_seen", "final",
                 "checkpoint_sha256", "wall_seconds", "stop_reason", "run_notes")
MANIFEST_SCHEMA = 1
CLASSES = ("benign", "dh", "ds")


class NonFiniteLossError(RuntimeError):
    """The training loss became NaN or infinite; the run was stopped and left without a final manifest."""


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
    target_loss: float = 0.0     # > 0: stop once the mean train loss of the last log interval is below it

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
    """(x, y, m, cls): inputs, next-character targets and loss mask as uint8 tensors, already shifted by one."""
    text = training_text(ep)
    ids, mask = encode(text), supervised_mask(text)
    return (torch.tensor(ids[:-1], dtype=torch.uint8), torch.tensor(ids[1:], dtype=torch.uint8),
            torch.tensor(mask[1:], dtype=torch.uint8), ep["cls"])


def make_batches(lengths: list, batch_tokens: int, seed: int, epoch: int) -> list:
    """Lists of sequence indices. Sequences are sorted by length and chunked greedily so that
    count x longest length stays within `batch_tokens` (a sequence longer than the budget is a batch
    of its own); the chunk order is then shuffled with a generator seeded from (seed, epoch)."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    chunks, current = [], []
    for i in order:
        if current and (len(current) + 1) * lengths[i] > batch_tokens:
            chunks.append(current)
            current = []
        current.append(i)
    if current:
        chunks.append(current)
    random.Random(seed * 1_000_003 + epoch).shuffle(chunks)
    return chunks


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


def _autocast(device):
    """bf16 autocast on the GPU (fp32 master weights, no GradScaler needed); nothing on the CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ── evaluation ──

@torch.no_grad()
def evaluate(model, dev_encoded: list, batch_tokens: int, device) -> dict:
    """Dev loss and teacher-forced line exact match.

    dev_loss is the mask-weighted mean cross-entropy over every supervised dev character. A line is
    one maximal run of supervised characters (one A: payload plus its newline); it is exact when the
    argmax equals the gold character at every position of the run given the gold prefix, which is
    exactly what greedy decoding would produce. Reported overall, per episode class and for the first
    action of each episode.
    """
    was_training = model.training
    model.eval()
    loss_sum, loss_count = 0.0, 0
    lines = {"all": [0, 0], "first": [0, 0], **{c: [0, 0] for c in CLASSES}}      # [exact, total]
    lengths = [len(e[0]) for e in dev_encoded]
    for idxs in make_batches(lengths, batch_tokens, 0, 0):
        x, y, m = collate(dev_encoded, idxs, device)
        with _autocast(device):
            logits, loss = model(x, y, m)
        count = int(m.sum().item())
        loss_sum, loss_count = loss_sum + float(loss.item()) * count, loss_count + count

        supervised = m.bool()
        wrong = ((logits.argmax(dim=-1) != y) & supervised).flatten()
        starts = (supervised & ~torch.nn.functional.pad(supervised, (1, 0))[:, :-1]).flatten()
        run_id = torch.cumsum(starts.long(), dim=0)                           # 1-based, valid where supervised
        runs = int(starts.sum().item())
        bad = torch.zeros(runs + 1, dtype=torch.long, device=wrong.device)
        bad.scatter_add_(0, run_id[supervised.flatten()], wrong[supervised.flatten()].long())
        exact = (bad[1:] == 0).tolist()
        row_of_run = (torch.nonzero(starts).flatten() // x.shape[1]).tolist()
        seen_rows = set()
        for ok, row in zip(exact, row_of_run):
            cls = dev_encoded[idxs[row]][3]
            for key in ("all", cls):
                lines[key][0] += ok
                lines[key][1] += 1
            if row not in seen_rows:
                seen_rows.add(row)
                lines["first"][0] += ok
                lines["first"][1] += 1
    model.train(was_training)

    def share(key):
        exact, total = lines[key]
        return exact / total if total else 0.0

    return {"dev_loss": loss_sum / max(1, loss_count),
            "dev_line_exact": share("all"),
            "dev_line_exact_by_cls": {c: share(c) for c in CLASSES if lines[c][1]},
            "dev_first_line_exact": share("first")}


# ── files ──

def _rel(path: str) -> str:
    """Repo-relative path with forward slashes; a file outside the repository is reduced to its name."""
    real, root = os.path.realpath(path), os.path.realpath(PROJECT_ROOT)
    if real == root or real.startswith(root + os.sep):
        return os.path.relpath(real, root).replace(os.sep, "/")
    return os.path.basename(real)


def _save_checkpoint(path: str, model, model_cfg: GPTConfig, meta: dict) -> None:
    """Tensors and primitives only, written next to the final name and moved into place."""
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
    """Train ScratchGPT on the episodes in `train_path`, select on `dev_path`, write the artifacts, return the manifest.

    Raises LeakageError before the first step if either file contains benchmark content, and
    NonFiniteLossError (after a FAILED line and a non_finite_loss log event, with no final manifest)
    if the loss ever stops being finite.
    """
    model_cfg = model_cfg or GPTConfig()
    train_cfg = train_cfg or TrainConfig()
    if model_cfg.context != CONTEXT_CHARS:
        raise ValueError(f"model context {model_cfg.context} must equal format.CONTEXT_CHARS {CONTEXT_CHARS}")
    if min(train_cfg.max_steps, train_cfg.eval_every, train_cfg.log_every) < 1:
        raise ValueError("max_steps, eval_every and log_every must be at least 1")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    on_cuda = torch.device(device).type == "cuda"
    started = time.time()

    train_eps, dev_eps = read_episodes(train_path), read_episodes(dev_path)
    if not train_eps or not dev_eps:
        raise ValueError("the train and dev files must each hold at least one episode")
    assert_clean(train_eps)       # a leaking file stops here, before the first step
    assert_clean(dev_eps)
    train_enc = [encode_episode(e) for e in train_eps]
    dev_enc = [encode_episode(e) for e in dev_eps]
    train_lengths = [len(e[0]) for e in train_enc]

    torch.manual_seed(train_cfg.seed)
    model = ScratchGPT(model_cfg).to(device)
    params = model.num_params()
    decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices and embeddings
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # biases and LayerNorm
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), fused=on_cuda)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "scratch_gpt.pt")
    log_path = os.path.join(out_dir, "train_log.jsonl")
    for stale in (log_path, os.path.join(out_dir, "manifest.json")):
        if os.path.exists(stale):
            os.remove(stale)          # a rerun into the same directory must not leave an old manifest behind
    log(f"device {'cuda' if on_cuda else 'cpu'} params {params}")

    history, best = [], {"dev_loss": float("inf"), "step": 0}
    tokens_seen, step, epoch, stale_evals = 0, 0, 0, 0
    since_eval, since_log, mark = [], [], (time.time(), 0)
    stop_reason = None
    model.train()
    while step < train_cfg.max_steps and stop_reason is None:
        for idxs in make_batches(train_lengths, train_cfg.batch_tokens, train_cfg.seed, epoch):
            lr = lr_at(step, train_cfg.max_steps, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y, m = collate(train_enc, idxs, device)
            with _autocast(device):
                _, loss = model(x, y, m)
            if not bool(torch.isfinite(loss)):
                log(f"FAILED non_finite_loss step {step + 1}")
                _append_jsonl(log_path, {"step": step + 1, "event": "non_finite_loss"})
                raise NonFiniteLossError(f"training loss is not finite at step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            step += 1
            tokens_seen += int(x.numel())
            value = float(loss.item())
            since_eval.append(value)
            since_log.append(value)

            if step % train_cfg.log_every == 0:
                now = time.time()
                rate = (tokens_seen - mark[1]) / max(1e-9, now - mark[0])
                mark = (now, tokens_seen)
                interval = sum(since_log) / len(since_log)
                since_log = []
                log(f"step {step}/{train_cfg.max_steps} loss {interval:.4f} lr {lr:.2e} tok/s {rate:.0f} "
                    f"elapsed {now - started:.0f}s")
                if train_cfg.target_loss > 0 and interval < train_cfg.target_loss:
                    stop_reason = "target_loss"

            if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps or stop_reason:
                result = evaluate(model, dev_enc, train_cfg.batch_tokens, device)
                record = {"step": step, "train_loss": sum(since_eval) / len(since_eval), **result, "lr": lr,
                          "tokens_seen": tokens_seen, "elapsed_s": round(time.time() - started, 1)}
                since_eval = []
                history.append(record)
                _append_jsonl(log_path, record)
                if result["dev_loss"] < best["dev_loss"]:
                    best, stale_evals = {"dev_loss": result["dev_loss"], "step": step}, 0
                    _save_checkpoint(ckpt_path, model, model_cfg,
                                     {"seed": train_cfg.seed, "step": step, "dev_loss": result["dev_loss"],
                                      "dev_line_exact": result["dev_line_exact"], "tokens_seen": tokens_seen})
                else:
                    stale_evals += 1
                log(f"eval step {step} dev_loss {result['dev_loss']:.4f} dev_line_exact "
                    f"{result['dev_line_exact']:.4f} best {best['dev_loss']:.4f}")
                if stop_reason is None and stale_evals >= train_cfg.patience:
                    stop_reason = "early_stop"
            if step >= train_cfg.max_steps or stop_reason:
                break
        epoch += 1
    stop_reason = stop_reason or "max_steps"

    _write_curve(os.path.join(out_dir, "loss_curve.png"), history, params, best["step"])
    last = history[-1]
    wall = round(time.time() - started, 1)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": "cuda" if on_cuda else "cpu", "seed": train_cfg.seed,
        "dataset": {"train_path": _rel(train_path), "train_sha256": file_sha256(train_path),
                    "dev_path": _rel(dev_path), "dev_sha256": file_sha256(dev_path),
                    "n_train": len(train_eps), "n_dev": len(dev_eps),
                    "class_counts": {c: sum(e["cls"] == c for e in train_eps) for c in CLASSES},
                    "synthetic_fraction": round(sum(bool(e["meta"]["synthetic"]) for e in train_eps) / len(train_eps), 6)},
        "held_out_sha256": held_out_sha256(),
        **current_stamps(),
        "config": {"model": model_cfg.to_dict(), "train": train_cfg.to_dict()},
        "params": params, "steps": step, "best_step": best["step"], "tokens_seen": tokens_seen,
        "final": {"train_loss": last["train_loss"], "dev_loss": last["dev_loss"],
                  "dev_line_exact": last["dev_line_exact"], "dev_line_exact_by_cls": last["dev_line_exact_by_cls"],
                  "dev_first_line_exact": last["dev_first_line_exact"]},
        "checkpoint_sha256": file_sha256(ckpt_path),
        "wall_seconds": wall,
        "stop_reason": stop_reason,
        "run_notes": list(run_notes or []),
    }
    assert tuple(manifest) == MANIFEST_KEYS, "manifest fields drifted from MANIFEST_KEYS"
    write_json_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    log(f"DONE best_step {best['step']} best_dev_loss {best['dev_loss']:.4f} wall {wall:.0f}s stop {stop_reason}")
    return manifest


# ── command line: python -m src.scratch_llm.train ──

def _flush_print(text: str) -> None:
    print(text, flush=True)


def _note(text: str) -> str:
    if len(text) > MAX_NOTE_CHARS or not text.isascii() or "\n" in text:
        raise argparse.ArgumentTypeError(f"a run note is one line of ASCII, at most {MAX_NOTE_CHARS} characters")
    return text


def main(argv=None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(prog="python -m src.scratch_llm.train",
                                     description="Train ScratchGPT from scratch on the synthetic episodes.")
    parser.add_argument("--train", default="data/local/train-seed0-n40000.jsonl")
    parser.add_argument("--dev", default="data/local/dev-seed0-n2000.jsonl")
    parser.add_argument("--out-dir", default="models/scratch")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--batch-tokens", type=int, default=defaults.batch_tokens)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-layer", type=int, default=GPTConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=GPTConfig.n_head)
    parser.add_argument("--d-model", type=int, default=GPTConfig.d_model)
    parser.add_argument("--dropout", type=float, default=GPTConfig.dropout)
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
    if args.d_model % args.n_head:
        parser.error("--d-model must be divisible by --n-head")

    model_cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
    train_cfg = TrainConfig(seed=args.seed, max_steps=args.max_steps, eval_every=args.eval_every,
                            log_every=args.log_every, batch_tokens=args.batch_tokens, lr=args.lr,
                            min_lr=min(defaults.min_lr, args.lr), warmup=args.warmup, patience=args.patience)
    os.makedirs(paths["out_dir"], exist_ok=True)
    try:
        train(paths["train"], paths["dev"], paths["out_dir"], model_cfg, train_cfg, args.device, _flush_print,
              run_notes=args.run_note)
    except NonFiniteLossError:
        return EXIT_NON_FINITE        # the FAILED line was already printed by train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
