"""
ScratchGPT inference - load a trained checkpoint and serve greedy completions
=============================================================================
LocalScratchProvider (src/llm/scratch.py) calls load_generator() once and then asks the returned
generator for the model's next reply, one chat transcript at a time.

This module imports only format.py and model.py. It never imports dataset.py or anything from
src/benchmark, so code that runs at evaluation time has no way to read the held-out tool list or
the benchmark text the training data was gated against (SCRATCH-02).

The reply is the model's raw text. Nothing repairs the JSON and nothing constrains the decoding:
a malformed reply is a real property of a weak, unaligned model, and the agent loop reports it
back to the model as a parse error like any other.

Checkpoint contract (decision D-05). train.py writes, and load_generator() requires, exactly the
keys in CHECKPOINT_KEYS. Five of them are stamps of the code that produced the checkpoint:

    format_version, vocab_sha256, render_fingerprint, context_chars, gen_reserve

A checkpoint whose stamps differ from the running code was trained on a different prompt format,
so evaluating it would measure the mismatch, not the model: load_generator() refuses it with a
RuntimeError, LocalScratchProvider fails to construct, and the benchmark runner prints a [skip]
line and writes no rows for that model. The file is read with torch.load(weights_only=True), so a
tampered checkpoint cannot run code while it is deserialised.
"""

import hashlib

import torch

from src.scratch_llm.format import (CONTEXT_CHARS, FORMAT_VERSION, GEN_RESERVE, STOI, VOCAB, VOCAB_SHA256, decode,
                                    encode, render_context, render_fingerprint)
from src.scratch_llm.model import GPTConfig, ScratchGPT

MAX_NEW_CHARS = 320   # the longest legitimate action is about 230 characters; this bounds a runaway reply
STOP_ID = STOI["\n"]

CHECKPOINT_KEYS = ("state_dict", "cfg", "format_version", "vocab_sha256", "render_fingerprint",
                   "context_chars", "gen_reserve", "train_meta")
STAMP_KEYS = ("format_version", "vocab_sha256", "render_fingerprint", "context_chars", "gen_reserve")


def current_stamps() -> dict:
    """The stamps of the running code: what train.py writes into a checkpoint and what load checks."""
    return {"format_version": FORMAT_VERSION, "vocab_sha256": VOCAB_SHA256,
            "render_fingerprint": render_fingerprint(), "context_chars": CONTEXT_CHARS,
            "gen_reserve": GEN_RESERVE}


def check_stamps(ckpt: dict) -> None:
    """Raise RuntimeError naming the field, the expected value and the found value on the first mismatch."""
    expected = current_stamps()
    for field in STAMP_KEYS:
        if ckpt[field] != expected[field]:
            raise RuntimeError(f"checkpoint {field} is {ckpt[field]!r} but the running code has {expected[field]!r}; "
                               f"retrain it with python -m src.scratch_llm.train")
    cfg = ckpt["cfg"]
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint cfg is {cfg!r}, expected a dict")
    if cfg.get("context") != CONTEXT_CHARS:
        raise RuntimeError(f"checkpoint cfg context is {cfg.get('context')!r} but the running code has {CONTEXT_CHARS!r}")
    if cfg.get("vocab_size") != len(VOCAB):
        raise RuntimeError(f"checkpoint cfg vocab_size is {cfg.get('vocab_size')!r} but the running code has {len(VOCAB)!r}")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScratchGenerator:
    """Callable (messages, temperature, max_tokens) -> the model's next reply as one line of text.

    After each call: last_prompt_chars (characters the model read), last_new_chars (characters it
    wrote, including the stop newline when there was one) and last_finish_reason ("stop" when the
    model ended its line, "length" when the cap cut it off). One character is one token.
    """

    def __init__(self, model: ScratchGPT, device: str, checkpoint_sha256: str):
        self.model = model
        self.device = device
        self.checkpoint_sha256 = checkpoint_sha256
        self.identity = {"checkpoint_sha256": checkpoint_sha256, "format_version": FORMAT_VERSION,
                         "params": model.num_params(), "context": model.cfg.context}
        self.last_prompt_chars = 0
        self.last_new_chars = 0
        self.last_finish_reason = None

    def __call__(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> str:
        prompt = render_context(messages)
        ids = encode(prompt)
        max_new = max(0, min(int(max_tokens), MAX_NEW_CHARS))
        self.last_prompt_chars = len(ids)
        if max_new == 0:
            self.last_new_chars, self.last_finish_reason = 0, "length"
            return ""
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new, STOP_ID, temperature)
        new = out[0, len(ids):].tolist()
        stopped = bool(new) and new[-1] == STOP_ID
        self.last_new_chars = len(new)
        self.last_finish_reason = "stop" if stopped else "length"
        return decode(new[:-1] if stopped else new)


def load_generator(path: str, device: str = None) -> ScratchGenerator:
    """Load a checkpoint written by train.py. RuntimeError if its keys or stamps differ from the running code."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"{path} is not a ScratchGPT checkpoint (expected a dict, found {type(ckpt).__name__})")
    missing = [k for k in CHECKPOINT_KEYS if k not in ckpt]
    unexpected = [k for k in ckpt if k not in CHECKPOINT_KEYS]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint keys differ from the contract: missing {missing}, unexpected {unexpected}")
    check_stamps(ckpt)
    model = ScratchGPT(GPTConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()      # fp32 weights, no autocast: greedy output does not depend on precision
    return ScratchGenerator(model, device, _sha256_file(path))
