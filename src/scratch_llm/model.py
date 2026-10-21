"""
ScratchGPT — a small decoder-only transformer, written from scratch
===================================================================
    tokens → token + position embeddings
           → N × [LayerNorm → causal multi-head self-attention → residual
                  LayerNorm → MLP (4×, GELU)                 → residual]
           → LayerNorm → linear head (tied to the token embedding) → logits

No pretrained weights, no external model code: every layer is defined here.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 96
    context: int = 2048   # must equal format.CONTEXT_CHARS (D-05); checkpoints are stamped with it
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(QKᵀ/√d + causal mask)·V — fused kernel, same maths as the textbook version
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ScratchGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        if mask is not None:  # learn only the agent's own action lines
            mask = mask.view(-1).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new: int, stop_id: int, temperature: float = 0.0):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1, :]
            if temperature > 0:
                nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == stop_id:
                break
        return idx
