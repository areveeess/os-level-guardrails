"""
From-scratch model provider
===========================
Wraps the small transformer trained in src/scratch_llm/ (Phase 2, python -m src.scratch_llm.train).
It is a deliberately weak, unaligned worst-case principal: it was trained to follow any
instruction it sees, including injected ones.

The provider returns the model's raw text. It never repairs a malformed reply; the agent loop
reports it back to the model as a parse error, like it does for every other model. Construction
fails (RuntimeError) when the checkpoint is missing or its format stamps differ from the running
code, and the benchmark runner turns that into a [skip] line instead of scoring a stale model.
"""

import os
import time

from src.llm.base import LLMProvider, LLMResponse


class LocalScratchProvider(LLMProvider):

    def __init__(self, name: str, checkpoint: str):
        self.name = name
        self.checkpoint = checkpoint
        if not os.path.exists(checkpoint):
            raise RuntimeError(
                f"Scratch model checkpoint not found: {checkpoint}. "
                f"Train it first: python -m src.scratch_llm.train"
            )
        # Lazy import: torch is only needed when this provider is built, not when the registry loads.
        from src.scratch_llm.inference import load_generator
        self._generate = load_generator(checkpoint)

    def generate(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        try:
            text = self._generate(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            # A prompt that does not fit the model's context is reported, never answered from a cropped prompt (WR-03).
            overflow = getattr(e, "finish_reason", None) == "context_overflow"
            return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name,
                               error=f"{type(e).__name__}: {e}",
                               prompt_tokens=getattr(self._generate, "last_prompt_chars", None) if overflow else None,
                               finish_reason="context_overflow" if overflow else None)
        # Provenance stamps (one character is one token for this model). getattr keeps test doubles
        # that do not expose these attributes working.
        sha = getattr(self._generate, "checkpoint_sha256", None)
        return LLMResponse(
            text=text.strip(), latency_ms=_elapsed_ms(start), model=self.name,
            prompt_tokens=getattr(self._generate, "last_prompt_chars", None),
            completion_tokens=getattr(self._generate, "last_new_chars", None),
            finish_reason=getattr(self._generate, "last_finish_reason", None),
            resolved_model=f"{self.name}@{sha[:12]}" if sha else None,
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
