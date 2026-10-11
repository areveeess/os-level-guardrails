"""
From-scratch model provider
===========================
Wraps the tiny transformer trained in src/scratch_llm/ (Phase 4). It is a
deliberately naive, unaligned worst-case principal: it follows any
instruction it sees, including injected ones.
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
        from src.scratch_llm.inference import load_generator  # Phase 4
        self._generate = load_generator(checkpoint)

    def generate(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        try:
            text = self._generate(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name,
                               error=f"{type(e).__name__}: {e}")
        return LLMResponse(text=text.strip(), latency_ms=_elapsed_ms(start), model=self.name)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
