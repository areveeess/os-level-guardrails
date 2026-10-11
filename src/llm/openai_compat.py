"""
OpenAI-compatible provider
==========================
Gemini, Groq, OpenRouter and Ollama all expose the OpenAI chat-completions
API, so one adapter covers every hosted and local model. Only the base URL,
API-key variable and model id differ (see config/models.yaml).
"""

import os
import time

from src.llm.base import LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):

    def __init__(self, name: str, model: str, base_url: str, api_key_env: str = None,
                 timeout: float = 60.0, client=None):
        self.name = name
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        # Local servers like Ollama ignore the key but the SDK requires one
        api_key = os.environ.get(api_key_env) if api_key_env else "not-needed"
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set (needed for model '{name}'). Add it to .env")
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self._client = client

    def generate(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            return LLMResponse(text="", latency_ms=_elapsed_ms(start), model=self.name,
                               error=f"{type(e).__name__}: {e}")

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text.strip(),
            latency_ms=_elapsed_ms(start),
            model=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
