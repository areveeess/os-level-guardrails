"""
LLM Provider interface
======================
The agent treats every model as an interchangeable, untrusted principal.
A provider only has to turn a list of chat messages into text; latency and
errors are captured so the benchmark can compare models on equal terms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    latency_ms: float
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMProvider(ABC):
    """A model the agent can consult. Implementations must not raise on API
    failures; they return an LLMResponse with `error` set instead."""

    name: str

    @abstractmethod
    def generate(self, messages: list, temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        """Return the model's reply to a list of {"role", "content"} messages."""
