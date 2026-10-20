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
    """One model reply. `finish_reason` (why generation stopped: stop | length |
    content_filter | ...) and `resolved_model` (the model id the server actually
    answered with) are filled when the backend reports them, else None."""
    text: str
    latency_ms: float
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    resolved_model: Optional[str] = None

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
