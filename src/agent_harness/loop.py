"""
Agent Loop — multi-step, tool-using agent
==========================================
    prompt → LLM → action → executor → observation → LLM → ... → respond

Observations (including "BLOCKED by policy_engine: ...") are fed back to the
model, so it can retry another way; that lets the benchmark measure whether
the guardrails hold up against a persistent (or injected) agent.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.agent_harness.executors import Executor, Observation
from src.agent_harness.prompting import (build_messages, parse_action, action_target, ActionParseError,
                                         OBSERVATION_TEMPLATE, INVALID_REPLY_TEMPLATE)
from src.llm.base import LLMProvider

MAX_ASSISTANT_CHARS = 2000


@dataclass
class Step:
    index: int
    raw: str
    latency_ms: float
    action: Optional[dict] = None
    parse_error: Optional[str] = None
    observation: Optional[Observation] = None


@dataclass
class Episode:
    prompt: str
    model: str
    steps: list = field(default_factory=list)
    final_message: str = ""
    stop_reason: str = ""          # responded | max_steps | llm_error
    llm_error: Optional[str] = None

    @property
    def actions(self) -> list:
        return [s.action for s in self.steps if s.action]


class AgentLoop:

    def __init__(self, provider: LLMProvider, executor: Executor, max_steps: int = 5,
                 temperature: float = 0.0, verbose: bool = False, tools: list = None):
        self.provider = provider
        self.tools = tools
        self.executor = executor
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose

    def run(self, prompt: str) -> Episode:
        episode = Episode(prompt=prompt, model=self.provider.name)
        messages = build_messages(prompt, self.tools)

        for index in range(1, self.max_steps + 1):
            response = self.provider.generate(messages, temperature=self.temperature)
            if not response.ok:
                episode.stop_reason, episode.llm_error = "llm_error", response.error
                self._say(f"  ✗ LLM error: {response.error}")
                return episode

            step = Step(index=index, raw=response.text, latency_ms=response.latency_ms)
            episode.steps.append(step)
            messages.append({"role": "assistant", "content": response.text[:MAX_ASSISTANT_CHARS]})

            try:
                step.action = parse_action(response.text)
            except ActionParseError as e:
                step.parse_error = str(e)
                self._say(f"\n[step {index}] ✗ unparseable reply: {response.text[:200]!r}")
                messages.append({"role": "user", "content": INVALID_REPLY_TEMPLATE.format(error=e)})
                continue

            kind = step.action["action"]
            self._say(f"\n[step {index}] {kind}: {str(action_target(step.action))[:200]}  ({response.latency_ms:.0f} ms)")
            if kind == "respond":
                episode.final_message = step.action.get("message", "")
                episode.stop_reason = "responded"
                self._say(f"  → final answer: {episode.final_message[:500]}")
                return episode

            step.observation = self.executor.execute(step.action)
            self._say(self._describe(step.observation))
            messages.append({"role": "user",
                             "content": OBSERVATION_TEMPLATE.format(result=step.observation.as_text())})

        episode.stop_reason = "max_steps"
        self._say(f"\n  Stopped after {self.max_steps} steps without a final answer.")
        return episode

    @staticmethod
    def _describe(obs: Observation) -> str:
        if obs.status == "blocked":
            return f"  ❌ BLOCKED by {obs.layer}: {obs.reason}"
        flag = " ⚠️ flagged" if obs.flagged else ""
        text = obs.output if obs.status == "success" else (obs.output or obs.reason)
        text = text.strip()
        if len(text) > 500:
            text = text[:500] + " ...(truncated)"
        icon = "✅" if obs.status == "success" else "✗"
        return f"  {icon}{flag} {text}" if text else f"  {icon}{flag} (no output)"

    def _say(self, text: str):
        if self.verbose:
            print(text)
