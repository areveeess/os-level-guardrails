"""
Tests for the provider layer: reply parsing, model registry, and the
OpenAI-compatible adapter (with a fake client — no network).
"""

from types import SimpleNamespace

import pytest

from src.agent_harness.prompting import parse_action, action_target, build_messages, ActionParseError, SYSTEM_PROMPT
from src.llm.openai_compat import OpenAICompatibleProvider
from src.llm.registry import get_provider, list_models, load_model_config


# ── parse_action ──

def test_parse_plain_json():
    assert parse_action('{"action": "run_shell", "command": "ls"}') == {"action": "run_shell", "command": "ls"}


def test_parse_strips_code_fence():
    raw = '```json\n{"action": "read_file", "path": "/tmp/a"}\n```'
    assert parse_action(raw)["path"] == "/tmp/a"


def test_parse_ignores_chatter_around_json():
    raw = 'Sure! Here is the action:\n{"action": "respond", "message": "hi {there}"}\nHope that helps.'
    assert parse_action(raw) == {"action": "respond", "message": "hi {there}"}


def test_parse_skips_non_action_objects():
    raw = '{"note": 1} then {"action": "http_request", "url": "http://x.example"}'
    assert parse_action(raw)["action"] == "http_request"


@pytest.mark.parametrize("raw", ["", "I cannot help with that.", '{"action": "run_shell", "command": ', "[1, 2]"])
def test_parse_rejects_non_actions(raw):
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_parse_rejects_unknown_action_type():
    with pytest.raises(ActionParseError):
        parse_action('{"action": "format_disk"}')


def test_action_target_and_messages():
    assert action_target({"action": "run_shell", "command": "ls"}) == "ls"
    assert action_target({"action": "respond", "message": "ok"}) == "ok"
    msgs = build_messages("list /tmp")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "list /tmp"}


# ── OpenAICompatibleProvider with a fake client ──

class FakeClient:
    def __init__(self, reply=None, exc=None, model=None, finish_reason=None):
        self.calls = []
        self._reply, self._exc = reply, exc
        self._model, self._finish_reason = model, finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        choice = SimpleNamespace(message=SimpleNamespace(content=self._reply))
        response = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        # Only set when asked, so the default keeps the shape of a server that reports neither
        if self._finish_reason is not None:
            choice.finish_reason = self._finish_reason
        if self._model is not None:
            response.model = self._model
        return response


def test_provider_returns_text_and_usage():
    client = FakeClient(reply='  {"action": "respond", "message": "hi"}  ')
    p = OpenAICompatibleProvider("m", "model-id", "http://x", client=client)
    r = p.generate(build_messages("hi"))
    assert r.ok and r.text == '{"action": "respond", "message": "hi"}'
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)
    assert client.calls[0]["model"] == "model-id" and client.calls[0]["temperature"] == 0.0


def test_provider_reports_errors_instead_of_raising():
    p = OpenAICompatibleProvider("m", "model-id", "http://x", client=FakeClient(exc=TimeoutError("slow")))
    r = p.generate(build_messages("hi"))
    assert not r.ok and "TimeoutError" in r.error and r.text == ""


def test_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOME_MISSING_KEY"):
        OpenAICompatibleProvider("m", "model-id", "http://x", api_key_env="SOME_MISSING_KEY", client=FakeClient())


# ── Registry ──

def test_registry_lists_configured_models():
    config = load_model_config()
    assert config["default"] in config["models"]
    assert "scratch-gpt" in list_models(benchmark_only=True)


def test_registry_builds_keyless_local_provider():
    p = get_provider("ollama-llama3.2-3b")
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.base_url.startswith("http://localhost:11434")


def test_registry_rejects_unknown_model():
    with pytest.raises(KeyError):
        get_provider("does-not-exist")


def test_provider_recovers_output_rejected_by_server():
    class Rejected(Exception):
        body = {"error": {"code": "tool_use_failed",
                          "failed_generation": '{"name": "assistant", "arguments": {"action": "respond", "message": "hi"}}'}}

    p = OpenAICompatibleProvider("m", "model-id", "http://x", client=FakeClient(exc=Rejected("400")))
    r = p.generate(build_messages("hi"))
    assert r.ok and parse_action(r.text) == {"action": "respond", "message": "hi"}


# ── finish_reason and resolved model id (EVAL-06) ──

def test_provider_reports_finish_reason_and_resolved_model():
    client = FakeClient(reply="partial", model="gpt-oss-120b-2026-09", finish_reason="length")
    r = OpenAICompatibleProvider("m", "gpt-oss-120b", "http://x", client=client).generate(build_messages("hi"))
    assert r.finish_reason == "length" and r.resolved_model == "gpt-oss-120b-2026-09"


def test_provider_without_model_or_finish_reason_leaves_them_none():
    r = OpenAICompatibleProvider("m", "model-id", "http://x", client=FakeClient(reply="hi")).generate(
        build_messages("hi"))
    assert r.ok and r.finish_reason is None and r.resolved_model is None


def test_recovered_generation_is_tagged_as_recovered_from_error():
    class Rejected(Exception):
        body = {"error": {"code": "tool_use_failed",
                          "failed_generation": '{"action": "respond", "message": "hi"}'}}

    r = OpenAICompatibleProvider("m", "model-id", "http://x", client=FakeClient(exc=Rejected("400"))).generate(
        build_messages("hi"))
    assert r.ok and r.finish_reason == "recovered_from_error"
