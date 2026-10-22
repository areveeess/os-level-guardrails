"""
Tests for the Hugging Face in-process provider (src/llm/hf_local.py): greedy decoding, stamps, registry wiring.
Cheapest first.

  - Fake-model contract tests need torch only (no transformers, no peft, no network, no GPU): the loader
    src.llm.hf_local.load_model is replaced by a fake that returns a scripted tokenizer and model.
  - The one live probe (test_live_base_model_answers_on_the_gpu) loads the real cached
    Qwen/Qwen2.5-0.5B-Instruct on the GPU and runs only when QWEN_LIVE=1 (it costs about 80 s of imports).

Torch is imported once per interpreter (about 17 s cold from the Windows drive); transformers and peft are never
imported at module level in this file.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from src.agent_harness.prompting import build_messages
from src.benchmark.injecagent import benign_cases, load_tool_specs
from src.llm import hf_local, registry
from src.llm.hf_local import GREEDY_KWARGS, STOP_TOKENS, HFLocalProvider, LoadedModel

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STOP_ID = 151645
PAD_ID = 151643
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


# ── fakes shared by the contract tests ──

class FakeTokenizer:
    """Renders any conversation to a 4-token prompt and decodes ids from a small table."""

    TABLE = {10: " Hello", 11: " world ", 12: " more", STOP_ID: "<|im_end|>", PAD_ID: "<|endoftext|>"}

    def __init__(self, prompt_len=4):
        self.prompt_len = prompt_len
        self.rendered = []

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append((messages, kwargs))
        ids = torch.arange(1, self.prompt_len + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=False):
        pieces = [self.TABLE.get(int(i), f" t{int(i)}") for i in ids]
        if skip_special_tokens:
            pieces = [p for p, i in zip(pieces, ids) if int(i) not in (STOP_ID, PAD_ID)]
        return "".join(pieces)


class FakeModel:
    """Records the kwargs of every generate call and appends a scripted tail to the prompt."""

    def __init__(self, tail=(10, 11, STOP_ID), decode_error=None):
        self.tail = list(tail)
        self.calls = []
        self.error = decode_error

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        prompt = kwargs["input_ids"]
        return torch.cat([prompt, torch.tensor([self.tail], dtype=prompt.dtype)], dim=1)


def fake_loaded(tail=(10, 11, STOP_ID), adapter_sha256=None, prompt_len=4, error=None):
    return LoadedModel(tokenizer=FakeTokenizer(prompt_len), model=FakeModel(tail, error), revision=REVISION,
                       adapter_sha256=adapter_sha256, stop_ids=[STOP_ID, PAD_ID], pad_id=PAD_ID, device="cpu")


def install_fake_loader(monkeypatch, loaded=None):
    """Replace the module-level loader; return the list of keyword arguments the provider passed to it."""
    calls = []
    loaded = loaded or fake_loaded()

    def fake_load_model(base_model, adapter=None, merge=True, device="cuda"):
        calls.append(dict(base_model=base_model, adapter=adapter, merge=merge, device=device))
        return loaded

    monkeypatch.setattr(hf_local, "load_model", fake_load_model)
    return calls


def notes_messages():
    return build_messages("list my notes", benign_cases(load_tool_specs())[0].tools)


# ── reply contract: stamps ──

def test_contract_reply_carries_text_tokens_finish_reason_and_resolved_model(monkeypatch):
    loaded = fake_loaded()
    install_fake_loader(monkeypatch, loaded)
    response = HFLocalProvider("qwen-test", QWEN_ID).generate(notes_messages())
    assert response.ok
    assert response.text == "Hello world"                      # decode of [10, 11], whitespace stripped
    assert response.prompt_tokens == 4
    assert response.completion_tokens == 3                      # the stop token counts
    assert response.finish_reason == "stop"
    assert response.model == "qwen-test"
    assert response.resolved_model == f"{QWEN_ID}@{REVISION[:12]}"
    assert response.latency_ms >= 0


def test_contract_adapter_row_appends_the_adapter_hash_to_resolved_model(monkeypatch):
    sha = "ab" * 32
    install_fake_loader(monkeypatch, fake_loaded(adapter_sha256=sha))
    response = HFLocalProvider("qwen-test", QWEN_ID, adapter="/x/adapter").generate(notes_messages())
    assert response.resolved_model == f"{QWEN_ID}@{REVISION[:12]}+qwen-test@{sha[:12]}"


def test_contract_the_prompt_is_rendered_by_the_chat_template_without_tools(monkeypatch):
    loaded = fake_loaded()
    install_fake_loader(monkeypatch, loaded)
    messages = notes_messages()
    HFLocalProvider("qwen-test", QWEN_ID).generate(messages)
    (rendered, kwargs), = loaded.tokenizer.rendered
    assert rendered is messages
    assert kwargs == dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    assert "tools" not in kwargs


# ── greedy decoding ──

def test_greedy_kwargs_are_explicit_on_every_call(monkeypatch):
    loaded = fake_loaded()
    install_fake_loader(monkeypatch, loaded)
    HFLocalProvider("qwen-test", QWEN_ID).generate(notes_messages())
    (call,) = loaded.model.calls
    assert call["do_sample"] is False
    assert call["temperature"] is None and call["top_p"] is None and call["top_k"] is None
    assert call["repetition_penalty"] == 1.0
    assert call["eos_token_id"] == [STOP_ID, PAD_ID]
    assert call["pad_token_id"] == PAD_ID
    assert {k: call[k] for k in GREEDY_KWARGS} == GREEDY_KWARGS
    assert STOP_TOKENS == ("<|im_end|>", "<|endoftext|>")


def test_greedy_temperature_is_accepted_and_ignored(monkeypatch):
    loaded = fake_loaded()
    install_fake_loader(monkeypatch, loaded)
    provider = HFLocalProvider("qwen-test", QWEN_ID)
    provider.generate(notes_messages(), temperature=0.0)
    provider.generate(notes_messages(), temperature=0.7)
    cold, hot = loaded.model.calls
    for key in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty"):
        assert cold[key] == hot[key], key
    assert hot["temperature"] is None and hot["do_sample"] is False


@pytest.mark.parametrize("max_tokens, cfg_new, cfg_ctx, expected", [
    (512, 512, 4096, 512),
    (100, 512, 4096, 100),          # the caller's cap wins
    (512, 64, 4096, 64),            # the configured cap wins
    (512, 512, 20, 16),             # the context leaves 20 - 4 tokens
])
def test_greedy_max_new_tokens_is_the_smallest_of_the_three_caps(monkeypatch, max_tokens, cfg_new, cfg_ctx, expected):
    loaded = fake_loaded()
    install_fake_loader(monkeypatch, loaded)
    provider = HFLocalProvider("qwen-test", QWEN_ID, max_new_tokens=cfg_new, max_context_tokens=cfg_ctx)
    provider.generate(notes_messages(), max_tokens=max_tokens)
    assert loaded.model.calls[0]["max_new_tokens"] == expected


# ── registry wiring ──

def write_hf_models_yaml(path, **fields):
    lines = ["default: qwen-test", "models:", "  qwen-test:", "    provider: hf_local", "    model: qwen-test"]
    lines += [f"    {key}: {value}" for key, value in fields.items()]
    lines += ["    family: test", "    benchmark: true"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_registry_hf_local_entry_builds_the_provider_with_its_config(tmp_path, monkeypatch):
    calls = install_fake_loader(monkeypatch)
    config = tmp_path / "models.yaml"
    write_hf_models_yaml(config, base_model=QWEN_ID, adapter="models/qwen-agentlora", device="cpu", max_new_tokens=64)
    provider = registry.get_provider("qwen-test", path=str(config))
    assert isinstance(provider, HFLocalProvider)
    assert calls == [dict(base_model=QWEN_ID, adapter=os.path.join(registry.PROJECT_ROOT, "models/qwen-agentlora"),
                          merge=True, device="cpu")]
    assert provider.max_new_tokens == 64 and provider.max_context_tokens == 4096


def test_registry_hf_local_adapter_null_passes_none(tmp_path, monkeypatch):
    calls = install_fake_loader(monkeypatch)
    config = tmp_path / "models.yaml"
    write_hf_models_yaml(config, base_model=QWEN_ID, adapter="null")
    registry.get_provider("qwen-test", path=str(config))
    assert calls[0]["adapter"] is None and calls[0]["device"] == "cuda" and calls[0]["merge"] is True


# ── opt-in live probe (GPU, real cached model) ──

@pytest.mark.skipif(os.environ.get("QWEN_LIVE") != "1", reason="set QWEN_LIVE=1 to load the real model on the GPU")
def test_live_base_model_answers_on_the_gpu():
    assert torch.cuda.is_available(), "QWEN_LIVE=1 needs a CUDA device"
    provider = registry.get_provider("qwen2.5-0.5b-base")
    case = benign_cases(load_tool_specs())[0]
    response = provider.generate(build_messages(case.prompt, case.tools))
    _, revision = hf_local.resolve_base_model(QWEN_ID)
    print(f"\nLIVE resolved_model={response.resolved_model}")
    print(f"LIVE latency_ms={response.latency_ms} prompt_tokens={response.prompt_tokens} "
          f"completion_tokens={response.completion_tokens} finish_reason={response.finish_reason}")
    print(f"LIVE reply={response.text!r}")
    assert response.ok, response.error
    assert response.resolved_model == f"{QWEN_ID}@{revision[:12]}"
    assert response.finish_reason in ("stop", "length")
    assert response.prompt_tokens > 242                         # the system prompt alone is 242 tokens
    assert response.completion_tokens >= 1
    assert torch.cuda.memory_allocated() > 0
