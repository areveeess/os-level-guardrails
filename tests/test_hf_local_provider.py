"""
Tests for the Hugging Face in-process provider (src/llm/hf_local.py): greedy decoding, stamps, registry wiring.
Cheapest first.

  - Fake-model contract tests need torch only (no transformers, no peft, no network, no GPU): the loader
    src.llm.hf_local.load_model is replaced by a fake that returns a scripted tokenizer and model.
  - The render contract (src/qwen_lora/chat.py) is proved against a real AgentLoop (torch only) and, for the
    chat template and the fingerprint, against the real cached Qwen tokenizer (needs transformers).
  - The tiny tests build a random-init Qwen2 (hidden size 64, 2 layers) next to the real cached tokenizer plus a
    LoRA adapter, and run the real transformers and peft code on the CPU: merged serving, adapter verification
    and refusal, offline loading. They need transformers, peft and the cached tokenizer (skipped without them,
    failed under STRICT_HF=1).
  - The one live probe (test_live_base_model_answers_on_the_gpu) loads the real cached
    Qwen/Qwen2.5-0.5B-Instruct on the GPU and runs only when QWEN_LIVE=1 (it costs about 80 s of imports).

Torch is imported once per interpreter (about 17 s cold from the Windows drive), transformers about 65 s and peft
about 74 s, so the Hugging Face tests share one module-scoped tiny model. transformers and peft are never imported
at module level in this file.
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import src.agent_harness.prompting as prompting
from src.agent_harness.executors import Executor, Observation
from src.agent_harness.loop import MAX_ASSISTANT_CHARS, AgentLoop
from src.agent_harness.prompting import SYSTEM_PROMPT, build_messages
from src.agent_harness.tool_broker import ToolBroker
from src.benchmark.injecagent import PROJECT_ROOT, benign_cases, load_tool_specs
from src.llm import hf_local, registry
from src.llm.base import LLMProvider, LLMResponse
from src.llm.hf_local import GREEDY_KWARGS, STOP_TOKENS, HFLocalProvider, LoadedModel
from src.qwen_lora.chat import chat_template_sha256, episode_to_chat, render_fingerprint
from src.scratch_llm import dataset
from src.scratch_llm.format import compact_action
from tests.qwen_fixtures import (build_tiny_qwen, copy_adapter, qwen_snapshot_or_skip, require_hf,
                                 write_lora_adapter)
from tests.test_format_parity import make_benign_episode, required_names

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


# ── the render contract: serving sees exactly what training renders (torch only) ──

class ScriptedGold(LLMProvider):
    """Replies with the gold actions in order and records every conversation it is shown."""

    name = "gold"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, temperature=0.0, max_tokens=512):
        self.seen.append([dict(m) for m in messages])
        return LLMResponse(text=self.replies.pop(0), latency_ms=0.0, model=self.name)


class ObservationExecutor(Executor):
    """Answers each action with the next recorded observation."""

    def __init__(self, observations):
        self.observations = list(observations)

    def execute(self, action):
        return Observation("success", output=self.observations.pop(0))


def replayed_conversations(ep):
    """Run the episode through a real AgentLoop; return the messages of every provider call."""
    provider = ScriptedGold([compact_action(a) for a, _ in ep["turns"]])
    executor = ObservationExecutor([o for _, o in ep["turns"][:-1]])
    AgentLoop(provider, executor, max_steps=len(ep["turns"]), tools=ep["tools"]).run(ep["user"])
    return provider.seen


def assert_replay_matches_chat(ep):
    chat = episode_to_chat(ep)
    seen = replayed_conversations(ep)
    assert len(seen) == len(ep["turns"])
    for k, messages in enumerate(seen):
        assert messages == chat[:2 + 2 * k], f"{ep['id']} step {k}"
        assert messages == dataset.episode_to_messages(ep, k), f"{ep['id']} step {k}"
    for action, observation in ep["turns"]:
        assert len(compact_action(action)) <= MAX_ASSISTANT_CHARS      # the loop never truncates a gold action
        if observation is not None:
            assert Observation("success", output=observation).as_text() == observation


@pytest.fixture(scope="module")
def free_specs():
    pool = dataset.free_tool_specs()
    chosen = [s for s in pool.values()
              if len([p for p in s["parameters"] if p.get("required")]) >= 2
              and all(p["type"] == "string" for p in s["parameters"] if p.get("required"))]
    assert len(chosen) >= 2
    return chosen


def test_replay_a_real_agent_loop_sends_exactly_the_episode_chat_prefixes():
    episodes = dataset.generate_episodes(30, 0)
    assert len(episodes) == 30
    for ep in episodes:
        assert_replay_matches_chat(ep)


def test_replay_matches_with_a_real_tool_broker_denial(free_specs):
    first, second = free_specs[0], free_specs[1]
    broker = ToolBroker(None, capabilities={first["name"]})
    second_args = {name: f"val{n}x" for n, name in enumerate(required_names(second))}
    denial = broker.execute({"action": "call_tool", "tool": second["name"], "args": second_args}).as_text()
    assert denial.startswith("BLOCKED by tool_broker: Tool '")
    ep = make_benign_episode(first)
    ep["tools"].append(dataset.episode_tool(second))
    ep["turns"].insert(1, [{"action": "call_tool", "tool": second["name"], "args": second_args}, denial])
    dataset.validate_episode(ep)
    assert_replay_matches_chat(ep)


def test_episode_chat_is_the_harness_messages_plus_the_gold_turns(free_specs):
    ep = make_benign_episode(free_specs[0])
    chat = episode_to_chat(ep)
    assert chat[:2] == build_messages(ep["user"], ep["tools"])
    assert chat[2:] == [
        {"role": "assistant", "content": compact_action(ep["turns"][0][0])},
        {"role": "user", "content": prompting.OBSERVATION_TEMPLATE.format(result=ep["turns"][0][1])},
        {"role": "assistant", "content": compact_action(ep["turns"][1][0])},
    ]


# ── the render contract against the real Qwen tokenizer ──

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(qwen_snapshot_or_skip(), local_files_only=True)


def test_system_prompt_is_rendered_first_and_qwen_defaults_are_absent(tokenizer):
    tool = {"name": "NoteSearchEntries", "summary": "Search notes.",
            "parameters": [{"name": "keyword", "type": "string", "required": True}]}
    text = tokenizer.apply_chat_template(build_messages("list my notes", [tool]), tokenize=False,
                                         add_generation_prompt=True)
    assert text.startswith("<|im_start|>system\n" + SYSTEM_PROMPT)
    assert "You are Qwen" not in text and "<tools>" not in text
    assert text.endswith("<|im_start|>assistant\n")


def test_fingerprint_is_stable_hex_and_follows_the_system_prompt(tokenizer, monkeypatch):
    first = render_fingerprint(tokenizer)
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert render_fingerprint(tokenizer) == first
    monkeypatch.setattr(prompting, "SYSTEM_PROMPT", prompting.SYSTEM_PROMPT + "One more rule.\n")
    assert render_fingerprint(tokenizer) != first


def test_fingerprint_module_chat_template_hash_matches_the_template(tokenizer):
    assert chat_template_sha256(tokenizer) == hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()


# ── import isolation ──

def test_isolation_light_modules_import_none_of_the_heavy_stack():
    code = ("import sys; import src.qwen_lora.chat, src.llm.hf_local, src.llm.registry; "
            "heavy = ('torch', 'transformers', 'peft', 'huggingface_hub', 'src.scratch_llm.dataset', "
            "'src.benchmark.injecagent'); "
            "bad = [m for m in heavy if m in sys.modules]; "
            "print('BAD', bad) if bad else print('CLEAN')")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "CLEAN", done.stdout


# ── tiny random-init Qwen2 with the real tokenizer and a LoRA adapter (real transformers and peft, CPU) ──

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    require_hf()
    work = tmp_path_factory.mktemp("tiny_qwen")
    base = os.path.realpath(build_tiny_qwen(str(work / "base")))
    adapter = write_lora_adapter(base, str(work / "adapter"))
    with open(os.path.join(adapter, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return SimpleNamespace(base=base, adapter=adapter, manifest=manifest, work=work)


@pytest.fixture(scope="module")
def tiny_providers(tiny):
    """The same adapter merged and unmerged, and the base alone."""
    def build(name, **kwargs):
        return HFLocalProvider(name, tiny.base, device="cpu", max_new_tokens=8, **kwargs)
    return SimpleNamespace(merged=build("tiny-lora", adapter=tiny.adapter),
                           unmerged=build("tiny-lora", adapter=tiny.adapter, merge=False),
                           base_only=build("tiny-base"))


def three_prompts():
    tools = benign_cases(load_tool_specs())[0].tools
    return [build_messages(text, tools) for text in ("list my notes", "what is on my calendar today", "say hello")]


def greedy_ids(provider, messages):
    """The token ids the provider's model writes for a prompt, with the provider's own decoding arguments."""
    enc = provider.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                 return_dict=True, return_tensors="pt")
    n_in = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = provider.model.generate(**enc, **GREEDY_KWARGS, max_new_tokens=8, eos_token_id=provider.stop_ids,
                                      pad_token_id=provider.pad_id)
    return out[0, n_in:].tolist()


def test_tiny_adapter_row_loads_merges_and_stamps_the_adapter_hash(tiny, tiny_providers):
    response = tiny_providers.merged.generate(three_prompts()[0])
    assert response.ok, response.error
    assert response.finish_reason in ("stop", "length")
    assert response.resolved_model == f"{tiny.base}@local+tiny-lora@{tiny.manifest['adapter_sha256'][:12]}"
    assert response.resolved_model.startswith(f"{tiny.base}@local")
    assert tiny_providers.merged.adapter_sha256 == tiny.manifest["adapter_sha256"]
    assert type(tiny_providers.merged.model).__name__ == "Qwen2ForCausalLM"        # merged: no PEFT wrapper left
    assert type(tiny_providers.unmerged.model).__name__ != "Qwen2ForCausalLM"      # merge=False keeps the wrapper
    assert tiny_providers.base_only.generate(three_prompts()[0]).resolved_model == f"{tiny.base}@local"


def test_tiny_merged_and_unmerged_greedy_outputs_are_identical_on_three_prompts(tiny_providers):
    written = []
    for messages in three_prompts():
        merged_ids = greedy_ids(tiny_providers.merged, messages)
        assert merged_ids == greedy_ids(tiny_providers.unmerged, messages)
        written.append(merged_ids)
    assert max(len(ids) for ids in written) > 1          # the comparison is over real generated text, not empty replies


def test_tiny_adapter_changes_the_logits_relative_to_the_base(tiny_providers):
    messages = three_prompts()[0]

    def last_logits(provider):
        enc = provider.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors="pt")
        with torch.inference_mode():
            return provider.model(**enc).logits[0, -1]

    assert (last_logits(tiny_providers.merged) - last_logits(tiny_providers.base_only)).abs().max() > 1e-3


# ── stale and missing adapters are refused at construction ──

def edit_json(path, change):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    change(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


STALE_EDITS = [
    ("manifest.json", lambda d: d.__setitem__("adapter_sha256", "0" * 64), "adapter_sha256"),
    ("manifest.json", lambda d: d["render"].__setitem__("fingerprint", "f" * 64), "render.fingerprint"),
    ("adapter_config.json", lambda d: d.__setitem__("base_model_name_or_path", "Qwen/Another-Model"),
     "base_model_name_or_path"),
    ("manifest.json", lambda d: d["base_model"].__setitem__("revision", "0123456789ab"), "revision"),
    ("manifest.json", lambda d: d["base_model"].__setitem__("id", "Qwen/Another-Model"), "base_model.id"),
]


@pytest.mark.parametrize("filename, change, field", STALE_EDITS, ids=[e[2] for e in STALE_EDITS])
def test_stale_adapter_is_refused_and_the_field_is_named(tiny, tmp_path, filename, change, field):
    stale = copy_adapter(tiny.adapter, str(tmp_path / "stale"))
    edit_json(os.path.join(stale, filename), change)
    with pytest.raises(RuntimeError, match=field) as raised:
        HFLocalProvider("tiny-lora", tiny.base, adapter=stale, device="cpu")
    assert "retrain it with python -m src.qwen_lora.train" in str(raised.value)


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors", "manifest.json"])
def test_missing_adapter_file_is_named(tiny, tmp_path, filename):
    damaged = copy_adapter(tiny.adapter, str(tmp_path / "damaged"))
    os.remove(os.path.join(damaged, filename))
    with pytest.raises(RuntimeError, match=re.escape(filename)):
        HFLocalProvider("tiny-lora", tiny.base, adapter=damaged, device="cpu")


def test_missing_adapter_directory_says_to_train_it(tiny, tmp_path):
    with pytest.raises(RuntimeError, match=re.escape("python -m src.qwen_lora.train")):
        HFLocalProvider("tiny-lora", tiny.base, adapter=str(tmp_path / "nowhere"), device="cpu")


def test_missing_adapter_config_without_its_fields_is_a_runtime_error_not_a_key_error(tiny, tmp_path):
    damaged = copy_adapter(tiny.adapter, str(tmp_path / "damaged"))
    with open(os.path.join(damaged, "adapter_config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(RuntimeError, match="adapter_config.json"):
        HFLocalProvider("tiny-lora", tiny.base, adapter=damaged, device="cpu")


# ── offline ──

def test_offline_uncached_base_model_is_refused_without_a_network_connection(monkeypatch):
    require_hf()
    attempts = []

    def refuse(self, address):
        attempts.append(address)
        raise OSError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    with pytest.raises(RuntimeError, match="local Hugging Face cache"):
        HFLocalProvider("x", "Qwen/not-a-cached-model-0000", device="cpu")
    assert attempts == []


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
