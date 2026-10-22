---
phase: 03-qwen-lora-hf-local-provider
plan: 01
subsystem: llm-provider
tags: [huggingface, transformers, peft, lora, qwen2.5, greedy-decoding, offline, chat-template]

requires:
  - phase: 02-scratch-llm
    provides: "provenance stamps convention, LocalScratchProvider pattern, runner [skip] path, Phase 2 episodes"
provides:
  - "HFLocalProvider: in-process Hugging Face causal LM behind LLMProvider (offline, explicit greedy, stop tokens, stamps, merged adapter)"
  - "src/qwen_lora/chat.py: the Qwen chat-render contract shared by training and serving, with render_fingerprint"
  - "verify_adapter: refuses a missing or stale adapter (files, base id, revision, hash, render fingerprint)"
  - "config entries qwen2.5-0.5b-base and qwen2.5-0.5b-agentlora; requirements-train.txt; STRICT_HF=1"
  - "tests/qwen_fixtures.py: tiny random-init Qwen2 + LoRA adapter writer for later plans"
affects: [03-02, 03-03, 03-04, phase-4-full-runs, phase-5-report]

actuals:
  tokens: 16600
  tasks: 3
  commits: 3
plan_head_before: 79e7f03cbf1bb09889cf83bb11f778a3988734c3
plan_head_after: b81e07a21b65d286dcc02e2257a1ff1f8d0ed570
commits: 3

tech-stack:
  added: [transformers 5.20.2, peft 0.21.2, accelerate 1.15.0 (pinned in requirements-train.txt, installed during research)]
  patterns:
    - "lazy heavy imports inside functions; module-level load_model seam replaced by a fake in tests"
    - "construction raises RuntimeError (runner prints [skip]); generate never raises"
    - "render fingerprint of a golden conversation stamped into the adapter manifest and checked at load"

key-files:
  created:
    - src/llm/hf_local.py
    - src/qwen_lora/__init__.py
    - src/qwen_lora/chat.py
    - tests/test_hf_local_provider.py
    - tests/qwen_fixtures.py
    - requirements-train.txt
  modified:
    - src/llm/registry.py
    - config/models.yaml
    - tests/conftest.py

key-decisions:
  - "resolved_model = '<base>@<rev12>' always, '+<name>@<adapter_sha12>' appended for an adapter row (assumption-delta promoted: base revision is the always-present identity)"
  - "Adapter file existence is checked before any Hugging Face import so a missing adapter skips in about a second instead of after the 65 s import"
  - "temperature accepted and ignored; the benchmark is greedy for every model"
  - "max_tokens 0 (or a full context) returns an empty 'length' reply without calling the model"

patterns-established:
  - "HF tests share one module-scoped tiny random-init Qwen2 with the real cached tokenizer (CPU, seconds)"
  - "Replay parity: a real AgentLoop with a scripted gold provider must send exactly episode_to_chat(ep)[:2+2k]"

requirements-completed: [LORA-01, LORA-02]

coverage:
  - id: D1
    description: "HFLocalProvider serves a Hugging Face causal LM in-process, greedy with explicit kwargs, stop tokens, token/finish_reason/resolved_model stamps, errors as values, context overflow as a recorded error"
    requirement: LORA-01
    verification:
      - kind: unit
        ref: "tests/test_hf_local_provider.py#test_contract_* / test_greedy_* / test_length_* / test_overflow_* / test_error_response_* / test_raw_text_*"
        status: pass
    human_judgment: false
  - id: D2
    description: "Adapter is merged at load and verified (files, base id, revision, sha256, render fingerprint); stale or missing adapter refused at construction and skipped by the runner; offline with no socket connection"
    requirement: LORA-01
    verification:
      - kind: integration
        ref: "tests/test_hf_local_provider.py#test_tiny_* / test_stale_* / test_missing_adapter_* / test_offline_* / test_runner_*"
        status: pass
    human_judgment: false
  - id: D3
    description: "Chat-render contract: serving sends exactly what training renders (real AgentLoop replay incl. a real ToolBroker denial), project system prompt first, no Qwen default prompt, no tools block"
    requirement: LORA-01
    verification:
      - kind: unit
        ref: "tests/test_hf_local_provider.py#test_replay_* / test_system_prompt_* / test_fingerprint_*"
        status: pass
    human_judgment: false
  - id: D4
    description: "Un-tuned qwen2.5-0.5b-base registered (LORA-02 configuration) and served live on the RTX 3060 from the offline cache"
    requirement: LORA-02
    verification:
      - kind: other
        ref: "QWEN_LIVE=1 pytest tests/test_hf_local_provider.py -k live (1 passed)"
        status: pass
    human_judgment: true
    rationale: "The base row's benchmark smoke run (rows, parse rate) is plan 03-04; this plan proves only the plumbing, not the row's results"

duration: 21min
completed: 2026-10-22
status: complete
---

# Phase 3 Plan 01: hf_local Provider and Chat-Render Contract Summary

**An in-process, offline, greedy Hugging Face provider (HFLocalProvider) that serves the cached Qwen2.5-0.5B-Instruct base row and a hash-verified, merged LoRA adapter row behind LLMProvider, plus the shared Qwen chat-render contract (render fingerprint) that the trainer of plans 03-02/03-03 must stamp.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-10-22T05:09:04Z
- **Completed:** 2026-10-22T05:29:43Z
- **Tasks:** 3 (1 tracer, 2 auto/tdd)
- **Files modified:** 9 (6 created, 3 modified)

## Accomplishments

- `get_provider("qwen2.5-0.5b-base")` builds an `HFLocalProvider` from the local Hugging Face cache with no network access (`HF_HUB_OFFLINE` set before any HF import, `local_files_only=True` on snapshot, tokenizer and model, no remote-code flag). Importing the registry, `hf_local` or `chat` imports none of torch, transformers, peft, huggingface_hub, the dataset or the benchmark (subprocess `sys.modules` test).
- Every reply is greedy by explicit per-call kwargs that override Qwen's sampling `generation_config.json`, stops on `<|im_end|>`/`<|endoftext|>`, and carries `prompt_tokens`, `completion_tokens` (stop token included), `finish_reason` and `resolved_model`.
- The render contract (`src/qwen_lora/chat.py`): the prompt is rendered only by `apply_chat_template` over `build_messages` output (system prompt first, no `tools` argument). A real `AgentLoop` replay (30 generated episodes plus a real `ToolBroker` denial episode) sends exactly `episode_to_chat(ep)[:2 + 2k]` at every step.
- `verify_adapter` refuses a stale adapter at construction, naming the field: `adapter_sha256`, `render.fingerprint`, `base_model_name_or_path`, `base_model.revision`, `base_model.id`, plus each missing file and a missing directory (with the `python -m src.qwen_lora.train` hint). The runner turns every refusal, an uncached base model included, into `[skip] <name>:` with no rows.
- On a tiny random-init Qwen2 with the real tokenizer: merged and unmerged greedy ids are identical on three prompts, a non-zero adapter moves the logits by more than 1e-3, and a tiny adapter row runs through the real `run_benchmark` with provider/config_model/resolved_model stamps.
- Live GPU probe passed once (plumbing evidence, below). The base row's benchmark smoke run is plan 03-04.

## Live probe evidence (QWEN_LIVE=1, RTX 3060, first benign benchmark case)

| Field | Value |
|-------|-------|
| resolved_model | `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf` |
| reply text | `{"action": "call_tool", "tool": "AmazonGetProductDetails", "args": {"product_id": "B08KFQ9HK5"}}` |
| latency_ms | 2560.5 (includes first-call CUDA warm-up) |
| prompt_tokens / completion_tokens | 296 / 34 |
| finish_reason | stop |

Test runtime: `tests/test_hf_local_provider.py` takes about 82 s (dominated by the one-time transformers/peft imports); the three-file run (provider + `test_llm_layer.py` + `test_scratch_inference.py`) took 124 s. Full suite: **943 passed, 1 skipped** (the QWEN_LIVE probe) in 264 s; baseline was 901 passing, so +42.

## Task Commits

1. **Task 1: hf_local provider, registry branch, qwen2.5-0.5b-base entry, contract tests, live probe** - `41f5552` (feat)
2. **Task 2: chat-render contract, verify_adapter, tiny-Qwen2 integration tests** - `863dba6` (feat)
3. **Task 3: agentlora entry, reply-contract edge tests, runner [skip], STRICT_HF, requirements-train.txt** - `b81e07a` (feat)

**Plan metadata:** the docs commit that adds this SUMMARY (see git log).

## Files Created/Modified

- `src/llm/hf_local.py` - HFLocalProvider, LoadedModel, resolve_base_model, load_model, verify_adapter, sha256_file, constants
- `src/qwen_lora/__init__.py`, `src/qwen_lora/chat.py` - light package; render contract (episode_to_chat, golden_chat, render_fingerprint, chat_template_sha256, has_special_token)
- `src/llm/registry.py` - lazy `hf_local` branch
- `config/models.yaml` - `qwen2.5-0.5b-base` (adapter null) and `qwen2.5-0.5b-agentlora` (adapter `models/qwen-agentlora`)
- `requirements-train.txt` - transformers>=5.20,<5.21, peft==0.21.2, accelerate>=1.15,<2
- `tests/test_hf_local_provider.py` (24 test functions, 33 cases), `tests/qwen_fixtures.py`, `tests/conftest.py` (HF header line, `STRICT_HF=1`)

## Decisions Made

- **Names match `<interfaces>`** (`STOP_TOKENS`, `GREEDY_KWARGS`, `ADAPTER_FILES`, `LoadedModel`, `resolve_base_model`, `load_model`, `verify_adapter`, `sha256_file`, `HFLocalProvider`; `IM_START`, `IM_END`, `ENDOFTEXT`, `ASSISTANT_HEADER`, `SPECIAL_TOKEN_STRINGS`, `RENDER_VERSION`, `episode_to_chat`, `golden_chat`, `render_fingerprint`, `chat_template_sha256`, `has_special_token`). Additions: `RETRAIN`, `_require_adapter_files` (private), and `DEFAULT_MAX_*` constants as specified.
- Adapter file existence is checked first, before the HF imports, so a missing adapter row is skipped in about a second; the hash/fingerprint checks run after the tokenizer load and before the base weights load.
- `max_new_tokens == 0` (caller passes `max_tokens=0`, or the context is exactly full) returns an empty `length` reply without a model call (HF `generate` rejects 0).
- SUS-by-recency rationale for the pinned packages: transformers 5.20.2 (released 2026-10-19), peft 0.21.2 and accelerate 1.15.0 are official Hugging Face packages rated SUS by the package-legitimacy seam only for release recency and unknown download counts (no postinstall, none deprecated, source repos under github.com/huggingface). They were installed and user-approved during research (`pip check` clean); this plan installed and downloaded nothing, it only pins them.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test expectation for the length-finish decode**
- **Found during:** Task 3
- **Issue:** `test_length_finish_reason_...` expected `"Hello world more"`, but the fake decode table (`" world "` carries a trailing space) yields two spaces. The provider was correct; the test expectation was wrong.
- **Fix:** expected `"Hello world  more"` with a comment. No source change.
- **Files modified:** tests/test_hf_local_provider.py
- **Commit:** b81e07a

**2. [Rule 1 - Bug] Docstring tripped an acceptance grep**
- **Found during:** Task 1
- **Issue:** the module docstring named the remote-code flag, making `grep -c trust_remote_code` print 1 (criterion: 0).
- **Fix:** reworded to "the remote-code flag". No behaviour change.
- **Commit:** 41f5552

**Total deviations:** 2 auto-fixed (both Rule 1, test/docstring only). **Impact:** none on behaviour or scope.

## Issues Encountered

None. Every plan-level verification passed first time once the code was written; the red steps (module missing, `verify_adapter` missing, yaml/requirements missing) failed as expected before implementation.

## Notes for plans 03-02 and 03-03 (trainer)

- `verify_adapter` compares `adapter_config.json` `base_model_name_or_path` against the configured base **id** exactly (`Qwen/Qwen2.5-0.5B-Instruct`). If the trainer loads the base from the snapshot directory, PEFT records the snapshot path there; the trainer must write the id (or load by id offline) or the adapter will be refused.
- The manifest must hold `adapter_sha256` (sha256 of `adapter_model.safetensors`), `base_model.id`, `base_model.revision` (the snapshot directory name, `7ae557604adf67be50417f59c2c2f167def9a775`) and `render.fingerprint` (`src.qwen_lora.chat.render_fingerprint(tokenizer)`), as `tests/qwen_fixtures.write_lora_adapter` does.
- `tests/qwen_fixtures.py` (`build_tiny_qwen`, `write_lora_adapter`, `qwen_snapshot_or_skip`) is ready for reuse; `has_special_token` is ready for the data guard (T-03-05).
- The `qwen2.5-0.5b-agentlora` row prints `[skip]` until plan 03-04 trains the adapter, by design. `run_train.sh` is referenced in the missing-adapter message and will arrive with the trainer.

## Known Stubs

None.

## Threat Flags

None. No new network endpoints, auth paths or trust boundaries beyond the plan's threat model (T-03-01 to T-03-04 and T-03-06 are mitigated and tested; T-03-05 accepted as planned).

## Self-Check: PASSED

- Created files exist: `src/llm/hf_local.py`, `src/qwen_lora/__init__.py`, `src/qwen_lora/chat.py`, `tests/test_hf_local_provider.py`, `tests/qwen_fixtures.py`, `requirements-train.txt` (all FOUND).
- Commits `41f5552`, `863dba6`, `b81e07a` are ancestors of HEAD.
- All task acceptance criteria re-run and passing (greps: local_files_only 4, remote-code flag 0, heavy top-level imports 0, registry branch 1, 24 test functions, STRICT_HF/STRICT_TORCH present, requirements pins 3, banned packages 0, provider hf_local entries 2).
- Plan verification: provider + llm_layer + scratch_inference run green (90 passed, 1 skipped); full suite 943 passed, 1 skipped; live probe passed once on the GPU.
