---
phase: 03-qwen-lora-hf-local-provider
plan: 02
subsystem: lora-training
tags: [lora, peft, qwen2.5, loss-mask, char-offsets, chat-template, gradient-checkpointing, early-stopping]

requires:
  - phase: 03-qwen-lora-hf-local-provider
    provides: "chat-render contract (episode_to_chat, render_fingerprint), HFLocalProvider + verify_adapter, tiny-Qwen2 fixtures"
  - phase: 02-scratch-llm
    provides: "episodes (generate_episodes, read_episodes, assert_clean), lr_at, provenance helpers, the CR-01 selection lesson"
provides:
  - "src/qwen_lora/data.py: one chat sequence per episode, character-offset loss mask (supervised = assistant action + <|im_end|>), action-level mask, guards, collate"
  - "src/qwen_lora/train.py: custom LoRA loop (core), supervised-position loss, evaluate() with action-level metrics, best-by-dev_loss_action selection, adapter + manifest hf_local verifies"
  - "tests proving LORA-03: decoded supervised tokens == gold actions (generated + Phase 2 files), prefix stability, loss equivalence, parameter count 8,798,208"
affects: [03-03, 03-04, phase-5-report]

actuals:
  tokens: 16592
  tasks: 3
  commits: 3
plan_head_before: f6fafa0837534336a5e26662695bbd5e5bb995c5
plan_head_after: 498fffd896f2c80df8b6b315b9f1ce602951d308
commits: 3

tech-stack:
  added: []
  patterns:
    - "loss mask from tokenizer offset_mapping over the rendered text, never from token ids"
    - "LM head applied only to supervised rows; fp32 cross-entropy normalised by the accumulation-window token count"
    - "evaluate() seam (logits_fn) and module-level evaluate lookup so tests script the selection metric"
    - "kept adapter written to adapter.best.tmp at each improvement and moved into place only when the run finishes"

key-files:
  created:
    - src/qwen_lora/data.py
    - src/qwen_lora/train.py
    - tests/test_qwen_lora_data.py
    - tests/test_qwen_lora_train.py
  modified: []

key-decisions:
  - "Mask by character offsets of the rendered text; ids are never compared (Qwen pad/eos can coincide and would hide the stop token)"
  - "Action-level mask excludes only respond message text, boundary BPE token excluded whole (conservative, deterministic)"
  - "all-linear resolves to full module names per layer in PEFT 0.21.2, so target modules are checked by leaf name plus a count of 7 x n_layers"
  - "PAD_ID = 151643 as a module constant (asserted equal to the tokenizer's <|endoftext|>) so evaluate() needs no tokenizer; padding is attention-masked and never decides a label"
  - "LORA-03 left unchecked in REQUIREMENTS.md: the mechanics are proven here, the real 600-step adapter is plan 03-04"

patterns-established:
  - "Tiny random-init Qwen2 + real tokenizer, module-scoped, few CPU steps: a full train() run in about 4 s after the import"
  - "Stand-in logits_fn (gold everywhere except respond text) to test the action-level metrics without a trained model"

requirements-completed: []

coverage:
  - id: D1
    description: "Loss covers exactly the assistant action plus <|im_end|>: decoded supervised tokens equal the gold actions; header, trailing newline, system, user and observation tokens are unsupervised; a pad-id token inside content stays supervised"
    requirement: LORA-03
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_data.py#test_supervised_*"
        status: pass
    human_judgment: false
  - id: D2
    description: "Train/serve prefix stability: every step prompt rendered by apply_chat_template is a token prefix of the training sequence and the next token is supervised"
    requirement: LORA-03
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_data.py#test_prefix_*"
        status: pass
    human_judgment: false
  - id: D3
    description: "Action-level mask, special-token guard and no-truncation length guard"
    requirement: LORA-03
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_data.py#test_action_mask_* / test_special_* / test_supervised_max_length_*"
        status: pass
    human_judgment: false
  - id: D4
    description: "Tracer: a tiny adapter trained by train() is verified and served by hf_local through run_benchmark (resolved_model +tiny-lora@<sha12>, no llm_error); a held-out tool in the training file raises LeakageError before any adapter file exists"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "tests/test_qwen_lora_train.py#test_tracer_* / test_leak_*"
        status: pass
    human_judgment: false
  - id: D5
    description: "Training maths: custom loss equals the HF labels= loss, LoRA parameter count (32,768 tiny; 366,592 per layer and 8,798,208 real), seven targets, gradients through gradient checkpointing"
    requirement: LORA-03
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_train.py#test_loss_equivalence_* / test_param_count_* / test_targets_* / test_checkpointing_*"
        status: pass
    human_judgment: false
  - id: D6
    description: "evaluate() returns exactly EVAL_KEYS; the kept adapter is the lowest dev_loss_action evaluation (not the last step, not the plain dev loss); early stop after patience"
    requirement: LORA-03
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_train.py#test_evaluate_* / test_selection_*"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-10-22
status: complete
---

# Phase 3 Plan 02: Qwen LoRA Data, Loss Mask and Core Training Loop Summary

**One-chat-sequence-per-episode training data with a character-offset loss mask (only the assistant action plus its `<|im_end|>` carries loss), and a custom LoRA loop whose supervised-position loss equals the Hugging Face loss, proven end to end on the CPU by serving a tiny trained adapter through `run_benchmark`.**

## Performance

- **Duration:** 30 min
- **Started:** 2026-10-22T05:33:18Z
- **Completed:** 2026-10-22T06:02:50Z
- **Tasks:** 3 (1 tracer, 2 auto/tdd)
- **Files modified:** 4 (all created)

## Accomplishments

- **LORA-03 proof.** `tokenizer.decode(supervised ids)` equals `"".join(compact_action(a) + "<|im_end|>")` for every episode checked, zero mismatches. The mask comes from `return_offsets_mapping` over the rendered text; no mask decision compares a token id (grep-checked), and a pad-id token placed inside assistant content stays supervised.
- **Coverage of the mask and prefix tests.** 300 generated episodes (200 train, 50 dev_tools, 50 dev_synthetic; all of benign/dh/ds present; 835 steps) plus the first 300 episodes of each Phase 2 file (`train-seed0-n40000.jsonl`: 852 steps; `dev-seed0-n2000.jsonl`: 829 steps) were decoded-checked: 900 episodes. The prefix test checks all 835 step prompts of the 300 generated episodes. **Maximum sequence length seen: 826 tokens** (generated and train file), 781 (dev file), under `MAX_LEN` 1024.
- **Prefix stability (D-03).** For every generated episode and step k, `apply_chat_template(episode_to_messages(ep, k), add_generation_prompt=True)` is an exact token prefix of the training ids and the next token is supervised, so a step's training context is exactly what the provider renders.
- **Loss equivalence.** On the tiny Qwen2 (dropout 0, non-zero B matrices) the supervised-position loss is 11.96544768 against the Hugging Face `labels=` loss 11.96544647: **difference 1.2e-06** (bound 1e-4). The 151,936-wide head sees only supervised rows.
- **Parameter counts asserted.** Tiny model 32,768 (`lora_param_count(64, 128, 4, 2, 2, 16)`); real dimensions 366,592 per layer and **8,798,208** in total, also computed from the cached `config.json`; `build_lora_model` asserts the loaded PEFT model's trainable count equals the formula, so a mismatch would stop a run. Target modules resolve to q/k/v/o/gate/up/down in every layer (no lm_head, no embeddings).
- **Tracer.** `train()` on the tiny base for 3 steps (micro-batch 2 x accumulation 2) writes `adapter_config.json` (r 16, alpha 32, dropout 0.05, `base_model_name_or_path` = realpath of the base), `adapter_model.safetensors`, `manifest.json` (adapter_sha256, base_model {id, revision "local"}, render fingerprint), no `README.md`; `hf_local` serves it in a benchmark row whose `resolved_model` ends `+tiny-lora@<adapter_sha256[:12]>` with `llm_error` None.
- **Selection (D-06).** A scripted `evaluate` with dev_loss_action minimum at the second evaluation (and a plain dev_loss that falls every call) stops with `early_stop` at step 4, `best_step` 2, and the final `adapter_model.safetensors` equals the LoRA tensors captured at the step-2 evaluation, not the last step's.

## Test runtimes (HF import included)

| Run | Result | Wall |
|-----|--------|------|
| `tests/test_qwen_lora_data.py` alone | 15 passed | about 83 s |
| `tests/test_qwen_lora_train.py` alone | 14 passed | about 90 s |
| both new files together | 29 passed | 92 s |
| new files + `tests/test_hf_local_provider.py` (plan verification) | 71 passed, 1 skipped (QWEN_LIVE probe) | 107 s |
| full suite | **972 passed, 1 skipped** (baseline 943 + 1; +29 tests) | 295 s |

`git status --porcelain models data/local` printed nothing: the tests write only under tmp_path.

## Task Commits

1. **Task 1 (tracer): tiny Qwen LoRA trained by the custom loop is served in a benchmark episode** - `86a8fab` (feat)
2. **Task 2: decoded supervised tokens, prefix stability, action mask and guards** - `d179bbe` (test)
3. **Task 3: LoRA loss, parameter count, action-level evaluation, best-by-dev_loss_action selection** - `498fffd` (feat)

**Plan metadata:** the docs commit that adds this SUMMARY (see git log).

## Files Created

- `src/qwen_lora/data.py` - `MAX_LEN`, `RESPOND_HEAD`, `RESPOND_TAIL`, `SequenceTooLongError`, `Example`, `labels_from_text`, `build_example`, `gold_supervised_text`, `collate`
- `src/qwen_lora/train.py` - `TrainConfig`, `LORA_TARGETS`, `lora_param_count`, `build_lora_model`, `supervised_logits`, `supervised_loss`, `evaluate`, `train`, `EVAL_KEYS`, `SELECTION_METRIC`, `IN_RUN_ADAPTER`
- `tests/test_qwen_lora_data.py` (14 test functions, 15 cases), `tests/test_qwen_lora_train.py` (14 test functions)

## Decisions Made

- Names match `<interfaces>`. Additions: `PAD_ID` (151643, asserted equal to the tokenizer's `<|endoftext|>` at run time), `MANIFEST_SCHEMA`, `STALE_FILES`, and the helpers `stratified_sample`, `epoch_stream`, `_model_id`, `_decoder_and_head`, `_n_supervised`, `_rel`, `_remove`, `_append_jsonl` (the last three copied rather than imported from the ScratchGPT trainer to avoid private cross-module imports). `supervised_loss` and `supervised_logits` take `(peft_model, input_ids, attention_mask, labels)`.
- The respond-text boundary token that BPE merges with the closing quote is excluded whole (documented in `labels_from_text`; plan 03-03 states it in the manifest's `selection.description`).
- The manifest of this plan holds schema, created_at, python, torch, device, seed, base_model, render, dataset, config, steps, best_step, tokens_seen, selection, best, last_eval, adapter_sha256, wall_seconds, stop_reason, run_notes; plan 03-03 adds the remaining fields and the `MANIFEST_KEYS` assertion.
- LORA-03 is not checked off: the mechanics are proven here, the real adapter is plan 03-04.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `all-linear` resolves to full module names, not the seven leaf names**
- **Found during:** Task 3 (first run of the parameter-count wiring)
- **Issue:** my assertion compared `peft_config.target_modules` with `set(LORA_TARGETS)`; PEFT 0.21.2 stores one full name per layer and projection (`model.layers.0.self_attn.q_proj`, ...).
- **Fix:** compare the leaf names with `LORA_TARGETS` and require `len == 7 x num_hidden_layers`; the test does the same and also asserts no `lm_head` or embedding is adapted.
- **Files modified:** src/qwen_lora/train.py, tests/test_qwen_lora_train.py
- **Commit:** 498fffd

**2. [Rule 3 - Blocking] A test name was not selected by the plan's `-k` expression**
- **Found during:** Task 2
- **Issue:** `test_max_length_...` matched none of "supervised or prefix or system_prompt or action_mask or special or collate or tracer or leak", so the plan's verify command would have skipped it.
- **Fix:** renamed to `test_supervised_max_length_...`.
- **Commit:** d179bbe

**Total deviations:** 2 auto-fixed (Rule 1 and Rule 3, assertion and naming only). **Impact:** none on scope. No file outside the four planned files changed.

## Issues Encountered

None beyond the two above. The tracer and the Task 2 mask tests were green on their first run against the implementation; the red steps (module missing; action mask, guards and `EVAL_KEYS` missing) failed as expected before implementation.

## Notes for plans 03-03 and 03-04

- `train()` currently raises a plain `RuntimeError` naming the metric on a non-finite training or dev loss; 03-03 turns that into the FAILED line, exit code 3 and the cleanup. `STALE_FILES` already spares `train_progress.log`, `train.done` and `gpu_preflight.log`.
- The epoch order is a generator (`epoch_stream`) deterministic in `(seed, epoch)`; resume in 03-03 can fast-forward it by step count.
- `evaluate` takes no tokenizer (it pads with `PAD_ID`); the final full-dev pass and the generation check run on the kept weights because `train()` reloads the best tensors into the model before it finishes.
- `train_log.jsonl` has one record per evaluation; `loss_curve.png`, the full manifest field list, the package versions and `gen_check` are still to come.

## Known Stubs

None.

## Threat Flags

None. T-03-07 (leakage gate before step 0, leak test), T-03-08 (special-token guard, benchmark files clean), T-03-09 (offset mask, decoded-token tests), T-03-10 (offline load, no remote-code flag, safetensors only) and T-03-11 (`refuse_foreign_manifest` before any deletion) are implemented as planned; T-03-SC accepted (no installs).

## Self-Check: PASSED

- Created files exist: `src/qwen_lora/data.py`, `src/qwen_lora/train.py`, `tests/test_qwen_lora_data.py`, `tests/test_qwen_lora_train.py` (all FOUND).
- Commits `86a8fab`, `d179bbe`, `498fffd` are ancestors of HEAD; `git rev-list --count` from `plan_head_before` gives 3.
- Acceptance greps: assert_clean 4, return_offsets_mapping=True 1, save_embedding_layers=False 2, enable_input_require_grads 2, remote-code flag 0, top-level transformers/peft imports 0, SequenceTooLongError 3, has_special_token 1, id-comparison in the mask 0, def lora_param_count 1, 8,798,208 in the tests 2, SELECTION_METRIC 8, set_peft_model_state_dict 2, test functions 14 (data) and 14 (train).
- Plan verification: both new files plus the provider tests 71 passed, 1 skipped in 107 s; full suite 972 passed, 1 skipped in 295 s; no change under models/ or data/local/.
