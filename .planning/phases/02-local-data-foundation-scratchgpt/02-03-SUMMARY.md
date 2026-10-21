---
phase: 02-local-data-foundation-scratchgpt
plan: 03
subsystem: testing
tags: [scratchgpt, training, inference, checkpoint-stamps, provenance, torch-cpu-tests]

requires:
  - phase: 02-local-data-foundation-scratchgpt (plan 01)
    provides: format v2, ScratchGPT model, render_context, supervised_mask, leakage gate
  - phase: 02-local-data-foundation-scratchgpt (plan 02)
    provides: generate_episodes, read/write_episodes, provenance helpers, dataset manifest
provides:
  - "python -m src.scratch_llm.train: masked next-character training with dev-selected best checkpoint, manifest, log and loss curve"
  - "inference.load_generator: refuses any checkpoint whose key set or format stamps differ from the running code"
  - "LocalScratchProvider filling prompt_tokens, completion_tokens, finish_reason and resolved_model (scratch-gpt@<sha12>)"
  - "config/models.yaml scratch-gpt entry with model: scratch-gpt (config_model stamp)"
affects: [02-04 GPU training run and smoke benchmark, 03 LoRA trainer, 04 full benchmark, 05 report disclosure]

plan_head_before: de9f49487df703f3169db3f66222a548b4211353
plan_head_after: 3dc735f616030326250116d6dace5b37b45a4083

actuals:
  tokens: 14500
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Writer and reader share one checkpoint contract: train.py builds the checkpoint from CHECKPOINT_KEYS and current_stamps(), inference.py checks the same objects"
    - "Evaluation-time code (inference.py) imports only format and model; a subprocess sys.modules test pins that it cannot reach the held-out list"
    - "Failure as exit code: a non-finite loss prints FAILED, logs an event, raises NonFiniteLossError and exits 3 with no final manifest"
    - "Teacher-forced line exact match computed with a vectorised run-length pass instead of generating each line"

key-files:
  created:
    - src/scratch_llm/train.py
    - src/scratch_llm/inference.py
    - tests/test_scratch_train.py
    - tests/test_scratch_inference.py
  modified:
    - src/llm/scratch.py
    - config/models.yaml
    - requirements.txt

key-decisions:
  - "Non-finite loss is checked right after the forward pass, before backward and the optimiser step, so a bad step never touches the weights (the plan said after the step)"
  - "A reconstructed-checkpoint mismatch (unknown cfg field, weights of another shape) is also a RuntimeError, so the runner skips instead of crashing"
  - "Provider tests pin the CPU by monkeypatching torch.cuda.is_available rather than adding a device argument to LocalScratchProvider"
  - "CLI --lr below the default floor lowers min_lr to --lr so the cosine never decays upward"

patterns-established:
  - "Progress lines are the polling contract for background runs (formats below)"

requirements-completed: [SCRATCH-03, SCRATCH-04, SCRATCH-05]

coverage:
  - id: D1
    description: "Training writes scratch_gpt.pt (weights_only-loadable, exact CHECKPOINT_KEYS, all stamps), manifest.json with exactly MANIFEST_KEYS, train_log.jsonl and loss_curve.png; no environment values in the manifest; checkpoint_sha256 equals the file hash"
    requirement: SCRATCH-04
    verification:
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_training_writes_the_four_artifacts_and_a_stamped_checkpoint"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_manifest_has_the_fixed_fields_no_environment_values_and_the_checkpoint_hash"
        status: pass
    human_judgment: false
  - id: D2
    description: "Loss counts only A: payloads plus newline; the model is causal; right padding is invisible; schedule and token-budget batching behave"
    requirement: SCRATCH-04
    verification:
      - kind: unit
        ref: "tests/test_scratch_train.py#test_decoded_supervised_targets_are_exactly_the_compact_actions"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_the_model_is_causal"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_right_padding_does_not_change_the_logits_of_real_positions"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_learning_rate_warms_up_peaks_and_decays_to_the_floor"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_batches_respect_the_token_budget_cover_every_sequence_once_and_follow_the_seed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Training converges on the CPU: a 60-step micro-run has finite falling loss; four episodes are memorised until greedy decoding through the saved checkpoint reproduces every gold action line and stops on its own"
    requirement: SCRATCH-04
    verification:
      - kind: unit
        ref: "tests/test_scratch_train.py#test_a_micro_run_has_a_finite_falling_loss"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_four_episodes_can_be_memorised_and_the_saved_checkpoint_regenerates_every_gold_line"
        status: pass
    human_judgment: false
  - id: D4
    description: "Dev selection and metrics: dev loss, line exact overall, per class and first action; stop reasons max_steps, early_stop, target_loss; run notes stored in order"
    requirement: SCRATCH-04
    verification:
      - kind: unit
        ref: "tests/test_scratch_train.py#test_evaluate_reports_loss_and_line_exact_overall_per_class_and_for_the_first_action"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_a_run_whose_dev_loss_stops_improving_ends_with_early_stop"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_the_command_line_trains_and_prints_the_lines_a_background_run_is_polled_for"
        status: pass
    human_judgment: false
  - id: D5
    description: "Failure paths: leaking train file stops before any checkpoint; paths outside the repository and missing inputs are argparse errors; a non-finite loss exits 3 with a FAILED line, a log event and no manifest"
    requirement: SCRATCH-04
    verification:
      - kind: unit
        ref: "tests/test_scratch_train.py#test_a_leaking_training_file_stops_training_before_any_checkpoint"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_the_command_line_refuses_paths_outside_the_repository_and_missing_inputs"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_train.py#test_a_non_finite_loss_fails_the_run_with_exit_code_3_and_no_final_manifest"
        status: pass
    human_judgment: false
  - id: D6
    description: "Load-time stamp enforcement (SCRATCH-03): each of format_version, vocab_sha256, render_fingerprint, context_chars, gen_reserve, an extra key and a missing key is refused by name; the runner prints [skip] and writes no row"
    requirement: SCRATCH-03
    verification:
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_each_changed_stamp_is_refused_and_named"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_an_extra_or_a_missing_checkpoint_key_is_refused_by_name"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_the_runner_skips_a_refused_checkpoint_and_writes_no_row_for_it"
        status: pass
    human_judgment: false
  - id: D7
    description: "Provider and benchmark integration (SCRATCH-05): a CPU-trained tiny model drives a real benchmark episode with every provenance stamp filled; runner rows carry config_model and resolved_model; generation is capped at 320 characters; errors, empty and garbage output come back as values; inference imports neither dataset nor benchmark modules"
    requirement: SCRATCH-05
    verification:
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_a_tiny_model_drives_a_real_benchmark_episode_end_to_end"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_runner_rows_of_a_good_checkpoint_carry_the_scratch_stamps"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_generation_is_capped_by_max_tokens_and_by_the_hard_limit"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_empty_and_garbage_replies_become_parse_errors_the_loop_reports_back"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_inference.py#test_inference_imports_neither_the_dataset_nor_the_benchmark"
        status: pass
    human_judgment: false
  - id: D8
    description: "Whether the real 5M-parameter run reaches a useful dev loss and line-exact rate on the GPU is only observable after the plan 02-04 run"
    requirement: SCRATCH-04
    verification: []
    human_judgment: true
    rationale: "No GPU training was run in this plan; CPU tests prove the machinery, not the quality of the final model"

duration: 173min
completed: 2026-10-21
status: complete
---

# Phase 2 Plan 3: ScratchGPT training script, stamped checkpoint contract and provider stamps Summary

**Masked next-character training (AdamW, warmup + cosine, bf16 on CUDA, dev-selected checkpoint, fixed manifest, exit-code-3 divergence guard) plus an inference loader that refuses any checkpoint whose key set or format stamps differ from the running code, and a provider that stamps finish_reason, token counts and `scratch-gpt@<sha12>` on every step; all proven on the CPU with a 2-layer model.**

## Performance

- **Duration:** about 173 min wall clock from 04:56Z to 07:49Z, which includes an API rate-limit interruption after Task 1 (active work was a fraction of that)
- **Tasks:** 3 (Task 1 tracer, Tasks 2 and 3 TDD), 3 commits
- **Files:** 7 (4 created, 3 modified)
- **CPU test runtimes (WSL, cold torch import about 50 s per pytest process):** `tests/test_scratch_inference.py` 21 tests in about 120 s together with test_llm_layer and test_provenance (the module-scoped 20-step training is a few seconds); `tests/test_scratch_train.py` 15 tests in 96 s (micro-run 36 s including first matplotlib and torch warm-up, overfit 13 s, the rest under 6 s each)
- **Full WSL suite:** 854 passed in 295 s (baseline 818 + 15 in test_scratch_train.py + 21 in test_scratch_inference.py)

## Final contracts (plan 02-04 and Phase 4 depend on these)

**CHECKPOINT_KEYS** (exact set, enforced both ways): `state_dict`, `cfg`, `format_version`, `vocab_sha256`, `render_fingerprint`, `context_chars`, `gen_reserve`, `train_meta`. `train_meta` holds primitives only: `seed`, `step`, `dev_loss`, `dev_line_exact`, `tokens_seen`.

**MANIFEST_KEYS** (fixed order): `schema, created_at, code_commit, code_dirty, python, torch, cuda, device, seed, dataset, held_out_sha256, format_version, vocab_sha256, render_fingerprint, context_chars, gen_reserve, config, params, steps, best_step, tokens_seen, final, checkpoint_sha256, wall_seconds, stop_reason, run_notes`. `dataset` = `train_path, train_sha256, dev_path, dev_sha256, n_train, n_dev, class_counts, synthetic_fraction` (train split values, repo-relative forward-slash paths). `final` = `train_loss, dev_loss, dev_line_exact, dev_line_exact_by_cls, dev_first_line_exact`. `device` is `cpu` or `cuda`.

**Progress lines on stdout** (flushed; poll these):

```
device <cpu|cuda> params <n>
step <s>/<max> loss <x> lr <y> tok/s <z> elapsed <t>s        (every --log-every steps; loss = mean of the interval)
eval step <s> dev_loss <x> dev_line_exact <y> best <z>       (every --eval-every steps and at the end)
DONE best_step <s> best_dev_loss <x> wall <t>s stop <max_steps|early_stop|target_loss>
FAILED non_finite_loss step <s>                              (exit code 3; no DONE, no manifest)
```

Exit codes: 0 success, 3 non-finite loss, 2 argparse errors (including a path outside the repository or a missing input). `--run-note TEXT` is repeatable (ASCII, one line, at most 200 characters) and stored in order in `run_notes`. `train_log.jsonl` has one JSON object per evaluation (`step, train_loss, dev_loss, dev_line_exact, dev_line_exact_by_cls, dev_first_line_exact, lr, tokens_seen, elapsed_s`), plus a final `{"step": s, "event": "non_finite_loss"}` line on divergence. `tok/s` counts padded positions, so it is slightly above the useful rate.

## Accomplishments

- **Tracer (Task 1):** a 20-step CPU-trained model is saved with all stamps, loaded by `load_generator`, wrapped by `LocalScratchProvider`, and drives `run_episode` on a real benign benchmark case with no `llm_error`. Gate re-run passed (7 tests), so expansion continued.
- **D-06 mask:** the test decodes `y[m == 1]` for four episodes and gets exactly `"".join(compact_action(a) + "\n")` over every action of the episode, which catches the off-by-one of research pitfall 6.
- **Memorisation proof:** four episodes (one shortest per class plus a second benign) train to the 0.02 target loss in under 1,500 steps with a d_model-64 model (`stop_reason` `target_loss`, final `dev_line_exact` 1.0); `load_generator` on the saved checkpoint then regenerates every gold `A:` line of every step from its gold prefix and stops on its own newline.
- **Line exact match without generating:** one batched forward pass computes the metric via run starts and a scatter-add over run ids (overall, per class, first action).
- **D-05 enforcement:** five stamps, cfg context and vocab size, the exact key set and the cfg/weights fit are all refused with a `RuntimeError` naming the field; `weights_only=True` blocks pickle code execution (T-02-11). The runner test shows `[skip] scratch-tampered: ...` and no row; the good-checkpoint runner test shows `config_model`, `resolved_model` (`scratch-tiny@<sha12>`), `finish_reasons` and token counts in the row.

## Task Commits

1. **Task 1 (tracer): CPU-trained tiny ScratchGPT drives a benchmark episode end to end** - `c990013`
2. **Task 2: token-budget training loop, dev line-exact metrics, early stopping, fixed manifest, CLI** - `9d537f6`
3. **Task 3: checkpoint stamp guards, runner skip, generation cap, scratch-gpt config stamp** - `3dc735f`

**Plan metadata:** committed separately (docs: complete plan).

## Files Created/Modified

- `src/scratch_llm/train.py` - TrainConfig, lr_at, make_batches, collate, evaluate, train, NonFiniteLossError, MANIFEST_KEYS, CLI
- `src/scratch_llm/inference.py` - CHECKPOINT_KEYS, current_stamps, check_stamps, ScratchGenerator, load_generator, MAX_NEW_CHARS
- `src/llm/scratch.py` - provider docstring corrected, stamps filled via getattr, lazy import comment
- `config/models.yaml` - `model: scratch-gpt` plus a comment on how the git-ignored checkpoint is produced
- `requirements.txt` - torch comment now names Phase 2, torch>=2.6 and the cu128 index
- `tests/test_scratch_train.py` - 15 tests; `tests/test_scratch_inference.py` - 21 test runs (17 test functions)

## Decisions Made

- The finiteness check sits right after the forward pass, before backward and `optimizer.step()`, so a bad step cannot corrupt weights; the reported step is the 1-based number of the failing step.
- `target_loss` is evaluated only at log boundaries on the mean train loss of the last log interval; the final evaluation and best-checkpoint logic still run, so a `target_loss` stop always leaves a checkpoint.
- The CLI keeps the default `min_lr` unless `--lr` is smaller, in which case `min_lr = --lr`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Checkpoint whose cfg or weights do not fit the model code**
- **Found during:** Task 3 (writing the tamper tests)
- **Issue:** `GPTConfig(**cfg)` with an unknown field raises `TypeError`, and a differently shaped state dict raises a raw error; the runner's `[skip]` path catches any exception, but the message would not say the checkpoint is the cause.
- **Fix:** both are wrapped in a `RuntimeError("checkpoint cfg and weights do not fit the running model code: ...")`.
- **Files modified:** src/scratch_llm/inference.py
- **Commit:** 3dc735f

**2. [Plan adjustment] Non-finite check before the optimiser step**
- **Reason:** the plan says "after every optimisation step"; checking immediately after the forward pass catches the same step (the test expects step 3) without applying a NaN update. Exit code, FAILED line, log event and missing manifest are as specified.
- **Commit:** 9d537f6

**3. [Plan adjustment] Task 1 implemented a simpler train loop first**
- **Reason:** as the plan prescribes, Task 1 shipped a fixed-size batch iterator and dev-loss-only evaluation behind the final `train()` signature; Task 2 replaced them. `encode_episode` gained a fourth tuple element (the class) in Task 2 for the per-class metric.
- **Commits:** c990013, 9d537f6

**4. [Rule 1 - Bug, test authoring] Shell escaping corrupted three test lines**
- **Found during:** Task 3 (collection error `unterminated string literal`)
- **Issue:** appending the new tests through a Python heredoc turned `"\n"` into real newlines in two places.
- **Fix:** re-edited those lines with the Edit tool; the file now collects and passes.
- **Commit:** 3dc735f (never committed in the broken state)

**Total deviations:** 1 auto-fixed (missing critical), 2 plan adjustments, 1 authoring slip fixed before commit. **Impact:** none on scope or on the interfaces plan 02-04 consumes.

### Process note (TDD)

Red was established by the missing modules (`ModuleNotFoundError` for `inference` and `train` before Task 1; the Task 2 test file imports `make_batches`, `evaluate` and `NonFiniteLossError`, which did not exist). The full test files were not run red before implementation because a cold torch import costs about a minute; every test then passed on its first green run, so there was no red-to-green fix cycle to observe for individual assertions.

## Known Stubs

None. No placeholder, TODO or hardcoded empty data flows to a consumer in the new files.

## Limitations to carry forward

- No GPU run happened here: the bf16 autocast, fused AdamW and CUDA device path in `train.py` are exercised only by code reading, not by a test (CPU tests cannot reach them). Plan 02-04 should treat its first minutes as the test of that path, using `--run-note` to record any relaunch.
- `tok/s` includes padding positions.
- A line counts as exact only if every character matches the gold line given the gold prefix; this equals greedy decoding but ignores the interaction of an early wrong character with later context (that is what the real benchmark measures).
- The held-out/benchmark list is stamped only as `held_out_sha256`; checkpoints do not carry the dataset hashes (the manifest does).

## Issues Encountered

- API rate limit interrupted the run after Task 1; resumed from the committed state with no loss.

## Authentication Gates

None.

## Threat Flags

None. The new surface (checkpoint deserialisation, CLI paths, generation loop) is all covered by the plan's threat model; mitigations T-02-11 to T-02-17 and T-02-23 are implemented and tested as listed in the coverage block.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-04 can run `python -m src.scratch_llm.train` on the GPU (defaults: `data/local/train-seed0-n40000.jsonl`, `data/local/dev-seed0-n2000.jsonl`, `models/scratch`), check the dataset hashes in `data/local/manifest.json` first, run `ollama ps` before starting, and poll the progress lines above; exit code 3 means divergence (retry with a lower `--lr`, recorded via `--run-note`).
- `config/models.yaml` already points `scratch-gpt` at `models/scratch/scratch_gpt.pt`; once that file exists the registry builds the provider.

## Self-Check: PASSED

- Files present: `src/scratch_llm/train.py`, `src/scratch_llm/inference.py`, `src/llm/scratch.py`, `config/models.yaml`, `requirements.txt`, `tests/test_scratch_train.py`, `tests/test_scratch_inference.py`.
- Commits `c990013`, `9d537f6`, `3dc735f` are ancestors of HEAD; `git rev-list --count de9f494..HEAD` was 3 before this summary commit.
- Acceptance criteria re-run: `weights_only=True` 2 in inference.py; `resolved_model` in scratch.py 1, `Phase 4` 0; test counts 17 (inference, needs 14) and 15 (train, needs 12); `bfloat16` 1, `assert_clean` 3, `MANIFEST_KEYS` 2, `NonFiniteLossError` 4 in train.py; `model: scratch-gpt` 1, `cu128` 2 and `cu124` 0.
- Plan verification: targeted files 22 passed (train + inference after Task 2), 54 passed with test_llm_layer and test_provenance after Task 3; full suite 854 passed. No file under `models/` or `data/local/*.jsonl` was staged.

---
*Phase: 02-local-data-foundation-scratchgpt*
*Completed: 2026-10-21*
