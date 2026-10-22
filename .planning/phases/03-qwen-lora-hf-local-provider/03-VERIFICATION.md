---
phase: 03-qwen-lora-hf-local-provider
verified: 2026-10-22T21:00:00Z
status: human_needed
score: 5/5 must-haves verified
covered_files:
  - ".gitignore"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-01-PLAN.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-01-SUMMARY.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-02-PLAN.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-02-SUMMARY.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-03-PLAN.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-03-SUMMARY.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-04-PLAN.md"
  - ".planning/phases/03-qwen-lora-hf-local-provider/03-04-SUMMARY.md"
  - "config/models.yaml"
  - "docs/viva/03-fine-tuning-lora-and-chat-templates.md"
  - "docs/viva/README.md"
  - "models/qwen-agentlora/adapter_config.json"
  - "models/qwen-agentlora/manifest.json"
  - "models/qwen-agentlora/train_log.jsonl"
  - "requirements-train.txt"
  - "src/benchmark/runner.py"
  - "src/llm/hf_local.py"
  - "src/llm/registry.py"
  - "src/qwen_lora/__init__.py"
  - "src/qwen_lora/chat.py"
  - "src/qwen_lora/checks.py"
  - "src/qwen_lora/data.py"
  - "src/qwen_lora/run_train.sh"
  - "src/qwen_lora/train.py"
  - "tests/conftest.py"
  - "tests/qwen_fixtures.py"
  - "tests/test_hf_local_provider.py"
  - "tests/test_qwen_lora_checks.py"
  - "tests/test_qwen_lora_data.py"
  - "tests/test_qwen_lora_train.py"
  - "tests/test_rate_limit_detection.py"
covered_digest: "v3:sha256:1e166a5188bce4d81935182384c372e24609cd27ad758e418c0f9c39f7f9d4ed"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Read docs/viva/03-fine-tuning-lora-and-chat-templates.md (Part A, B, C, D) as a first-time reader"
    expected: "Every term is defined before use, the Part C numbers read naturally, and the 16+ Q&A answers are each speakable in under a minute. Part C line ~341 still names results/phase3-smoke-lora (the pre-fix directory) in its source sentence; the v2 re-run is explained later on the page (cosmetic)."
    why_human: "Readability and teaching quality cannot be checked by grep. checks viva only proves structure, verbatim manifest quotes, README row and the secret scan (part-c-ok)."
  - test: "Open models/qwen-agentlora/loss_curve.png and read it against the viva description"
    expected: "Train loss falls 0.151 -> 0.045, dev action loss falls and flattens, dev all-token loss drifts up (respond text is unpredictable, held-back wording) and the page explains that rise honestly"
    why_human: "Visual reading. I viewed the image: three series, step 100..600, best-step marker at 600, matches the manifest numbers; whether it is presentable is a human call."
  - test: "unverified-prohibition, human review recommended: 03-01 prohibition 'hf_local must never repair, re-format, constrain or post-process a reply beyond stripping whitespace'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. generate() returns tokenizer.decode(...).strip() only (src/llm/hf_local.py:326); test_raw_text_prose_around_json_is_returned_verbatim_only_stripped passes. decode uses skip_special_tokens=True, which silently drops special-token strings from the reply; the reviewer should confirm that is acceptable."
    why_human: "Judgment-tier prohibition, plan status unverified / flagged. Never a silent pass."
  - test: "unverified-prohibition, human review recommended: 03-01 prohibition 'No code path builds a Qwen prompt by hand, omits the project system prompt, or passes tools to apply_chat_template'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. grep for tools= in src/llm/hf_local.py and src/qwen_lora/*.py finds nothing; prompts come from build_messages through tokenizer.apply_chat_template (chat.py, data.py, hf_local.py); replay-parity and system-prompt-first tests pass."
    why_human: "Judgment-tier prohibition, plan status unverified / flagged."
  - test: "unverified-prohibition, human review recommended: 03-03 and 03-04 prohibition 'No hyper-parameter, step budget, data subset, code path or retraining decision chosen from benchmark or smoke outcomes'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS on the evidence available. Adapter weights and config mtime 15:09:39 precede the base smoke (15:22) and first LoRA smoke (15:25); the weights hash 270501a67998 is unchanged afterwards (WR-04 and WR-05 touched only manifest.json); the remediation rule is mechanical, read dev_action_exact 0.9396 >= 0.90 and did not fire; no retrain occurred. The decision history between commits cannot be proven from files alone."
    why_human: "Judgment-tier prohibition about process order, plan status unverified / flagged."
  - test: "unverified-prohibition, human review recommended: 03-04 prohibition 'The tuned model must never be presented as safety-tuned or as evidence that fine-tuning broke safety'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. The viva page opens by saying the tuned Qwen is 'not a safe or safety-tuned model' but a second injection-compliant worst-case principal (D-01), repeats this in Limits of this phase, and config/models.yaml carries the same disclosure."
    why_human: "Judgment-tier prohibition about wording, plan status unverified / flagged."
  - test: "Review-fix items WR-01, WR-02, WR-03 were marked 'fixed: requires human verification' in 03-REVIEW-FIX.md (rate-limit predicate, run_train.sh stale marker, staged adapter promotion)"
    expected: "Spot read of src/benchmark/runner.py _is_rate_limited, src/qwen_lora/run_train.sh launch (rm -f train.done before preflight) and train.py _promote_run. Automated evidence: 118 tests in test_qwen_lora_train/checks/test_rate_limit_detection and 131 runner/report tests all pass."
    why_human: "Logic and shell control flow whose tests pin the intended behaviour only; the fixer asked for a human read."
---

# Phase 3: Qwen LoRA & hf_local Provider Verification Report

**Phase Goal:** A pretrained small open model that the student fine-tuned locally with LoRA, and its un-tuned base as a control, both run through the benchmark behind the same provider interface as every other model.
**Verified:** 2026-10-22
**Status:** human_needed (all automated truths verified; five flagged judgment-tier prohibitions and the viva and loss-curve review remain for a human)
**Re-verification:** No, initial verification

The goal is achieved in the codebase. Nothing below is taken from SUMMARY.md: I read the code, re-ran the tests in WSL, re-ran the repository's own gate commands against the kept adapter and re-computed the two smoke checks and the decoded-supervised-tokens property myself. The current state was verified, including the superseding WR-05 stamp (`@924ac021f0cb`) and `results/phase3-smoke-lora-v2`.

## Goal Achievement

### Observable Truths (ROADMAP success criteria, the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `hf_local` provider serves a HF causal LM in-process (optionally with a PEFT adapter) behind `LLMProvider`, greedy, project system prompt always supplied; contract test with a fake model passes; a missing adapter makes the runner skip, not crash | VERIFIED | `HFLocalProvider(LLMProvider)` in `src/llm/hf_local.py:273`; registry branch `src/llm/registry.py:48-60` (lazy import). Explicit `GREEDY_KWARGS` (do_sample False, temperature/top_p/top_k None, repetition_penalty 1.0) passed on every `generate` call (line 320) and a fresh `GenerationConfig` at load. Stop ids `<|im_end|>`/`<|endoftext|>`. Fake-model contract tests (`test_contract_*`, `test_greedy_*`) pass. Runner skip: `src/benchmark/runner.py:329-333` logs `[skip]`; `test_runner_skips_a_missing_adapter_and_writes_no_row` and `test_runner_offline_uncached_base_is_skipped_with_no_row_and_no_connection` pass. `66 passed, 1 skipped` (the skip is the opt-in `QWEN_LIVE` probe). |
| 2 | Un-tuned Qwen2.5-0.5B-Instruct runs end to end through the benchmark on a smoke run as the base control row | VERIFIED | `results/phase3-smoke-base/raw.jsonl`: 26 rows, all `provider hf_local`, `config_model qwen2.5-0.5b-base`, `resolved_model Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf`, 0 `llm_error`, 40-hex `code_commit`. `checks smoke --expect-rows 26` printed `smoke-plumbing-ok` (my own run). Parse rate 48/105; reported, not graded (a finding). Config entry `qwen2.5-0.5b-base` with `adapter: null`. |
| 3 | Custom LoRA loop fine-tunes Qwen2.5-0.5B on the Phase 2 episodes within 6 GB VRAM, writes adapter plus manifest; a decoded-supervised-tokens test shows only assistant action tokens are supervised | VERIFIED | `models/qwen-agentlora/` holds `adapter_model.safetensors` (35,237,104 bytes, git-ignored by `.gitignore:56`), `adapter_config.json`, `manifest.json`, `train_log.jsonl`, `loss_curve.png` (the last four git-tracked). Manifest: r 16, alpha 32, seven target modules, 8,798,208 / 502,830,976 trainable (1.75 %), 600 steps, device cuda, peak VRAM 2,317 MiB allocated / 2,584 MiB reserved of 6,143. `checks artifacts --require-cuda` printed `artifacts-ok` (my run). Decoded-supervised-tokens: `test_supervised_tokens_decode_to_the_gold_actions_*` pass on generated episodes and on the first 300 episodes of each Phase 2 file; I independently built 200 real dev episodes with `build_example` and decoded the supervised ids: 0 mismatches against `gold_supervised_text`, supervised fraction 0.149. Mask is by character offset (`data.py:53-91`), not by token id; test `..._decided_by_offsets_not_token_ids` passes. |
| 4 | LoRA-tuned Qwen runs end to end through the benchmark and produces valid actions on a smoke run | VERIFIED | Current evidence `results/phase3-smoke-lora-v2`: 26 rows, `config_model qwen2.5-0.5b-agentlora`, `resolved_model ...@7ae557604adf+qwen2.5-0.5b-agentlora@924ac021f0cb` on all 26, 0 `llm_error`; `smoke-plumbing-ok`. Valid actions 77/77, parse 77/77. Pre-registered LORA-04 dev gate `gen_check.tuned.parse_rate 1.0000 >= 0.90` on 100 dev first-turn prompts (`checks artifacts`: `LORA04_GATE pass`). Base on the same prompts: parse 0.69. Old `results/phase3-smoke-lora` (stamp `@270501a67998`) is the superseded pre-fix run; same behaviour numbers. |
| 5 | Viva page(s) for fine-tuning and LoRA, chat templates and loss masking, and decoding added to `docs/viva/` | VERIFIED (readability: human) | `docs/viva/03-fine-tuning-lora-and-chat-templates.md`, 608 lines, Parts A-D plus "Limits of this phase"; `docs/viva/README.md` row 03 marked "written (Phase 3)". `checks viva` printed `part-c-ok` (manifest values quoted verbatim, current identity prefix `924ac021f0cb` quoted). Quality of explanation is in the human list. |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified; the state-transition and cleanup invariants have passing tests, see below).

### Plan-level must-haves (merged, none reduced)

| Plan truth | Status | Evidence |
|-----------|--------|----------|
| 03-01 offline, lazy imports, `local_files_only` | VERIFIED | `_offline_environment()` before HF import, every `from_pretrained`/`snapshot_download` passes `local_files_only=True`; `test_isolation_light_modules_import_none_of_the_heavy_stack`, `test_offline_uncached_base_model_is_refused_without_a_network_connection` pass. (IN-04: `setdefault` yields to an explicit `HF_HUB_OFFLINE=0`; calls still pass `local_files_only`, so no network path. Info.) |
| 03-01 stamps and context overflow | VERIFIED | `finish_reason` stop/length, `prompt_tokens`, `completion_tokens`, `resolved_model = base@rev12 (+name@identity12)`; overflow returns an error response without calling the model; `test_overflow_*`, `test_length_finish_reason_*` pass. |
| 03-01 stale or missing adapter refused at construction | VERIFIED | `verify_adapter` (`hf_local.py:170-226`) checks files, base id, revision, weights hash, config hash, identity hash and render fingerprint; `test_stale_adapter_is_refused_and_the_field_is_named`, `test_wr05_*`, `test_missing_adapter_*` pass. |
| 03-01 merge is output-preserving | VERIFIED | `merge_and_unload` at load; tiny-model merged/unmerged identical on three prompts and non-zero adapter changes logits (tests pass). Review measured 0/50 first-action differences on the real adapter (IN-03: approximation, not bit-identity; docstring wording only). |
| 03-01 config, pins, STRICT_HF | VERIFIED | `config/models.yaml:74-91` both entries; `requirements-train.txt` pins `transformers>=5.20,<5.21`, `peft==0.21.2`, `accelerate>=1.15,<2`; `tests/conftest.py:50` STRICT_HF. |
| 03-02 prefix stability, over-long error, special-token guard | VERIFIED | `test_prefix_every_step_prompt_is_a_token_prefix...`, `test_supervised_max_length_...is_an_error_and_is_never_truncated`, `test_special_token_strings_in_text_are_rejected...`, `test_special_token_strings_are_absent_from_the_benchmark_files` pass. |
| 03-02 loss equals HF labels loss; param count 8,798,208; every LoRA param gets a gradient under checkpointing | VERIFIED | `test_loss_equivalence_*`, `test_param_count_for_the_real_qwen_dimensions_is_8798208`, `test_checkpointing_every_lora_parameter_gets_a_finite_gradient...` pass (35+ train tests). |
| 03-02 best-by-`dev_loss_action`, patience | VERIFIED | `test_selection_keeps_the_lowest_dev_loss_action_and_stops_after_patience`, `..._runs_to_max_steps_...` pass. |
| 03-02 tracer: tiny adapter served through the runner | VERIFIED | `test_tracer_tiny_lora_adapter_is_served_in_a_benchmark_episode` passes. |
| 03-03 resume, fail-closed exits 3/4, staged promotion (behavior-dependent) | VERIFIED | `test_resume_continues_at_the_next_step_and_equals_an_uninterrupted_run`, `test_resume_refuses_*`, `test_non_finite_*_exits_3_*`, `test_vram_guard_exits_4_*`, `test_cuda_out_of_memory_exits_4_*`, `test_wr03_*` all pass: named behavioural tests exercise each transition, so none is left PRESENT_BEHAVIOR_UNVERIFIED. |
| 03-03 pre-registered remediation and LORA-04 gate as constants applied mechanically from dev metrics | VERIFIED | `REMEDIATION_RULE`/`LORA04_GATE_RULE` in `train.py`, copied into the manifest `selection` block; `test_remediation_*` pass. Fired? No: `final_dev_full.dev_action_exact 0.9396 >= 0.90`, no retry, converged-below-bar case not reached. |
| 03-03 launcher and checks commands | VERIFIED | `run_train.sh` (185 lines, tracked); 49 checks/launcher tests pass including the WR-02 stale `train.done` regression tests. |
| 03-04 trained, artifacts committed, remediation and gate recorded before smoke | VERIFIED | Weights/config mtime 15:09:39 precede smoke runs (15:22, 15:25); `git ls-files models` lists manifest, adapter_config, train_log, loss_curve; commit b68a707. |
| 03-04 `resolved_model` stamp ends `+qwen2.5-0.5b-agentlora@<adapter_sha256[:12]>` | VERIFIED (superseded) | The plan named the weights prefix `270501a67998`. WR-05 (commit 64c0c40) deliberately replaced the suffix with the identity hash `924ac021f0cb` (weights + canonical config), which meets the same property (stamp binds the served adapter to the row) more strongly. Current rows carry `@924ac021f0cb`; manifest `adapter_identity_sha256` begins with it. Treated as satisfied per the caller's instruction. |
| 03-04 viva page, 16+ questions, README row | VERIFIED | `checks viva` -> `part-c-ok`; page 608 lines (plan minimum 300); README row present. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llm/hf_local.py` | provider, load, verify, merge | VERIFIED | 338 lines, substantive, wired from registry |
| `src/qwen_lora/chat.py` | shared render contract | VERIFIED | 87 lines; used by `data.py`, `hf_local.verify_adapter`, tests |
| `src/qwen_lora/data.py` | loss mask, examples, collate | VERIFIED | 140 lines; used by `train.py` |
| `src/qwen_lora/train.py` | trainer, manifest, resume, gates | VERIFIED | 1,023 lines |
| `src/qwen_lora/checks.py`, `run_train.sh` | gates and launcher | VERIFIED | 397 and 185 lines; executed above |
| `src/llm/registry.py`, `config/models.yaml` | hf_local branch, two rows | VERIFIED | present and exercised by registry tests and real smoke runs |
| `requirements-train.txt` | pinned stack | VERIFIED | contains `peft==0.21.2` |
| `models/qwen-agentlora/*` | adapter, config, manifest, log, curve | VERIFIED | weights on disk and ignored; four small files tracked |
| `docs/viva/03-*.md`, `docs/viva/README.md` | viva page, index row | VERIFIED | see truth 5 |
| `tests/test_hf_local_provider.py`, `tests/test_qwen_lora_{data,train,checks}.py`, `tests/qwen_fixtures.py`, `tests/conftest.py` | test suites | VERIFIED | 66+1 skip, 45, 49 pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `registry.get_provider` | `HFLocalProvider` | lazy import in the `hf_local` branch | WIRED | `registry.py:48-60`; real smoke rows prove it end to end |
| `verify_adapter` | `chat.render_fingerprint` | manifest `render.fingerprint` compared to running code | WIRED | `hf_local.py:221-225`; stale-fingerprint test passes |
| `AgentLoop` messages | `episode_to_chat` | replay parity test | WIRED | `test_replay_a_real_agent_loop_sends_exactly_the_episode_chat_prefixes` and the broker-denial variant pass |
| `data.build_example` | `chat.episode_to_chat` | one episode-to-messages function | WIRED | `data.py:101` |
| `train()` | `assert_clean` | leakage gate before step 0 | WIRED | `train.py:672-673` |
| `train gen_check` | `HFLocalProvider` | finished adapter loaded through stamp checks | WIRED | manifest `gen_check` populated (tuned 1.0 / base 0.69) |
| `run_benchmark` | provider stamps | RuntimeError -> `[skip]`; `resolved_model`, `finish_reasons`, token counts on rows | WIRED | `runner.py:329-333`; raw rows carry all stamps |
| `checks smoke` | `results/*/raw.jsonl` | exact `resolved_model` | WIRED | run for both rows |
| manifest dataset hashes | `data/local/manifest.json` | `checks artifacts --data-manifest` | WIRED | `artifacts-ok` |
| viva Part C | manifest | verbatim quotes | WIRED | `part-c-ok` |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| smoke `raw.jsonl` rows | `final_message`, `actions`, `finish_reasons`, `prompt_tokens` | live `model.generate` on cached Qwen base (+ merged adapter) through the runner | Yes (non-trivial attack and parse counts, differing between base and tuned) | FLOWING |
| manifest `final_dev_full`, `gen_check` | dev metrics | computed from the real adapter on the real dev file; WR-04 values reproduced pre-registered ones bit for bit | Yes | FLOWING |
| viva Part C | numbers | quoted from manifest and smoke summaries | Yes, `part-c-ok` | FLOWING |

### Behavioral Spot-Checks (my own runs, WSL)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| provider + data tests | `pytest tests/test_hf_local_provider.py tests/test_qwen_lora_data.py` | 66 passed, 1 skipped | PASS |
| trainer + checks + rate-limit tests | `pytest tests/test_qwen_lora_train.py tests/test_qwen_lora_checks.py tests/test_rate_limit_detection.py` | 118 passed | PASS |
| runner / report regression (WR-01 touched `runner.py`) | `pytest tests/test_runner_queue.py tests/test_benchmark.py tests/test_report.py tests/test_review_fix_benchmark.py` | 131 passed | PASS |
| kept adapter integrity, budget, gates | `checks artifacts --out-dir models/qwen-agentlora --data-manifest data/local/manifest.json --require-cuda` | `artifacts-ok`; REMEDIATION no; LORA04_GATE pass | PASS |
| base smoke plumbing | `checks smoke --results results/phase3-smoke-base --model qwen2.5-0.5b-base --expect-rows 26` | `smoke-plumbing-ok` | PASS |
| tuned smoke plumbing (current v2) | `checks smoke --results results/phase3-smoke-lora-v2 --model qwen2.5-0.5b-agentlora --expect-rows 26` | `smoke-plumbing-ok` | PASS |
| viva structure | `checks viva --page docs/viva/03-... --manifest ... --readme ...` | `part-c-ok` | PASS |
| supervised tokens = gold actions (independent) | script: `build_example` on 200 real dev episodes, decode supervised ids | 200 episodes, 0 mismatch | PASS |

I did not re-run the Phase 1-2 remainder of the 1,085-test suite (about 900 tests unrelated to Phase 3); the 184 Qwen tests and the 131 runner/report tests that the review-fix touched were run. The caller recorded 1,085 passed + 1 skipped for the full suite.

### Probe Execution

SKIPPED. No `probe-*.sh` scripts exist or are declared by the Phase 3 plans; the phase's own gate commands (checks artifacts/smoke/viva) are the equivalent and were run above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| LORA-01 | 03-01 | `hf_local` provider behind `LLMProvider`, optional PEFT adapter, greedy, system prompt always supplied | SATISFIED | truth 1; `hf_local.py`, registry, tests |
| LORA-02 | 03-01, 03-04 | Un-tuned Qwen2.5-0.5B-Instruct benchmarked as base control row | SATISFIED | truth 2; `results/phase3-smoke-base`, 26 rows |
| LORA-03 | 03-02, 03-03, 03-04 | Custom LoRA loop on same synthetic episodes within 6 GB, assistant action tokens only (decoded-tokens test), adapter + manifest | SATISFIED | truth 3; 2,317 / 2,584 MiB peak, decoded-token tests plus my own 200-episode check |
| LORA-04 | 03-03, 03-04 | LoRA-tuned Qwen runs end to end in the benchmark, valid actions on a smoke run | SATISFIED | truth 4; 77/77 valid actions, dev gate pass |

No orphaned requirements: REQUIREMENTS.md maps exactly LORA-01..04 to Phase 3 and all four appear in the plans' `requirements:` fields.

### Anti-Patterns Found

None blocking. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK` in any Phase 3 source, test or config file. No stub returns. `git diff HEAD` over Phase 3 paths shows only CRLF warnings (working tree equals HEAD).

| Item | Severity | Note |
|------|----------|------|
| Info findings IN-01..IN-08 from 03-REVIEW.md deliberately out of scope and unfixed | Info | IN-01 (control-token guard covers 3 of 22 added tokens, serving does not flag them); IN-02 (`code_dirty: true`, `code_commit 75bedd3e` in the manifest); IN-03; IN-04; IN-05 (D-06 says "last two evaluations", code window is `steps - 2 x eval_every`, i.e. three evaluation points, identical to Phase 2; did not matter, rule did not fire); IN-06..IN-08. None blocks a roadmap success criterion. Worth carrying into Phase 4 because IN-01 and IN-05 touch full-run claims. |
| `results/` is untracked (not ignored, not committed) | Info | The smoke-run evidence (`phase3-smoke-base`, `-lora`, `-lora-v2`) exists on disk but not in git; the user's working-tree decision. Re-generating needs the GPU. |
| Viva Part C line ~341 cites `results/phase3-smoke-lora` as the smoke source | Info | Old pre-fix directory; the v2 re-run and the stamp change are documented later on the same page (lines ~394 and ~449). Cosmetic. |
| Mixed evaluation overlap | Info | Final dev numbers are dev numbers that overlap the selection subset (300 of 2,000; 16 of the 100 gen-check prompts); disclosed in the manifest `evaluation_overlap` and by `checks artifacts`. Not a benchmark leak. |

### Honest limits (not gaps)

- Per-class `ds` action exactness is 0.8392 (non-respond 0.7841) and is below the 0.90 bar; the aggregate pre-registered gate (0.9396) is what the rule reads and was not changed. This is disclosed (`dev_min_class_action_exact`, `gate_context`) and is the intended outcome of WR-04 (report, do not re-gate).
- Smoke runs are plumbing and behaviour evidence only (n = 26 rows each); the base parse rate of 48/105 and the tuned model's 10/10 attack attempts are reported findings, not scores. Full 342-episode runs are Phase 4.

### Gaps Summary

No gaps. All five ROADMAP success criteria, all four requirement IDs and every plan must-have are verified against the code, the on-disk adapter and my own re-runs. Status is `human_needed` rather than `passed` only because (a) five judgment-tier prohibitions are carried by the plans as `status: unverified, flagged: true` (my non-authoritative verdict is that each holds, with evidence above), and (b) the viva page readability, the loss-curve review and the three review-fix items that the fixer marked "requires human verification" need a person. Nothing in the automated evidence is failing.

---

_Verified: 2026-10-22_
_Verifier: Claude (gsd-verifier)_
