---
phase: 03-qwen-lora-hf-local-provider
reviewed: 2026-10-22T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - src/llm/hf_local.py
  - src/llm/registry.py
  - config/models.yaml
  - requirements-train.txt
  - .gitignore
  - src/qwen_lora/__init__.py
  - src/qwen_lora/chat.py
  - src/qwen_lora/data.py
  - src/qwen_lora/train.py
  - src/qwen_lora/checks.py
  - src/qwen_lora/run_train.sh
  - tests/conftest.py
  - tests/qwen_fixtures.py
  - tests/test_hf_local_provider.py
  - tests/test_qwen_lora_data.py
  - tests/test_qwen_lora_train.py
  - tests/test_qwen_lora_checks.py
findings:
  critical: 0
  warning: 5
  info: 8
  total: 13
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-10-22
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

I read every listed file and the code they call into (`agent_harness/loop.py`, `prompting.py`, `executors.py`, `scratch_llm/format.py`, `provenance.py`, `benchmark/runner.py`, `benchmark/outcomes.py`). I ran the test files in WSL:

- `test_qwen_lora_data.py` + `test_qwen_lora_checks.py`: 56 passed.
- `test_hf_local_provider.py`: 42 passed, 1 skipped (the opt-in live probe).
- `test_qwen_lora_train.py`: 35 passed.

I also ran three probes against the real adapter in `models/qwen-agentlora`, on the GPU with about 4.6 GB free and no other process touched:

- **Data scan.** I scanned the two Phase 2 files and the benchmark files for lengths and control-token strings.
- **Merge probe.** I compared the merged and unmerged adapters over 50 dev episodes.
- **Tokenizer probe.** I checked how the Qwen tokenizer handles control strings.

**No Critical issue found.** The core numerics hold up.

- **Loss and mask.** The supervised-position LM head shifts correctly: `hidden[:, :-1][labels[:, 1:] != -100]`. The accumulation window is normalised by the window's supervised-token count, which equals the head's row count. The character-offset mask is correct, and I found no way for a pad-id token to hide the stop token.
- **Train/serve parity.** The loop's `MAX_ASSISTANT_CHARS` is 2000 and the longest gold action in 111,607 train spans is 344 chars, so serving never truncates a gold-style history. Training observations are at most 488 chars and the benchmark's longest tool response is about 331, so `Observation.as_text()` truncation (2000) never fires. No Qwen control string occurs in either data file or in the benchmark files. Training and serving share one render path, and the fingerprint covers the system prompt, tool list, observation template and `compact_action`.
- **Merge fidelity.** The merged bf16 adapter matches the unmerged one in all 50 first-action texts. Teacher-forced argmax differs at 1 of 3,481 action positions, with a max logit difference of 1.31 (bf16 rounding). See IN-03.
- **Pre-registered gates.** Remediation did not fire (`final_dev_full.dev_action_exact` 0.9396 against 0.90). The LORA-04 gate passes at `gen_check.tuned.parse_rate` 1.0.
- **Offline.** `HF_HUB_OFFLINE` is set and every call that resolves a model passes `local_files_only=True`. The uncached-base tests assert zero `socket.connect` attempts.

The weaknesses are in operational robustness: one provider-to-runner interaction, the launcher's done-marker, destructive defaults in the trainer, and gate metrics that are weaker than their names suggest.

## Warnings

### WR-01: hf_local error strings can be misread as rate limiting, which sleeps up to about 40 minutes and then aborts the whole model run

**File:** `src/llm/hf_local.py:271-275, 293-296` (consumer: `src/benchmark/runner.py:56-57, 205-220`)
**Issue:** The runner's `_is_rate_limited(error)` returns true if the substring `"429"` appears anywhere in `llm_error`. The provider puts numbers it does not control into those strings: the prompt token count (`prompt of {n_in} tokens ...`) and, for CUDA out-of-memory, PyTorch's message with sizes such as `of which 429.50 MiB is free`.

I confirmed both cases:
- `_is_rate_limited("ContextOverflowError: context_overflow: prompt of 4296 tokens leaves no room in 4096")` returns `True`.
- `_is_rate_limited("OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. ... 429.50 MiB is free.")` returns `True`.

`_run_with_backoff` then sleeps through `RATE_LIMIT_BACKOFF_S = (20, 45, 90, 180, 300, 600, 600, 600)`, which is about 40 minutes of sleeping in total, re-runs the episode, and finally raises `QuotaExhausted`. The model's remaining queue is dropped with "[stop] ... quota exhausted". This is a local GPU on a machine shared with another project, so an out-of-memory episode is a realistic event in the Phase 4 342-episode runs. A random 3-digit match is rare, but the cost is a lost model run reported under the wrong cause.
**Fix:** Keep digits that a rate-limit heuristic could match out of the provider's error text. For example, report the overflow as `context_overflow: prompt exceeds the configured context window (see prompt_tokens)`. For other exceptions, `re.sub(r"\d", "#", ...)` is too blunt, so pass only `type(e).__name__` plus the message with digit runs of three or more collapsed. Better, tighten the runner:
```python
def _is_rate_limited(error):
    return bool(error) and bool(re.search(r"\b429\b|RateLimit|rate limit", error, re.I)) and "OutOfMemory" not in error
```

### WR-02: `run_train.sh poll` reports the previous run's result while a new launch is still waiting in preflight

**File:** `src/qwen_lora/run_train.sh:92-108, 136`
**Issue:** `launch` removes `OUT_DIR/train.done` only after the preflight has passed (line 108). The preflight loop can wait `PREFLIGHT_RECHECKS x PREFLIGHT_WAIT_S`, which is 6 x 300 s = 30 minutes, with the old `train.done` still on disk.

`poll` breaks as soon as `train.done` exists (line 136), prints the old `train_progress.log`, and says `STATUS done-ok` (or `done-failed`) for a run that has not started. This is exactly the sequence of the pre-registered remediation retry. The retry would be launched into the same out-dir as the first run, whose `train.done` is `0`. The poller then reports success, and the operator or the orchestrating plan could proceed to smoke runs and registration against the first adapter while the retry is queued or running.
**Fix:** Remove the stale marker at the very start of `launch`, before the preflight loop. `train.done` is the launcher's own file.
```bash
launch() {
  local out="$1"; shift
  mkdir -p "$out"
  rm -f "$out/train.done"
  ...
```
Line 108 then becomes redundant. Optionally truncate or rotate `train_progress.log` at the same point so `poll` cannot tail a previous run.

### WR-03: A finished or resumable adapter is destroyed by routine failure and relaunch paths

**File:** `src/qwen_lora/train.py:619-621, 761, 824-835, 850-852`
**Issue:** Four paths destroy work that took about 41 minutes of GPU time:
1. After training, `set_peft_model_state_dict`, the file move into `out_dir`, and `_remove(resume_path)` (line 761) run first. Then `generation_check` runs through `HFLocalProvider` (lines 829-830). If it raises for any reason, the blanket `except BaseException` (line 850) calls `_wipe_run_files`, which deletes `adapter_model.safetensors`, `adapter_config.json` and `manifest.json`. The resume file was already deleted, so nothing can be recovered. The CUDA out-of-memory branch also wipes. A transient failure of the post-hoc check, such as the other project taking memory while the card is reloaded for serving, costs the entire trained adapter.
2. `resume=False` unconditionally runs `_wipe_run_files(out_dir)` plus the `STALE_FILES` removal before doing anything (lines 620-621). That removes the resume file and the in-run best adapter of a killed run. Launching without `--resume` after a crash silently throws away the state the Phase 2 pattern exists to keep, and `run_train.sh launch` also truncates `train_progress.log` in that case.
3. The same wipe removes the previous run's valid adapter and manifest before the new run has produced anything. The pre-registered retry ("train once more from scratch" in `models/qwen-agentlora`) would delete the first, valid adapter on its first step. If the retry then fails, the project has no adapter at all.
4. `_wipe_run_files` removes `manifest.json` on any failure after the adapter has been moved. A "fail closed" manifest and a generic crash are not the same thing, and the code cannot tell them apart.

**Fix:**
- Train into a staging directory (the in-run `adapter.best.tmp` already is one) and promote the files into `out_dir` only after the generation check passes. Alternatively, keep the adapter and write `gen_check: {"error": ...}` into the manifest, so it can be re-checked with `checks`.
- Keep the resume file until the final manifest is written.
- Refuse to start a non-resume run when `RESUME_FILE` exists unless `--force` or `--fresh` is passed.
- Do not delete a previous valid adapter until its replacement is promoted.

### WR-04: The pre-registered gate metrics are diluted, and the one class that is below the bar does not move the gate

**File:** `src/qwen_lora/train.py:338-346, 522-533` (numbers from `models/qwen-agentlora/manifest.json`)
**Issue:** `final_dev_full.dev_action_exact` is the share of all assistant spans whose action-level tokens are argmax-correct. In the data, 2,000 of the 5,579 dev spans (35.8 %) are `respond` spans. Their action-level tokens are `{"action":"respond","message":"`, the closing `"}` and `<|im_end|>`, which are boilerplate. So more than a third of the denominator is nearly free credit.

From 0.9396, the non-respond spans score at least (0.9396 x 5579 - 2000) / 3579 = 0.906, even if every respond span is perfect. Per class, `ds` is at 0.839, well under the 0.90 bar that triggers remediation. It does not matter because the rule is aggregate. The 0.90 threshold therefore does not mean what the Phase 2 CR-01 lesson was after. Nothing is wrong with the arithmetic, but the headline is a weak gate for "the model does the action right".

Two smaller confounds sit in the same area:
- The closing `"}` and `<|im_end|>` after a held-back respond message are still counted in `dev_loss_action` and in exactness. They depend on where the held-back wording ends, and only a token that BPE merges across the boundary is excluded.
- The 300-episode subset that selects the checkpoint (`dev_eval_n`) is a subset of the dev set used for `final_dev_full` and for the 100 gen-check prompts. The "final" numbers are not independent of the selection. This is dev data only, so it is not a benchmark leak, but it biases the reported figure upwards.

**Fix:** Report, and ideally gate on, non-respond span exactness and per-class minimums, for example `min(by_cls.values())`. `dev_first_action_exact` (0.9875) is the more honest headline and is already computed. State the respond share next to `dev_action_exact` in `SELECTION_DESCRIPTION`, and draw the gen-check and final evaluation from dev episodes outside the 300-episode selection subset.

### WR-05: The adapter integrity stamp does not cover `adapter_config.json`

**File:** `src/llm/hf_local.py:146-190, 223-228`
**Issue:** `verify_adapter` hashes only `adapter_model.safetensors`. `adapter_config.json` is checked only for `peft_type` and `base_model_name_or_path`, and `manifest.json` is trusted as written.

Fields in `adapter_config.json` change the served model without changing the hash:
- `lora_alpha` and `r` set the scaling.
- `use_rslora` and `use_dora` change the update rule.
- `target_modules`, `rank_pattern` and `alpha_pattern` change what is adapted.

`resolved_model` is `...+name@adapter_sha256[:12]`, so a hand-edited config yields a different model under an identical stamp. `checks artifacts` re-hashes the weights and checks `r`/`alpha`, but only when run explicitly, never at benchmark time. The trainer already has everything needed: `manifest["lora"]` records `r`, `alpha` and `target_modules`.
**Fix:** Add `adapter_config_sha256` to the manifest at train time and include it in `verify_adapter`. Derive the stamp from a hash of both files:
```python
sha256 = hashlib.sha256(sha256_file(weights).encode() + sha256_file(config).encode()).hexdigest()
```
If the manifest key must stay the weights hash, add a second key and compare both.

## Info

### IN-01: The control-token guard covers 3 of the tokenizer's 22 special tokens, and serving does not sanitise or flag them

**File:** `src/qwen_lora/chat.py:42-46, 86-87`; `src/qwen_lora/data.py:102-106`; `src/llm/hf_local.py:266`
**Issue:** `SPECIAL_TOKEN_STRINGS` is `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`. The tokenizer's added vocabulary has 22 entries, and I checked that `<tool_call>`, `</tool_call>`, `<|fim_prefix|>`, `<|file_sep|>`, `<|vision_start|>` and others each tokenise to a single id (151657, 151659, 151664, 151652). The guard returns `False` for all of them.

At serving, a forged turn inside a tool result is parsed as real control tokens: I built one and got five `<|im_end|>` ids from what is a three-message conversation. The docstring says this is intentional ("every model must see identical text") and the current benchmark data has no such string, so nothing is exploitable today. The consequences are:
- A Qwen-only injection vector exists that does not apply to the other principals.
- The tuned adapter never saw it.
- Nothing in the row records that it happened.

**Fix:** Derive the list from `tokenizer.get_added_vocab()` in `build_example` and in a test over the benchmark files. At serving, count control-token strings in `messages` and expose the count (for example in `finish_reason` or a new `LLMResponse` field) so a result can be flagged. Do not rewrite the text.

### IN-02: The shipped manifest was produced from a dirty tree

**File:** `models/qwen-agentlora/manifest.json` (`code_commit`, `code_dirty`)
**Issue:** `"code_commit": "75bedd3e..."` with `"code_dirty": true`. `git_dirty()` ignores untracked files, so tracked files differed from HEAD during the 41-minute run. The commit therefore does not identify the code that made the adapter. The render fingerprint and the dataset hashes do identify the behaviour, so the practical risk is small. Nothing in `checks artifacts` flags it.
**Fix:** Make `checks artifacts` print `code_dirty` as a WARNING. For a run that is cited in the report, record a hash of `git diff HEAD -- src/ config/` in the manifest.

### IN-03: "Identical" merged and unmerged outputs is a measured approximation, and the test that stands for it runs in fp32

**File:** `src/llm/hf_local.py:23-26`; `tests/test_hf_local_provider.py:405-411`
**Issue:** The docstring says the merged output is "identical (measured on 20 prompts and 12 episodes)". The unit test checks that on a tiny fp32 CPU model. The served path is bf16 on CUDA, where `merge_and_unload` rounds the update into bf16 weights. On the real adapter over 50 dev episodes I measured: 0 of 50 first-action texts differ; teacher-forced argmax differs at 1 of 3,481 action positions; max absolute logit difference is 1.31. That is acceptable for the benchmark, but the claim is not protected by a test and the training-time dev metrics are computed unmerged.
**Fix:** Reword the docstring to "greedy outputs matched on N prompts; teacher-forced argmax differs at about 0.03 % of positions". The opt-in `QWEN_LIVE` test could compare merged and unmerged first-action text on a handful of dev prompts.

### IN-04: Offline mode is `setdefault`, and a directory base model has no revision identity

**File:** `src/llm/hf_local.py:87-90, 99-100`
**Issue:** `os.environ.setdefault("HF_HUB_OFFLINE", "1")` yields to an explicit `HF_HUB_OFFLINE=0` in the user's environment. The calls still pass `local_files_only=True`, so I found no path that reaches the network. `TRANSFORMERS_OFFLINE` is not set, and importing `huggingface_hub` earlier in the process (which this module does not control) makes the later `setdefault` a no-op because the constant is read at import time.

For a directory base model the revision is the literal string `"local"`, so `verify_adapter`'s revision check is vacuous and replaced base weights go unnoticed. That only affects tests, since production uses the Hub id.
**Fix:** Assign `os.environ["HF_HUB_OFFLINE"] = "1"` unconditionally, since a benchmark must not depend on the network. For a directory base, use a hash of `config.json` plus `model.safetensors` size, or document the limit.

### IN-05: The remediation window and the `early_stop` label do not match the wording of D-06

**File:** `src/qwen_lora/train.py:117-125, 529-533, 742-743`
**Issue:** D-06 says "best step within the last two evaluations". The code uses `best_step >= steps - 2 x eval_every`, which with evaluations at 100..600 is 400, 500 or 600, so three evaluations. This is identical to the Phase 2 rule and tested at 400/399, so it is consistent with the earlier phase and the text in the manifest, only not with the CONTEXT sentence. It did not matter here (0.9396 >= 0.90).

Separately, if `stale_evals >= patience` happens at the final evaluation (`step == max_steps`), the run is labelled `early_stop` although it ran every step, and `remediation_needed` then says "converged".
**Fix:** Align the CONTEXT/rule text with the code ("last three evaluation points"), or change the constant to `steps - every`. Set `stop_reason` to `max_steps` when `step == cfg.max_steps`.

### IN-06: Launcher details in `run_train.sh`

**File:** `src/qwen_lora/run_train.sh:52, 90, 128-129, 116-118`
**Issue:**
- `preflight`/`launch` run `mkdir -p "$out"` and write `gpu_preflight.log`, `train_progress.log` and `train.done` into any OUT_DIR before `train.py` applies its repository confinement (`resolve_under_root`). `launch /some/other/dir` creates files there before Python rejects the argument. The risk is low (operator-run), but the confinement belongs at the entry point as well.
- `alive()` interpolates `${out}` into a `pgrep -f` regular expression without escaping. A `.`, `+` or `(` in the path matches more or fewer processes than intended. It is still read-only, which I confirmed: only `pgrep`, `nvidia-smi`, `/proc/meminfo` and a read-only `ollama ps` are called, and nothing signals or stops a process.
- A user-supplied `--out-dir` among `"$@"` overrides the launcher's own (argparse keeps the last), so the done-marker and log land in a directory the trainer is not using.

**Fix:** Reject an OUT_DIR that resolves outside the repository in the script. Escape `${out}` with `sed 's/[][\.*^$+?(){}|]/\\&/g'`. Fail if `"$@"` contains `--out-dir`.

### IN-07: Small issues in `checks.py`

**File:** `src/qwen_lora/checks.py:148, 188, 204-208`
**Issue:**
- `open(done, encoding="utf-8").read()` leaks a file handle.
- The 4096 MiB limit is a bare literal.
- `cmd_artifacts` prints the REMEDIATION and LORA04_GATE decisions and then returns `artifacts-ok` with exit code 0 even when the gate fails or remediation is needed. The docstring says this is informational, but an automated caller reading only the exit code cannot tell.

**Fix:** Use `with open(...)`. Name the constant. Add `--enforce-gates` that returns 1 when `lora04_gate` fails or `remediation_needed` is true.

### IN-08: `train()` is a 290-line function, determinism is not enforced, and absolute paths can reach the manifest

**File:** `src/qwen_lora/train.py:567-852, 623, 193-195`
**Issue:**
- `train()` mixes setup, resume, the loop, promotion, the final evaluation, manifest building and the generation check in one function with a nested closure (`guard_vram`) and long `except` handling. That is why WR-03 is easy to introduce.
- `manifest["seed"]` suggests reproducibility, but only `torch.manual_seed` is set. There are no deterministic-algorithm flags, and bf16 plus checkpointing on CUDA is not bitwise repeatable. Say so in `run_notes`.
- `_model_id` stores a directory base model's real path in `adapter_config.json` and `manifest.json`. For the default Hub id this is harmless, but a local base would write the user's absolute path (and name) into tracked files.

**Fix:** Split out `_promote_adapter`, `_final_eval`, `_build_manifest`. Record `"deterministic": false` in the manifest. Store `os.path.relpath(real, PROJECT_ROOT)` for local base models, as `_rel` does for the data paths.

---

_Reviewed: 2026-10-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
