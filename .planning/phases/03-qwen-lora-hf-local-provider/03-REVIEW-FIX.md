---
phase: 03-qwen-lora-hf-local-provider
fixed_at: 2026-10-22T16:00:00Z
review_path: .planning/phases/03-qwen-lora-hf-local-provider/03-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 3: Code Review Fix Report

**Fixed at:** 2026-10-22
**Source review:** .planning/phases/03-qwen-lora-hf-local-provider/03-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (WR-01 to WR-05; the eight Info findings IN-01 to IN-08 were out of scope and are untouched)
- Fixed: 5
- Skipped: 0

The adapter was not retrained. Its weights (`adapter_model.safetensors`, sha256 prefix 270501a67998) and
`adapter_config.json` are byte-identical to before; only `manifest.json` was extended.

**Verification environment:** `workflow.use_worktrees` is `false` in `.planning/config.json`, so every edit, test run and
commit happened in the main checkout (no worktree, no temp branch, no recovery sentinel). Tests ran under WSL
(`venv/bin/python -m pytest`), the GPU recomputation and the smoke run under WSL on the RTX 3060 after a read-only
preflight (4661 MiB GPU free, about 10.2 GB RAM available; no other project's process was touched). Each fix has a
regression test that was run against the old code first and failed there.

Full suite after the last fix, in four split calls: 925 passed (Phase 1-2 and everything outside the Qwen files, was 901),
66 passed + 1 skipped (`test_hf_local_provider.py` + `test_qwen_lora_data.py`, was 57 + 1), 45 passed
(`test_qwen_lora_train.py`) and 49 passed (`test_qwen_lora_checks.py`); 1085 passed + 1 skipped in all, against the baseline
of 1034 + 1 (51 new tests). The skip is the opt-in `QWEN_LIVE` probe.

## Fixed Issues

### WR-01: hf_local error strings can be misread as rate limiting

**Status:** fixed: requires human verification (a wrong-condition logic bug; the tests pin the intended behaviour)
**Files modified:** `src/benchmark/runner.py`, `src/llm/hf_local.py`, `tests/test_rate_limit_detection.py` (new),
`tests/test_hf_local_provider.py`
**Commit:** ad617a4
**Applied fix:** `_is_rate_limited` no longer searches for the substring `429`. It matches a compiled, case-insensitive
pattern for real signals only: `rate limit` / `RateLimitError` / `rate_limit_exceeded`, `too many requests`,
`RESOURCE_EXHAUSTED`, a status field (`status 429`, `status_code=429`, `Error code: 429`, `'code': 429`, `HTTP 429`, each
with a negative lookahead so `4296` and `429.50` never match), and the status-line words (`429 Client Error`, `429 Too Many
Requests`, `429 quota exceeded`, ...). The hf_local context-overflow error text no longer carries bare numbers (the count is
in `prompt_tokens`, the limit is `max_context_tokens` in `config/models.yaml`); other exception text from the model call is
unchanged, since the runner no longer collides with it. Tests: "prompt of 4296 tokens", "prompt of 429 tokens", "429.50 MiB
is free", "Tried to allocate 429.00 MiB", "expected 4290 elements" are all False; 13 real rate-limit strings (including
every string the Phase 1 tests use) are True; the overflow message has no digit. The Phase 1 runner and report tests
(`test_runner_queue.py`, `test_benchmark.py`, `test_report.py`, `test_review_fix_benchmark.py`) pass unchanged.

### WR-02: `run_train.sh poll` reports the previous run's result while a launch waits in preflight

**Status:** fixed: requires human verification (shell control flow)
**Files modified:** `src/qwen_lora/run_train.sh`, `tests/test_qwen_lora_checks.py`
**Commit:** 101bf5b
**Applied fix:** `launch` runs `rm -f "$out/train.done"` right after `mkdir -p`, before the preflight loop (the later, now
redundant removal is gone). Two tests: a fake `nvidia-smi` records whether `train.done` exists each time the preflight
reads the card (old code: "stale" on the first reading, new code: never), and a real `poll` against a launcher that is
still waiting in preflight reports `STATUS running`, never `done-ok`. `train_progress.log` rotation (the optional part of
the suggestion) was not added.

### WR-03: A finished or resumable adapter is destroyed by routine failure and relaunch paths

**Status:** fixed: requires human verification (state handling)
**Files modified:** `src/qwen_lora/train.py`, `src/qwen_lora/checks.py`, `tests/test_qwen_lora_train.py`,
`tests/test_qwen_lora_checks.py`
**Commit:** f0e6133
**Applied fix:**
- Outputs are staged. The best adapter lives in `adapter.best.tmp/` as before, which now also receives `manifest.json` and
  `loss_curve.png`; the log goes to `train_log.new.jsonl`. `generation_check` serves the staged directory through
  `HFLocalProvider`. Only after the final dev pass and both generation checks succeed does `_promote_run` move the files into
  `--out-dir` (`os.replace`, manifest last, every staged file checked first, a stale curve removed if the new one failed to
  draw). A crash in the middle of the promotion leaves new weights next to the old manifest, which the hash check refuses
  (fails closed).
- A fresh (non-resume) start no longer deletes a previous adapter, manifest, curve or log; it discards only staging leftovers.
  `--resume` no longer deletes them either (the old line that removed the final artifacts on resume is gone).
- The resume file is kept until the promotion succeeds.
- A failure while training (non-finite loss, VRAM guard, CUDA out of memory during training) still removes the staging
  directory and the resume file, keeps the staged log as the diagnostic, and exits 3 or 4 as before. A failure after the
  last step (final dev pass, generation check, including an out-of-memory error there) keeps the staged adapter and the
  resume file, prints `FAILED post_training: ...`, and leaves the previous adapter untouched.
- A fresh launch that finds a killed run's `qwen_lora.resume.tmp` refuses (ValueError from `train()`, exit 2 from the CLI)
  unless `--fresh` is given; `--resume` and `--fresh` exclude each other.
- `checks artifacts` now also flags a leftover `train_log.new.jsonl` (a failed re-run next to a valid adapter).
- Tests (all five new ones failed on the old code): a failing generation check leaves the previous adapter byte-identical and
  keeps the staged new one; with no previous adapter nothing loadable appears in `--out-dir`; a successful fresh run over a
  previous adapter replaces it only at promotion (a spy on `generation_check` sees the old files during both checks; the log
  is the new one, not appended to); resuming does not delete the previous adapter; a fresh launch refuses without `--fresh`.
  The old "fresh run removes stale files" test was rewritten for the new contract; the failure-exit tests read the staged log.
- Not done from the suggestion: the manifest `gen_check: {"error": ...}` alternative (the staged-adapter route was taken).

### WR-04: The pre-registered gate metrics are diluted, and the one class below the bar does not move the gate

**Status:** fixed
**Files modified:** `src/qwen_lora/train.py`, `src/qwen_lora/checks.py`, `tests/test_qwen_lora_train.py`,
`tests/test_qwen_lora_checks.py`, `models/qwen-agentlora/manifest.json`
**Commit:** 5e45f70
**Applied fix:** the pre-registered gates, thresholds, `REMEDIATION_RULE`, `LORA04_GATE_RULE` and `SELECTION_DESCRIPTION` are
unchanged (a test pins that the rule text does not mention the new metrics). `evaluate()` additionally returns, and the log
records, `best`, `last_eval` and `final_dev_full` blocks now hold: `dev_respond_share`, `dev_nonrespond_action_exact`,
`dev_nonrespond_action_exact_by_cls` and `dev_min_class_action_exact` (appended to `EVAL_KEYS`). A new manifest key
`evaluation_overlap` (after `gen_check` in `MANIFEST_KEYS`) states how many selection-subset episodes are in the final pass
and how many gen-check prompts are in the selection subset and the final pass, with a note that the final numbers are dev
numbers not independent of the selection. `checks artifacts` prints `ARTIFACT gate_context ...` and `ARTIFACT overlap ...`
(or "not recomputed" for a manifest that predates the metrics). The decisions (REMEDIATION, LORA04_GATE) are untouched.

**Recomputation for the existing run** (kept adapter, GPU, a couple of minutes in all, peak 1,883 MiB
allocated / 2,088 MiB reserved, preflight first): the evaluation sets were rebuilt from the manifest's seed and config
(`Random(seed + 1)` selection subset of 300, the whole 2,000-episode dev file, `seed + 3` gen-check prompts) and evaluated
with the same `evaluate()` and the same loading path as the trainer. Every pre-registered value (all eight, for both the
300-episode subset and the full dev file, including per-class values) reproduced **exactly** (bit for bit), so the new
numbers measure the adapter that produced the old ones. Step-100 to step-500 log records were not recomputed (no
checkpoints of those steps exist); `train_log.jsonl` was not touched.

| Metric (final_dev_full, 2,000 episodes) | Old manifest | New manifest |
|---|---|---|
| dev_action_exact (pre-registered, gate 0.90) | 0.9396 | 0.9396 (unchanged) |
| dev_action_exact by class | benign 0.9896, dh 1.0000, ds 0.8392 | unchanged |
| dev_first_action_exact | 0.9875 | 0.9875 (unchanged) |
| dev_respond_share | not recorded | 0.3585 |
| dev_nonrespond_action_exact | not recorded | 0.9058 |
| dev_nonrespond_action_exact by class | not recorded | benign 0.9804, dh 1.0000, ds 0.7841 |
| dev_min_class_action_exact | not recorded | 0.8392 (ds) |

On the 300-episode selection subset at step 600 (`best` and `last_eval`): respond share 0.3593, non-respond exactness
0.9009 (ds 0.7674), minimum class 0.8270. Overlap: all 300 selection episodes are inside the 2,000-episode final pass; of the
100 gen-check prompts, 16 are inside the selection subset and all 100 are inside the final pass. The manifest also got one
`run_notes` entry saying the metrics were recomputed (no retraining).

### WR-05: The adapter integrity stamp does not cover `adapter_config.json`

**Status:** fixed
**Files modified:** `src/llm/hf_local.py`, `src/qwen_lora/train.py`, `src/qwen_lora/checks.py`, `tests/qwen_fixtures.py`,
`tests/test_hf_local_provider.py`, `tests/test_qwen_lora_train.py`, `tests/test_qwen_lora_checks.py`,
`models/qwen-agentlora/manifest.json`
**Commit:** 64c0c40
**Applied fix:** `hf_local` gains `canonical_json_sha256`, `adapter_config_sha256` (sorted keys, no insignificant whitespace,
so a reformatting passes and any value change does not) and `adapter_identity_sha256(weights, config)`. The manifest keeps
`adapter_sha256` as the plain weights file hash and gains `adapter_config_sha256` and `adapter_identity_sha256`
(`MANIFEST_KEYS` updated, after `adapter_sha256`). `verify_adapter` recomputes both hashes from the files on disk, refuses a
mismatch naming the field (an edited `lora_alpha`, `r`, `use_rslora`, `use_dora` and so on, or a legacy manifest without the
fields), and returns the identity hash, which is now what `resolved_model` carries (`LoadedModel.adapter_identity`,
`HFLocalProvider.adapter_identity`; the old `adapter_sha256` attribute is gone). `checks artifacts` re-hashes the config and
checks the identity, `checks smoke`/`probe` expect the identity stamp, and `checks viva` requires the identity prefix to be
quoted. Tests (failing on the old code): lora_alpha, use_rslora and r edits are refused; the manifest-side edits are refused;
a coordinated re-stamp of an edited config yields a different `resolved_model` than the original (same weights); a
reformatted config still verifies; `checks artifacts` refuses an edited config and a wrong identity.

**Re-stamp of the existing adapter** (a CPU script, nothing retrained, weights and config untouched, no existing manifest value
changed, two keys inserted plus one `run_notes` entry):

| | Old | New |
|---|---|---|
| `adapter_sha256` (weights) | 270501a67998495164dc... | unchanged |
| `adapter_config_sha256` | not recorded | 9338083a13546c284c935b01e2d9fa0758bd846b24f997855d711e2c0218d74d |
| `adapter_identity_sha256` | not recorded | 924ac021f0cb5787336b6de401168c4b8f13a7de77d449eb4b59845df43258d5 |
| `resolved_model` suffix | `+qwen2.5-0.5b-agentlora@270501a67998` | `+qwen2.5-0.5b-agentlora@924ac021f0cb` |

`checks artifacts --require-cuda` on the re-stamped adapter prints `artifacts-ok` (REMEDIATION no, LORA04_GATE pass, as before).

**Smoke re-run:** the stamp format changed, so the LoRA smoke run was repeated into a fresh directory with the same arguments as
03-04 (`run_benchmark.py --models qwen2.5-0.5b-agentlora --runs 1 --dh 5 --ds 5 --benign 3 --resume
results/phase3-smoke-lora-v2`, 146 s, exit 0, no `[skip]` line). `checks smoke --expect-rows 26` printed
`smoke-plumbing-ok`; every row's `resolved_model` is
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf+qwen2.5-0.5b-agentlora@924ac021f0cb`. Behaviour numbers are identical to the first run
(parse 77/77, valid actions 77/77, unprotected dh/ds attempted 10/10 and succeeded 10, protected succeeded 0, benign 3/3,
outcomes attacked=10 ignored_injection=3 in both arms). `results/phase3-smoke-lora` (old stamp) and
`results/phase3-smoke-base` (no adapter stamp) were left as they were; both are untracked.

## Skipped Issues

None.

## Not in scope (Info findings, untouched)

IN-01 (control-token guard covers 3 of 22 special tokens), IN-02 (`code_dirty: true` in the manifest), IN-03 (merged versus
unmerged claim), IN-04 (offline mode `setdefault`), IN-05 (remediation window wording, `early_stop` label at the last step),
IN-06 (launcher details), IN-07 (`checks.py` small issues), IN-08 (`train()` length, determinism note). IN-05 interacts with
WR-04 only in that `REMEDIATION_RULE` was deliberately not edited.

## Notes for the reader

- The manifest's `code_commit` (75bedd3e...) and `code_dirty: true` still describe the training run; the WR-04 and WR-05
  edits to the manifest are recorded in its `run_notes`, not in `code_commit`.
- The viva page `docs/viva/03-fine-tuning-lora-and-chat-templates.md` has a new "What the code review found and what we fixed"
  section and the changed stamp (commit 815069f); `checks viva` still prints `part-c-ok`.
- The working tree reports many files as modified when `git` is run inside WSL (line-ending and stat differences of the Windows
  drive); Windows `git status` is the reference and showed only the files named above plus the user's own unrelated changes.
- Nothing under `papers/`, `docs/*.pdf|pptx`, `docs/images/`, `results/`, `scripts/`, `.impeccable/`, `.planning/state.json`,
  `.planning/milestone.lock` or `src/dashboard/app.py` was staged.

---

_Fixed: 2026-10-22_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
