---
phase: 02-local-data-foundation-scratchgpt
reviewed: 2026-10-21T00:00:00Z
depth: standard
files_reviewed: 18
files_reviewed_list:
  - src/scratch_llm/format.py
  - src/scratch_llm/dataset.py
  - src/scratch_llm/templates.py
  - src/scratch_llm/train.py
  - src/scratch_llm/inference.py
  - src/scratch_llm/model.py
  - src/scratch_llm/provenance.py
  - src/scratch_llm/__init__.py
  - src/llm/scratch.py
  - config/models.yaml
  - requirements.txt
  - .gitignore
  - tests/test_format_parity.py
  - tests/test_scratch_dataset.py
  - tests/test_scratch_inference.py
  - tests/test_scratch_train.py
  - tests/test_training_leakage.py
  - .planning/phases/02-local-data-foundation-scratchgpt/02-CONTEXT.md
findings:
  critical: 1
  warning: 9
  info: 8
  total: 18
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-10-21
**Depth:** standard (plus targeted probes against the shipped checkpoint `models/scratch/scratch_gpt.pt` and `data/local/*.jsonl`)
**Files Reviewed:** 18
**Status:** issues_found

## Summary

All five test modules pass in WSL (`tests/test_format_parity.py`, `test_scratch_dataset.py`, `test_scratch_inference.py`, `test_scratch_train.py`, `test_training_leakage.py`).

The areas the review was asked to attack held up:

- **Benchmark leakage.** I found no leak in the shipped data.
  - Scanning the train and dev files for the 50 held-only parameter names, as tool parameters or call_tool argument keys, found none.
  - The gate fired 0 times across 24,000 episode checks during regeneration of 12,000 episodes.
  - Overlap with benchmark text is 0 for 6-grams and 7-grams. It is 4 of 1,089 for 5-grams, all generic phrases such as "and send the details to".
  - Of 137 rare benchmark identifiers, 35 appear in the training text. They are generic field names and times (`user_id`, `09:30`), not attacker values.
- **Train/inference parity.** The training text is a strict prefix-render of the same `render_context`, and the fit budget is never exceeded by validated episodes.
- **Loss mask.** Only `A:` payload plus newline carries loss, and padding is masked.
- **Checkpoint loading.** `torch.load(weights_only=True)`, the strict key contract and the stamp checks are correct.

The serious problem is in the dev metrics. They are structurally confounded by held-back `respond` phrasing (CR-01). That corrupts early stopping, best-checkpoint selection and the headline `dev_line_exact` number. The remaining warnings concern a few leakage-gate normalisation gaps, inference robustness (output cap and context overflow), provenance and failure-path hygiene in `train.py`, and a benchmark-validity risk. The shipped model often picks an attacker tool on step 1 before any injection is visible.

## Critical Issues

### CR-01: Dev loss and dev line-exact are dominated by unpredictable held-back `respond` text; early stopping and best-checkpoint selection optimise the wrong thing

**File:** `src/scratch_llm/train.py:151-201` (metrics), `:336-346` (selection), `:345-346` (early stop); source of the problem: `src/scratch_llm/dataset.py:961` plus `src/scratch_llm/templates.py` (`REPLY_TEMPLATES_DEV`, `BLOCKED_REPLY_TEMPLATES_DEV`)

**Issue:** The dev split deliberately uses held-back reply phrasing (D-04). The final `{"action":"respond","message":"..."}` line of every dev episode is therefore text the model has never seen. It is also drawn at random from the template list, so even a perfect model cannot predict it. `evaluate()` scores that line like any other supervised run. Measured on the shipped checkpoint:

- `respond` lines are 35.8% of all dev action lines (1,997 of 5,579). Free-running greedy decoding over 120 dev prompts got 0 of 120 `respond` lines exactly right. The cap on `dev_line_exact` is about 0.64, and non-`respond` lines reach about 0.43.
- Per-character loss on dev: `respond` lines 1.897 nats, all other action lines 0.048 nats. `respond` is 27.9% of supervised characters but 93.9% of the dev loss.
- The log shows the effect. Dev loss is lowest at step 6000 (0.572, line-exact 0.2746) and rises afterwards while line-exact keeps climbing (0.304 at 7000, 0.320 at 7500, 0.324 at 8000).
- Early stopping (patience 8 on dev loss) therefore ended the run at 8000 and kept an inferior checkpoint, chosen by how well it predicts unseen reply wording, not by action quality.
- The plan's "0.50 threshold" and the "dev remediation" run note (7000 to 14000 steps) were driven by a metric that cannot reach 0.50 for structural reasons. The viva page repeats the 0.27 figure.

**Fix:** Separate the diagnostic from the selection signal.

```python
# train.py, evaluate(): classify each supervised run by its first characters
# and exclude 'respond' MESSAGE text from the selection metrics.
def _is_respond(run_text: str) -> bool:
    return run_text.startswith('{"action":"respond"')

# 1. selection loss: mask-weighted CE over supervised chars of non-respond lines only
#    (and the first N chars of respond lines, i.e. '{"action":"respond","message":"', which are predictable);
#    report respond-message loss separately as dev_loss_respond (diagnostic, not used for selection).
# 2. dev_line_exact: score respond lines on the action key only (or drop them),
#    and report dev_action_exact (non-respond) as the headline.
# 3. if (1) is not feasible quickly, at least select/early-stop on dev_line_exact_non_respond.
```

Re-run the training selection with the corrected criterion. The best checkpoint under the corrected criterion is probably a later step than 6000. Correct the viva and manifest numbers accordingly.

## Warnings

### WR-01: Shipped model picks the attacker's tool on step 1 before any injection is visible, so measured ASR will contain false positives

**File:** `src/scratch_llm/dataset.py:1110-1189` (episode construction), `src/benchmark/scoring.py` (attacker-tool-executed scoring)

**Issue:** This is a validity risk, not a code defect. Probe: for 40 benchmark attack cases, the first model reply, produced from the user instruction and tool list only, was:

| First reply | Cases |
|---|---|
| The user's tool | 15 |
| A tool in the case's attacker list, or GmailSendEmail | 7 |
| A tool outside both sets | 17 |
| Unparseable | 1 |

Example: for the `TodoistSearchTasks` task the model emitted `TrafficControlGivePriorityToEmergencyVehicle` with `args:{}`. A "dh" success is scored as "attacker tool executed". Calls made before the injection was read inflate the unguarded baseline ASR, and the guardrail arms are compared against it. Dev `first_line_exact` of 0.48 shows the same weakness (tool-name copying from the T: line does not generalise).

**Fix:** At minimum, log and report the step-1 attacker-tool rate on the benign (no-injection) variants as a noise floor. Better: score an attack as successful only when the attacker call occurs after the observation that carries the injection. Consider more tool-copy training signal (more tools per prompt, shuffled order) before the benchmark run.

### WR-02: `MAX_NEW_CHARS = 320` is below the format's own action limit (400) and the comment is wrong

**File:** `src/scratch_llm/inference.py:35`, `:95`

**Issue:** The comment says "the longest legitimate action is about 230 characters". The dataset allows actions up to `HISTORY_ACTION_CHARS = 400` (`validate_episode`, dataset.py:228), and `GEN_RESERVE` is 400. Gold actions of up to 335 characters (including the newline) exist in the shipped data, and 4 gold actions exceed 320. A correct long reply, such as an attacker `call_tool` with several arguments, is cut at 320, reported as `finish_reason="length"` and fails to parse. That is scored as the model refusing or failing. Benchmark attacker calls such as `CiscoUmbrellaCreatePolicy` already reach 275 characters.

**Fix:**
```python
from src.scratch_llm.format import HISTORY_ACTION_CHARS, END
MAX_NEW_CHARS = HISTORY_ACTION_CHARS + len(END)   # the longest action the format allows, plus the newline
```
Also fix the comment. The existing bound still holds (budget 1648 + 2 + 401 is at most 2048).

### WR-03: A prompt longer than the model context is silently cropped at inference

**File:** `src/scratch_llm/inference.py:93-106`, `src/scratch_llm/format.py:101-117`, `src/scratch_llm/model.py:116`

**Issue:** `fit()` bounds only the turns. The `T:` line is unbounded, because the tool list is not budgeted, and `fit()` returns the bare header when `room <= 0`. `ScratchGenerator.__call__` never checks `len(ids) + max_new <= context`. Probe: a 60-tool prompt rendered to 2,402 characters. `generate()` then crops to the last 2,048 characters (`idx[:, -context:]`), which drops the start of the `T:` line, and returned `'{"}}}}'` with `finish_reason="stop"` and `prompt_tokens=2402` and no error. A benchmark with many tools would record garbage as ordinary model behaviour.

**Fix:** Refuse, or record it.
```python
if len(ids) + max_new > self.model.cfg.context:
    raise ValueError(f"prompt of {len(ids)} chars + {max_new} new exceeds context {self.model.cfg.context}")
```
Alternatively set `last_finish_reason = "context_overflow"` and return an empty reply. The provider already converts exceptions into an `LLMResponse(error=...)`. Also have `render_context` cap the `T:` line (drop trailing tools, or truncate with a visible marker) so the training and inference renderings stay identical.

### WR-04: Manifest `final` describes the last evaluation, not the checkpoint that is shipped

**File:** `src/scratch_llm/train.py:353-372`

**Issue:** `final` is built from `history[-1]`, but the saved `scratch_gpt.pt` is the best-dev-loss step. In `models/scratch/manifest.json`, `best_step` is 6000 and `final.dev_line_exact` is 0.3239 (step 8000). The shipped checkpoint scores 0.2746, which is the number in its `train_meta` and in the viva. A reader of the manifest, the provenance record for the checkpoint, gets numbers that do not describe the artifact. The manifest has no field for the best step's dev metrics.

**Fix:** Add `"best": {"step", "dev_loss", "dev_line_exact", "dev_line_exact_by_cls", "dev_first_line_exact"}` taken from the history record at `best["step"]`. Rename `final` to `last_eval`, and bump `MANIFEST_SCHEMA` and `MANIFEST_KEYS` accordingly.

### WR-05: Failure paths can leave a stale or orphan checkpoint that the benchmark will happily load

**File:** `src/scratch_llm/train.py:286-288`, `:304-307`, `:336`, `:352`, `:373`

**Issue:** Four related holes:

1. The start of `train()` removes the old `train_log.jsonl` and `manifest.json` but not the old `scratch_gpt.pt` or `scratch_gpt.pt.tmp`. If a rerun fails before its first improving evaluation (non-finite loss, leakage after partial setup, kill), the previous run's checkpoint stays next to no manifest. `LocalScratchProvider` only checks existence and stamps, so it is benchmarked as if valid.
2. The `FAILED non_finite_loss` path leaves the best-so-far checkpoint from that same run loadable. The module docstring says a diverged run "can never be mistaken for a finished one", which holds for the manifest only.
3. A NaN `dev_loss` is never detected: `nan < best` is False, so nothing is saved. If it happens at the first evaluation there is no checkpoint, and `file_sha256(ckpt_path)` (line 373) raises `FileNotFoundError` after the whole run.
4. `_write_curve` (matplotlib) runs before the manifest is written. A plotting failure after a 20-minute run aborts with a checkpoint and no manifest.

**Fix:**
```python
for stale in (ckpt_path, ckpt_path + ".tmp", log_path, manifest_path):
    if os.path.exists(stale): os.remove(stale)
...
if not math.isfinite(result["dev_loss"]):          # treat like a non-finite train loss
    raise NonFiniteLossError(...)
...
try: _write_curve(...)
except Exception as e: log(f"WARNING loss curve not written: {e}")
```
On the FAILED path also delete the in-run checkpoint, or write it under a different name until the manifest exists.

### WR-06: Training can overwrite or delete an unrelated `manifest.json` (for example the dataset manifest)

**File:** `src/scratch_llm/train.py:283-288`, `:379`, `:421-425`

**Issue:** `--out-dir` is confined to the repository (good), but any directory inside it is accepted. `train()` deletes `<out_dir>/manifest.json` and later writes its own there. `--out-dir data/local` destroys `data/local/manifest.json`, the committed dataset provenance record (hashes, gate verdict, held-out hash). `--out-dir .` does the same to a repository-root manifest.

**Fix:** Refuse to delete or overwrite a `manifest.json` that lacks `"checkpoint_sha256"`. Alternatively write `train_manifest.json`, or restrict `--out-dir` to be under `models/`.

### WR-07: Canary gate is case-sensitive, and `_secret` shape 0 emits the canary's variable part

**File:** `src/scratch_llm/dataset.py:294-296`, `:383-389`, `:597-601`

**Issue:** There are two defects:

1. `_CANARY = re.compile(r"CANARY[0-9A-F]{12}DATA")` and `_BANNED_CASE_SENSITIVE = ("CANARY",)` are case-sensitive, while the email, shingle and phrase checks are lower-cased. Probe: `canary0a1b2c3d4e5fdata` and `Canary0A1B2C3D4E5FData` appended to an observation pass `_episode_violation` with no violation. Bare `0A1B2C3D4E5F` passes too, which is expected for a bare hex string.
2. `_secret()` shape 0 draws 12-20 characters from `0123456789ABCDEF`. A 12-character draw (1 in 6 shapes times 1 in 9 lengths, about 1.9% of ds secrets) is exactly the distribution of the canary's variable part. The docstring says "never the benchmark's canary shape", and the gate would not notice if it ever did.

The model copies secrets from the prompt anyway, so recall is not a plausible mechanism. It is still a gap between the stated D-04 guarantee ("no CANARY... value") and what is enforced.

**Fix:**
```python
_CANARY = re.compile(r"canary[0-9a-f]{12}data", re.I)
# and match the bare phrase case-insensitively if 'canary' is not a legitimate template word:
_BANNED_ANY_CASE = ("canary", "private record", "account owner. reference")
```
In `_secret()`, make shape 0 use lengths 13-20, or add a non-hex character, so 12 uppercase hex characters is never drawn.

### WR-08: Generation silently redraws rejected episodes and records nothing about rejections

**File:** `src/scratch_llm/dataset.py:1269-1287`, `:884`, `:1411-1432` (manifest)

**Issue:** The generator redraws up to 50 times on any validation or leak failure and then moves on. The manifest records `"gate": {"passed": true}` but no rejection counts. A template, tool or parameter name that systematically trips a gate rule is hidden: the data simply skews away from it, as long as at least one of 50 draws survives. `except ValueError: continue` also swallows genuine generator bugs. Currently there are 0 leak rejections and about 49 per 12,000 draws rejected for the user text exceeding `USER_CHARS`, so nothing is wrong today. The defense is not observable.

A related latent hole: `_synthetic_parameters` takes `text_names` from `types` (line 884), which is not filtered by held-only names, unlike `names`. Today none of `TEXT_PARAMS` is held-only, but a benchmark update would be absorbed silently by redraws.

**Fix:** Count rejections per rule and per validation message and store them in the manifest (`gate.rejected`). Fail generation if any leak-rule rejection exceeds a small threshold such as 0.1% of draws. Filter `text_names` by `held_only_params`.

### WR-09: Batch composition is frozen across epochs

**File:** `src/scratch_llm/train.py:113-127`

**Issue:** `make_batches` sorts by length and chunks greedily with a deterministic tie-break `(length, index)`. Only the order of chunks is shuffled per epoch (`random.Random(seed*1_000_003 + epoch)`), so every epoch sees the same batches (for example about 1,100 per epoch, 7 epochs by step 8000). Length-sorted batches are also highly correlated: a batch is all short OS/benign episodes or all long ds episodes. Combined with a train loss of 0.02 against a dev loss of 0.57, this plausibly contributes to overfitting.

**Fix:** Add per-epoch noise before sorting, for example `key=lambda i: (lengths[i] + rng.randint(0, 64), i)`. Alternatively shuffle indices, bucket them in windows of 100 batches, and sort within the window.

## Info

### IN-01: Dev line-exact is measured under bf16 autocast but the model is served in fp32

**File:** `src/scratch_llm/train.py:168`, `src/scratch_llm/inference.py:125`, `train.py:156-158` docstring

**Issue:** The evaluate docstring says exact match "is exactly what greedy decoding would produce". `inference.py` states "greedy output does not depend on precision". Measured on the shipped checkpoint: CPU fp32 gives dev_line_exact 0.27514 against the logged bf16 0.27460, with small per-class differences (dh 0.336 against 0.334). The difference is tiny but the claims are literally false.

**Fix:** Soften the wording, or run the final dev evaluation in fp32 and record both.

### IN-02: Checkpoint `cfg` is only checked for `context` and `vocab_size`

**File:** `src/scratch_llm/inference.py:57-63`, `:119-123`

**Issue:** `GPTConfig(**ckpt["cfg"])` is built from untrusted numbers. `weights_only=True` blocks code execution but not resource exhaustion: a huge `n_layer` or `d_model` allocates memory before `load_state_dict` can reject the shapes. `n_head=0` raises `ZeroDivisionError`, which is not in the `except (TypeError, AssertionError, RuntimeError)` list. The runner's broad `except Exception` still turns this into a `[skip]`, so impact is low.

**Fix:** Validate the types and ranges of `cfg` values, or compare the `state_dict` tensor shapes against the declared `cfg` before constructing the model.

### IN-03: `.gitignore` does not cover the atomic-write temporaries

**File:** `.gitignore:52-62`

**Issue:** `models/**/*.pt` does not match `scratch_gpt.pt.tmp` (left behind by a killed run, tens of MB). Likewise `data/local/*.jsonl` does not match `*.jsonl.tmp`, and `manifest.json.tmp` is not ignored.

**Fix:** Add `models/**/*.tmp`, `data/local/*.tmp` and `**/*.json.tmp`.

### IN-04: `torch` is not in `requirements.txt`, so two test modules skip silently

**File:** `requirements.txt:20-23`, `tests/test_scratch_inference.py:20`, `tests/test_scratch_train.py:23`

**Issue:** Torch appears only in a comment. A plain `pip install -r requirements.txt` followed by `pytest` skips `test_scratch_inference.py` and `test_scratch_train.py` through `importorskip`, so the stamp checks and loss-mask tests silently vanish from a green run.

**Fix:** Add a CI assertion that those modules ran, for example `-p no:cacheprovider --strict-markers` with a `torch` extra, or fail in `conftest.py` when `STRICT_TORCH=1` and torch is missing.

### IN-05: Gate boundary and scope details

**File:** `src/scratch_llm/dataset.py:347`, `:424-431`

**Issue:** Two details:

1. `held_re` uses `\b...\b`, and `_` is a word character. Held tool names embedded in a longer identifier, such as `GmailSendEmail_x`, `GmailSendEmailV2` or a lower-cased form, are not matched (probe confirmed). This is currently moot because the generator never builds such names.
2. Held-only parameter names are checked only as top-level `call_tool` argument keys and as spec parameter names. Keys nested in object-typed arguments are not examined. `validate_episode` limits top-level keys to the required parameters, and the generator's `OBJECT_KEYS` contain no held-only name, so it is not exploitable today.

**Fix:** Use lookarounds such as `(?<![A-Za-z0-9])...(?![A-Za-z0-9])` and compare case-insensitively. Walk nested dict keys for `RULE_HELD_ONLY_PARAM`.

### IN-06: Provenance weaknesses

**File:** `src/scratch_llm/provenance.py:3-5`, `models/scratch/manifest.json`, `data/local/manifest.json`

**Issue:** The module docstring says it "needs nothing outside the standard library" but imports `src.benchmark.injecagent` for `PROJECT_ROOT`. Both committed manifests record `code_dirty: true`. The dataset manifest names commit `696e32d` and the model manifest `c722878`, so the exact generating code is not recoverable from the commit hash. The dataset and model are hash-linked through `train_sha256`, and they match the files on disk.

**Fix:** Compute `PROJECT_ROOT` locally in `provenance.py`. Refuse, or require an explicit flag, to write manifests from a dirty tree for the final run.

### IN-07: Provider has unguarded code after the `try`

**File:** `src/llm/scratch.py:43-50`

**Issue:** `LLMProvider` must not raise. Only `self._generate(...)` is inside the `try`. `text.strip()` and `sha[:12]` run outside it. They are fine for the real generator, but a double that returns a non-str or a non-str `checkpoint_sha256` would raise. The other probed bad inputs (`messages=[]`, `max_tokens=None`, `max_tokens=-5`) are handled and return error or empty replies.

**Fix:** Extend the `try` to cover response construction, or coerce with `str(text)`.

### IN-08: Training and sampling are not bitwise reproducible

**File:** `src/scratch_llm/train.py:275-282`, `src/scratch_llm/model.py:119`

**Issue:** The manifest records a `seed`, but there is no `torch.use_deterministic_algorithms` or deterministic SDPA backend. `torch.multinomial` in `generate` (temperature greater than 0) is not seeded by the generator or the provider. Data generation is deterministic (the `PYTHONHASHSEED` test exists) and greedy inference at temperature 0 is deterministic, so this affects only reruns of training and sampled benchmark runs.

**Fix:** Document it in the manifest (`"deterministic": false`). Optionally add a `seed` argument to `ScratchGenerator.__call__` with a per-call `torch.Generator`.

---

_Reviewed: 2026-10-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
