---
phase: 02-local-data-foundation-scratchgpt
verified: 2026-10-22T09:30:00Z
status: passed
score: 12/12 must-haves verified
covered_files:
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-01-PLAN.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-01-SUMMARY.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-02-PLAN.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-02-SUMMARY.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-03-PLAN.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-03-SUMMARY.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-04-PLAN.md"
  - ".planning/phases/02-local-data-foundation-scratchgpt/02-04-SUMMARY.md"
  - "config/models.yaml"
  - "data/local/manifest.json"
  - "docs/viva/02-llm-foundations-and-synthetic-data.md"
  - "docs/viva/README.md"
  - "models/scratch/loss_curve.png"
  - "models/scratch/manifest.json"
  - "models/scratch/train_log.jsonl"
  - "src/benchmark/injecagent.py"
  - "src/benchmark/report.py"
  - "src/benchmark/scoring.py"
  - "src/llm/scratch.py"
  - "src/scratch_llm/__init__.py"
  - "src/scratch_llm/dataset.py"
  - "src/scratch_llm/format.py"
  - "src/scratch_llm/inference.py"
  - "src/scratch_llm/model.py"
  - "src/scratch_llm/provenance.py"
  - "src/scratch_llm/templates.py"
  - "src/scratch_llm/train.py"
  - "tests/test_format_parity.py"
  - "tests/test_scratch_dataset.py"
  - "tests/test_scratch_inference.py"
  - "tests/test_scratch_train.py"
  - "tests/test_training_leakage.py"
covered_digest: "v3:sha256:210474ccbd0e0dfeaded73518334ec91af209e45b762ffe2d75fcccf384556fb"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 2: Local Data Foundation & ScratchGPT Verification Report

**Phase Goal:** A language model trained from scratch on this laptop's GPU drives the agent in the benchmark as `scratch-gpt`, trained on synthetic episodes that provably contain nothing from the benchmark and rendered by the same function at training and inference time.
**Verified:** 2026-10-22T09:30:00Z
**Status:** passed
**Re-verification:** No, initial verification

The verification judges the CURRENT state. The 02-04 SUMMARY's checkpoint `83b0763a6d26`, its numbers and its 0.50 remediation threshold are superseded by the code-review fix and retrain (02-REVIEW-FIX.md): checkpoint `97bdace56294`, selection on `dev_loss_action`, rule `dev_action_exact < 0.75`, smoke run `results/phase2-smoke-v2`. Plan must-haves that name the old values are treated as satisfied because the superseding fix meets the same property (a dev-metric-only, pre-registered, one-shot remediation rule).

## Goal Achievement

### Observable Truths (ROADMAP success criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `generate_episodes(n, seed)` reproducibly gives the 50/25/25 mix with self-written injections on non-InjecAgent tools; same seed gives same hash; leakage test passes on it and fails when a held-out tool (incl. `GmailSendEmail`), attacker address, canary or 8-word shingle is planted | VERIFIED | Independent check (own script, not the repo gate): train file 20,000 benign / 10,000 dh / 10,000 ds, dev 1,000/500/500; 0 episodes use any of the 79 held-out tools (`GmailSendEmail` is in the held-out set, absent from the data); no held-out tool name, "canary" or "private record" in the raw data; none of the 7 benchmark e-mail addresses in the data; 0 benchmark 8-word shingles (13,451 built from benchmark string values) occur in the rendered training text of 42,000 episodes. File sha256 values on disk equal `data/local/manifest.json` (train `a2586548c7f7...`, dev `1f6333eb8ac8...`). `tests/test_training_leakage.py` (planted-violation tests) and `tests/test_scratch_dataset.py` (determinism across PYTHONHASHSEED, mix, dev split) passed fresh: 59 + leakage file in the 122-test run. |
| 2 | Training and inference render through the same function (compact format v2, parameter names on the `T:` line), a parity test proves identical output, and a checkpoint whose format-version or vocab-hash stamp mismatches is rejected | VERIFIED | `inference.py` line 109 calls `format.render_context(messages)`; `dataset.training_text` builds on `render_context(episode_to_messages(...))`. `FORMAT_VERSION = 2`, CONTEXT_CHARS 2048, GEN_RESERVE 400 (printed from the import). `tests/test_format_parity.py` passed. Fresh tamper check on the REAL checkpoint: setting `format_version` to 1, `vocab_sha256` to zeros, or `render_fingerprint` to ones each makes `load_generator` raise `RuntimeError` naming the field; the untampered checkpoint loads. |
| 3 | `python -m src.scratch_llm.train` trains ScratchGPT on the RTX 3060 and writes `models/scratch/scratch_gpt.pt`, a manifest (seed, dataset hash, config, git SHA) and a loss curve | VERIFIED | `models/scratch/manifest.json`: device `cuda`, torch 2.11.0+cu128, seed 0, `params` 4,763,648, code_commit `1e2a020...`, dataset train/dev sha256 equal to the files on disk and to `data/local/manifest.json`, config block, `checkpoint_sha256` `97bdace562946f51...` equals the fresh `sha256sum` of `scratch_gpt.pt`. `train_log.jsonl` and `loss_curve.png` are tracked in git; weights are git-ignored by design and present on disk. Best step 11,750, early stop at 13,750; `dev_action_exact` 0.8037 (benign 0.897, dh 0.837, ds 0.677). Pre-registered remediation (0.656 < 0.75, still improving) fired exactly once and is recorded in `run_notes`. 122 CPU tests in `test_scratch_train.py` and siblings passed (non-finite loss exit 3, stale-checkpoint cleanup, masked loss decode, resume equality). |
| 4 | `scratch-gpt` runs end to end through the benchmark (provider loads the checkpoint via `inference.load_generator`) and produces valid actions on a smoke run | VERIFIED | `config/models.yaml` has `scratch-gpt` (provider `scratch`, checkpoint `models/scratch/scratch_gpt.pt`, `benchmark: true`); `src/llm/scratch.py` lazily calls `load_generator` and stamps `resolved_model`. Fresh read of `results/phase2-smoke-v2/raw.jsonl`: 26 rows, 26 unique (case, run, arm), `resolved_model` `scratch-gpt@97bdace56294` on all 26, provider `scratch`, config_model `scratch-gpt`, 0 `llm_error`, every `code_commit` 40 characters. 74 steps with 9 parse errors, so 65 valid actions (matches the REVIEW-FIX 65/74); the SCRATCH-05 "zero valid actions" failure condition is not met. Behaviour is plumbing evidence only, as the plan states. |
| 5 | Viva pages for LLM foundations and for synthetic-data/leakage design are in `docs/viva/` | VERIFIED | `docs/viva/02-llm-foundations-and-synthetic-data.md` (708 lines): Part A (tokenization, transformer, attention, loss mask, AdamW, overfitting, decoding), Part B (leakage gate, 50/25/25 mix, shared format), Part C (real numbers, code-review findings), limits, Part D with 18 Q&A. Part C figures match the manifest (params 4,763,648; best step 11,750; `dev_action_exact` 0.8037; hash prefixes) and cite no stale `83b0763a6d26`. `docs/viva/README.md` lists it as "written (Phase 2)". |

### Plan must-haves (non-redundant truths)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | Loss mask supervises exactly the `A:` payload plus newline (D-06) | VERIFIED | `supervised_mask` decode tests in `test_format_parity.py` and `test_scratch_train.py` passed; manifest `supervised_fraction` 0.405. |
| 7 | Prefix parity: `render_context(episode_to_messages(ep,k))` is a prefix of `training_text(ep)`, also against a real AgentLoop replay (D-06) | VERIFIED | `test_format_parity.py` passed (behavior-dependent, test-backed). |
| 8 | Leakage gate runs inside generation and in train (training refuses a leaking file before step 0); D-04 held-only parameter-name scope is a named constant written to the manifest | VERIFIED | `generate_episodes` -> `assert_clean`; `train.py` imports and calls `assert_clean` (grep + passing tests); `gate.held_only_param_scope` present in `data/local/manifest.json`; gate rejected counts recorded (WR-08): 0 leak rejections. |
| 9 | `inference.py` imports neither `dataset.py` nor `src.benchmark` | VERIFIED | Header comment and import block (lines 27-33) import only `format` and `model`; enforced by a test in `test_scratch_inference.py` (passed). |
| 10 | Over-long prompt is an `llm_error` (`context_overflow`), reply cap is 401 chars (WR-02, WR-03) | VERIFIED | `test_scratch_inference.py` passed (371-char action, 2,402-char prompt, runner-level episode). |
| 11 | Pre-registered remediation is dev-metric-only and applied at most once; nothing changed in response to the smoke run (superseded rule 0.75 meets the same property) | VERIFIED | Manifest `selection.remediation_rule` plus two `run_notes`; `remediation_needed(manifest)` tested; REVIEW-FIX states no change after the smoke run. |
| 12 | Plan 02-04 artifact links: manifest dataset hash == data manifest; every smoke row's `resolved_model` ends with `checkpoint_sha256[:12]` | VERIFIED | Both confirmed fresh (above). |

**Score:** 12/12 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `src/scratch_llm/{format,dataset,templates,provenance,model,train,inference,__init__}.py` | VERIFIED | Substantive, wired (train -> dataset/inference/provenance; inference -> format/model; dataset -> format/prompting/injecagent). |
| `src/llm/scratch.py`, `config/models.yaml` entry | VERIFIED | Wired to `load_generator`; config entry present. |
| `tests/test_format_parity.py`, `test_training_leakage.py`, `test_scratch_dataset.py`, `test_scratch_train.py`, `test_scratch_inference.py` | VERIFIED | 181 tests passed fresh across these files; the full WSL suite (901 passed) was reported by the caller. |
| `data/local/manifest.json` (tracked), `data/local/*.jsonl` (ignored, on disk) | VERIFIED | Hashes match. |
| `models/scratch/manifest.json`, `train_log.jsonl`, `loss_curve.png` (tracked); `scratch_gpt.pt` (ignored, on disk) | VERIFIED | Checkpoint hash matches manifest. |
| `docs/viva/02-...md`, `docs/viva/README.md` | VERIFIED | 708 lines, 18 Q&A. |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| dataset.py | format.py | `training_text` uses `render_context` | WIRED |
| dataset.py | agent_harness/prompting.py | `build_messages`, `OBSERVATION_TEMPLATE` | WIRED (plan-declared; parity tests pass) |
| dataset.py | benchmark/injecagent.py | `held_out_tools`, benchmark JSONL under `benchmarks/injecagent` build the corpus | WIRED |
| generate_episodes | assert_clean | gate on every batch | WIRED |
| train.py | inference.py | `current_stamps` / `CHECKPOINT_KEYS` shared contract | WIRED (real checkpoint loads; tamper rejected) |
| LocalScratchProvider | inference.load_generator | lazy import in `__init__` | WIRED |
| runner row | provider | `resolved_model` / `finish_reason` / `prompt_tokens` stamped | WIRED (26/26 rows) |
| models manifest | data manifest | `dataset.train_sha256` | WIRED (equal) |

### Data-Flow Trace (Level 4)

Not a rendering phase. The data chain was traced end to end: generator -> JSONL (hash-pinned) -> train.py (hash re-checked, gate re-run) -> checkpoint (stamped, hash in manifest) -> `load_generator` -> provider -> runner rows (resolved_model carries the checkpoint hash). All FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Data hashes and mix | `sha256sum` + own episode counter (WSL venv) | hashes equal manifest; 20k/10k/10k and 1k/500/500 | PASS |
| Independent leakage scan | own script: held-out names, emails, canary, 8-word shingles on rendered text | 0 / 0 / 0 / 0 | PASS |
| Stamp rejection on real checkpoint | tamper `format_version`, `vocab_sha256`, `render_fingerprint` | all three raise `RuntimeError` | PASS |
| Smoke rows | read `results/phase2-smoke-v2/raw.jsonl` | 26 unique rows, 0 llm_error, stamped | PASS |
| Targeted tests | pytest leakage + parity + inference + train | 122 passed in 157 s | PASS |
| Dataset tests | pytest `tests/test_scratch_dataset.py` | 59 passed in 19 s | PASS |

### Probe Execution

Step 7c: no probe scripts declared by the phase or present under `scripts/*/tests/probe-*.sh` for this phase. SKIPPED.

### Requirements Coverage

Requirement IDs declared in PLAN frontmatter: 02-01 [SCRATCH-01, 02, 03]; 02-02 [SCRATCH-01, 02]; 02-03 [SCRATCH-03, 04, 05]; 02-04 [SCRATCH-04, 05]. REQUIREMENTS.md maps SCRATCH-01..05 to Phase 2, all marked Complete; no orphaned IDs.

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCRATCH-01 | 02-01, 02-02 | Reproducible 50/25/25 synthetic episodes, self-written injections, non-InjecAgent tools | SATISFIED | Truth 1 |
| SCRATCH-02 | 02-01, 02-02 | Automated leakage test fails on planted held-out tool / attacker address / canary / 8-word shingle | SATISFIED | Truth 1, 8; planted-violation tests pass |
| SCRATCH-03 | 02-01, 02-03 | Same render function at train and inference; parity test; format-version/vocab-hash stamp | SATISFIED | Truth 2, 7 |
| SCRATCH-04 | 02-03, 02-04 | `train` CLI trains on the RTX 3060; checkpoint, manifest, loss curve | SATISFIED | Truth 3 |
| SCRATCH-05 | 02-03, 02-04 | `scratch-gpt` runs end to end in the benchmark with valid actions | SATISFIED | Truth 4 |

### Anti-Patterns Found

None blocking. Debt-marker scan (`TBD|FIXME|XXX`) over the phase's source, test and viva files returned no matches. Info items IN-01, IN-02, IN-05..IN-08 from the code review were deliberately left open by the user (non-blocking). `torch` is intentionally not in `requirements.txt` (documented, with a pytest header and `STRICT_TORCH=1`).

### Human Verification Required

None outstanding. The UAT (02-UAT.md) records 6/6 passed, including the human judgments on training quality, viva readability and acceptance of ScratchGPT as a weak worst-case principal (dev action exact 0.80, smoke parse 65/74, unprotected attempts 3/10, protected successes 0).

### Advisory notes (not blocking)

- `coincidental_reliance`: none found. Checkpoint, data and results are pinned by hash and verified against each other.
- The model weights, the JSONL datasets and `results/` are git-ignored or untracked by design. A fresh clone needs `python -m src.scratch_llm.dataset` (deterministic, hash-checked against the committed manifest) and the train CLI to reproduce them; retraining is not bit-reproducible, which the manifest and viva page state.
- Manifests record `code_dirty: true` because unrelated user files (papers, dashboard) are modified; the code commits are named. `src/dashboard/app.py` was excluded from the covered files.
- Smoke-run behaviour (unprotected attempts 3/10, ds weakest class at 0.677 on dev) is a reported finding for Phase 4, not a Phase 2 gap.

### Gaps Summary

No gaps. All five roadmap success criteria and the plan must-haves hold in the current codebase, including after the post-review fix and retrain. Phase goal achieved.

---

_Verified: 2026-10-22T09:30:00Z_
_Verifier: Claude (gsd-verifier)_
