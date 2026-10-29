---
phase: 05-comparison-report-deliverables
verified: 2026-10-29T13:10:00Z
status: human_needed
score: 4/4 must-haves verified (criteria 1 to 3 verified on the 07:12Z snapshot, by user decision; the final Groq numbers need the post-freeze re-run)
covered_files:
  - ".planning/phases/05-comparison-report-deliverables/05-01-PLAN.md"
  - ".planning/phases/05-comparison-report-deliverables/05-01-SUMMARY.md"
  - ".planning/phases/05-comparison-report-deliverables/05-02-PLAN.md"
  - ".planning/phases/05-comparison-report-deliverables/05-02-SUMMARY.md"
  - ".planning/phases/05-comparison-report-deliverables/05-03-PLAN.md"
  - ".planning/phases/05-comparison-report-deliverables/05-03-SUMMARY.md"
  - ".planning/phases/05-comparison-report-deliverables/05-04-PLAN.md"
  - ".planning/phases/05-comparison-report-deliverables/05-04-SUMMARY.md"
  - "README.md"
  - "docs/SDD.md"
  - "docs/viva/04-namespaces-seccomp-and-local-inference.md"
  - "docs/viva/05-evaluation-statistics-and-examiner-qa.md"
  - "docs/viva/README.md"
  - "src/benchmark/report.py"
  - "src/deliverables/__init__.py"
  - "src/deliverables/__main__.py"
  - "src/deliverables/blocks.py"
  - "src/deliverables/build.py"
  - "src/deliverables/charts.py"
  - "src/deliverables/consistency.py"
  - "src/deliverables/sections.py"
  - "src/deliverables/slides.py"
  - "tests/test_deliverables_build.py"
  - "tests/test_deliverables_check.py"
  - "tests/test_deliverables_sections.py"
  - "tests/test_deliverables_slides.py"
covered_digest: "v3:sha256:caa99bb54b1ddbc6576542e24e0ba89a5d80daeaa63286c3298f8a5ae042552a"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Open docs/OS_Level_Guardrails_Review_3.pptx in PowerPoint and read all 15 slides and their speaker notes aloud once"
    expected: "No clipped or overlapping text, the three charts are legible at projector size, the notes are speakable (slide 14 notes run about 500 words, slide 12 about 250), and the story is yours to defend."
    why_human: "Visual layout and speech quality cannot be checked by grep. The slide images I viewed (slides 8, 12, 14) came from the 14:21 render; the deck was rebuilt at 18:21 after the fix round. The numbers on those slides match the current numbers.json, and the deck check passes on the current file, but I did not re-render it."
  - test: "After the 2026-10-29 15:30 UTC freeze run the one-command rebuild, then re-run the check: PYTHONUTF8=1 py -3.12 -m src.deliverables build (a fresh freeze), then PYTHONUTF8=1 py -3.12 -m src.deliverables check and deck --check; close WINDOWS.md entry 2 (and entry 4's WR-01/IN-04/IN-05 decision)"
    expected: "freeze.json says data_state final (or snapshot with a NOT FINAL reason if a Groq model is not in a terminal state), the README, SDD and viva blocks are regenerated with the final Groq n, check prints deliverables-check-ok docs 4, and the deck is rebuilt with the same numbers."
    why_human: "Documented follow-up by user decision (D-01). The final Groq numbers do not exist yet. Every deliverable carries the snapshot time and the per-model n, so the drift is labelled, not hidden. I did not run build, as instructed."
  - test: "Export docs/SDD.md to PDF/DOCX with scripts/export_sdd.py (needs the CDN, so network) and commit the new SDD.pdf, SDD.docx and docs/build files if you want them in the Review 3 pack"
    expected: "The exported SDD says version 3.0 and has the generated results section. The tracked docs/SDD.pdf, docs/SDD.docx and docs/build/*.html are still the older exports."
    why_human: "Manual step by plan decision (D-06). The exporter needs network access. The Markdown source is current; the binaries are stale."
  - test: "Read the three review-fix items marked 'fixed: requires human verification' in 05-REVIEW-FIX.md: CR-01 (claim blocks that branch on numbers.json, src/deliverables/blocks.py), WR-01 (per-paragraph fraction rule, consistency.py), CR-03 (data_state final rule, build.py)"
    expected: "The generated sentences read correctly for the all-zero case today, and would read correctly if one Groq model had a protected success after the freeze; data_state cannot say final on a wall-clock alone."
    why_human: "The fixer asked for a human read because these change logic. Automated evidence: 122 deliverables tests pass in WSL (including the branch, tamper and data-state tests), and check passes on the real docs."
  - test: "unverified-prohibition, human review recommended: 05-01 'Nothing is written under results/phase4-* or results/final-20261011 and no process is started, stopped or signalled except the build itself'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS on the evidence I can see. results/phase5-final/sources/*/raw.jsonl hashes equal the freeze.json hashes and equal my own sha256 of each frozen copy; the live results/phase4-groq-* files keep growing (mtime later than the 07:12Z copy), which is the Groq loop, not this phase. I sent no signal and ran no build. I cannot prove the process history of earlier plan runs."
    why_human: "Judgment-tier prohibition, plan status unverified / flagged. Never a silent pass."
  - test: "unverified-prohibition, human review recommended: 05-01 to 05-04 'No step opens or prints .env, key values are never written to any output'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. check runs a secret scan over 33 frozen files (SECRETS results/phase5-final files 33 ok), the widened SECRET_RE is pinned by tests, and the deck check rejects key-shaped text. My own tool call that mentioned the .env path was blocked by the secret-read guard before it read anything."
    why_human: "Judgment-tier prohibition, plan status unverified / flagged."
  - test: "unverified-prohibition, human review recommended: 05-02, 05-03, 05-04 'No attack command, bypass recipe or payload text appears in README, SDD, viva pages 04/05 or any slide; the Phase 1 table is cited by row number only'"
    expected: "Non-authoritative LLM-judge verdict: HOLDS. A grep of README, SDD and viva 04/05 for the benchmark's known payload strings and destructive commands found nothing; slide 12 cites rows 1, 2, 8, 9, 10, 15, 16, 20, 21, 22 and 3-7, 11-14, 17-19, 23 by number and outcome class only. I read slides 8, 12 and 14 only; a human should skim the rest."
    why_human: "Judgment-tier prohibition about wording, plan status unverified / flagged."
---

# Phase 5: Comparison Report & Deliverables Verification Report

**Phase Goal:** The student can present and defend the multi-LLM + local-model result: one frozen data set drives a cross-model report, charts, README, SDD and Review-3 deck that cannot drift apart, backed by a complete set of viva notes.
**Verified:** 2026-10-29T13:10Z (the Groq loop was still running; the frozen data is the 07:12Z snapshot)
**Status:** human_needed (all automated truths verified; the deck review, the post-freeze re-run and the SDD export need a person or the clock)
**Re-verification:** No, initial verification

The goal is achieved for everything that can be achieved before the freeze. I took nothing from SUMMARY.md. I recounted the per-model attack counts and the per-layer blocks from the frozen `raw.jsonl` files with my own script, compared the sha256 of every frozen copy with `freeze.json`, ran the deliverables check on Windows, ran the 4 deliverables test modules in WSL, opened the three PNG charts, opened the real `.pptx` with python-pptx and viewed three rendered slides. I did not run `build`, wrote nothing under `results/phase4-*`, signalled no process and did not open `.env`.

## Goal Achievement

### Observable Truths (ROADMAP success criteria, the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | One script regenerates the cross-model comparison report from a single frozen results directory, with the headline table, the confounds table (size, quantization, local/hosted, context), the where-blocked-by-layer table and a threats-to-validity section | VERIFIED (on the snapshot) | `python -m src.deliverables build [--reuse-frozen]` exists (`src/deliverables/__main__.py`, `build.py` 721 lines); `results/phase5-final/` holds `freeze.json`, `numbers.json`, `summary.md`, `summary.csv`, `layer_blocks.csv`, `charts/`, `sources/` and `inputs/`. `summary.md` has "Headline: complete runs" (l.64), "Confounds between the models" (l.88; parameters, quantization, local/hosted, context, decoding, provenance), "Blocks by layer, per model and pooled (EVAL-08)" (l.240), "Threats to validity" (l.313) and a separate "Legacy rows (older engine, not compared)" table (l.103). Sources hash check: all 11 frozen `raw.jsonl` sha256 values I computed equal `freeze.json`. |
| 2 | The report includes static charts for ASR with confidence intervals per model, a capability-vs-vulnerability scatter, and a utility-vs-ASR arrow plot (unprotected to protected) | VERIFIED (cosmetic warnings) | `charts/asr_ci.png`, `capability_vs_vulnerability.png`, `utility_vs_asr_arrows.png` exist and I looked at each. All three are labelled with k/n, mark the three Groq rows hollow as partial, state the snapshot and freeze time, and carry the control reference line on the first. Legacy rows are not plotted. Chart geometry (error-bar span, hollow partial markers) is pinned by a test. Cosmetic: the `111/120` label sits on top of the error-bar cap in `asr_ci.png`; in the arrow plot the filled and hollow markers of the near-zero models overlap at the bottom right and the leader lines cross. Legible, not polished. |
| 3 | The README and the IEEE 1016 SDD describe the multi-LLM + local-model system and its results (no longer Gemini-only), and a Review-3 slide deck presents the reviewer remark, the local models and the comparison results | VERIFIED (deck visuals: human) | README: "Review 3: a locally made LLM and other LLMs" (l.10), "Models compared" (l.69), "Results (frozen Phase 5 data)" (l.200) with 5 generated blocks; Gemini appears only as a key note and legacy rows. SDD: version 3.0, sections 5.13 (provider layer), 5.14 (local training), 5.15 (run tooling and deliverables), 7.3 (results, 9 generated blocks), Appendix C (Review 3 remark to design changes). Deck: `docs/OS_Level_Guardrails_Review_3.pptx` opens with python-pptx, 15 slides, 16:9, 15 of 15 have notes, three embedded charts; slides cover the remark (2), architecture (3), ScratchGPT (4), Qwen LoRA (5), line-up and confounds (6), method (7), headline (8), the charts (9 to 11), blocks by layer with the honest zero (12), the control (13), limitations (14) and next steps (15). `deck --check` prints `DECK ... slides 15 notes 15 charts 3 ok`. |
| 4 | `docs/viva/` is complete: OS security, LLM foundations, prompt injection and defenses, evaluation statistics, plus examiner Q&A; every earlier phase's page is present and this phase adds the Q&A and threats-to-validity page | VERIFIED (teaching quality: human) | Pages 01 to 05 plus `README.md` index exist (658, 708, 608, 678, 923 lines); the index lists all five with topics and phase. Page 05 has Part A (prompt injection and defenses), B (statistics: Wilson, rule of three, paired arms, ASR vs ASR-valid vs conditional block rate, completeness gating, pseudo-replication), C (confounds and threats), D (what the numbers say, generated), E (defend the result in one page) and F (40 Q&A across OS security, namespaces and seccomp, LLM, LoRA, injection, statistics, results). Page 04's former placeholder "What the code review found and what we fixed" (l.471) is now filled (CR-01 and the other findings). No `TODO`, `TBD`, `placeholder` or `To be filled` text in README, SDD or any viva page. |

**Score:** 4/4 truths verified (0 present-but-behavior-unverified).

### Plan-level checks (not reduced from the ROADMAP)

| Plan-level truth | Status | Evidence |
|-----------------|--------|----------|
| One source of numbers: docs and deck cannot drift (D-03) | VERIFIED | `PYTHONUTF8=1 py -3.12 -m src.deliverables check` printed `DOC README.md blocks 5 ok`, `DOC docs/SDD.md blocks 9 ok`, `DOC ...viva/04... blocks 11 ok`, `DOC ...viva/05... blocks 17 ok`, `SECRETS results/phase5-final files 33 ok`, `deliverables-check-ok docs 4` (the required line). `load_numbers` ties `numbers.json` to `freeze.json` (IN-08). The deck check applies the same rules to every slide and note. |
| The numbers are real, not just consistent | VERIFIED | My own recount from the frozen `raw.jsonl` (non-benign, non-error rows): unprotected successes lora 111/120, scratch 33/120, base 3/120, ollama-qwen 3/120, ollama-llama 7/120, control 120/120; protected successes 0 for every model and the control. Groq rows 4/95, 2/51, 0/98 match after the report's pair dropping. Blocks by layer: tool broker 245 and policy engine 60 for the 8 language models (I summed the same totals, control excluded), sandbox and seccomp 0 in all current-engine rows (the only sandbox and seccomp blocks in my scan are in the legacy Gemini and Gemma files, which are not compared). |
| Data labelled as a snapshot, partial rows not ranked (D-01, D-02) | VERIFIED | `freeze.json`: `data_state: snapshot`, snapshot 2026-10-29T07:12Z, freeze 15:30Z. Every table, chart footer and slide footer carries the data line; partial Groq rows show `partial n/342` and appear hollow; the legacy Gemini and Gemma rows are only in their own table and the threats text. |
| Freeze safety and rebuild behaviour (CR-02, CR-03, WR-03: behavior-dependent state transitions) | VERIFIED | Named behavioural tests in the 122 that passed: overlap and symlink refusal of `--frozen`, render failure leaves the previous frozen directory byte-identical, `data_state` returns final only with terminal hosted states plus freeze evidence. Not left PRESENT_BEHAVIOR_UNVERIFIED. The 2 skipped tests are the python-pptx render tests; they ran on Windows `py -3.12` in the plan runs, and I ran `deck --check` on the real file there. |
| Review findings fixed | VERIFIED | `05-REVIEW-FIX.md`: 11 of 11 in scope fixed (CR-01 to CR-03, WR-01 to WR-06, IN-02 symlink, IN-08), each with its commit in `git log` (675aecc, b1d5fe2, 2ecd532, e0312c6, 50682a5, 875af7d, 96e8fea, 3996ef5, 40b8624, 4d7d4e3). Info items IN-01, IN-03 to IN-06 are recorded as skipped, out of scope. |
| Plan requirements coverage | VERIFIED | 05-01 DOC-01, DOC-02; 05-02 DOC-04; 05-03 DOC-03; 05-04 DOC-05. REQUIREMENTS.md DOC-01..05 are checked and marked Complete. No orphaned requirement ids. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/deliverables/{__init__,__main__,build,sections,charts,blocks,consistency,slides}.py` | the one-command engine, 3641 lines | VERIFIED | Imported by the CLI and exercised by the tests and my `check` and `deck --check` runs. No `TODO`, `TBD`, `FIXME`, `XXX` or `NotImplemented` in `src/deliverables` or its tests. |
| `src/benchmark/report.py` (extended, not a parallel report) | confounds, legacy and threats data | VERIFIED | D-04 followed: the deliverables build calls `generate_report`, and fails if its summary differs from the one `numbers.json` came from. |
| `results/phase5-final/{freeze.json,numbers.json,summary.md,summary.csv,layer_blocks.csv,charts/*.png,sources/,inputs/}` | frozen data set and outputs | VERIFIED | Present; hashes tie out; untracked by design (results/ is not staged). |
| `README.md`, `docs/SDD.md` | updated in place with generated blocks | VERIFIED | 347 and 962 lines; blocks checked above. |
| `docs/viva/05-...md`, `docs/viva/04-...md`, `docs/viva/README.md` | page 05, page 04 with generated Part C, index | VERIFIED | Above. |
| `docs/OS_Level_Guardrails_Review_3.pptx` | 15 slides with notes, template-based | VERIFIED (visuals: human) | Binary, untracked by design (D-08). |
| `tests/test_deliverables_{build,sections,check,slides}.py` | tests | VERIFIED | `venv/bin/python -m pytest` over the four modules in WSL: 122 passed, 2 skipped (python-pptx render tests, not importable in the WSL venv), 65 s. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `build` | `results/phase5-final` | freeze, hash, stage, swap | WIRED | `freeze.json` lists 11 sources with raw hashes that match my recompute; inputs (models.yaml, two manifests) hashed too. |
| `numbers.json` | README, SDD, viva 04/05, deck | marker blocks (`deliverables:begin/end`) and `blocks.render_block` | WIRED | 5 + 9 + 11 + 17 blocks, all `ok`; the deck is generated from the same file. |
| `check` | docs and frozen directory | `consistency.scan_text`, `tie_to_freeze`, secret scan | WIRED | The check ran green; the tamper tests (edited number, key-shaped text, edited `numbers.json`) are among the 122 that pass. |
| deck charts | `results/phase5-final/charts` | sha256 comparison in `check_deck` | WIRED | `charts 3 ok`. |
| README charts | `results/phase5-final/charts/*.png` | relative image links | WIRED, but see warning W1 | The files exist on this machine; `results/` is untracked, so a fresh clone would show broken images. |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| `summary.csv`, `summary.md` | counts, k/n | frozen `raw.jsonl` of 9 current sources | Yes; my recount matches | FLOWING |
| `layer_blocks.csv` | blocks per layer | `blocked_by` in protected rows | Yes; zeros are real (no sandbox or seccomp key in current rows) | FLOWING |
| README, SDD, viva blocks | headline, findings, claims | `numbers.json` | Yes; the claim sentences branch on the data | FLOWING |
| deck slides | tables, charts | `numbers.json` and frozen PNGs | Yes; slide 8 and 12 values equal the csv and my recount | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| docs and secrets consistent | `PYTHONUTF8=1 py -3.12 -m src.deliverables check` (Windows) | `deliverables-check-ok docs 4` | PASS |
| deck matches data | `PYTHONUTF8=1 py -3.12 -m src.deliverables deck --check` | `DECK docs/OS_Level_Guardrails_Review_3.pptx slides 15 notes 15 charts 3 ok` | PASS |
| deliverables tests | WSL, `pytest` over the 4 modules (scratchpad `v5_tests.sh`) | 122 passed, 2 skipped | PASS |
| frozen hashes | own script `v5_recount.py` vs `freeze.json` | all 11 equal | PASS |
| per-model counts | same script | match the report | PASS |

### Probe Execution

SKIPPED. Phase 5 declares no probe scripts (`scripts/*/tests/probe-*.sh`).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOC-01 | 05-01 | one script, frozen directory, headline, confounds, by-layer, threats | SATISFIED | Truth 1 |
| DOC-02 | 05-01 | three static charts | SATISFIED | Truth 2 |
| DOC-03 | 05-03 | viva notes for every component plus Q&A, each phase adds its page | SATISFIED (teaching quality: human) | Truth 4 |
| DOC-04 | 05-02 | README and SDD multi-LLM and local | SATISFIED | Truth 3 |
| DOC-05 | 05-04 | Review-3 deck | SATISFIED (visuals: human) | Truth 3 |

No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/deliverables/*.py`, tests | n/a | `TBD`, `FIXME`, `XXX`, `TODO`, `NotImplemented` | none found | Debt-marker gate clear |
| `docs/viva/*.md`, README, SDD | n/a | `TODO`, `placeholder`, `To be filled` | none found | The Phase 4 placeholder is gone |
| `.planning/WINDOWS.md` entry 3 | n/a | open ledger entry for a placeholder that is now filled | Warning | Stale ledger: `docs/viva/04` l.471 holds the real section. Close it with `gsd-tools windows fixed 3`. Entry 2 stays open until the post-freeze re-run. |
| `results/phase5-final/charts/asr_ci.png`, `utility_vs_asr_arrows.png` | n/a | label on top of an error-bar cap; overlapping markers and crossing leader lines | Info | Legible, cosmetic. |
| `docs/SDD.pdf`, `docs/SDD.docx`, `docs/build/*.html` | n/a | tracked exports older than `docs/SDD.md` 3.0 | Warning | Known manual step (D-06). |

### Warnings (not gaps)

- **W1: README charts point into untracked `results/`.** D-05 allowed referencing the charts from `results/phase5-final/` when copying to `docs/images/` is not allowed. A clone of the repository will show broken image links unless the results directory travels with it. Copy the three PNGs to `docs/images/phase5/` if the README must render from git alone.
- **W2: the deck I rendered is older than the deck on disk.** See human item 1.
- **W3: Info findings IN-01, IN-03 to IN-06 from 05-REVIEW remain open** (silent arm drop in `asr_ci.png` when n is 0, hard-typed "57 cases (20 dh, 20 ds, 17 benign)" in the threats text, undecodable lines not surfaced, timestamp notation, stale SDD exports). None affects a number in the deliverables today.

### Human Verification Required

See the `human_verification` list in the frontmatter: (1) deck visual and speaker-notes read, (2) the post-freeze `build` re-run and WINDOWS.md closure, (3) the SDD PDF/DOCX export, (4) a human read of the three logic fixes, and (5 to 7) three judgment-tier prohibitions flagged, not silently passed.

### Gaps Summary

No gaps. Every ROADMAP criterion is verified against the artefacts, the numbers reproduce from the frozen raw data, the check and the deck check pass, and 122 tests pass. The status is `human_needed` because the work is explicitly staged around a clock and a human: the Groq numbers are a 07:12Z snapshot until the post-freeze re-run, the SDD binaries are a manual export, the deck needs a person to look at it and rehearse the notes, and three fixes that change logic were marked by the fixer for a human read.

---

_Verified: 2026-10-29T13:10Z_
_Verifier: Claude (gsd-verifier)_
