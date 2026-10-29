---
phase: 05-comparison-report-deliverables
fixed_at: 2026-10-29T13:05:00Z
review_path: .planning/phases/05-comparison-report-deliverables/05-REVIEW.md
iteration: 1
findings_in_scope: 11
fixed: 11
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-10-29T13:05:00Z
**Source review:** .planning/phases/05-comparison-report-deliverables/05-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 11 (CR-01..CR-03, WR-01..WR-06, IN-02 symlink part, IN-08)
- Fixed: 11 (CR-01, CR-03 and WR-01 change logic and are marked for human verification)
- Skipped: 0 in scope; the other Info findings were out of scope and are listed under Skipped Issues

**Where verification ran.** `workflow.use_worktrees` is `false`, so every edit, build and commit happened in the
main checkout on branch `docs/sdd` (no worktree). The builds ran there on Windows with
`PYTHONUTF8=1 py -3.12 -m src.deliverables build --reuse-frozen`. The tests ran in the WSL venv of the same
checkout (`venv/bin/python -m pytest`). The two python-pptx render tests ran on Windows `py -3.12` through the
05-04 stand-in runner. No re-freeze happened: the frozen sources and their hashes are those of the 07:12Z
snapshot, and nothing was written under `results/phase4-*`.

## Fixed Issues

### CR-01: The headline result is restated in hand-written prose outside the generated blocks

**Files modified:** `src/deliverables/blocks.py`, `src/deliverables/consistency.py`, `tests/test_deliverables_check.py`, `README.md`, `docs/SDD.md`, `docs/viva/04-namespaces-seccomp-and-local-inference.md`, `docs/viva/05-evaluation-statistics-and-examiner-qa.md`
**Commit:** 675aecc
**Status:** fixed: requires human verification (the claim sentences branch on the data)
**Applied fix:**
- Five new generated blocks, each one paragraph that branches on `numbers.json`:
  - `claim-protected`: whether any model succeeded with guardrails, and the control's protected result.
  - `claim-blocked`: whether every attempted attack was blocked at the attack action, with the pooled k/n.
  - `claim-kernel`: the sandbox and seccomp counts, and whether the layers in front blocked everything first.
  - `claim-overflow`: the context-overflow count behind the WR-01 bias.
  - `claim-state`: snapshot or final, and which rows are partial.
- A claim block may repeat in a doc. Every other block still appears at most once.
- Every outcome sentence the review listed in README, SDD and viva 04/05 now comes from one of these blocks, or is reworded so that it states no result and points to the block. This covers the framing paragraphs, Part C/E, Q8, Q12, Q21, Q33-Q35 and Q37.
- The viva/04:291 contradiction is gone. Part C now says the prose between the tables and generated sentences states no Groq-dependent result.
- `check` fails on a paragraph outside the blocks that states a headline outcome in its own words. It matches `OUTCOME_PATTERNS` across wrapped lines.
- Hard-typed numbers that the review called stable (local rows) were kept where they are conditional or design figures, for example "a zero is 0 of 120, upper bound 3.1 %". The llama "0/342" lines became "none of its 342 episodes".

### WR-01: The fraction rule is per line and only for lines naming a model

**Files modified:** `src/deliverables/consistency.py`, `tests/test_deliverables_check.py`, `README.md`, `docs/viva/04-namespaces-seccomp-and-local-inference.md`, `docs/viva/05-evaluation-statistics-and-examiner-qa.md`
**Commit:** b1d5fe2
**Status:** fixed: requires human verification (new check rules)
**Applied fix:**
- In the checked docs, fractions are attributed to the models named anywhere in their paragraph, so a wrapped model name still counts.
- `k of n` and `k of its n` count as fractions.
- A fraction in a paragraph that names no model must still be some generated value. Any bare `P%` or `P %` must be a generated pct, lo or hi, or the allowed 95.
- The test that blessed a model-less `999/1000` is removed. Doc-level tests cover wrapping, `k of n`, model-less fractions and bare percentages.
- The README attack-simulation table, which is not benchmark data, is now written in words.
- The viva/04 and viva/05 sentences about what `check` does now match it.
- Not done: restricting a fraction to the metric family the sentence names. The module docstring states this limit.
- The deck's `scan_text` keeps its per-line rule, because every slide line is generated.

### CR-02: `--frozen` replaces and then deletes whatever directory it names

**Files modified:** `src/deliverables/build.py`, `src/deliverables/__main__.py`, all four `tests/test_deliverables_*.py`
**Commit:** 2ecd532
**Applied fix:**
- `check_frozen_target` runs before any write, both on a fresh freeze and on `--reuse-frozen`.
- After resolving symlinks with realpath, the target must be a `phase5-*` directory directly under the results root.
- It must not overlap any source.
- If it exists and is not empty, it must hold a `freeze.json`.
- `_under` now compares realpaths. This covers the IN-02 symlink part.
- The tests keep their frozen directories under the synthetic results root. New tests cover `phase4-jobs`, a source, the legacy root, `docs/`, a nested path, `.`, a symlink to `phase4-jobs` and a non-frozen `phase5-*` sibling. Each is refused and left untouched, through `freeze()`, `build()` and the CLI.

### CR-03: The post-freeze build labels its snapshot `final` from the wall clock alone

**Files modified:** `src/deliverables/build.py`, `src/deliverables/__init__.py`, `src/deliverables/blocks.py`, `tests/test_deliverables_build.py`
**Commit:** e0312c6
**Status:** fixed: requires human verification (state rule)
**Applied fix:**
- `data_state()` returns `final` only when all three of these hold:
  1. The snapshot clock is at or after `FREEZE_UTC`.
  2. Every hosted model's frozen `hosted_state.json` is in `jobs.HOSTED_TERMINAL`. That tuple is mirrored in `src/deliverables/__init__.py` and pinned by a test that reads `jobs.py`.
  3. The frozen states show that the loop reached the freeze on its own clock: a recorded timestamp at or after the freeze (launch start or end, `next_launch_utc`, or a time in a stop reason), or a model in `stopped_at_freeze`. A run where every hosted model completed needs neither.
- Otherwise the freeze is `snapshot` and records a `data_state_reason`. The build prints it as a `NOT FINAL ...` line.
- A post-freeze snapshot gets its own data line: "copied after the D-09 freeze ... not final".
- `hosted_state.json` is now copied before `raw.jsonl`. The loop writes a terminal state only after the model's child has exited, so a terminal state copied first means the raw copy is complete. This also fixes the IN-02 copy-order bullet.
- Point for the human check: the loop sets `stopped_at_freeze` with no timestamp for a model that is waiting on its quota at 15:30Z. The rule therefore accepts the state name itself as evidence that the freeze was reached.

### WR-03: A render failure after a successful freeze destroys the previous good frozen snapshot

**Files modified:** `src/deliverables/build.py`, `tests/test_deliverables_build.py`
**Commit:** 50682a5
**Applied fix:**
- `build()` writes everything into the hidden staging sibling first:
  - A fresh run freezes into it (`stage_freeze`).
  - `--reuse-frozen` copies the verified sources and `freeze.json` into it (`stage_reuse`).
- The report, charts and `numbers.json` are rendered in the staging directory. It is swapped in only after the render succeeds, and the old directory is deleted only after the swap.
- A test makes the render fail after a good freeze (an unknown model), and makes chart drawing fail after the report was written on `--reuse-frozen`. In both cases the previous frozen directory stays byte-identical, and no staging directory is left behind.
- A real `--reuse-frozen` render left `numbers.json` and `summary.md` byte-identical (sha256 83c8e770..., e0529397...).

### WR-04: `numbers.json` is not a pure function of the frozen directory

**Files modified:** `src/deliverables/build.py`, `src/deliverables/sections.py`, `tests/test_deliverables_sections.py`
**Commit:** 875af7d
**Applied fix:**
- The freeze copies `config/models.yaml`, `models/scratch/manifest.json` and `models/qwen-agentlora/manifest.json` into `frozen/inputs/` and records their sha256 in `freeze.json`. `read_freeze` verifies them.
- `render()` reads the confounds and local-training figures only from `inputs/`.
- A freeze from before this change adopts the inputs once on `--reuse-frozen`. The build prints `INPUTS adopted 3 ...`. Sources and their hashes are unchanged.
- The real 07:12Z freeze adopted its inputs this way. The working-tree files are unchanged since that freeze's commit, and the re-rendered `numbers.json` is byte-identical.
- The hf_local precision and context now follow the model's `device` and `max_context_tokens` config entries, the same rule the provider uses. The current output is unchanged.
- The hosted and Ollama precision strings stay as they were.

### WR-05: The build path drops the template-fallback signal

**Files modified:** `src/deliverables/build.py`, `src/deliverables/slides.py`, `src/deliverables/__main__.py`, `tests/test_deliverables_slides.py`
**Commit:** 96e8fea
**Applied fix:**
- `deck_step` prints `TEMPLATE <how>` before the `DECK` line.
- When the template could not be opened (`slides.is_blank_fallback`), the build fails with a `FAIL deck:` line instead of ending `BUILD ok`.
- `run_deck`, behind the `deck` command, does the same.
- `--allow-blank-template` on `build` and `deck` accepts a blank deck explicitly.
- The real build printed `TEMPLATE .../docs/template.pptx`.

### WR-02: The secret scan only recognises Groq keys

**Files modified:** `src/deliverables/consistency.py`, `src/benchmark/run_checks.py`, `tests/test_deliverables_check.py`
**Commit:** 3996ef5
**Applied fix:**
- `SECRET_RE` now also matches `AIza...`, `sk-or-v1-...`, any other `sk-` key and `hf_` tokens. It is changed in `run_checks` and in its pinned copy, which the doc check and the deck check use.
- Guards stop ordinary words such as `risk-...` and `hf_local...` from matching. No tracked file, frozen file or deck matches.
- `scan_frozen` reports a file it cannot open as a FAIL line instead of skipping it.
- It reads UTF-16 files that start with a byte-order mark.
- `run_checks.py` is not in `RUN_AFFECTING_PATHS`, and the live loop never calls the scan.

### IN-08: `check` trusts `numbers.json` without tying it to `freeze.json`

**Files modified:** `src/deliverables/consistency.py`, `tests/test_deliverables_check.py`
**Commit:** 40b8624
**Applied fix:**
- `load_numbers` calls `tie_to_freeze` before `blocks`, `check`, `deck` or `build` read `numbers.json`.
- The schema, snapshot, freeze, data state and source hashes must equal `freeze.json`, and every frozen `raw.jsonl` must still match its hash.
- A hand-edited `numbers.json` or a changed frozen copy fails with one FAIL line.

### IN-02 (symlink part): `_under` uses `abspath`, not `realpath`

**Files modified:** `src/deliverables/build.py`
**Commit:** 2ecd532 (with CR-02)
**Applied fix:** `_under` resolves both paths with `os.path.realpath`. The frozen target is checked on its realpath, and a symlink test covers it. The copy-order bullet of IN-02 was fixed under CR-03 (e0312c6). The stale `raw_lines` bullet is out of scope; see below.

### WR-06: Test gaps on the riskiest paths

**Files modified:** `src/deliverables/build.py`, `src/deliverables/slides.py`, all four `tests/test_deliverables_*.py`
**Commit:** 4d7d4e3 (plus the tests that landed with CR-01, WR-01, CR-02, CR-03 and WR-03)
**Applied fix:**
- **Hermetic fixtures.** `build()` resolves `repo_root` at call time. `CODE_ROOT` keeps the git HEAD lookup on the real checkout. An autouse `hermetic_repo` fixture, shared by the four modules, points the build at a fake repository, and a test pins that no real input reaches a test freeze.
- **Chart geometry.** The ASR chart test now checks the drawn geometry: each error bar spans `pct-(pct-lo)` to `pct+(hi-pct)` at its model's x, and a partial model's marker is hollow.
- **Deck check without python-pptx.** `check_deck`'s text rules moved into the pure `slides.text_problems`. A WSL test runs it on the deck spec: a hand-edited number, the widened key regex, a legacy id and a missing data line.
- **Earlier tests.** The overlap/misuse, render-failure, data-state and prose-outside-blocks tests landed with the findings above.

## Skipped Issues

These Info findings were out of scope for this run, as the request specified.

### IN-01: `asr_ci.png` silently drops an arm with n = 0

**File:** `src/deliverables/charts.py:101-102`
**Reason:** out of scope (Info, not requested)
**Original issue:** the ASR chart `continue`s on an empty arm without the "Not plotted" note the other two charts add.

### IN-02 (stale `raw_lines` bullet)

**File:** `src/deliverables/build.py` (copy loop)
**Reason:** out of scope (the request asked only for the realpath part of IN-02). The copy-order bullet was fixed incidentally under CR-03.
**Original issue:** `raw_lines`/`raw_sha` carry over from the previous source if `raw.jsonl` vanishes between the pre-check and the copy.

### IN-03: The threats text hard-codes the benchmark shape

**File:** `src/deliverables/sections.py:214-215`
**Reason:** out of scope (Info, not requested)
**Original issue:** "57 cases (20 dh, 20 ds, 17 benign)" is typed in, not derived the way `slides._bench` derives it.

### IN-04: Undecodable lines in frozen copies are never surfaced

**File:** `src/deliverables/build.py:245-249`, `src/benchmark/report.py:906-909`
**Reason:** out of scope (Info, not requested)
**Original issue:** `load_frozen_rows` discards the skipped-line count, so a corrupt mid-file line never reaches `numbers.json` or the check.

### IN-05: The timestamp and percentage rules see only one notation

**File:** `src/deliverables/consistency.py:46-47`
**Reason:** out of scope as an Info item. WR-01 now covers percentages with a space ("3.1 %"). Timestamps in the "15:30 UTC on 2026-10-29" form are still not checked.
**Original issue:** only ISO minute timestamps and "P% (k/n" percentages were checked.

### IN-06: Tracked SDD exports are not regenerated or checked

**File:** `docs/SDD.pdf`, `docs/SDD.docx`, `docs/build/SDD.html`, `docs/build/SDD.static.html`
**Reason:** out of scope (Info, not requested)
**Original issue:** the exports predate Phase 5 and are outside `DOC_TARGETS`.

### IN-07: Hand-typed counts in the README

**File:** `README.md:282-284, 304`
**Reason:** out of scope (Info, not requested). Under WR-01 the attack-simulation table is now written in words. The "1308 passed and 1 skipped" test count is unchanged.
**Original issue:** static counts that do not come from `numbers.json`.

## Verification

- `PYTHONUTF8=1 py -3.12 -m src.deliverables build --reuse-frozen` (Windows, main checkout) ended with
  `BUILD ok`, after `deliverables-check-ok docs 4`, `TEMPLATE .../docs/template.pptx` and
  `DECK docs/OS_Level_Guardrails_Review_3.pptx slides 15 notes 15 charts 3 ok`. The deck was regenerated.
- An unscoped `PYTHONUTF8=1 py -3.12 -m src.deliverables check` printed DOC lines for README (5 blocks),
  the SDD (9), viva 04 (11) and viva 05 (17), then `SECRETS results/phase5-final files 33 ok` and
  `deliverables-check-ok docs 4`.
- WSL venv, `pytest tests` without the four GPU-heavy files (`test_qwen_lora_train.py`,
  `test_hf_local_provider.py`, `test_scratch_train.py`, `test_scratch_inference.py`): 1229 passed, 2 skipped,
  0 failed (279 s). None of those four files touches code changed here. The 2 skips are the python-pptx tests,
  which ran on Windows.
- Windows `py -3.12` (python-pptx): `test_render_on_the_template_then_check` and
  `test_check_rejects_a_stale_chart_and_a_number_edited_by_hand` passed (2 passed, 0 failed).

---

_Fixed: 2026-10-29T13:05:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
