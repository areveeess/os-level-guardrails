---
phase: 05-comparison-report-deliverables
reviewed: 2026-10-29T12:11:13Z
depth: standard
files_reviewed: 18
files_reviewed_list:
  - src/benchmark/jobs.py
  - src/benchmark/report.py
  - src/benchmark/run_checks.py
  - src/deliverables/__init__.py
  - src/deliverables/__main__.py
  - src/deliverables/blocks.py
  - src/deliverables/build.py
  - src/deliverables/charts.py
  - src/deliverables/consistency.py
  - src/deliverables/sections.py
  - src/deliverables/slides.py
  - tests/test_deliverables_build.py
  - tests/test_deliverables_check.py
  - tests/test_deliverables_sections.py
  - tests/test_deliverables_slides.py
  - tests/test_jobs.py
  - tests/test_phase4_runner.py
  - tests/test_run_checks.py
findings:
  critical: 3
  warning: 6
  info: 8
  total: 17
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-10-29
**Depth:** standard
**Files Reviewed:** 18
**Status:** issues_found

## Summary

**Scope.** I reviewed the source and tests changed in `6017829..HEAD -- src tests`. That covers the new `src/deliverables/` package, the Phase 5 sections added to `src/benchmark/report.py`, the WR-02/03/04 follow-ups in `jobs.py` and `run_checks.py`, and the new tests. As the request asked, I also checked the generated-block mechanism against `README.md`, `docs/SDD.md`, `docs/viva/04-...` and `docs/viva/05-...`.

**Code read alongside.** `src/benchmark/stats.py` (Wilson), `outcomes.classify_llm_error`, `report.generate_report`, the hosted loop's freeze path in `jobs.py`, `config/models.yaml`, and `.env.example` (variable names only).

**What I ran.** All of it was read-only:

- `python -m src.deliverables check` against the existing frozen directory. It printed `deliverables-check-ok docs 4`.
- A script that compares every `numbers.json` metric with `results/phase5-final/summary.csv`.
- Scripts that list numeric and result-claim lines outside the generated blocks.
- A count-only grep for key-shaped strings. No match was printed.

I did not run `build`, sent no signal, did not read `.env` and did not commit.

**What holds up:**

- **Number correctness.** All 9 rows and every `PREFIX_{k,n,pct,lo,hi}` in `numbers.json` equal `summary.csv` (0 differences). `render()` also guards this by comparing the report's own summary with the one `numbers.json` came from.
- **Pooled figures.** The pooled conditional block rate, 200/200, is the sum of the per-model values (111+36+6+10+9+18+7+3). The pooled attack-arm n, 844, is the sum of `attack_pairs`.
- **Chart intervals and labels.** The charts draw their error bars from the stored Wilson `lo`/`hi`. The k/n labels come from the same metric dicts as the headline block, so the n labels match the tables.
- **Block re-rendering.** The stale-block rule really does re-render and compare, so a hand edit inside a block fails `check`.
- **Freeze copy.** The freeze opens sources only with `rb` and copies complete lines only. A failed freeze keeps the old frozen directory.

**What is wrong:**

- **Result claims outside the blocks (main issue).** The D-03 guarantee covers only the generated blocks and model-named k/n tokens. Both viva pages state the headline result in hand-written prose outside the blocks: zero protected successes "for any model, local or hosted", every attempt blocked, kernel layers zero, no context overflow, and the hosted rows "labelled partial". The Groq data is still changing until 15:30Z, and the check cannot see any of these claims (CR-01, WR-01).
- **Deleting directories.** `--frozen` can delete any existing directory (CR-02).
- **Post-freeze build.** The post-freeze build is labelled `final` by the wall clock alone, while the hosted loop can still be appending (CR-03).
- **Secret scan.** It only knows Groq keys (WR-02).

## Critical Issues

### CR-01: The headline result is restated in hand-written prose outside the generated blocks, so it can become false after the post-freeze build while `check` still passes

**Files:**
- `docs/viva/05-evaluation-statistics-and-examiner-qa.md:11-17, 299-300, 313, 332-333, 348, 568-587, 599-601, 610-612, 616, 753, 830-833, 836, 845, 857`
- `docs/viva/04-namespaces-seccomp-and-local-inference.md:10-14, 291, 338-341, 495, 575-577`
- `README.md:311`

**Mechanism:** `src/deliverables/consistency.py:112-135`.

**Issue:** The check verifies only three things outside blocks:
- a `k/n` token on a line that names a model or says "pooled";
- an ISO `YYYY-MM-DDTHH:MM` timestamp;
- a `P% (k/n` percentage.

Qualitative and "k of n" claims are invisible to it. The pages assert, outside any block, facts that depend on the Groq rows that are still being collected:

| Doc:line | Claim outside a block | Depends on live data |
|---|---|---|
| viva/05:11-13, 830-831 | "no attack succeeded for any current-engine model ... to the hosted models" | hosted `success_protected.k` |
| viva/05:348, 576, 845; viva/04:338 | "zero in every row" (the headline includes the 3 Groq rows) | hosted `success_protected.k` |
| viva/05:13-15, 577, 579, 600, 836; viva/04:13-14, 341, 576 | "every attempted attack was blocked at the attack action" | hosted `blocked_given_attempt` |
| viva/05:13-14, 586, 599, 835; viva/04:10-11, 575 | sandbox/seccomp "recorded no block at all" / "zero" | pooled kernel-layer k |
| viva/05:299-300, 313, 593, 857; viva/04:495 | "No current episode had a context overflow" | hosted `context_overflow_episodes` |
| viva/05:15-17, 589, 610-612; README:311 | hosted rows "were still running when the data was frozen ... labelled partial" | false after the post-freeze build |
| viva/04:291 | "the prose here quotes no Groq or pooled number" | contradicted by line 338 and lines 10-14 |

**Other hand-typed result numbers outside blocks.** These are on complete local rows, so they are stable, but the check does not cover them:
- viva/04:339 ("0 of 120 ... 3.1 %", "0 of 40 cases ... 8.8 %")
- viva/04:363-370 (951 / 1251 steps, largest prompt 1002 / 1042 tokens, probes 6402 / 5273, "0/342": the model name is on line 367, so line 368 is not attributed)
- viva/04:526, 609, 616 ("0/40 cases", "0/342")
- viva/05:332-333, 796, 819 (8.8% / 3.1%, which depend on n)
- viva/04:315 and 342 (`342/342` control, `21/120` scratch-gpt). These two are checked, because the line names the model.

**What happens.** If one Groq episode before 15:30Z leaks with guardrails, or is attempted and not blocked, or overflows, the post-freeze `build` correctly rewrites every block, and the slides switch wording through `_clean()`. The viva pages, however, keep telling the panel the opposite, and `build` ends `BUILD ok`. That is exactly the drift D-03 is meant to prevent.

The deck does not have this problem: `slides._remark_slide`, `_headline_slide` and the others branch on the data.

**Fix:** Move every data-dependent sentence into a generated block that branches on the data, as `_findings` and the slides already do. Add a block such as `claim`:

```python
def _claim(numbers):
    leaks = [m for m in _models(numbers) if m["metrics"]["success_protected"]["k"]]
    unblocked = [m for m in _models(numbers)
                 if m["metrics"]["blocked_given_attempt"]["k"] != m["metrics"]["blocked_given_attempt"]["n"]]
    ...  # emit "no attack succeeded for any of the N models" only when leaks == []
```

Rewrite the prose around it so it states no outcome itself. Also add a check rule that fails on outcome words outside blocks (for example `zero in every row|no attack succeeded|every attempt|no block|had a context overflow|still running`), or on any `\d+ of \d+` and `\d+(\.\d+)? ?%` outside blocks that is not in an allow-list.

### CR-02: `--frozen` replaces and then deletes whatever directory it names; the "never write into results/phase4-*" invariant is enforced only for the 11 source directories

**File:** `src/deliverables/build.py:160-162, 166-217`

**Issue:** `freeze()` refuses a `frozen_dir` only when it overlaps one of the 11 source origins. Otherwise it does the following, whatever `frozen_dir` holds:

1. `os.rename(frozen_abs, aside)` (line 206)
2. `_remove(aside)` (line 217), which is `shutil.rmtree`

So any of these is accepted and permanently deletes the target:
- `python -m src.deliverables build --frozen results/phase4-jobs` deletes the live hosted loop's jobs directory (pidfiles, logs) while the loop runs.
- `--frozen results/phase4-report`, `--frozen results/phase4-smoke-scratch-gpt`, and so on.
- `--frozen docs` deletes `docs/`, including the untracked `docs/OS_Level_Guardrails_Review_3.pptx`, `docs/template.pptx` and the Review-2 deck. Those are not recoverable from git.

No check confirms that the directory being replaced was a frozen directory, and no test covers misuse of the overlap guard.

**Fix:** Before the swap, refuse to replace an existing directory unless it is empty or holds a valid `freeze.json` (schema 1). Also refuse any `frozen_dir` whose basename starts with `phase4-` or that lies under `results_root/phase4-*` or `final-20261011`:

```python
if os.path.exists(frozen_abs) and os.listdir(frozen_abs) \
        and not os.path.isfile(os.path.join(frozen_abs, FREEZE_FILE)):
    raise DeliverablesError(f"{posix(frozen_dir)} exists and is not a frozen directory; refusing to replace it")
if os.path.basename(frozen_abs).startswith("phase4-"):
    raise DeliverablesError("the frozen directory may not be a phase4-* directory")
```

Add tests for `--frozen` pointing at a source, at a non-frozen sibling, and at a non-empty directory without `freeze.json`.

### CR-03: The post-freeze build labels its snapshot `final` from the wall clock alone, while the hosted loop can still be appending rows and has not yet written its terminal `hosted_state.json`

**Files:** `src/deliverables/build.py:158, 197`; `src/benchmark/jobs.py:1467-1486, 1510-1515, 801-815`

**Issue:** `data_state` is `"final" if snapshot >= FREEZE_UTC`. The hosted loop only notices the freeze on its next poll:
- `HOSTED_POLL_S = 10` while children run;
- up to 60 s when it is sleeping on a quota wait (line 1515).

It then stops its children one after another, each with up to `FREEZE_STOP_WAIT_S = 60` of SIGTERM grace, so up to about 3 minutes for 3 models. An episode the child finishes inside that window is appended to `raw.jsonl`. `stopped_at_freeze` is written to `hosted_state.json` only after all children are stopped (lines 1480-1483).

A build run at about 15:30-15:33Z therefore produces a snapshot stamped "Data: final ... after the D-09 freeze" that:
- misses rows that exist in the source;
- shows a non-terminal loop state (`running` / `waiting_quota`) in the hosted block, under the sentence "A row still labelled partial stopped at the freeze or on its quota".

The clocks make this worse:
- the loop runs in WSL on the WSL clock;
- the deck build is run with `py -3.12` on Windows on the Windows clock;
- WSL2 clock drift after sleep is common.

**Fix:** Make `final` a property of the data, not of the clock. In `freeze()`:
- Set `data_state = "final"` only when `snapshot >= freeze_at` and every hosted source's frozen `hosted_state.json` has a state in `jobs.HOSTED_TERMINAL`. Mirror that tuple as a constant pinned by a test.
- Otherwise raise, or record `"snapshot"` with a clear `FREEZE ... loop not terminal` line.

Optionally, also require that no `progress.log` ends after the snapshot without a `FREEZE stopped`/`HOSTED freeze reached` line.

## Warnings

### WR-01: The fraction rule is per line and only for lines naming a model, so wrapped prose, "k of n" wording and percentages escape; viva/04 overstates what the check does

**File:** `src/deliverables/consistency.py:45-47, 112-126`

**Issue:** Four gaps:

- **Hard-wrapped prose.** The viva pages wrap at about 110 columns, so the model name and its number often land on different lines. Example: viva/04:367-368, "llama3.2:3b re-check ..." on one line and "0/342 episodes" on the next. The token on line 368 is never attributed to a model and passes.
- **Other wordings.** `FRACTION_RE` requires `d/d`, so "0 of 120", "111 of 120" and "120 of its 120" are never checked. `PERCENT_RE` requires `P% (k/n`, so "3.1 %" and a bare "92.5%" are never checked.
- **Loose matching.** A line that names two models accepts either model's fractions. Each model's allowed set is the union of every metric, including `episodes_done/expected` and all `by_kind` and `layers` fractions, so a stale value often still matches some metric (for example `0/120`).
- **Tests and docs.** `tests/test_deliverables_check.py:218` asserts that `999/1000` on a line without a model passes, which writes the blind spot into the test suite. viva/04:286-289 tells the examiner that `check` "fails ... if the prose around it quotes a number that is not in the data", which it does not.

**Fix:**
- Attribute model names per paragraph (split on blank lines) rather than per line.
- Treat `(\d+) of (\d+)` as a fraction.
- Flag any `k/n` or `P%` outside blocks that is not in the union of all generated values, even with no model named.
- Restrict each metric to its own row: a fraction on a line must match the metric family the line names, or be inside a block.
- Correct the viva/04 sentence.

### WR-02: The secret scan only recognises Groq keys, but the frozen directory also holds Gemini-era data and the project uses Gemini and OpenRouter keys

**Files:** `src/deliverables/consistency.py:28, 138-139, 176-198`; `src/deliverables/slides.py:1265-1267`

**Issue:** `SECRET_RE = gsk_[A-Za-z0-9]{20,}` is copied from `run_checks`. `.env.example` and `config/models.yaml` name `GEMINI_API_KEY` (key shape `AIza[0-9A-Za-z_-]{35}`) and `OPENROUTER_API_KEY` (`sk-or-v1-...`). The Qwen download path may also see `hf_...` tokens.

The freeze copies `legacy-gemini-flash-lite/raw.jsonl` and `legacy-gemma-4-31b/raw.jsonl`, rows from Gemini-key runs whose `llm_error` text is free-form provider output, into `results/phase5-final/sources/`. `check` then prints `SECRETS ... ok` without being able to see a Gemini key. The deck check uses the same regex for slide text and notes.

A count-only grep today finds 0 matches for `AIza`, `sk-or-`, `sk-` and `hf_` shapes, so this is a coverage gap, not a leak.

Two more gaps:
- `scan_frozen` silently skips files it cannot open (`except OSError: continue`).
- It skips any file with a NUL byte in its first 8 KiB, so a UTF-16 text file is never scanned.

**Fix:** Widen the pattern in one shared place, and make the `run_checks` pin test compare against it:

```python
SECRET_RE = re.compile(r"gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{35}|sk-or-v1-[0-9a-f]{32,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}")
```

Report unreadable files as `FAIL check: path: unreadable`, not as a skip.

### WR-03: A render failure after a successful freeze destroys the previous good frozen snapshot

**File:** `src/deliverables/build.py:203-217, 486-490, 413-419`

**Issue:**
- **Fresh freeze.** `freeze()` swaps in the new `sources/` and `freeze.json` and `rmtree`s the old frozen directory (including its `numbers.json`, `summary.md` and charts) before `render()` runs. If `render()` then fails, the last verified snapshot is gone and `load_numbers` fails for `blocks`, `check` and `deck`. `render()` fails on, for example, `_validate_models` (a row with an unexpected model or provider), the summary-equality guard, or a matplotlib error. The module docstring promises "On any error ... the old frozen directory is kept", but that holds only for freeze errors.
- **`--reuse-frozen`.** `generate_report` writes `summary.md`, `summary.csv` and `layer_blocks.csv` non-atomically before the charts and `numbers.json`. A failure in `write_phase5_charts` leaves a new `summary.md` next to an old `numbers.json`, and `check` still passes against the old one.

**Fix:**
- Render into the staging directory and swap only after `render()` succeeds. `render(staging)` works because every path is relative to `frozen_dir`.
- On `--reuse-frozen`, render into a temporary sibling and swap.

### WR-04: `numbers.json` is not a pure function of the frozen directory, and some confound values are asserted rather than "read from a recorded source"

**Files:** `src/deliverables/build.py:14-16, 407-409`; `src/deliverables/sections.py:9-11, 91-95, 113-121, 224-249`

**Issue:** `confound_rows` and `local_training` read three files from the working tree at render time:
- `config/models.yaml`
- `models/scratch/manifest.json`
- `models/qwen-agentlora/manifest.json`

None of them is copied into the frozen directory or hashed in `freeze.json`. Re-rendering a frozen directory after any of these changes therefore silently changes `numbers.json`, the confounds and threats blocks, and the deck. That contradicts "re-rendering a frozen directory rewrites it byte-identically".

Some values are also hard-coded, contradicting the sections docstring's "nothing is guessed":
- precision `"bf16 on the GPU"` and `"fp32"`, although `hf_local` falls back to fp32 on CPU;
- context `HF_LOCAL_CONTEXT = 4096`, which ignores a per-model `max_context_tokens` override the provider supports.

**Fix:**
- Copy `config/models.yaml` and both manifests into `frozen/inputs/` during `freeze()`, record their sha256 in `freeze.json`, and have `render()` read only those copies.
- Read precision and context from the rows or manifests where they are recorded, or print "not recorded".

### WR-05: The build path drops the template-fallback signal, so a deck rendered on a blank 16:9 fallback passes silently

**File:** `src/deliverables/build.py:455`; `src/deliverables/slides.py:1115-1133`

**Issue:** `_open_template` catches every exception while opening `docs/template.pptx` and falls back to a blank deck. It reports this only through its return value (`how`). `run_deck` prints that as a `TEMPLATE ...` line, but `deck_step` discards the return value of `slides.render_deck`. `build` therefore prints `DECK ... ok` / `BUILD ok` for a deck drawn without the Review template, and nothing in the build output says so.

**Fix:** Emit `TEMPLATE {how}` from `deck_step`. When `how` starts with `none (`, fail with `DeliverablesError` unless the user passes an explicit `--allow-blank-template`.

### WR-06: Test gaps on the riskiest paths

**Files:** `tests/test_deliverables_build.py`, `tests/test_deliverables_check.py`, `tests/test_deliverables_slides.py`, `tests/test_deliverables_sections.py`

**Issue:** Seven gaps:

- **Overlap guard.** No test covers the `overlaps` guard or `--frozen` misuse (CR-02).
- **Render failure.** No test covers a render failure after a successful freeze (WR-03).
- **Data state.** No test covers `data_state` versus a non-terminal hosted state (CR-03).
- **Prose outside blocks.** No test proves that a data-dependent sentence outside a block fails `check`. `test_fraction_rules` asserts the opposite for model-less lines.
- **Chart values.** The chart tests check only the returned label strings. Nothing asserts that the error bars equal `pct-lo` / `hi-pct`, or that a hollow marker is used for partial rows.
- **Deck check.** The only two tests that exercise `check_deck` on a real deck call `pytest.importorskip("pptx")`, so they are skipped in the WSL venv where the full suite runs. The deck's number and secret scanning is therefore untested there.
- **Hermeticity.** `build()` in the build and slides tests uses the default `repo_root=REPO_ROOT`, so those tests read the real `config/models.yaml` and `models/*/manifest.json`. They are not hermetic, despite the docstring "nothing reads results/".

**Fix:**
- Add the missing tests.
- Pass `repo_root=fake_repo(...)` in the build fixtures.
- Make one deck test run without python-pptx by checking `slide_text(spec)` with `scan_text` plus the widened secret regex.

## Info

### IN-01: `asr_ci.png` silently drops an arm with n = 0

**File:** `src/deliverables/charts.py:101-102`

The other two charts add a "Not plotted, an axis has no episodes yet" note. The ASR chart just `continue`s, which leaves a tick label with no marker. Add the same `_not_plotted` note.

### IN-02: Small freeze edge cases

**File:** `src/deliverables/build.py:92-94, 174-189`

- **Stale `raw_lines`.** `raw_lines` and `raw_sha` carry over from the previous source when `raw.jsonl` vanishes between the pre-check and the copy. On the first source this is a `NameError`. Assert that `raw.jsonl` was copied.
- **Copy order.** `hosted_state.json` is read after `raw.jsonl`, so the two can describe different moments.
- **Symlinks.** `_under` uses `abspath`, not `realpath`, so a symlinked frozen or source path bypasses the overlap guard.

### IN-03: The threats text hard-codes the benchmark shape

**File:** `src/deliverables/sections.py:214-215`

`threat_items` hard-codes "57 cases (20 dh, 20 ds, 17 benign)". `slides._bench` derives the same figures from the control. The docstring says each sentence is "built from numbers.json values". Derive the figures the same way.

### IN-04: Undecodable lines in frozen copies are never surfaced

**File:** `src/deliverables/build.py:245-249`; `src/benchmark/report.py:906-909`

`load_rows_counted` counts skipped lines. `load_frozen_rows` discards that count, and `generate_report` only `print`s it, which breaks the one-fact-per-line output contract. A corrupt mid-file line therefore never reaches `numbers.json`, a block or the check. Record `skipped_lines` per source in `numbers.json` and fail or label on a non-zero value.

### IN-05: The timestamp and percentage rules see only one notation

**File:** `src/deliverables/consistency.py:46-47`

The docs write times as "15:30 UTC on 2026-10-29", "18:45Z" and "18:48 UTC on 2026-10-28" (viva/04:102, 219, 313, 416). They write percentages as "3.1 %" with a space. Neither form is checked. Either widen the patterns or note in the module docstring that only these forms are guarded.

### IN-06: Tracked SDD exports are not regenerated or checked

**Files:** `docs/SDD.pdf`, `docs/SDD.docx`, `docs/build/SDD.html`, `docs/build/SDD.static.html` (all dated 2026-10-11)

These tracked exports of the SDD predate Phase 5. They are outside `DOC_TARGETS`, are not regenerated by `build` and are not scanned by `check`. Regenerate them in the build, or mark them as superseded.

### IN-07: Hand-typed counts in the README

**File:** `README.md:282-284, 304`

- The `run_attack_simulation.py` table (7/7, 0/7, 1/1) is static and correctly labelled as separate from the benchmark.
- The "1308 passed and 1 skipped" test count is hand-typed and will drift with the suite.

Neither comes from `numbers.json`. The table is acceptable as is. Consider dropping the exact test count.

### IN-08: `check` trusts `numbers.json` without tying it to `freeze.json`

**File:** `src/deliverables/consistency.py:201-203, 225-228`

`check` (and `blocks`/`deck`) load `numbers.json` without comparing its `sources[*].raw_sha256` with `freeze.json` or the frozen `raw.jsonl` files. A stale or hand-edited `numbers.json` therefore yields `deliverables-check-ok`. Call `read_freeze(frozen_dir)` and compare the sha lists before checking.

---

_Reviewed: 2026-10-29T12:11:13Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
