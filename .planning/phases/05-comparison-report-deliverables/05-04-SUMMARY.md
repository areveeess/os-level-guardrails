---
phase: 05-comparison-report-deliverables
plan: 04
subsystem: deliverables
status: complete
tags: [deck, doc-05, python-pptx, consistency-check, one-command-build]
requires:
  - results/phase5-final/numbers.json and charts/ (plan 05-01, snapshot 2026-10-29T07:12Z)
  - src/deliverables consistency.scan_text, blocks.data_line, report.LAYER_EVIDENCE (plan 05-01)
  - docs/template.pptx (Review-2 template, 18 slides, 29 layouts, 16:9)
provides:
  - "python -m src.deliverables deck [--frozen F] [--out P] [--template T] [--check]"
  - "build --no-deck / --deck-out; the deck step at the end of build (exit 3 without python-pptx)"
  - src/deliverables/slides.py (SLIDE_TITLES, deck_spec, render_deck, check_deck, run_deck)
  - docs/OS_Level_Guardrails_Review_3.pptx (binary output, never staged)
affects: [post-freeze rebuild (one command now also regenerates the deck), phase 05 verification]
tech-stack:
  added: []
  patterns: [pure slide spec from numbers.json, python-pptx imported only inside render/check, deck check reuses the D-03 consistency rules plus chart sha256]
key-files:
  created:
    - src/deliverables/slides.py
    - tests/test_deliverables_slides.py
  modified:
    - src/deliverables/__main__.py
    - src/deliverables/build.py
decisions:
  - "The template's 29 layouts all carry gradient-filled decoration on a dark background, so every slide uses its plain BLANK layout with the chart surface colour (#FCFCFB), its own 24 pt Arial title, one thin rule in the chart palette's guardrails blue (#0072B2), and a 9 pt data-line footer"
  - "build() draws the deck only when the caller passes deck_out (the CLI always does unless --no-deck), so library callers and the existing build tests never write the real deck from synthetic data; build also gained --deck-out for tests"
  - "check_deck also checks each slide's title against the spec and that every shape lies inside the slide, besides the planned rules"
  - "The legacy threat sentence is rewritten in the slide-14 notes without the legacy ids, because the deck rejects legacy ids anywhere"
metrics:
  duration: 33min
  completed: 2026-10-29
  tasks: 3
  files: 4
estimate:
  tokens: 110000
actuals:
  tokens: 24397
  tasks: 3
  commits: 3
plan_head_before: f3dd9f4a2418a3497213a6aba2944d22029d2b6a
plan_head_after: 4bdf802263fd4f656056f27ce2d17e3462f8b1ad
---

# Phase 5 Plan 04: Review-3 Slide Deck Summary

The Review-3 deck (DOC-05) is now generated, not hand-made. `PYTHONUTF8=1 py -3.12 -m src.deliverables deck` turns `results/phase5-final/numbers.json` and the three frozen charts into `docs/OS_Level_Guardrails_Review_3.pptx`. It uses python-pptx 1.0.2 on `docs/template.pptx` and removes the template's own 18 slides.

The deck has 15 slides, and every slide has spoken-script speaker notes. A mechanical check then reopens the file and:

- applies the 05-01 consistency rules (fractions, timestamps, percentages, key-shaped text) to every slide and every note;
- compares each embedded chart with its frozen file by sha256;
- rejects legacy model ids.

The one-command `build` now ends with the deck step. An interpreter without python-pptx fails loudly with exit 3 and names the old deck as stale.

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 (tracer): title slide end to end, deck subcommand | 1ba6354 | src/deliverables/slides.py, src/deliverables/__main__.py, tests/test_deliverables_slides.py |
| 2: full 15-slide deck, notes, tables, charts | bb81cea | src/deliverables/slides.py, tests/test_deliverables_slides.py |
| 3: deck step in build, exit 3 without python-pptx | 4bdf802 | src/deliverables/build.py, src/deliverables/__main__.py, tests/test_deliverables_slides.py |

I checked each commit with `git show --stat`. Each lists exactly its task's files, and none contains a .pptx. The deck file is still untracked (`?? docs/OS_Level_Guardrails_Review_3.pptx`).

## Red runs (TDD)

- **Task 1:** `tests/test_deliverables_slides.py` failed at collection with `ImportError: cannot import name 'slides' from 'src.deliverables'` (exit 2). It then passed: 5 passed, 1 skipped.
- **Task 2:** the extended tests failed with 13 failed, 7 passed and 2 skipped. The full-deck tests got an IndexError or a title mismatch on the one-slide spec. They then passed: 20 passed, 2 skipped.
- **Task 3:** the build-step tests failed with 4 failed, 21 passed and 2 skipped (`unrecognized arguments: --deck-out`). They then passed: 25 passed, 2 skipped.

## Test counts

**WSL venv (no python-pptx):**

- `tests/test_deliverables_slides.py`: 25 passed, 2 skipped. The 2 skips are the render and check tests (`could not import 'pptx'`).
- The four deliverables files (slides, build, sections, check): 73 passed, 2 skipped, in 55 s.

**Windows py -3.12 (python-pptx, no pytest):** I ran the two render and check tests through a pytest stand-in in the session scratchpad (fixtures wired by hand). Both passed:

- `test_render_on_the_template_then_check`
- `test_check_rejects_a_stale_chart_and_a_number_edited_by_hand`

## DECK lines

- **Task 1 verify:** `DECK docs/OS_Level_Guardrails_Review_3.pptx slides 1 notes 1 charts 0 ok`
- **Task 2 verify:** `DECK docs/OS_Level_Guardrails_Review_3.pptx slides 15 notes 15 charts 3 ok`. Each `deck` run first prints `TEMPLATE docs/template.pptx`.

## Slide titles

1. OS-Level Guardrails for Autonomous AI Agents: Review 3
2. The reviewer's remark and our answer. The remark is quoted verbatim on the slide and in the notes.
3. Architecture: one reference monitor, four layers. This slide is a diagram drawn with shapes:
   - the untrusted model;
   - the four layers inside the monitor;
   - the Linux kernel;
   - the decision log;
   - the two arms.
4. Locally made model 1: ScratchGPT, trained from scratch. It shows 4.8M parameters, 6 layers, 8 heads, a 2048-character context, and best step 11750 of 13750.
5. Locally made model 2: Qwen2.5-0.5B, LoRA-tuned here. It shows r = 16 and alpha = 32, 8.8M trainable of 502.8M parameters, and 94.0% dev exact.
6. The model line-up and its confounds. This is the confounds table at 11 pt.
7. Method: InjecAgent, two arms, honest statistics. It shows 57 cases (20 dh, 20 ds, 17 benign) with 3 runs each; these figures are read back from the control.
8. Headline: attack success with and without guardrails. The table:
   - lists the 8 models in MODEL_ORDER;
   - labels the Groq rows partial with their done/expected and greys them;
   - ends with the control as a reference row;
   - has no rank column.
9. Attack success per model, with 95% intervals. It embeds `asr_ci.png`.
10. Capability versus vulnerability. It embeds `capability_vs_vulnerability.png`.
11. From unprotected to protected. It embeds `utility_vs_asr_arrows.png`.
12. Where attacks were stopped: blocks by layer. It shows:
   - pooled layer counts;
   - sandbox and seccomp at 0/1173, called a lower bound;
   - Phase 1 residual-bypass rows cited by number only.
13. The model-independent control. It shows:
   - attempted 120/120 and success 120/120 without guardrails, then 0/120 with guardrails;
   - blocked 120/120;
   - 40 distinct cases.
14. Limitations and threats to validity. It has six short bullets:
   - partial runs;
   - the kernel-layer lower bound;
   - format failures;
   - WR-01 with its generated count ("0 context-overflow episodes in the current data, so no current number is affected");
   - one benchmark at temperature 0;
   - confounds.

   The full threat sentences are in the notes.
15. Next steps (v2). It has the "Final Groq numbers after the 2026-10-29T15:30Z freeze" bullet, which appears only while the data state is snapshot, then CMP-V2-01, EVAL-V2-01, EVAL-V2-02, EVAL-V2-03 and SERVE-V2-01.

Every content slide has a footer with the data line: "Data: snapshot of 2026-10-29T07:12Z (UTC), not final; the D-09 freeze is 2026-10-29T15:30Z. Partial rows show their episodes done." Slide 1 has the full report data line instead.

Notes length per slide ranges from 547 characters (slide 11) to 3281 (slide 14).

## Design choice after inspecting the template

The template's master and layouts (TITLE, TITLE_ONLY, TITLE_AND_BODY and the custom ones) all carry gradient-filled decoration: columns of teal pixel squares on a dark #0C343D/#0E2A47 background. The theme's text colour (dk1) is white. That is the case the plan names, so I followed its rule:

- Every slide uses the template's own plain **BLANK** layout. The deck is still based on `docs/template.pptx`, but its 18 slides are dropped through the slide-id list and `drop_rel`.
- The background is the chart surface colour, #FCFCFB, so the charts sit flush on the slide.
- Text is Arial in ink #0B0B0B and muted #52514E, always set explicitly, because the theme default is white.
- There is one accent, the chart palette's guardrails blue #0072B2. It is used for the title rule, the bullets and the diagram frame.
- Tables are borderless and editorial: a 1.25 pt ink rule under the header, hairline row rules, right-aligned numbers, partial rows in grey, and the control row in italics.
- Sizes: titles 24 pt, body 16-18 pt, tables 11-13 pt, charts 7 in wide.

I exported every slide through PowerPoint COM (read-only, PowerPoint quit afterwards) and checked them. A scratchpad inspector compares each text frame's `BoundHeight` with its box, checks shape bounds against the slide, and lists overlapping boxes.

- **First render:** 9 issues. The confounds and headline tables ran past the footer, and text overflowed on slides 2, 3, 7, 10 and 14.
- **Fix, two passes:** I tightened the table widths and line spacing, shortened the bullets and the monitor label, and set 16 pt where needed.
- **Final render:** `EXPORTED 15 ISSUES 0`. The only overlaps left are the intended nesting in the slide-3 diagram.

## Build outputs (Task 3)

**WSL, without python-pptx (`build --reuse-frozen --no-docs`): exit 3**

```
REUSE 2026-10-29T07:12Z state snapshot sources 11
SOURCE ... (11 lines)
REPORT results/phase5-final/summary.md models 8 control 1 legacy 2
CHARTS 3
NUMBERS results/phase5-final/numbers.json
FAIL deck: python-pptx is not importable by /mnt/c/Users/rvsre/.gemini/antigravity/scratch/os_level_guardrails/venv/bin/python; run with py -3.12 on Windows; docs/OS_Level_Guardrails_Review_3.pptx was not rebuilt and is stale
```

**WSL `--no-deck`: exit 0.** The output ends with `DECK skipped (--no-deck)` and then `BUILD ok`.

**Windows (`build --reuse-frozen --no-docs`, run last): exit 0.** The output ends with:

```
DECK docs/OS_Level_Guardrails_Review_3.pptx slides 15 notes 15 charts 3 ok
BUILD ok
```

All four Task 3 verify commands passed.

## Phase-level one-command check

05-02 and 05-03 are done, so I ran it:

```
PYTHONUTF8=1 py -3.12 -m src.deliverables build --reuse-frozen
```

It exited 0, and the check command passed. The output was:

```
BLOCKS README.md 4 unchanged
BLOCKS docs/SDD.md 9 unchanged
BLOCKS docs/viva/04-namespaces-seccomp-and-local-inference.md 5 unchanged
BLOCKS docs/viva/05-evaluation-statistics-and-examiner-qa.md 7 unchanged
DOC ... 4 docs ok
SECRETS results/phase5-final files 30 ok
deliverables-check-ok docs 4
DECK docs/OS_Level_Guardrails_Review_3.pptx slides 15 notes 15 charts 3 ok
BUILD ok
```

`git status --porcelain --untracked-files=no` was identical before and after the run, so no tracked file changed.

The last chart render is the Windows one. The deck embeds those bytes, which the check confirmed (`charts 3`).

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] A chart with another aspect ratio ran off the slide**
- **Found during:** Task 2, Windows render test with synthetic 1 x 1 PNGs.
- **Issue:** `add_picture` at a fixed width produced a square chart 7 in tall.
- **Fix:** A chart taller than the body is scaled down to the body height, keeping its aspect ratio.
- **Commit:** bb81cea.

**2. [Rule 1 - Test bug] Mutating numbers.json did not prove that the fraction rule fires**
- **Found during:** Task 2.
- **Issue:** The planned "changed number" test raised one k. The new k/n equalled another of the same model's allowed fractions (attempted), so the check rightly passed.
- **Fix:** The test now edits a headline cell in the saved deck to `99.0% (99/100)`, and asserts a `slide 8 text ... fraction 99/100` problem. The test is renamed `test_check_rejects_a_stale_chart_and_a_number_edited_by_hand`.
- **Commit:** bb81cea.

**3. [Rule 2 - Correctness] build() never draws the deck unless asked**
- **Found during:** Task 3.
- **Issue:** Defaulting `build()` to draw the deck would make every existing synthetic-data build test overwrite the real `docs/` deck wherever python-pptx exists. In WSL, it would turn those tests into exit-3 failures.
- **Fix:** `build(deck_out=None)` skips the step. The CLI passes the repository deck path unless `--no-deck` is given. `build --deck-out` lets the tests write to tmp.
- **Commit:** 4bdf802.

**4. [Rule 3 - Interface] numbers.json `local_training` is flat**
- **Issue:** The plan's interfaces describe nested `lora` and `best` keys. The real file has `r`, `alpha`, `trainable_params`, `total_params` and `dev_action_exact` at the top level, and `base_model` is a string.
- **Fix:** I read the actual shape. Missing values render as "not recorded"; a test covers this.

### Content choices within Claude's discretion

- **Slide 6 shortened text:** the slide drops the parenthetical part of some confounds cells, for example "(per the model id, not verified here)" and "(num_ctx)". The hosted parameter counts get an asterisk, and the caption explains it.
- **Slide 14 notes:** the legacy item is rewritten without the ids, because the check rejects legacy ids. Every other threat sentence appears verbatim, which a test checks.

## Known Stubs

None. Every slide value comes from numbers.json, the frozen charts or `report.LAYER_EVIDENCE`.

## Pending human check (from the plan)

Open `docs/OS_Level_Guardrails_Review_3.pptx` in PowerPoint or Google Slides and page through all 15 slides with the notes visible. Check that:

- the visual tone and legibility from the back of a room work;
- the notes read naturally as a spoken script.

The slide images from the last render are in the session scratchpad (`deck/slide01.png` to `slide15.png`).

## Notes

- A `POWERPNT.EXE` process (pid 35876) was running after the exports. My first availability probe created a PowerPoint COM object without calling Quit, so the process may be that leftover. I did not signal it.
- No `.env` was opened, nothing was installed, nothing was written under `results/phase4-*`, and no process was started or stopped apart from the builds, tests and PowerPoint exports.

## Self-Check: PASSED

- FOUND: src/deliverables/slides.py, tests/test_deliverables_slides.py, docs/OS_Level_Guardrails_Review_3.pptx (untracked)
- FOUND commits: 1ba6354, bb81cea, 4bdf802 (all ancestors of HEAD)
