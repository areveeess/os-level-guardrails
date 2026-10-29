---
phase: 05-comparison-report-deliverables
plan: 01
subsystem: deliverables
status: complete
tags: [report, charts, numbers, consistency-check, freeze]
requires:
  - Phase 4 result directories (results/phase4-*) and the legacy October directories (results/final-20261011)
  - src/benchmark/report.py (summarize, layer_block_summary, control_summary, generate_report)
provides:
  - "python -m src.deliverables build | blocks | check (contract in 05-01-PLAN.md interfaces)"
  - results/phase5-final/ (frozen sources, freeze.json, summary.md/.csv, layer_blocks.csv, numbers.json, charts/)
  - blocks.render_block, blocks.BLOCK_NAMES, blocks.DOC_TARGETS, consistency.scan_text, consistency.MODEL_ALIASES, consistency.SECRET_RE
affects: [05-02 README/SDD, 05-03 viva pages 04/05, 05-04 deck]
tech-stack:
  added: []
  patterns: [read-only freeze with staging and swap, numbers.json as single source of numbers, marker-delimited generated doc blocks]
key-files:
  created:
    - src/deliverables/__init__.py
    - src/deliverables/__main__.py
    - src/deliverables/build.py
    - src/deliverables/sections.py
    - src/deliverables/charts.py
    - src/deliverables/blocks.py
    - src/deliverables/consistency.py
    - tests/test_deliverables_build.py
    - tests/test_deliverables_sections.py
    - tests/test_deliverables_check.py
  modified:
    - src/benchmark/report.py
decisions:
  - "numbers.json is computed before the report (the threats section quotes it); the build fails if generate_report's own summary differs from the one numbers.json came from"
  - "Pinned constants (FREEZE_UTC, HOSTED_DS_EXPECTED, SECRET_RE, DEFAULT_MAX_CONTEXT_TOKENS) are checked by tests that parse the source with ast, so no test or module imports jobs, run_checks or src.llm for them"
  - "D-05 charts use Okabe-Ito blue/vermillion for the arms and blue/orange for local/hosted, plus one marker shape per model and a right-hand label column with leader lines, so labels never overlap"
  - "Doc check runs only on the selected target docs; an unscoped check fails on viva page 04 until plan 05-03 converts it (expected baseline)"
metrics:
  duration: 50min
  completed: 2026-10-29
  tasks: 3
  files: 11
estimate:
  tokens: 150000
actuals:
  tokens: 35219
  tasks: 3
  commits: 3
plan_head_before: 78e26e065aaec9cc4b9b926f25073389f791e8f9
plan_head_after: 6134a0c7ae4e2c8b9cfbede6ce6e4c3ed9c8d1cb
---

# Phase 5 Plan 01: Deliverables Engine Summary

One command (`PYTHONUTF8=1 py -3.12 -m src.deliverables build`) now does four things:

- It freezes the nine Phase 4 directories and the two legacy October directories, read-only and complete lines only, into `results/phase5-final/`.
- From that frozen copy only, it regenerates the cross-model report through `report.py`. The report gains confounds, legacy-rows, charts and threats-to-validity sections.
- It writes the three colour-blind-safe D-05 charts and `numbers.json`.
- It refreshes the generated doc blocks and runs a D-03 consistency check. The check fails on any number that differs from the data, and on key-shaped text.

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 (tracer): freeze, report, numbers.json, CLI | acb89b7 | src/deliverables/__init__.py, __main__.py, build.py, src/benchmark/report.py, tests/test_deliverables_build.py |
| 2: confounds, legacy, threats, D-05 charts | e651643 | src/benchmark/report.py, src/deliverables/sections.py, charts.py, build.py, tests/test_deliverables_sections.py, tests/test_deliverables_build.py (see deviation 1) |
| 3: doc blocks, consistency check, --doc scoping | 6134a0c | src/deliverables/blocks.py, consistency.py, __main__.py, build.py, tests/test_deliverables_check.py |

Each commit was checked with `git show --stat`. Only the listed paths were committed.

## Red runs (TDD)

- **Task 1:** `tests/test_deliverables_build.py` failed at collection with `ImportError: cannot import name 'deliverables' from 'src'` (exit 2). It then passed: 13 passed.
- **Task 2:** `tests/test_deliverables_sections.py` failed at collection with an ImportError (`legacy_lines` and `src.deliverables.charts` did not exist; exit 2). It then passed: 11 passed.
- **Task 3:** `tests/test_deliverables_check.py` failed at collection with an ImportError (`blocks` and `consistency` did not exist; exit 2). It then passed: 24 passed, including 5 parametrized marker cases.

## Verify results

| Verify | Result |
|--------|--------|
| T1 build tests >= 9 passed | 13 passed (about 1.5 s) |
| T1 regression: test_report, test_layer_blocks_report, test_control_agent | 82 passed (28 s) |
| T1 freeze.json holds 11 sources, 2 of them legacy | `11 2` |
| T1 numbers.json groups | `5 3 1` |
| T2 sections tests >= 10 passed | 11 passed |
| T2 regression plus build tests | 95 passed (29 s) |
| T2 four new headings, once each | 4 |
| T2 "WR-01 touches none of the current data" once | 1 |
| T2 three chart files | 3 |
| T2 numbers.json counts: legacy, confounds, charts | `2 8 3` |
| T2 acceptance: headline and partial table lines, legacy ids | `10 0` |
| T2 acceptance: docs/images/phase5 absent | absent |
| T3 check tests >= 14 passed | 24 passed (2 s) |
| T3 deliverables, report and run_checks tests | 139 passed (63 s) |
| T3 scoped check prints 4 lines | ok |
| T3 `build --reuse-frozen --doc README.md --doc docs/SDD.md` ends with BUILD ok | ok |
| T3 `git status --porcelain README.md docs/SDD.md docs/viva/04-...` | empty (none of the three changed) |

The full WSL suite ran in four calls: 1308 passed and 1 skipped, with no failures.

| Call | Result | Time |
|------|--------|------|
| 1: everything except the 7 heavy files | 983 passed | 309 s |
| 2: test_hf_local_provider and test_qwen_lora_data | 66 passed, 1 skipped | 132 s |
| 3: test_qwen_lora_train and test_qwen_lora_checks | 94 passed | 230 s |
| 4: test_jobs, test_run_checks and test_phase4_runner | 165 passed | 11 s |

## The first build (D-01 "build it now")

The run started at 2026-10-29T07:12:22Z and took about 1 s, before the charts existed. It printed:

```
FREEZE 2026-10-29T07:12Z state snapshot sources 11
SOURCE phase4-qwen2.5-0.5b-agentlora lines 342 sha256 d36a6c6c26dd
SOURCE phase4-scratch-gpt lines 342 sha256 bed1d6f2c2d4
SOURCE phase4-qwen2.5-0.5b-base lines 342 sha256 0cc50b1284f5
SOURCE phase4-ollama-qwen2.5-3b lines 342 sha256 5d663d06d722
SOURCE phase4-ollama-llama3.2-3b lines 342 sha256 9f4ec0f61f35
SOURCE phase4-groq-gpt-oss-120b lines 919 sha256 779c33d45614
SOURCE phase4-groq-gpt-oss-20b lines 798 sha256 b67af4ee192d
SOURCE phase4-groq-qwen3.8-27b lines 939 sha256 1dcceaef8511
SOURCE phase4-control lines 342 sha256 dbfab047b275
SOURCE legacy-gemini-flash-lite lines 199 sha256 8be175ff349c
SOURCE legacy-gemma-4-31b lines 360 sha256 a2e6b049353d
REPORT results/phase5-final/summary.md models 8 control 1 legacy 2
NUMBERS results/phase5-final/numbers.json
BUILD ok
```

Every later run in this plan used `--reuse-frozen`, which keeps this snapshot. Those runs add `CHARTS 3` after Task 2, and the BLOCKS, DOC, SECRETS and check-ok lines after Task 3.

The frozen directory (`results/phase5-final/`) is in this state:

- **Snapshot:** 2026-10-29T07:12Z
- **Data state:** `snapshot`
- **Freeze:** 2026-10-29T15:30Z
- **Report data line:** "Data: snapshot of 2026-10-29T07:12Z (UTC); the D-09 freeze is 2026-10-29T15:30Z. Partial rows are labelled with their episodes done and are not final."

## Model labels and headline numbers (snapshot 07:12Z)

| Model | Group | Label | Success without -> with guardrails | Benign task (no guardrails) |
|-------|-------|-------|-------------------------------------|-----------------------------|
| qwen2.5-0.5b-agentlora | local | complete | 111/120 -> 0/120 | 51/51 |
| scratch-gpt | local | complete | 33/120 -> 0/120 | 36/51 |
| qwen2.5-0.5b-base | local | complete | 3/120 -> 0/120 | 15/51 |
| ollama-qwen2.5-3b | local | complete | 3/120 -> 0/120 | 49/51 |
| ollama-llama3.2-3b | local | complete | 7/120 -> 0/120 | 48/51 |
| groq-gpt-oss-120b | hosted | partial, 251/342 episodes | 4/95 -> 0/95 | 25/30 |
| groq-gpt-oss-20b | hosted | partial, 117/342 episodes (error-dominated) | 2/51 -> 0/51 | 5/7 |
| groq-qwen3.8-27b | hosted | partial, 272/342 episodes | 0/98 -> 0/98 | 37/37 |
| control-always-comply | control | complete | 120/120 -> 0/120 | 51/51 |

Pooled and control figures:

- **Pooled conditional block rate:** 200/200, which is 100.0% [98.1, 100.0].
- **Sandbox:** 0/1173 pooled protected-arm episodes (844 attack, 329 benign).
- **Seccomp:** also 0/1173 over the same episodes.
- **Control:** guardrails blocked 120/120 of its attempted attacks, and the broker denied all 120.
- **Hosted loop state:** all three hosted models are `waiting_quota` in the frozen `hosted_state.json`. Their quota stops are 3, 3 and 4.
- **Context overflow:** 0 context-overflow episodes for every model.

## Confounds table (as printed in summary.md)

| Model | Parameters | Quantization or precision | Local or hosted | Context window | Decoding | Training provenance | Largest prompt seen |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | 494.0M + 8.8M LoRA | bf16 on the GPU | local, in-process | 4096 tokens | temperature 0 (greedy) | LoRA-tuned here on the Phase 2 synthetic mix | 731 |
| scratch-gpt | 4.8M | fp32 | local, in-process | 2048 characters | temperature 0 (greedy) | trained from scratch here | 1293 |
| qwen2.5-0.5b-base | 494.0M | bf16 on the GPU | local, in-process | 4096 tokens | temperature 0 (greedy) | pretrained instruct, not tuned here | 1566 |
| ollama-qwen2.5-3b | 3.1B | Q4_K_M | local, Ollama server | 8192 tokens (num_ctx) | temperature 0 (greedy) | pretrained instruct, pulled Ollama tag (qwen2.5:3b) | 1002 |
| ollama-llama3.2-3b | 3.2B | Q4_K_M | local, Ollama server | 8192 tokens (num_ctx) | temperature 0 (greedy) | pretrained instruct, pulled Ollama tag (llama3.2:3b) | 1042 |
| groq-gpt-oss-120b | 120B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 1027 |
| groq-gpt-oss-20b | 20B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 1039 |
| groq-qwen3.8-27b | 27B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 662 |

The legacy table holds two rows:

| Model | Episodes done | Paired attack episodes | Success, no guardrails | Success, guardrails |
|-------|---------------|------------------------|------------------------|---------------------|
| gemini-flash-lite | 195/342 | 46 | 5/47 | 0/46 |
| gemma-4-31b | 156/342 | 19 | 0/19 | 0/19 |

## Threats to validity (as printed, in order)

1. **Partial runs.** Each Groq model is listed with its label and its paired attack n: 95, 51 and 98.
2. **Legacy rows.** Gemini 195/342 with 46 pairs, and Gemma 156/342 with 19 pairs. They come from an older engine and are not compared, pooled or charted.
3. **Kernel layers.** Sandbox 0/1173 and seccomp 0/1173. The item explains that the policy engine stops attacks first, then gives LAYER_EVIDENCE (the lower bound and the Phase 1 residual-bypass rows).
4. **Format failures.** Raw success is shown next to ASR-valid for four models:

   | Model | Without guardrails (raw vs ASR-valid) | With guardrails (raw vs ASR-valid) |
   |-------|---------------------------------------|------------------------------------|
   | qwen2.5-0.5b-base | 3/120 vs 3/81 | 0/120 vs 0/75 |
   | ollama-qwen2.5-3b | 3/120 vs 3/105 | 0/120 vs 0/111 |
   | ollama-llama3.2-3b | 7/120 vs 7/120 | 0/120 vs 0/119 |
   | groq-gpt-oss-20b | 2/51 vs 2/47 | 0/51 vs 0/49 |

5. **WR-01 (deferred).** The item explains the upward bias on ASR-valid, then says: "No current-engine episode had a context overflow (0 for every model), so WR-01 touches none of the current data; the bias would apply only to future runs." It notes that the fix waits until after the freeze because `outcomes.py` is run-affecting.
6. **One benchmark.** InjecAgent, 57 cases (20 dh, 20 ds, 17 benign), with simulated tools.
7. **Temperature-0 decoding.** Repeats of a case are near-duplicates, so the episode-level intervals are optimistic.
8. **Confounds.** The item points to the confounds table.

## Charts

The three charts are in `results/phase5-final/charts/`:

- `asr_ci.png`
- `capability_vs_vulnerability.png`
- `utility_vs_asr_arrows.png`

All three are 10 x 5.6 in at 160 dpi. I opened all three and checked them; after one adjustment that moved the asr_ci legend below the tick labels, nothing overlaps.

`docs/images/phase5` was not created. docs/images is on the never-stage list, so per D-05 the deck embeds the charts from `results/phase5-final`.

## Doc check baseline

**Scoped check (`--doc README.md --doc docs/SDD.md`), passes:**

```
DOC README.md blocks 0 ok
DOC docs/SDD.md blocks 0 ok
SECRETS results/phase5-final files 30 ok
deliverables-check-ok docs 2
```

**Unscoped check, expected FAIL until plan 05-03:**

The unscoped `check` exits 1 with 33 `FAIL check` lines. All of them are on `docs/viva/04-namespaces-seccomp-and-local-inference.md`. The page quotes the 2026-10-29T03:06Z snapshot by hand.

Old timestamps:

- 2026-10-29T03:06 on lines 11, 287, 289, 396 and 398
- 2026-10-28T18:48 on line 402
- 2026-10-29T02:56 on line 409
- 2026-10-29T02:33 on line 410

Groq done/expected from the older snapshot:

- 240/342 on lines 302, 408 and 416
- 109/342 on lines 303, 409 and 417
- 246/342 on lines 304, 410 and 418

Other Groq fractions:

| Line | Fractions |
|------|-----------|
| 357 | 0/92 |
| 358 | 0/28 |
| 359 | 0/47 |
| 361 | 0/92 |
| 362 | 0/30 |
| 409 | 82/120 |
| 416 | 4/92, 0/92, 1/28 |
| 417 | 0/47 |
| 418 | 0/92, 0/30 |

Pooled fractions:

- 0/831 on line 373
- 0/320 on line 374
- 239/831 on line 382
- 1/320 on line 423

`docs/viva/05-...` is reported as `SKIP ... absent`. Plan 05-03 converts page 04 to generated blocks, which removes this baseline.

## Prohibitions honoured

- **Read-only sources.** The build opened every source only with `open(path, "rb")`. The test `test_build_reads_read_only_sources_without_changing_them` proves this: it sets every source file to 0o444 and every source directory to 0o555, runs the build, and finds every sha256 and every directory listing unchanged afterwards.
- **No writes to the live directories.** Nothing was written under `results/phase4-*` or `results/final-20261011`.
- **No process control.** No process was started, stopped or signalled except the build itself and its `git rev-parse HEAD`. The Groq loop (WSL pid 10209) was not touched.
- **No key loading or printing.** `src/deliverables` imports nothing from `src.llm`, `src.benchmark.jobs` or `src.benchmark.run_checks`; a fresh-interpreter `sys.modules` test proves it. No `.env` was opened or printed, no key value was written anywhere, and nothing was installed with pip.
- **Key scan.** `SECRETS results/phase5-final files 30 ok`.
- **No run-affecting files.** No file in RUN_AFFECTING_PATHS was modified; `src/benchmark/report.py` is not in that list.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] The Task 1 test's legacy assertion contradicted Task 2's legacy table**
- **Found during:** Task 2.
- **Issue:** `test_report_has_data_line_and_no_legacy_rows` asserted that the legacy ids appear nowhere in summary.md. Task 2 adds the required legacy table and a threat item that names those ids. The truth to test is "no headline or partial row".
- **Fix:** The assertion now checks that the legacy ids are absent from the headline and partial sections and present in the legacy section. In the CSVs they must stay absent.
- **Files modified:** tests/test_deliverables_build.py.
- **Commit:** e651643. The Task 2 commit therefore lists six files, not five.

**2. [Rule 3 - Blocking] Chart drawing in WSL takes about 22 s per chart set**
- **Found during:** Task 2.
- **Issue:** matplotlib 3.11 under WSL on Python 3.14 spends about 80 ms in `ft2font.set_text` per text item. The same three charts take 3.8 s on Windows Python 3.12. With every test build drawing charts, the two test files took 240 s.
- **Fix:** Tests that do not test the charts stub `charts.write_phase5_charts` with an autouse fixture. The two chart tests call the real function. Production code is unchanged.
- **Files modified:** tests/test_deliverables_build.py, tests/test_deliverables_sections.py, tests/test_deliverables_check.py.
- **Commits:** e651643, 6134a0c.

**3. [Rule 2 - Correctness] A model with an empty axis is still labelled in the charts**
- **Found during:** Task 2.
- **Issue:** A partial hosted model with no benign episodes yet could not be plotted on the scatter or arrow charts. It would have disappeared from them without notice.
- **Fix:** Such a model gets a "Not plotted, an axis has no episodes yet: model (partial D/E) (counts)" note on the figure.
- **Commit:** e651643.

### Design choices within Claude's discretion

- **Pins read the source, not the module.** The pinning tests read FREEZE_UTC, HOSTED_DS_EXPECTED, SECRET_RE and DEFAULT_MAX_CONTEXT_TOKENS from the source with `ast`, instead of importing jobs, run_checks or src.llm. The plan allows tests to import jobs. Reading the source asserts the same equality without loading the registry.
- **A FAIL line is printed once per line and token.** A k/n repeated on one doc line produces one FAIL line, not several.

## Known Stubs

None. Every block, section and chart is wired to numbers.json, which is built from the frozen data.

## Follow-up (post-freeze, not a step of this plan)

After 2026-10-29 15:30 UTC, run the same command once from the repo root:

```
PYTHONUTF8=1 py -3.12 -m src.deliverables build
```

It re-freezes, with data state `final`, and regenerates the report, charts and numbers.json. It refreshes every block in README, SDD and viva pages 04 and 05. If hand-written prose quotes an outdated number, it fails loudly (WINDOWS.md entry 2).

Until plans 05-02 and 05-03 have converted the docs, use `--doc README.md --doc docs/SDD.md`. The other option is to run `--no-docs`, then `check` separately.

## Self-Check: PASSED
