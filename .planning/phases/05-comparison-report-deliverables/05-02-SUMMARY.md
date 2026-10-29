---
phase: 05-comparison-report-deliverables
plan: 02
subsystem: docs
status: complete
tags: [readme, sdd, ieee-1016, generated-blocks, consistency-check, doc-04]
requires:
  - results/phase5-final/numbers.json (plan 05-01, snapshot 2026-10-29T07:12Z)
  - python -m src.deliverables blocks | check (plan 05-01)
provides:
  - README.md for the multi-LLM and local-model system with 4 generated blocks (status, headline, findings, charts)
  - docs/SDD.md version 3.0 with sections 5.13, 5.14, 5.15, 7.3 (9 generated blocks) and Appendix C
affects: [05-03 viva pages (README index links docs/viva), 05-04 deck, post-freeze rebuild]
tech-stack:
  added: []
  patterns: [result numbers only inside marker-delimited generated blocks, per-file line endings kept (README CRLF, SDD LF)]
key-files:
  created: []
  modified:
    - README.md
    - docs/SDD.md
decisions:
  - "README headings drop the old emojis so the new required headings and the old ones share one style"
  - "README benchmark examples use --prefix results/run1, not results/phase4, so a reader cannot write into the live Phase 4 directories"
  - "SDD prose names the marker lines without writing the literal marker text, because the block parser rejects a marker that is not alone on its line and the plan's grep counts every 'deliverables:begin ' occurrence"
  - "SDD 5.13-5.15 are presented as design-element sections combining the composition, interface and algorithm viewpoints, listed in section 4 under the twelve IEEE viewpoints"
  - "Four design-rationale rows added to section 6 (comply-trained local models, scripted control, committed stopping rule, generated blocks)"
metrics:
  duration: 15min
  completed: 2026-10-29
  tasks: 3
  files: 2
estimate:
  tokens: 70000
actuals:
  tokens: 18632
  tasks: 3
  commits: 3
plan_head_before: da1f57472aee93a73230daf6e572dc9897179850
plan_head_after: 6a8948c6891b37df599c307ea1a595bba7ef7901
---

# Phase 5 Plan 02: README and SDD for the Multi-LLM System Summary

README.md and docs/SDD.md now describe the guardrails as holding for any LLM: hosted Groq, local Ollama, the Qwen base and LoRA models served in-process, ScratchGPT trained from scratch, and the scripted control. Every result number in the two files sits in generated blocks filled from `results/phase5-final/numbers.json`, and the scoped consistency check passes on both (DOC-04, D-03, D-06).

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 (tracer): README results section becomes four generated blocks | 73a5dd4 | README.md |
| 2: the rest of the README covers the multi-LLM and local-model system | 6591892 | README.md |
| 3: SDD version 3.0 | 6a8948c | docs/SDD.md |

`git show --stat --format= SHA` lists exactly one file for each commit. `git log --format= --name-only da1f574..HEAD` lists only README.md (2) and docs/SDD.md (1).

## Check output (final, scoped)

`PYTHONUTF8=1 py -3.12 -m src.deliverables check --doc README.md --doc docs/SDD.md` exits 0:

```
DOC README.md blocks 4 ok
DOC docs/SDD.md blocks 9 ok
SECRETS results/phase5-final files 30 ok
deliverables-check-ok docs 2
```

The last `blocks --doc README.md --doc docs/SDD.md` printed `BLOCKS README.md 4 unchanged` and `BLOCKS docs/SDD.md 9 unchanged`.

## Verify results

| Verify | Result |
|--------|--------|
| T1: README holds 4 begin markers | 4 |
| T1: scoped check on README prints both lines | 2 (exit 0) |
| T2: the four new README headings, once each | 4 |
| T2: README contains `py -3.12 -m src.deliverables build` | 1 |
| T2: README names `hf_local` at least twice | 5 |
| T2: Models compared table has 9 rows, all ids in config/models.yaml | 9 rows, 0 unknown |
| T2: README LF-only lines | 0 (CRLF kept) |
| T3: the five new SDD headings, once each | 5 |
| T3: one `| 3.0 |` revision row | 1 |
| T3: SDD holds 9 begin markers | 9 |
| T3: combined scoped check, 4 lines | 4 (exit 0) |
| T3: SDD CRLF count | 0 (LF kept) |
| T3: `git status --porcelain docs/SDD.pdf docs/SDD.docx` | empty |

## What changed

**README.md (CRLF, 343 lines):**

- New opening paragraph: the monitor mediates every action whichever model drives the agent.
- New sections:
  - Review 3: a locally made LLM and other LLMs (the remark quoted verbatim, with four bullets)
  - Models compared (9 rows: 5 local, 3 hosted, 1 control, plus a legacy sentence)
  - Reproducing the results (the one build command, what it writes, the check, the post-freeze re-run)
  - Viva notes (a link to docs/viva/README.md and one line per page)
- The Solution table lists the four layers in mediation order. The architecture diagram shows "AI agent (any LLM)" and a provider-layer box.
- Project structure: adds src/llm, the benchmark jobs and run_checks, src/scratch_llm, src/qwen_lora, src/deliverables, config, models/ and results/.
- Quick start: keys are optional and per provider, plus a "Run the Benchmark" block.
- InjecAgent added to the references.
- The results section is four generated blocks. The scripted attack-simulation table is kept as its own subsection.
- Test line: "Test suite (full run in WSL on 29 October 2026): 1308 passed and 1 skipped, no failures."

**docs/SDD.md (LF, 962 lines), version 3.0:**

- Front matter, header table and a `| 3.0 |` revision row that quotes the remark.
- 1.2 Scope names every provider and adds the training pipeline and the run tooling. 1.5 is updated.
- Section 4 gains a table for the new sections.
- Mermaid diagrams updated:
  - Section 3: the provider node
  - 5.1: the LLM providers node and the edge label
  - 5.2: adds HFLocalProvider, ScriptedControlProvider and modules M8–M10
  - 5.3: two more provider classes
- Text sections updated:
  - 5.5: the registry fields
  - 5.7: CLI rows for jobs, run_checks, the training scripts and deliverables
  - 5.8: the structure tree
  - 5.11.4: the context is now 2 048 characters, and training points to 5.14
  - 5.12: GPU, memory and quotas
- New sections:
  - 5.13 Multi-LLM provider layer (concerns C2, C6)
  - 5.14 Local training pipeline (C2, C7)
  - 5.15 Run tooling and deliverables (C7, C5, C2)
- Section 6 gains four rationale rows.
- 7.1 lists the test areas added since 2.0 and the full-suite count: 1308 passed, 1 skipped, dated 29 October 2026, run in four WSL calls.
- 7.2 gives the current-engine line-up as a table, with Gemini and Gemma as legacy only. The procedure is now `jobs local`, `jobs hosted` and the deliverables build.
- New 7.3 Results (frozen Phase 5 data) holds nine blocks in order: status, headline, findings, confounds, layers, control, legacy, threats, charts.
- Appendix A: updated concern rows, plus a table for the sections added in 3.0.
- New Appendix C maps the Review 3 remark to design changes.

## Prose rules honoured (D-01, D-03)

- No Groq number appears outside a block.
- The freeze is written as "15:30 UTC on 2026-10-29", and no other ISO timestamp appears in prose.
- No k/n appears on a line that names a model. The only fractions in prose are the scripted-simulation table (7/7, 0/7, 1/1), which names no model.
- Phase 1 residual-bypass evidence appears only through the generated layers and threats blocks, which cite rows by number.
- No attack command, payload or bypass text was copied into either document.

## Manual step: SDD PDF/DOCX export

`scripts/export_sdd.py` was not run. It loads marked 12.0.2 and mermaid 11.4.1 from cdn.jsdelivr.net, and it drives Chrome and Word. Per D-06, regenerating docs/SDD.pdf and docs/SDD.docx is a manual step. Run `python scripts/export_sdd.py` on Windows with network access, preferably after the post-freeze rebuild so the exported tables carry the final Groq numbers. Both binaries are unchanged and were not staged.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] Writing the marker syntax in SDD prose broke `blocks` and then the marker count**
- **Found during:** Task 3.
- **Issue:**
  - Section 5.15 first quoted the full marker comment inline. `blocks` stopped with `FAIL blocks: docs/SDD.md:692: malformed deliverables marker (it must be alone on its line)`.
  - The next wording still contained `deliverables:begin NAME`. The plan's `grep -c "deliverables:begin "` then counted 10 instead of 9.
- **Fix:** The sentence now reads "between its begin and end marker lines (HTML comments naming the block, invisible when rendered)".
- **Commit:** 6a8948c.

**2. [Rule 2 - Correctness] README benchmark examples would have targeted the live Phase 4 prefix**
- **Found during:** Task 2.
- **Issue:** The natural example, `--prefix results/phase4`, matches the directories the Groq loop is writing until the freeze.
- **Fix:** The examples use `--prefix results/run1`.
- **Commit:** 6591892.

### Notes

- Only `blocks` and `check` were run, always scoped with `--doc README.md` and/or `--doc docs/SDD.md`. No `build` was run, so the snapshot is unchanged.
- No test run was needed, because no code changed. The full-suite count is quoted from 05-01-SUMMARY.md.
- No process was started, stopped or signalled except the Python calls above.
- .env was not opened. Nothing was installed.

## Known Stubs

None. The README Viva notes list mentions page 05. Plan 05-03 creates that page, and this plan only links the viva index.

## Follow-up (post-freeze, not a step of this plan)

After 15:30 UTC on 2026-10-29, run `PYTHONUTF8=1 py -3.12 -m src.deliverables build` once. It refreshes all 13 blocks in README.md and docs/SDD.md, and no hand edit is needed. Then export the SDD PDF/DOCX manually as described above.

## Self-Check: PASSED

- README.md and docs/SDD.md exist.
- 73a5dd4, 6591892 and 6a8948c are ancestors of HEAD.
- The scoped check exits 0 with all four lines.
