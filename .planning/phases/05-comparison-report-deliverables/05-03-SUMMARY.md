---
phase: 05-comparison-report-deliverables
plan: 03
subsystem: docs
status: complete
tags: [viva, doc-03, generated-blocks, consistency-check, code-review, threats-to-validity]
requires:
  - results/phase5-final/numbers.json and summary.csv (plan 05-01, snapshot 2026-10-29T07:12Z)
  - python -m src.deliverables blocks | check with --doc scoping (plan 05-01)
  - .planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-REVIEW-FIX.md (commit 880e98a)
provides:
  - docs/viva/05-evaluation-statistics-and-examiner-qa.md (Parts A-F, 7 generated blocks, 40 examiner questions)
  - docs/viva/04-namespaces-seccomp-and-local-inference.md with a generated Part C (5 blocks) and a filled code-review section
  - docs/viva/README.md row 05, written (Phase 5)
affects: [05-04 deck (can cite the viva set as complete), post-freeze rebuild (refreshes pages 04 and 05)]
tech-stack:
  added: []
  patterns: [result numbers only inside marker-delimited generated blocks; prose points to blocks instead of quoting Groq or pooled numbers]
key-files:
  created:
    - docs/viva/05-evaluation-statistics-and-examiner-qa.md
  modified:
    - docs/viva/04-namespaces-seccomp-and-local-inference.md
    - docs/viva/README.md
decisions:
  - "Page 05 Part F answers point to Part D for every result number, so none of the 40 answers can go stale after the post-freeze build"
  - "Page 04 keeps final local numbers in prose (0 of 120, 0 of 40 cases, scratch-gpt's random-choice baseline) but quotes no Groq or pooled number; those live only in the five blocks"
  - "Page 04's Phase 1 row list after the layers block is replaced by a pointer to the block's own LAYER_EVIDENCE paragraph, which cites the same rows, to avoid saying it twice"
metrics:
  duration: 45min
  completed: 2026-10-29
  tasks: 3
  files: 3
estimate:
  tokens: 120000
actuals:
  tokens: 24316
  tasks: 3
  commits: 3
plan_head_before: c856e11d9d8f982d875718caf94c15f122ca5923
plan_head_after: 258c99f245c29d51598e29042d53be0e5746061c
---

# Phase 5 Plan 03: Viva Page 05 and Generated Page 04 Summary

The viva set is complete (DOC-03). Page 05 covers prompt injection and its defenses, the evaluation statistics behind every number, the confounds and threats to validity, the frozen results, a one-page "defend the result" summary and 40 examiner questions across all five phases. Page 04's Part C now takes its numbers from the same numbers.json as the report, README, SDD and deck (D-03), and its code-review section is filled from 04-REVIEW-FIX.md.

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 (tracer): page 05 with Part D's five generated blocks, index row 05 | afd3b13 | docs/viva/05-evaluation-statistics-and-examiner-qa.md, docs/viva/README.md |
| 2: glossary, Parts A-C (confounds and threats blocks), Part E | b06bb8c | docs/viva/05-evaluation-statistics-and-examiner-qa.md |
| 3: Part F; page 04 Part C generated; page 04 code-review section | 258c99f | docs/viva/04-namespaces-seccomp-and-local-inference.md, docs/viva/05-evaluation-statistics-and-examiner-qa.md |

`git show --stat --format= SHA` listed exactly the task's files for each commit. Nothing on the never-stage list was staged.

## Page 05

- 876 lines, LF only (0 CRLF).
- 66 lines starting `- **` (38 glossary terms plus the bold list items in Part A and Part E's evidence).
- 40 questions (`**Q1.` to `**Q40.`), each followed by its answer line (awk gate: q=40, a=40, bad=0).
- Parts in order: glossary, A (prompt injection and defenses), B (evaluation statistics), C (confounds and threats; blocks confounds, threats), D (blocks status, findings, headline, layers, control), E (defend the result in one page), F (examiner questions).
- Part F areas: OS security 6 (Q1-Q6), namespaces and seccomp 5 (Q7-Q11), LLM foundations 5 (Q12-Q16), LoRA and chat templates 5 (Q17-Q21), prompt injection and defenses 5 (Q22-Q26), evaluation statistics 6 (Q27-Q32), results, confounds and threats 8 (Q33-Q40).
- Part B states the Wilson formula with z = 1.96 and works 0 of 120 to [0.0, 3.1]; `wilson_interval(0, 120)` returns (0.0, 0.03102), and `wilson_interval(0, 40)` returns (0.0, 0.0876), the 8.8% quoted for the case-level view.

## Page 04

Sections converted in Part C (heading "## Part C — What the numbers say now" kept):

- Lead paragraph rewritten: numbers come from results/phase5-final/numbers.json and are checked by `check`; the status block gives the snapshot; local rows final; Groq rows final after the build re-runs after the freeze at 15:30 UTC on 2026-10-29.
- "Completeness per model": table replaced by the status block.
- "Headline: the five complete local models" renamed "Headline: every current-engine model"; table replaced by the headline block; the prose keeps only local final numbers and no Groq or pooled number.
- "The model-independent control": numbers paragraph replaced by the control block and one interpretation sentence.
- "Ollama window evidence (D-06)": unchanged.
- "Blocks by layer, per model and pooled (EVAL-08)": table and quoted report sentences replaced by the layers block; interpretation prose kept with every pooled count removed.
- Hosted subsection renamed "The hosted re-run at the data snapshot"; the d4ee817 sentence kept with the loop start written as "18:48 UTC on 2026-10-28"; both tables replaced by the hosted block; outage, false-positive and ds-first explanations kept without counts.

Outside Part C, three places no longer quote the earlier report's pooled count or time, and each now points to the layers block:

- the opening paragraph's zero-block sentence;
- the "Limits" bullet that quoted the pooled count (the second bullet; the first holds only Phase 1 row numbers and is unchanged);
- the Q8 answer.

Code-review section "What the code review found and what we fixed":

- **Fixed:** CR-01 (loader record ranking; tests/test_phase4_runner.py), WR-02 (ownership re-check before SIGKILL; tests/test_jobs.py), WR-03 (freeze checked after preflight, after Ollama setup and before spawn; three tests in tests/test_jobs.py), WR-04 (progress gate split from provenance; seven tests in tests/test_run_checks.py and tests/test_jobs.py), WR-05 (key-check ordering test that can fail, exit-before-freeze test), plus the first bullet of IN-01 and the client_env half of IN-07.
- **Deferred until after the freeze (run-affecting, WINDOWS.md entry 4):** WR-01 (upward bias on ASR-valid; no Phase 4 episode had a context overflow; the report's threats section states that count), IN-04, IN-05.
- **Not addressed:** IN-02, IN-03, IN-06, the rest of IN-01 and the rest of IN-07, in one line.

## Index

docs/viva/README.md row 05 now links `05-evaluation-statistics-and-examiner-qa.md` with the planned topics, phase "Phase 5" and status "written (Phase 5)". `git diff --stat` before the commit showed 1 insertion and 1 deletion; rows 01-04 unchanged. LF kept.

## Check output

Scoped check (Task 3 verify):

```
DOC docs/viva/04-namespaces-seccomp-and-local-inference.md blocks 5 ok
DOC docs/viva/05-evaluation-statistics-and-examiner-qa.md blocks 7 ok
SECRETS results/phase5-final files 30 ok
deliverables-check-ok docs 2
```

The last `blocks --doc` on both pages printed `5 unchanged` and `7 unchanged`.

Unscoped check (`PYTHONUTF8=1 py -3.12 -m src.deliverables check`, exit 0), which failed with 33 lines on page 04 at the 05-01 baseline:

```
DOC README.md blocks 4 ok
DOC docs/SDD.md blocks 9 ok
DOC docs/viva/04-namespaces-seccomp-and-local-inference.md blocks 5 ok
DOC docs/viva/05-evaluation-statistics-and-examiner-qa.md blocks 7 ok
SECRETS results/phase5-final files 30 ok
deliverables-check-ok docs 4
```

run_checks viva for page 04 (WSL venv, against results/phase5-final/summary.csv): `viva-numbers-ok`, exit 0.

## Verify results

| Verify | Result |
|--------|--------|
| T1: page 05 holds 5 begin markers | 5 |
| T1: scoped check prints DOC ... blocks 5 ok and check-ok docs 1 | 2 |
| T1: index links page 05 written (Phase 5) once | 1 |
| T2: glossary lines >= 30 | 66 |
| T2: Parts A-E once each | 5 |
| T2: Part E heading once | 1 |
| T2: scoped check blocks 7 ok | 2 |
| T3: awk question/answer gate | exit 0 (40/40/0) |
| T3: page 05 >= 350 lines | 876 |
| T3: Parts A-F once each | 6 |
| T3: page 04 holds 5 begin markers | 5 |
| T3: scoped check on both pages, 4 lines | exit 0 |
| T3: run_checks viva page 04 | viva-numbers-ok |
| T3: page 04 prose has no "1151" or "03:06" | 0 |
| T3: code-review section names the 8 IDs | 8 |
| Acceptance: page 04 CRLF count | 0 |
| Acceptance: page 04 keeps Part A-D and Limits headings | yes |

## Prohibitions honoured

- No attack command, bypass recipe, payload text or key on pages 04 or 05. The injection example in Part A is described in words. The Phase 1 residual-bypass table is cited by row number only, and the review findings by ID. The SECRETS scan and run_checks viva's key scan both passed.
- No code changed. The only commands run were `blocks` and `check` scoped with `--doc` to these pages, one unscoped read-only `check`, and the read-only run_checks viva. No `build` was run, so the snapshot is unchanged.
- No .env was opened or printed, nothing was written under results/, no process was signalled, and nothing was installed.

## Deviations from Plan

None that change the plan's outputs. Two notes:

- The plan names "the first bullet of Limits" as quoting the pooled count. On the page that bullet was the second one; the first bullet holds only Phase 1 row numbers. The second bullet was rewritten, and the first was left as it was.
- Writing Part F through a shell heredoc failed (a quoting error in the tool, and nothing was written). It was written to a scratchpad file and appended with Python, normalised to LF.

## Notes for the orchestrator

- **WINDOWS.md entry 3** (the page 04 code-review placeholder) can be marked fixed: the section is filled from 04-REVIEW-FIX.md (commit 258c99f).
- **WINDOWS.md entry 2** (page 04's Groq rows): page 04 no longer quotes any Groq number by hand. Its five blocks are refreshed by the post-freeze `PYTHONUTF8=1 py -3.12 -m src.deliverables build`, so the entry is resolved once that build runs. Page 05's seven blocks refresh the same way.

## Known Stubs

None. Every result table on both pages is a generated block filled from numbers.json.

## Self-Check: PASSED

- FOUND: docs/viva/05-evaluation-statistics-and-examiner-qa.md
- FOUND: docs/viva/04-namespaces-seccomp-and-local-inference.md
- FOUND: docs/viva/README.md
- FOUND commits: afd3b13, b06bb8c, 258c99f (ancestors of HEAD)
