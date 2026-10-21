---
phase: 02-local-data-foundation-scratchgpt
plan: 01
subsystem: testing
tags: [scratchgpt, compact-format, synthetic-data, leakage-gate, parity, injecagent, torch-free]

requires:
  - phase: 01-evaluation-hardening-policy-fix
    provides: prompting.py single source of truth, AgentLoop, ToolBroker, manifest conventions
provides:
  - "Compact transcript format version 2 (required parameter names on the T: line, context 2048) with version / vocab / render-fingerprint stamps"
  - "Episode schema with full tool specs, training_text (= render_context + gold action) and the A:-payload loss mask"
  - "Strict D-04 leakage gate (assert_clean, check_templates) derived from benchmarks/injecagent and enforced inside generate_episodes"
  - "Reproducible benign generator slice, LF JSONL writer/reader with file sha256, ignore rules for weights and episode files"
affects: [02-02 full 50/25/25 generator and dev splits, 02-03 train and inference, 03 LoRA trainer]

plan_head_before: 494285fc764cb05689c39a167484fb5a215a3151
plan_head_after: 2741b9d7a9cd5c97c8401dcea8b415219427e2bd

actuals:
  tokens: 21452
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Training text is the inference renderer's output (parity by construction), never a second renderer"
    - "Leakage gate derived programmatically from the benchmark files and run inside the generator"
    - "One random.Random per call, every pool sorted before drawing; LF-only JSONL for cross-OS hashes"

key-files:
  created:
    - src/scratch_llm/__init__.py
    - src/scratch_llm/format.py
    - src/scratch_llm/dataset.py
    - src/scratch_llm/templates.py
    - src/scratch_llm/model.py
    - tests/test_format_parity.py
    - tests/test_training_leakage.py
  modified:
    - .gitignore

key-decisions:
  - "write_episodes keeps dict insertion order (no sort_keys): sorting would reorder call_tool args away from the spec's required order and the {action,tool,args} gold text"
  - "Gate normalisation is a superset of the research one: shingles are collected and checked with and without edge-dot stripping; the exact-sentence check strips edge dots"
  - "render_context takes an optional budget so validate_episode can measure the untruncated length (fit always shortens, so the truncated render cannot reveal truncation)"
  - "The render fingerprint uses the literal template text for the golden observation so a changed OBSERVATION_TEMPLATE changes the hash"
  - "Free-pool tools with a held-out required signature are kept; the signature rule applies to synthetic tools only (measured: 77 of 251 free tools would be lost)"

patterns-established:
  - "Planted-violation tests: one hand-made clean episode, one broken rule, assert the rule name in the LeakageError"
  - "Benchmark strings in tests are read from the benchmark files or build_corpus() at test time, never pasted"

requirements-completed: [SCRATCH-01, SCRATCH-02, SCRATCH-03]

coverage:
  - id: D1
    description: "Format version 2: T: line with required parameter names parsed from the harness system prompt (all 330 specs round-trip), context 2048, GEN_RESERVE 400, line-boundary fit"
    requirement: SCRATCH-03
    verification:
      - kind: unit
        ref: "tests/test_format_parity.py#test_every_vendored_tool_spec_round_trips_through_the_t_line"
        status: pass
      - kind: unit
        ref: "tests/test_format_parity.py#test_render_context_cuts_old_turns_at_a_line_boundary"
        status: pass
    human_judgment: false
  - id: D2
    description: "Train/inference parity by construction: every step prompt is an exact prefix of the training text, also through a real AgentLoop and a real ToolBroker denial; loss mask covers exactly the A: payloads"
    requirement: SCRATCH-03
    verification:
      - kind: unit
        ref: "tests/test_format_parity.py#test_agent_loop_replay_matches_with_a_real_tool_broker_denial"
        status: pass
      - kind: unit
        ref: "tests/test_format_parity.py#test_generated_episodes_satisfy_parity_mask_and_replay"
        status: pass
      - kind: unit
        ref: "tests/test_format_parity.py#test_supervised_mask_decodes_to_the_gold_actions"
        status: pass
    human_judgment: false
  - id: D3
    description: "Version, vocab hash and render fingerprint stamps (pinned golden fingerprint, notices a changed OBSERVATION_TEMPLATE)"
    requirement: SCRATCH-03
    verification:
      - kind: unit
        ref: "tests/test_format_parity.py#test_render_fingerprint_is_pinned"
        status: pass
      - kind: unit
        ref: "tests/test_format_parity.py#test_render_fingerprint_notices_a_changed_observation_template"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-04 leakage gate: 13 planted violations each raise LeakageError naming the rule, free text with held-only words passes, clean 2,000-episode set passes, gate inside generation cannot be skipped"
    requirement: SCRATCH-02
    verification:
      - kind: unit
        ref: "tests/test_training_leakage.py (25 tests)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Reproducible benign generator slice with copyable values, LF JSONL and stable sha256 across processes and hash seeds"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_format_parity.py#test_generation_does_not_depend_on_the_hash_seed"
        status: pass
      - kind: unit
        ref: "tests/test_format_parity.py#test_write_and_read_round_trip_with_lf_only_and_a_file_hash"
        status: pass
    human_judgment: false
  - id: D6
    description: "SCRATCH-01 full scope (50/25/25 mix with injected dh and ds episodes, synthetic tools, dev splits) is NOT in this plan; only the benign slice exists"
    requirement: SCRATCH-01
    verification: []
    human_judgment: true
    rationale: "The 50/25/25 quota is delivered by plan 02-02; this plan only provides the foundation it builds on"

duration: 20min
completed: 2026-10-21
status: complete
---

# Phase 2 Plan 1: Format v2, episode schema and leakage gate Summary

**Compact transcript format v2 (required parameter names on the T: line, 2048-char context) with training text that is literally the inference renderer's output, a D-04 leakage gate derived from InjecAgent that runs inside a reproducible generator, and 59 new torch-free tests.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-10-21T04:00:42Z
- **Completed:** 2026-10-21T04:21:03Z
- **Tasks:** 3
- **Files modified:** 8 (7 created, 1 modified)

## Accomplishments

- Format version 2: `T:GmailSendEmail(to,subject,body)|Other()` parsed from the system prompt `build_messages` writes (all 330 vendored specs round-trip), `CONTEXT_CHARS` 2048, `GEN_RESERVE` 400, `fit` now cuts at a line boundary, `supervised_mask` marks exactly each `A:` payload plus its newline.
- Parity by construction: `training_text(ep) = render_context(messages before the last step) + gold action + "\n"`; every per-step prompt is an exact prefix, proven for generated episodes through a real `AgentLoop` and for a hand-built episode containing the exact BLOCKED text a real `ToolBroker` returns.
- Stamps for the checkpoint: `FORMAT_VERSION = 2`, `VOCAB_SHA256`, `render_fingerprint()` (pinned in a test; changes when `OBSERVATION_TEMPLATE`, the limits or the fit rule change).
- Leakage gate (T-02-01): held-out tools (79, `GmailSendEmail` included), 50 held-only parameter names as tool parameters / `call_tool` argument keys only, synthetic-tool signature rule, 6 benchmark e-mail addresses, canary shape and "Private record" phrasing, 8-word shingles, exact short sentences, tool summaries, plus a scan of `templates.py`; runs inside `generate_episodes`.
- Reproducible benign generator: 251 real non-benchmark tools, return-schema observations, copyable values, LF-only JSONL whose sha256 is identical across processes and `PYTHONHASHSEED` values.

## Measured numbers (for the viva and the dataset manifest)

| Item | Value |
|------|-------|
| Pinned render fingerprint | `650ca1c4193c2e07e9fdee4990c4fb37b2edd7964aa5026a64c53b9af8fadf04` |
| VOCAB_SHA256 | `a5444bb0573ce8bbe523ca78bfd0e064568459ac55615c59f1e5f1cbbc0aad4b` |
| Held-out tool names | 79 (GmailSendEmail included) |
| Held-only parameter names | 50 |
| Benchmark e-mail addresses | 6 |
| Distinct 8-word shingles | 2,035 under the research normalisation; 2,780 in the gate corpus (union with the edge-stripped normalisation) |
| Exact benchmark sentences | 79 (user + attacker instructions) |
| Held-out non-empty required signatures | 52, of which 27 contain no held-only name |
| Free-pool tools dropped by the gate | 0 of 251 |
| Gate time | 0.48 s for 2,000 episodes, about 10 s for 40,000 (generation plus gate for 40,000: 20 s) |
| Longest generated U: line / training text | 240 / 801 characters |

`HELD_ONLY_GATE_SCOPE` (plan 02-02 writes it into the dataset manifest), exact text:

> Held-only parameter names (names that occur only in held-out benchmark tools, such as to, subject or url) are rejected as tool parameter names (on the T: line / in tool specs) and as call_tool argument keys. The same words in free text, that is U: requests, O: observation values and respond messages, are allowed and are not gated.

## Task Commits

1. **Task 1: format v2 render, episode schema, parity spine (tracer)** - `e0bf369` (feat)
2. **Task 2: benign generator, LF JSONL, context 2048, ignore rules** - `ef08e11` (feat)
3. **Task 3: strict leakage gate with planted-violation tests** - `2741b9d` (feat)

**Plan metadata:** committed separately (docs: complete plan).

Tracer gate: the Task 1 `<verify>` was automated-only; it was re-run end to end after the commit and passed, so expansion continued (no checkpoint, `end-of-phase` mode).

## Files Created/Modified

- `src/scratch_llm/__init__.py` - package marker with docstring, no imports
- `src/scratch_llm/format.py` - format v2: constants, `tools_from_system`, `render_header`, `fit`, `render_context`, `supervised_mask`, stamps
- `src/scratch_llm/dataset.py` - schema, `episode_to_messages`, `training_text`, `validate_episode`, held-out sets, corpus + `assert_clean` + `check_templates`, generator, JSONL I/O
- `src/scratch_llm/templates.py` - 44 train + 8 dev instruction templates, 18 reply templates, synonym map, word lists (all self-written)
- `src/scratch_llm/model.py` - now tracked; `GPTConfig.context` default 2048
- `.gitignore` - weights, logs, pilot dir, `data/local/*.jsonl` (manifests, train logs, loss curves stay tracked)
- `tests/test_format_parity.py` - 26 tests (format, parity, mask, stamps, generator, determinism)
- `tests/test_training_leakage.py` - 25 tests (13 planted rules, boundaries, templates, generation gate)

## Decisions Made

- Gold-action key order is `{"action","tool","args"}` as the system prompt documents it; the JSONL writer therefore preserves insertion order instead of sorting keys.
- The gate is deliberately a superset of the research design (two normalisations, A: lines in the sentence check, held-out names matched as whole words in the whole training text).
- Real free-pool tools are exempt from the signature rule (only synthetic tools are checked), as the plan's objective states.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `write_episodes` must not use `sort_keys=True`**
- **Found during:** Task 2 (design of the JSONL round trip)
- **Issue:** The plan asked for `json.dumps(..., sort_keys=True)`. Sorting reorders `args` alphabetically, so `read_episodes` -> `validate_episode` would reject every multi-argument call (args keys must equal the required names in spec order) and the gold text would become `{"action","args","tool"}`, differing from the documented action grammar.
- **Fix:** Compact `json.dumps(ep, separators=(",", ":"), ensure_ascii=True)` with insertion order preserved; generation is deterministic, and the cross-process / `PYTHONHASHSEED` test proves the hash is stable.
- **Files modified:** src/scratch_llm/dataset.py
- **Verification:** `test_write_and_read_round_trip_with_lf_only_and_a_file_hash`, `test_generation_does_not_depend_on_the_hash_seed`
- **Committed in:** ef08e11

**2. [Rule 1 - Bug] The planned "no truncation" check cannot detect truncation**
- **Found during:** Task 1 (`validate_episode`)
- **Issue:** `len(render_context(...)) - 2 <= CONTEXT_CHARS - GEN_RESERVE` is always true, because `fit` itself cuts the context to that budget.
- **Fix:** `render_context(messages, budget=None)` takes an optional budget; `validate_episode` renders with a huge budget and compares the untruncated length with the fit budget, and checks the final text is a prefix-equal extension of that render and at most `CONTEXT_CHARS`.
- **Files modified:** src/scratch_llm/format.py, src/scratch_llm/dataset.py
- **Verification:** `test_validate_episode_rejects[_too_long_context]`
- **Committed in:** e0bf369

**3. [Rule 2 - Missing critical] Gate normalisation made stricter**
- **Found during:** Task 3
- **Issue:** The research tokenizer keeps edge dots (`door.` != `door`), so a benchmark sentence with different trailing punctuation, or an 8-word run shifted by punctuation, would slip through. Stripping alone also lowered the shingle count to 1,988 (below the plan's ">2000").
- **Fix:** Shingles are collected and checked under both normalisations (union 2,780, research-only 2,035); the exact-sentence check uses edge-stripped tokens and also scans `A:` lines (respond messages), not only `U:` and `O:`.
- **Files modified:** src/scratch_llm/dataset.py
- **Verification:** `test_rule_7_*`, `test_rule_8_*`, `test_corpus_is_derived_from_the_benchmark_files`
- **Committed in:** 2741b9d

**4. [Rule 2 - Missing critical] `validate_episode` rules beyond the plan's list**
- **Found during:** Task 1
- **Issue:** An observation containing a newline or exceeding `OBS_CHARS` would be silently flattened or cut in the render, and a final training text longer than the model context (header+turns 1,648 plus a 403-char action line can reach 2,051) would not fit the model.
- **Fix:** Observations must be single-line and at most `OBS_CHARS`; the training text must be at most `CONTEXT_CHARS`; `meta`/`cls`/`split` keys are schema-checked.
- **Files modified:** src/scratch_llm/dataset.py
- **Committed in:** e0bf369

---

**Total deviations:** 4 auto-fixed (2 bug, 2 missing critical), plus the process note below.
**Impact on plan:** All four are necessary for correctness of the parity and gate claims; no scope creep. No architectural change.

### Process note (TDD)

Task 1's tests were written after `format.py`/`dataset.py` (pre-existing untracked design files that I rewrote in the same sitting), so no RED run was recorded for Task 1. Tasks 2 and 3 were red-first (collection `ImportError` for `templates` / `RULE_*` confirmed before implementation).

## Known Stubs

- `src/scratch_llm/dataset.py:676` - `generate_episodes` raises `NotImplementedError` for `dev_tools` / `dev_synthetic` splits; the dh and ds classes, synthetic tools, OS-action and blocked-tail episodes and the 50/25/25 quota are added by plan 02-02 behind the same signature. Recorded in `.planning/WINDOWS.md` (entry 1, open) so it is visible at ship time.

## Issues Encountered

- Git Bash heredocs mangled backslash escapes in one scripted test edit (`"\n"`, `b"\r"` became raw control characters); fixed by editing with the Edit tool and a script written via the Write tool. No effect on committed files.

## Authentication Gates

None.

## Threat Flags

None - no new network endpoints, auth paths or trust-boundary file access beyond the plan's threat model (the generator only reads the vendored benchmark files and writes caller-supplied paths).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-02 can import `generate_episodes`, `assert_clean`, `HELD_ONLY_GATE_SCOPE`, `dropped_free_tools()`, `held_out_sha256()`, `write_episodes` / `read_episodes` / `file_sha256` and extend the generator (dh/ds, synthetic tools, dev splits, manifest, CLI); `tests/test_training_leakage.py` already covers whatever it emits.
- Plan 02-03 can stamp checkpoints with `FORMAT_VERSION`, `VOCAB_SHA256`, `render_fingerprint()`, `CONTEXT_CHARS`, `GEN_RESERVE` and assert `GPTConfig().context == CONTEXT_CHARS` under torch.
- Full WSL suite: 766 passed (baseline 707 plus 59 new), no regression.

## Self-Check: PASSED

- Files: all 8 files of the plan present on disk.
- Commits: `e0bf369`, `ef08e11`, `2741b9d` are ancestors of HEAD; `git rev-list --count` from `plan_head_before` is 3.
- Acceptance criteria of all three tasks re-run and passing; `git diff --stat 494285f..HEAD -- src/dashboard/app.py` prints nothing.
- Plan verification: targeted files 91 passed; full suite 766 passed.

---
*Phase: 02-local-data-foundation-scratchgpt*
*Completed: 2026-10-21*
