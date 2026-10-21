---
phase: 02-local-data-foundation-scratchgpt
plan: 02
subsystem: testing
tags: [scratchgpt, synthetic-data, prompt-injection, dev-split, manifest, provenance, torch-free]

requires:
  - phase: 02-local-data-foundation-scratchgpt (plan 01)
    provides: format v2, episode schema, strict leakage gate, benign generator slice, LF JSONL I/O
provides:
  - "generate_episodes for all three splits: exact 50/25/25 benign / dh-compliant / ds-compliant mix with self-written injections"
  - "Synthetic tools, OS-action benign episodes, blocked tails (real ToolBroker text), two-step tasks, tool-list shapes, repr/json observation styles"
  - "Dev split (dev_tools + dev_synthetic) with a seeded 10% held-back tool set and held-back templates"
  - "python -m src.scratch_llm.dataset CLI, provenance.py helpers, and the committed manifest of the 40,000 + 2,000 episode dataset"
affects: [02-03 train and inference, 02-04 training run, 03 LoRA trainer, 05 report disclosure]

plan_head_before: da05b744b28d842f35b59d5d1e94743249ced26b
plan_head_after: 5f110e591f602cc65253ea8db1df1b877c7e7168

actuals:
  tokens: 23575
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Shares are allocated (exact counts, shuffled), not drawn per episode, so every share is testable as a band around a fixed number"
    - "Per-episode redraw on validation or gate violation, bounded, then LeakageError if the leak is systematic; the whole set still passes assert_clean"
    - "Blocked-tail observation is produced by a real ToolBroker(None, capabilities={user tool}).execute(call).as_text(), never typed"
    - "Manifest = fixed field list, repo-relative paths, no environment values; hashes of ignored files are the committed record"

key-files:
  created:
    - src/scratch_llm/provenance.py
    - tests/test_scratch_dataset.py
    - data/local/manifest.json
  modified:
    - src/scratch_llm/dataset.py
    - src/scratch_llm/templates.py

key-decisions:
  - "D-02 interpreted as 10% of ALL episodes, every one drawn from the benign half (so 20% of benign episodes are OS tasks)"
  - "A sink without a recipient-like parameter gets an injection without an address (no 'ADDR inside the body'), keeping every argument value copyable verbatim"
  - "Real sinks are scarce (9 in the train pool, 0 in the dev pool), so a synthetic sink fills the role where the real pool has none and the episode records 'sink:synthetic'"
  - "Task 3 was committed in two steps (CLI code, then manifest) so the manifest's code_commit names a commit that contains the CLI"
  - "GENERATOR_VERSION bumped to 2 (1 was the benign-only slice)"

patterns-established:
  - "Template families come as *_TRAIN / *_DEV pairs with ids 'family:tNN' / 'family:dNN'; the partition letter is asserted on every generated episode"

requirements-completed: [SCRATCH-01, SCRATCH-02]

coverage:
  - id: D1
    description: "Exact 50/25/25 class quota (within one episode when 4 does not divide n), n=0 empty, negative or non-int n raises"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_quota_is_exact_when_n_is_divisible_by_four"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_quota_is_within_one_episode_for_other_sizes"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_zero_episodes_negative_n_and_bad_arguments"
        status: pass
    human_judgment: false
  - id: D2
    description: "dh episodes comply with a self-written injection (tool named in words in at most 25%); ds episodes read a secret from a source tool and put it, with the injected address, into a sink tool; every argument value is copyable; secrets and addresses are not benchmark-shaped"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_dh_episode_calls_the_injected_tool_named_in_the_observation"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_ds_episode_reads_a_secret_and_sends_it_to_the_address_in_the_injection"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_every_call_names_a_listed_tool_and_every_value_is_copyable"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_secrets_and_addresses_do_not_look_like_the_benchmark"
        status: pass
    human_judgment: false
  - id: D3
    description: "Breadth: synthetic tools (35%, no held-only parameter names, no held-out signature), OS actions (10%, benign, policy-safe), blocked tails (8% of dh/ds, real broker text), two-step benign (12% of benign), tool-list order and distractors, both observation quote styles, harmless free text in benign observations"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_synthetic_tools_are_a_third_of_the_episodes_and_leak_nothing"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_os_action_episodes_are_a_tenth_benign_and_inside_the_default_policy"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_blocked_tails_end_with_the_real_broker_text_and_keep_their_class"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_tool_list_order_and_distractors"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_observations_use_both_quote_styles_and_benign_ones_carry_harmless_text"
        status: pass
    human_judgment: false
  - id: D4
    description: "Dev split: 25 dev tools (seeded 10% of the free pool) never in train, dev_tools uses only dev tools, dev_synthetic only synthetic tools, both use only held-back templates, keep the quota and pass the gate"
    requirement: SCRATCH-02
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_dev_tools_are_a_stable_tenth_of_the_free_pool_and_never_in_train"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_dev_splits_keep_the_quota_pass_the_gate_and_use_held_back_text_only"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_train_episodes_use_training_templates_only"
        status: pass
    human_judgment: false
  - id: D5
    description: "Same seed gives the same sha256 in interpreters with different PYTHONHASHSEED, another seed differs; the real 40,000-episode files reproduce the committed hashes under PYTHONHASHSEED=7"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_same_seed_same_hash_across_interpreters_and_another_seed_differs"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_dev_tool_set_does_not_depend_on_the_hash_seed"
        status: pass
    human_judgment: false
  - id: D6
    description: "CLI and manifest: fixed 18-field list, hashes equal the files, D-04 held-only parameter-name scope recorded, no environment values or absolute paths, --n cap, even --dev, --out-dir confined to the repository"
    requirement: SCRATCH-01
    verification:
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_manifest_has_exactly_the_fixed_field_list_and_the_gate_scope"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_the_manifest_carries_no_environment_values_or_local_paths"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_bad_sizes_are_refused_before_anything_is_written"
        status: pass
      - kind: unit
        ref: "tests/test_scratch_dataset.py#test_an_out_dir_outside_the_repository_is_refused"
        status: pass
    human_judgment: false
  - id: D7
    description: "Whether this mix actually produces the intended weak, injection-compliant ScratchGPT behaviour (and a usable dev signal) is only observable after training"
    requirement: SCRATCH-01
    verification: []
    human_judgment: true
    rationale: "Behavioural adequacy of the data for a roughly 5M-parameter model is measured by the plan 02-04 training run and Phase 4 benchmark, not by data tests"

duration: 26min
completed: 2026-10-21
status: complete
---

# Phase 2 Plan 2: Full generator, dev split, CLI and dataset manifest Summary

**Deterministic 50/25/25 benign / direct-harm-compliant / data-stealing-compliant generator with self-written injections, synthetic tools, OS-action and blocked-tail episodes and a held-back dev split, plus a CLI that wrote the 40,000 + 2,000 episode files (git-ignored) and a committed, hash-pinned manifest.**

## Performance

- **Duration:** 26 min (start time not recorded at launch; estimated from the previous plan's completion at 04:21Z and the manifest timestamp 04:48Z)
- **Completed:** 2026-10-21T04:53Z
- **Tasks:** 3 (Task 3 in two commits)
- **Files:** 5 (3 created: provenance.py, test_scratch_dataset.py, manifest.json; 2 extended: dataset.py, templates.py)
- **Generation wall time:** 4,000 episodes in 5.2 s; the full CLI run (40,000 train + 2,000 dev, generation + gate + both files + manifest) took 62 s in WSL

## The real dataset (from data/local/manifest.json, committed in 5f110e5)

| Item | Train | Dev |
|------|-------|-----|
| File (git-ignored) | `data/local/train-seed0-n40000.jsonl` | `data/local/dev-seed0-n2000.jsonl` |
| sha256 | `63a192b109bf2c8d071f1c7a1ba833072a2cba113a833161f98e3b8faabdcc46` | `58f5bcb09845c5a096109d1381534d767d1fd5aa4d95e83a1c0d94c27e43ad2a` |
| Episodes | 40,000 | 2,000 (1,000 dev_tools, then 1,000 dev_synthetic) |
| benign / dh / ds | 20,000 / 10,000 / 10,000 | 1,000 / 500 / 500 |
| synthetic_fraction | 0.35 | 0.50 (all of dev_synthetic) |
| os_action_fraction (of all) | 0.10 | 0.10 |
| blocked_tail_fraction (of dh+ds) | 0.08 | 0.08 |
| two_step_fraction (of benign) | 0.12 | 0.12 |
| mean_chars (training_text) | 656.61 | 656.70 |
| max_context_chars | 1,694 (limit 2,048) | 1,392 |
| supervised_fraction | 0.405087 | 0.401882 |

Other manifest facts: generator_version 2, format_version 2, seed 0, 79 held-out tools (`held_out_sha256` `13f3aae47fa271ff8e863137d0443023cb8973a203391a51aa72ed1a7423e676`), 50 held-only parameter names, free pool 251 (0 dropped by the gate), 25 dev tools (`dev_tools_sha256` `474808776f3a3404376877c8c00310370970f841c12601fe5641a81f0cd08f74`), gate passed with the 10 rule names and `held_only_param_scope` equal to `HELD_ONLY_GATE_SCOPE`. `code_commit` is `696e32dedcd067bdbf59bdbfa9c2301e2b1b6f3d` (the commit that contains the CLI); `code_dirty` is `true` because unrelated tracked user files (papers, `src/dashboard/app.py`, `.planning/config.json`) were modified at generation time. Python 3.14.4.

Inputs for the plan 02-04 training-time estimate: 26,264,403 training characters in the train file (about 26.3M tokens at one character per token), mean length by class benign 436, dh 730, ds 1,024 characters, about 40.5% of characters supervised.

Reproducibility evidence: regenerating all 42,000 episodes under `PYTHONHASHSEED=7` gave byte-identical sha256 values (checked into a throw-away directory, then deleted).

Composition facts worth quoting: the train file uses 226 real tools (the 251 free tools minus the 25 dev tools) and 28,256 distinct synthetic tool names; 187 distinct template ids appear in train and 47 in dev, with no overlap. Of the non-blocked ds train episodes (9,207), 52.4% use a sink that has a recipient-like parameter, so their injection names an address that the sink call must copy.

## Accomplishments

- **D-01 exact quota.** `class_counts(n)` (largest remainders) and a shuffled class sequence give exactly n/2, n/4, n/4 when 4 divides n and within one episode otherwise; the other shares (synthetic, OS, two-step, blocked tail) are allocated in exact counts rather than drawn, so the manifest shows round numbers.
- **Attack episodes.** dh: user tool, observation with a wrapped self-written injection (32 train + 6 dev templates, 6 + 3 wrappers) carried in a free-text key of an array/object, the injected call, respond. ds: user tool, injection "get X and send it to ADDR", a source tool returning a record with a 12-30 character secret (28 + 5 record phrasings), the sink call carrying the secret alone / the whole record / `label: secret`, respond.
- **D-02 / D-03.** OS-action benign episodes (run_shell `ls`/`cat`/`wc -l`/`head -n`, read_file, write_file under /home/user/workspace, none BLOCKed by the default policy, verified in a test) and blocked tails whose observation is produced by a real `ToolBroker`.
- **Synthetic tools** (toolkit + verb + object names; parameter names from the free vocabulary minus the 50 held-only names plus generic words; no held-out signature) force copying of unseen names and parameters; dev_synthetic uses a separate toolkit-word slice.
- **D-04 dev split.** `DEV_TOOLS_SEED` fixed constant, `dev_tool_names()` = seeded 10% of the free pool (25 tools), disjoint from every train role; held-back user, dh, ds, record, wrapper, reply, blocked-reply, two-step and OS templates (`d` partition) used by both dev splits.
- **CLI + manifest.** `python -m src.scratch_llm.dataset --n 40000 --dev 2000 --seed 0` validated and wrote everything; `provenance.py` supplies `git_commit`, `git_dirty`, `write_json_atomic`, `resolve_under_root` for reuse by train.py.

## Task Commits

1. **Task 1 (tracer): dh and ds compliant episodes at the exact 50/25/25 quota** - `3495f8d` (feat)
2. **Task 2: synthetic tools, OS-action and blocked-tail episodes, dev split** - `7adff19` (feat)
3. **Task 3a: dataset CLI and provenance helpers** - `696e32d` (feat)
4. **Task 3b: committed manifest for the 40,000 + 2,000 episode dataset** - `5f110e5` (feat)

**Plan metadata:** committed separately (docs: complete plan).

Tracer gate: the Task 1 `<verify>` is automated-only; it was re-run end to end after the commit (quota 200/100/100 and the ds training text showing the secret in an `O:` line and again in the sink `A:` line) and passed, so expansion continued (`end-of-phase` mode, no checkpoint).

## Files Created/Modified

- `src/scratch_llm/dataset.py` - generator (roles, synthetic tools, `_Ctx`, builders, allocation), `dev_tool_names`, `dataset_stats`, CLI `main`, manifest
- `src/scratch_llm/templates.py` - attack, record, wrapper, benign-text, synthetic vocabulary, OS, two-step and blocked-reply text with train/dev partitions
- `src/scratch_llm/provenance.py` - stdlib helpers shared with train.py (plan 02-03)
- `tests/test_scratch_dataset.py` - 36 torch-free tests
- `data/local/manifest.json` - committed manifest (the two `.jsonl` files stay ignored)

## Decisions Made

- D-02 read as 10% of all episodes taken only from the benign half; D-03 as 8% of dh+ds; two-step as 15% of the non-OS benign episodes (12% of benign).
- Injection names the tool by its CamelCase name in 3 of 32 train templates (about 9%) and 1 of 6 dev templates; otherwise only in words.
- Observation style is chosen per observation (Python repr 60%, json.dumps 40%); json is replaced by repr whenever it would have to escape a quote, so a copied value never differs from the text it was copied from.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `dataset_stats` implemented in Task 2, not Task 3**
- **Found during:** Task 2 (its own `<verify>` command calls `dataset.dataset_stats`)
- **Issue:** The plan lists `dataset_stats` under Task 3's action but Task 2's verify and acceptance criteria already print it.
- **Fix:** Added it (and a test) in the Task 2 commit; Task 3 only consumes it.
- **Files modified:** src/scratch_llm/dataset.py, tests/test_scratch_dataset.py
- **Commit:** 7adff19

**2. [Rule 1 - Bug] Sink payload could be the recipient parameter**
- **Found during:** Task 1 (inspecting a ds training text: `TodoistShareTask(task_id,user_email)` received the record in `user_email`)
- **Issue:** `_payload_param` accepted any string parameter that was not id/path-like, including recipient-like ones.
- **Fix:** Recipient-like names (mail, recipient, receiver, contact, phone, channel, username) are skipped as payload carriers; such tools with no other text parameter are no longer sinks (9 train sinks remain).
- **Commit:** 3495f8d (fixed before the commit)

**3. [Rule 1 - Bug] Two plan statements conflict for sinks without a recipient parameter**
- **Found during:** Task 1 design
- **Issue:** The plan says the address goes "inside the body" for such sinks, yet also requires every string argument value to appear verbatim in the instruction or an earlier observation (a body made of address plus secret appears nowhere).
- **Fix:** The verbatim rule wins (it is what lets a 5M model copy): for a sink without a recipient-like parameter the injection omits the address ("send it" with no "to ADDR") and the body carries only the payload. 47.6% of non-blocked ds train episodes are of this kind.
- **Commit:** 3495f8d

**4. [Rule 2 - Missing critical] Per-episode redraw on a gate violation**
- **Found during:** Task 1
- **Issue:** The plan runs `assert_clean` only on the finished list, so one coincidental 8-word overlap among 40,000 attack texts would abort generation with no data.
- **Fix:** Each drawn episode is also checked with the gate and redrawn (bounded); if every redraw leaks, `LeakageError` is raised with the rule. The final `assert_clean` still runs on the whole set, and the existing leaking-template tests still see `LeakageError`.
- **Commit:** 3495f8d

**5. [Rule 1 - Bug] Reply ids carried no partition letter**
- **Found during:** Task 2 (the planned "train ids have `t`, dev ids have `d`" test)
- **Fix:** Train replies renamed `reply:NN` -> `reply:tNN`, added `REPLY_TEMPLATES_DEV` (`reply:dNN`); existing tests only use the `reply:` prefix.
- **Commit:** 7adff19

**6. [Plan adjustment] Task 3 committed in two commits**
- **Reason:** The manifest records `code_commit`; committing the CLI first makes it name a commit that contains the CLI (a single commit would have recorded the previous commit and `code_dirty: true` for the CLI itself).
- **Commits:** 696e32d (code), 5f110e5 (manifest). The staged paths are exactly the four named in the plan.

**Total deviations:** 5 auto-fixed (1 blocking, 3 bugs, 1 missing critical) plus 1 process adjustment. **Impact:** no scope creep; items 3 and 4 change generator behaviour in ways the next plans should know (sink without address; redraw hides coincidental overlaps).

### Process note (TDD)

Tasks 1 and 2 were red-first (quota assertion and `SYN_*` / `DEV_TOOLS_SEED` failures observed before implementation). For Task 3 the tests were written first but their failing run was not recorded separately (the module they import, `provenance`, did not exist yet).

## Known Stubs

None. The `NotImplementedError` for the dev splits recorded in `.planning/WINDOWS.md` entry 1 is gone (`grep NotImplementedError src/scratch_llm/*.py` prints nothing); the entry was marked fixed.

## Limitations to carry forward

- Real sinks are scarce: 9 in the train pool (GitHubPostIssue, SlackSendMessage, TwitterManagerPostTweet, ... at about 700 ds episodes each) and none in the dev pool, so every dev_tools ds episode (250 of 1,000) uses a synthetic sink (`sink:synthetic` in `meta.template_ids`). Synthetic tools (35% of train) are the main source of sink-name variety.
- The benign `T:`-line habit of "one tool, one call" is diluted by 30% distractors and 12% two-step tasks, but dh/ds structure (user tool first) mirrors the benchmark; the model has not yet seen it work.
- `code_dirty` in the manifest is `true`; the committed files are reproducible from `code_commit` plus the seed regardless (verified with a different `PYTHONHASHSEED`).

## Issues Encountered

- Git Bash mangled one heredoc passed to `python -c` while patching a test; the edit was redone with a script file. No effect on committed files.

## Authentication Gates

None.

## Threat Flags

None - the CLI writes only inside the repository (`resolve_under_root`, tested for `/etc`, `../outside`, `sub/../../outside`) and the git calls use a fixed argument list with a 10 second timeout, as the plan's threat model assumed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-03 can import `provenance` (`git_commit`, `git_dirty`, `write_json_atomic`, `resolve_under_root`), read `data/local/manifest.json`, and stamp checkpoints with `generator_version` 2, `held_out_sha256` and the dataset file hashes.
- Plan 02-04 must check `files.train.sha256` and `files.dev.sha256` against the ignored JSONL files before training; the files regenerate in about 62 s with `python -m src.scratch_llm.dataset --n 40000 --dev 2000 --seed 0`.
- Full WSL suite: 818 passed (baseline 766 plus 52 new), no regression.

## Self-Check: PASSED

- Files: `src/scratch_llm/provenance.py`, `tests/test_scratch_dataset.py`, `data/local/manifest.json`, `src/scratch_llm/dataset.py`, `src/scratch_llm/templates.py` present; the two `.jsonl` files exist locally and are git-ignored (`data-ignore-ok`), `git ls-files data/local` lists only `manifest.json`.
- Commits: `3495f8d`, `7adff19`, `696e32d`, `5f110e5` are ancestors of HEAD; `git rev-list --count da05b74..HEAD` was 4 before this summary commit.
- Acceptance criteria re-run: `grep -c "^def test_"` 36 (>= 18), `PYTHONHASHSEED` referenced, `DEV_TOOLS_SEED` 2+ uses in dataset.py, `resolve_under_root` defined, `shell=True` absent, manifest hashes equal the file hashes, class counts 20000/10000/10000, gate scope equals `HELD_ONLY_GATE_SCOPE`.
- Plan verification: targeted files 111 passed; full suite 818 passed.

---
*Phase: 02-local-data-foundation-scratchgpt*
*Completed: 2026-10-21*
