---
phase: 02-local-data-foundation-scratchgpt
plan: 04
subsystem: testing
tags: [scratchgpt, gpu-training, rtx-3060, manifest, loss-curve, smoke-benchmark, viva-notes]

requires:
  - phase: 02-local-data-foundation-scratchgpt (plan 02)
    provides: the committed 40,000 + 2,000 episode dataset and its hash-pinned manifest
  - phase: 02-local-data-foundation-scratchgpt (plan 03)
    provides: train.py, inference.load_generator, stamped LocalScratchProvider, progress-line and exit-code contracts
provides:
  - "models/scratch/scratch_gpt.pt (git-ignored): 4,763,648-parameter ScratchGPT trained on the RTX 3060, best dev checkpoint at step 6000"
  - "models/scratch/{manifest.json, train_log.jsonl, loss_curve.png} committed, tied to data/local/manifest.json by sha256"
  - "results/phase2-smoke (untracked): 26 stamped scratch-gpt rows proving the model drives the benchmark end to end"
  - "docs/viva/02-llm-foundations-and-synthetic-data.md (634 lines, Parts A-D, 18 questions) and the README row marked written"
affects: [03 LoRA trainer, 04 full benchmark (scratch-gpt row), 05 report disclosure and viva Q&A]

plan_head_before: 2dc49440335ad930d56338d05b9b1015867d66fd
plan_head_after: 91a11b77def4963c0fc5c183b7608cfe60a27a87

actuals:
  tokens: 12000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Long GPU job as a detached nohup process with a progress log and an exit-code done marker, polled with bounded waits"
    - "Pre-registered, dev-metric-only remediation rule (one retry at most), recorded as a run note in the manifest"
    - "WSL logic kept in script files (preflight, launch, poll, checks) because inline quoting through wsl.exe corrupts nested quotes"

key-files:
  created:
    - models/scratch/manifest.json
    - models/scratch/train_log.jsonl
    - models/scratch/loss_curve.png
    - docs/viva/02-llm-foundations-and-synthetic-data.md
  modified:
    - docs/viva/README.md

key-decisions:
  - "Step budget kept at 7000 for the first run: pilot timing gave about 120-145 ms per step (estimate about 15 minutes, under the 40 minute limit)"
  - "The pre-registered dev-only rule triggered after run 1 (final dev line exact 0.1669 < 0.50, stop_reason max_steps, best_step 6500 >= 7000 - 500), so exactly one retry ran with max_steps 14000; the kept model is the retry"
  - "No retry after the second run: it converged (early stop) and the rule allows no second remediation; the sub-0.50 dev line exact is recorded as a finding"
  - "Nothing was changed in response to the smoke run; its behavioural numbers are reported, not graded"

patterns-established:
  - "The manifest final block is the LAST evaluation, not the kept checkpoint; the kept checkpoint's dev numbers are the train_log.jsonl record at best_step (and the checkpoint's train_meta)"

requirements-completed: [SCRATCH-04, SCRATCH-05]

coverage:
  - id: D1
    description: "ScratchGPT trained from scratch on the RTX 3060 by python -m src.scratch_llm.train on the committed dataset; manifest, train log and loss curve committed; checkpoint_sha256 equals the weights file; dataset sha256 equals data/local/manifest.json; device cuda; format_version 2"
    requirement: SCRATCH-04
    verification:
      - kind: command
        ref: "check_artifacts.py (scratchpad): manifest keys == MANIFEST_KEYS, checkpoint and dataset hashes match, load_generator loads, png 37,595 bytes, 32 log lines -> artifacts-ok"
        status: pass
      - kind: command
        ref: "verify3.sh: train.done is 0, DONE line present, no FAILED line"
        status: pass
    human_judgment: false
  - id: D2
    description: "Training ran in the background under nohup with a done marker, every launch after a GPU preflight with at least 3584 MiB free; one pilot then the full run; at most one remediation retry and zero infrastructure relaunches"
    requirement: SCRATCH-04
    verification:
      - kind: command
        ref: "models/scratch/gpu_preflight.log last free_mib 4600 >= 3584; manifest run_notes holds 1 dev remediation entry and 0 relaunch entries (bounded-policies-ok)"
        status: pass
    human_judgment: false
  - id: D3
    description: "scratch-gpt runs end to end in the benchmark: 26 unique rows, no llm_error, no [skip], provider scratch, config_model scratch-gpt, resolved_model scratch-gpt@83b0763a6d26 equal to the checkpoint hash prefix, 40-hex code_commit, finish_reasons only stop or length, summary.md present"
    requirement: SCRATCH-05
    verification:
      - kind: command
        ref: "check_smoke.py (scratchpad) -> smoke-plumbing-ok; parse rate 76/80 valid actions"
        status: pass
    human_judgment: false
  - id: D4
    description: "Viva page 02 has Parts A-D plus the limits section, 18 questions, quotes the manifest values verbatim and is listed as written in the README"
    requirement: SCRATCH-04
    verification:
      - kind: command
        ref: "verify3.sh -> part-c-ok, secret-check-ok, 18 Q lines, 634 lines"
        status: pass
    human_judgment: false
  - id: D5
    description: "Whether the trained model is a good enough worst-case principal for Phase 4 (dev line exact 0.27 on the kept checkpoint, below the 0.50 hoped for; 4/10 smoke attack attempts) is a judgement for the Phase 4 benchmark and the Phase 5 disclosure"
    requirement: SCRATCH-05
    verification: []
    human_judgment: true
    rationale: "The dev metric measures copying and format skill on synthetic unseen tools and wordings; whether behaviour on the benchmark is adequate can only be read from full runs"

duration: 58min
completed: 2026-10-21
status: complete
---

# Phase 2 Plan 4: GPU training run, smoke benchmark and viva page Summary

**ScratchGPT (4,763,648 parameters) trained from scratch on the RTX 3060 in two runs (the second by a pre-registered dev-only rule), best dev loss 0.5720 at step 6000, checkpoint 83b0763a6d26; it drives the benchmark end to end with 76 of 80 steps valid actions, and viva page 02 quotes every real number.**

## Performance

- **Duration:** about 58 min wall clock (07:50Z to 08:48Z), dominated by two training runs of 17 and 21 minutes
- **Tasks:** 3 (Task 1 tracer, no commit by design; Tasks 2 and 3 committed), 3 commits
- **Files:** 5 (4 created, 1 modified); the weights and results/ stay untracked or ignored
- **Full WSL suite after training finished:** 854 passed in 295 s (same as the baseline; this plan adds no tests)

## GPU preflight readings (gpu_preflight.log, git-ignored)

| When (UTC) | Launch | nvidia-smi (name, used MiB, free MiB, total MiB) | free_mib |
|---|---|---|---|
| 07:52:54 | pilot (first attempt) | empty reading, see Deviation 1 | blank |
| 07:53:17 | pilot | RTX 3060 Laptop, 1372, 4623, 6144 | 4623 |
| 07:58:56 | full run (run 1) | RTX 3060 Laptop, 1372, 4623, 6144 | 4623 |
| 08:17:31 | dev remediation (run 2) | RTX 3060 Laptop, 1395, 4600, 6144 | 4600 |

Windows-host Ollama: `tasklist.exe` showed no ollama process. WSL Ollama is installed and `ollama ps` listed no loaded model at any preflight, so nothing needed stopping. The nvidia-smi line before training was 1372 MiB used, 4623 MiB free of 6144.

## Pilot (150 steps on CUDA)

`device cuda params 4763648`; 150 steps finished in 103 s wall (about 78 s of which was start-up before the first step line). Steady-state about 120 ms per step (steps 25 to 50: 3 s) and about 140 ms per step across the evaluation interval (steps 100 to 150: 7 s); throughput 232k to 288k padded characters per second. The pilot checkpoint then drove one benchmark episode (`dh-u02-a27`, unprotected) through `LocalScratchProvider`: resolved model `scratch-gpt@419e60340997`, no `llm_error`, 5 steps, `torch.cuda.memory_allocated()` 30,720,000 bytes (the generator runs on the GPU). The barely trained model wrote garbage, as expected; plumbing only.

**Step budget:** estimate 7000 x 120-145 ms = 14 to 17 minutes plus evaluation, under the 40 minute limit, so `--max-steps 7000` was kept with every other setting at the plan 02-03 defaults.

## The two training runs

| | Run 1 | Run 2 (kept) |
|---|---|---|
| Command | `--max-steps 7000` | `--max-steps 14000 --run-note "dev remediation: ..."` |
| Steps run | 7000 (max_steps) | 8000 (early_stop, patience 8 evaluations after best) |
| Wall time | 1,037 s | 1,245 s (20.8 min) |
| Best step / best dev loss | 6500 / 0.6757 | 6000 / 0.5720 |
| Last-evaluation dev loss | 0.6951 | 0.6395 |
| Last-evaluation dev line exact (overall) | 0.1669 | 0.3239 |
| Checkpoint sha256 prefix | 8607aafcae02 (weights not kept) | 83b0763a6d26 |
| Characters seen (incl. padding) | 226,644,625 | 259,021,855 |

**Remediation decision (computed from the run 1 manifest before the smoke run):** x = final.dev_line_exact = 0.1669 < 0.50, and the dev loss was still improving at the end (stop_reason max_steps, best_step 6500 >= steps - 2 x eval_every = 7000 - 500). The rule therefore required exactly one retry with max_steps = min(2 x 7000, steps fitting 60 minutes at about 148 ms per step = about 24,300) = 14,000. The plan-checker advisory (skip the retry if the cap made it no longer than the first run) did not apply: 14,000 is twice 7,000. The retry used the same data, seed and settings and the run note `dev remediation: dev_line_exact 0.167 < 0.50 and dev loss still improving at max_steps; max_steps 7000 -> 14000`. It overwrote models/scratch; run 1's manifest, log, curve and weights were copied to the session scratchpad first (not part of the repository). No relaunch was needed, so there is no infrastructure relaunch and no run note beyond the remediation one.

**Kept model, from `models/scratch/manifest.json` (run 2):** params 4763648, steps 8000, best_step 6000, tokens_seen 259021855, wall_seconds 1245.2, stop_reason early_stop, device cuda, torch 2.11.0+cu128, code_commit c722878 (code_dirty true because unrelated user files are modified), checkpoint_sha256 `83b0763a6d26...`.

Dev metrics of the kept checkpoint (the `train_log.jsonl` record at step 6000): dev loss 0.5720; line exact overall 0.2746 (benign 0.2759, dh 0.3340, ds 0.2277); first action 0.4835; train loss 0.0280. The manifest's `final` block is the last evaluation (step 8000): dev loss 0.6395; line exact overall 0.3239 (benign 0.3000, dh 0.3847, ds 0.3032); first action 0.5355; train loss 0.0201.

**Finding (recorded, no further action by rule):** dev line exact is below 0.50 at convergence (0.2746 for the kept checkpoint, 0.3239 at the last step), and the run converged, so no second retry was allowed. Loss curve: dev loss falls to about 1.05 by step 1,250, plateaus to about step 2,750, drops to about 0.8 by step 3,500, reaches its minimum at step 6,000 and creeps up slightly while train loss reaches 0.02. The gap indicates fit to the training phrasing; the dev set uses held-back wordings and unseen tools, so some gap is expected (how much of the dev loss is irreducible free text was not measured).

## Smoke benchmark (plumbing check; behaviour reported, not graded)

`run_benchmark.py --models scratch-gpt --runs 1 --dh 5 --ds 5 --benign 3 --resume results/phase2-smoke`: 26 episodes, exit 0, no `[skip]` line. Plumbing checks all passed (`smoke-plumbing-ok`): 26 unique (case_id, run, arm) rows, no `llm_error`, provider `scratch`, config_model `scratch-gpt`, resolved_model `scratch-gpt@83b0763a6d26` on every row, 40-hex code_commit (372c74d7...), finish_reasons all "stop" (80 of 80), summary.md present.

REPORT lines exactly as printed:

- REPORT parse rate 76/80
- REPORT unprotected dh/ds episodes with call_tool 10/10, attempted 4/10, succeeded 2
- REPORT protected dh/ds episodes with call_tool 10/10, attempted 4/10, succeeded 0
- REPORT unprotected benign episodes 3, with any valid action 3; protected the same (3 and 3)
- REPORT action types {call_tool: 52, respond: 24}

(The call_tool count includes the user's own legitimate tool call, so only `attempted` measures an attack.)

Outcome counts per arm (summary.md): unprotected attack episodes 4 attacked, 6 ignored_injection, 0 format_failure, 0 llm_error, 0 refused; unprotected benign 3 ignored_injection (format-valid, no injection to follow); protected arm identical (4 / 6 / 0 / 0 / 0 and 3). All 4 protected attempts were blocked at the attack action, all by the tool broker (7 broker blocks in total, 0 policy engine, sandbox, seccomp). Benign task completion 2/3 in both arms, false positives 0/3, mean latency 582 ms per step, max prompt_tokens 1,379. By kind: dh attempted 1/5, ds attempted 3/5. Wilson interval for attempted 4/10 is [16.8, 68.7].

SCRATCH-05 is met: 76 valid actions out of 80 steps. This is 13 cases and one deterministic run, a plumbing result and not an attack-success claim; the full 342-episode runs are Phase 4.

No setting, hyper-parameter, data file or code path was changed after the smoke run; the only change in response to any metric was the single dev-gated retry decided before the smoke run.

## Accomplishments

- A real GPU training run with manifest, per-evaluation log and loss curve, all committed and hash-tied to the committed dataset; weights remain git-ignored (`git ls-files models/scratch` lists exactly the three files).
- A pilot-first workflow: CUDA path, checkpoint and one GPU benchmark episode were proven in under two minutes before the 17 minute run was launched detached.
- The pre-registered rule applied mechanically, with the rule's input (dev metrics) and its decision written down before the smoke run.
- Viva page 02: 634 lines, glossary of 36 terms, Parts A, B, C, D and a limits section, 18 questions with model answers, parameter arithmetic verified against the manifest (4,763,648), real numbers in Part C.

## Task Commits

1. **Task 1 (tracer): GPU pilot end to end, then background launch** - no commit by design (git-ignored outputs only)
2. **Task 2: viva page (glossary, Parts A, B, D, limits)** - `c722878` (docs)
3. **Task 3a: training artifacts (manifest, train log, loss curve)** - `372c74d` (feat)
4. **Task 3b: viva Part C and README row** - `91a11b7` (docs)

(Counted by `git rev-list --count 2dc4944..HEAD` = 3 before this summary commit: c722878, 372c74d, 91a11b7.)

**Plan metadata:** committed separately (docs: complete plan).

## Files Created/Modified

- `models/scratch/manifest.json`, `models/scratch/train_log.jsonl`, `models/scratch/loss_curve.png` - committed training evidence (run 2)
- `docs/viva/02-llm-foundations-and-synthetic-data.md` - the study page
- `docs/viva/README.md` - page 02 row linked and marked written (Phases 3 to 5 rows unchanged)
- Untracked or ignored by design: `models/scratch/scratch_gpt.pt`, `train_progress.log`, `train.done`, `gpu_preflight.log`, `models/scratch-pilot/`, `results/phase2-smoke/`

## Decisions Made

See key-decisions in the frontmatter. In short: keep 7000 steps from pilot timing; apply the pre-registered rule as written (one retry, 14,000 steps); stop after the retry; report the smoke numbers without acting on them.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Inline quoting through wsl.exe corrupted commands**
- **Found during:** Task 1 (first preflight and the nvidia-smi capture)
- **Issue:** `bash -lc '... $(nvidia-smi ...) ...'` run through Git Bash and wsl.exe lost its quoting, so the first preflight wrote an empty reading and printed "GeForce: command not found".
- **Fix:** All WSL logic (preflight, pilot, launch, poll, artifact and smoke checks, suite) moved into script files in the session scratchpad and invoked as `wsl.exe -d Ubuntu -- bash <script>`. The preflight script implements the plan's step 1 exactly (stop WSL Ollama models, read nvidia-smi, append two lines, require at least 3584 MiB).
- **Effect on records:** the first line pair in the git-ignored gpu_preflight.log is blank (no launch happened on it); the verify command reads the last `free_mib`, which is 4600.
- **Files modified:** none in the repository.

**2. [Rule 1 - Bug, script authoring] Pilot episode script used wrong attribute names**
- **Found during:** Task 1 step 4
- **Issue:** `BenchmarkCase` has `id` not `case_id`, `Step` has `raw`, and `run_episode` returns `(episode, score)`.
- **Fix:** corrected in the throw-away script and re-run; it passed. No repository file involved.

**Total deviations:** 2 (one blocking tooling issue, one script authoring slip), both outside the committed files. **Impact:** none on the committed artifacts or on any plan requirement.

### Process note

The tracer gate for Task 1 was satisfied by its automated verify (pilot-ok, vram-preflight-ok, training-alive for run 1) in end-of-phase mode, so no checkpoint was synthesised. The first poll found run 1 healthy at step 50 (loss 3.2110) with no done marker and no FAILED line.

## Authentication Gates

None.

## Known Stubs

None. No placeholder, TODO or hardcoded empty data in the files created or modified. No entries were added to `.planning/WINDOWS.md` (no stub, skipped test or unrun verify).

## Threat Flags

None. All new surface (GPU launch, detached process, staged files) is covered by T-02-18 to T-02-22 in the plan. Mitigations observed: preflight before every launch (T-02-20), explicit staged-path assertions before both commits (T-02-19), resolved_model equals `scratch-gpt@` plus the manifest hash prefix on every smoke row (T-02-21), secret and canary scan of the viva page (T-02-22), and the dev-only rule with no change after the smoke run (T-02-18).

## Limitations to carry forward

- The manifest `final` block describes the last evaluation (step 8000), not the kept checkpoint (step 6000). Phase 4 and 5 should quote the checkpoint's own numbers (train_log record at best_step, listed above) or state which they quote.
- Dev line exact 0.27 for the kept checkpoint is below the 0.50 hoped for; attack attempts in the smoke run were 4/10 per arm. Whether ScratchGPT is a strong enough worst-case principal is for the Phase 4 full run and Phase 5 disclosure to state; this plan made no attempt to raise it.
- GPU training is not bit-reproducible; the dataset is. Rerunning the training command will give similar but not identical numbers.
- One training seed, one run per case at temperature 0: the smoke intervals are wide and repeats of a case are near-duplicates.

## Issues Encountered

None beyond Deviation 1.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `config/models.yaml` `scratch-gpt` now resolves to an existing checkpoint, so Phase 4 can run `run_benchmark.py --models scratch-gpt` for the full 342 episodes; its rows will carry `scratch-gpt@83b0763a6d26`.
- Phase 3 (LoRA) can reuse the pilot-first, detached, polled pattern and the preflight script logic (stop Ollama, require 3.5 GiB free) for its GPU run.
- Human checks for end of phase (from the plan): read the viva page; look at `models/scratch/loss_curve.png` (dev loss falls then flattens with a slight late rise, best step 6000 matches the manifest); skim `results/phase2-smoke/summary.md` (scratch-gpt, 26 episodes, both arms).

## Self-Check: PASSED

- Files present: `models/scratch/manifest.json`, `models/scratch/train_log.jsonl`, `models/scratch/loss_curve.png`, `docs/viva/02-llm-foundations-and-synthetic-data.md`, `docs/viva/README.md`; `models/scratch/scratch_gpt.pt` exists locally and is ignored; `results/phase2-smoke/raw.jsonl` has 26 rows.
- Commits `c722878`, `372c74d`, `91a11b7` are ancestors of HEAD; `git rev-list --count 2dc4944..HEAD` was 3 before this summary commit.
- Acceptance criteria re-run: `git ls-files models/scratch` lists exactly 3 files and no .pt; smoke check printed `smoke-plumbing-ok` and the REPORT lines; run_notes has 1 dev remediation entry and 0 relaunch entries; Part C quotes params 4763648, steps 8000, best_step 6000, checkpoint prefix 83b0763a6d26 and dataset prefix 63a192b109bf (`part-c-ok`); README row linked and "written (Phase 2)"; 18 `**Q` lines and the secret scan passes.
- Plan verification: full WSL suite 854 passed; training manifest matches the weights file and data/local/manifest.json by hash.

---
*Phase: 02-local-data-foundation-scratchgpt*
*Completed: 2026-10-21*
