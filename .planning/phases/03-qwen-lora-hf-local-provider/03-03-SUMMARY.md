---
phase: 03-qwen-lora-hf-local-provider
plan: 03
subsystem: lora-training
tags: [lora, cli, manifest, resume, fail-closed, gpu-launcher, checks, remediation-rule]

requires:
  - phase: 03-qwen-lora-hf-local-provider
    provides: "train() core loop, evaluate(), HFLocalProvider + verify_adapter, chat render contract (plans 03-01, 03-02)"
  - phase: 02-scratch-llm
    provides: "provenance helpers, the remediation / resume / exit-code patterns of the ScratchGPT trainer"
provides:
  - "src/qwen_lora/train.py: complete trainer CLI (python -m src.qwen_lora.train), MANIFEST_KEYS manifest, final full-dev pass, gen_check through HFLocalProvider for tuned and base, loss curve, --resume, FAILED lines and exit codes 3/4, REMEDIATION_RULE + remediation_needed, LORA04 gate"
  - "src/qwen_lora/run_train.sh: read-only preflight / foreground launch with train.done marker / bounded poll"
  - "src/qwen_lora/checks.py: probe, artifacts, smoke and viva gates for plan 03-04"
affects: [03-04, phase-5-report]

actuals:
  tokens: 31368
  tasks: 3
  commits: 3
plan_head_before: f6b21fdc8345f5135657454b0172c9d594ed8d2b
plan_head_after: f57266fc5b67f54e397fd8af3c4abd23d8b15e54
commits: 3

tech-stack:
  added: []
  patterns:
    - "manifest assembled in MANIFEST_KEYS order and asserted; written once before gen_check and rewritten with gen_check filled"
    - "resume file = trainable tensors + optimizer + RNG + counters, torch.load(weights_only=True), identity refusal naming dotted fields"
    - "failure wrapper removes every loadable artifact; Ctrl-C keeps resume file and in-run adapter"
    - "check commands return (problems, ok_line); main prints FAIL lines and exits 1"

key-files:
  created:
    - src/qwen_lora/run_train.sh
    - src/qwen_lora/checks.py
    - tests/test_qwen_lora_checks.py
  modified:
    - src/qwen_lora/train.py
    - tests/test_qwen_lora_train.py
    - .gitignore

key-decisions:
  - "gen_check and final_dev samples are stratified by (class, split) with fixed seeds (seed+2, seed+3); n = 0 means the whole dev file"
  - "MANIFEST_SCHEMA stays 1 as in the plan interfaces"
  - "poll treats a still-waiting launcher process as alive, so a preflight wait is never reported as dead"
  - "A CUDA OOM inside gen_check (returned by the provider as an error value) is re-raised as torch.cuda.OutOfMemoryError so it also exits 4"

requirements-completed: []

coverage:
  - id: D1
    description: "CLI on a tiny model prints the polling lines and writes a manifest with exactly MANIFEST_KEYS, package versions, dataset hashes, LoRA counts, selection block with both pre-registered rules, final_dev_full and gen_check; no environment value"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "tests/test_qwen_lora_train.py#test_cli_*"
        status: pass
    human_judgment: false
  - id: D2
    description: "Kill and --resume gives the same steps, log and adapter tensors as an uninterrupted run; identity / missing-file refusals name the field"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "tests/test_qwen_lora_train.py#test_resume_*"
        status: pass
    human_judgment: false
  - id: D3
    description: "Non-finite loss (train and dev) exits 3, VRAM guard and CUDA OOM exit 4, each leaving nothing loadable; leakage stops before step 0; bad paths and foreign manifests exit 2"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "tests/test_qwen_lora_train.py#test_non_finite_* / test_vram_* / test_cuda_* / test_leak_* / test_cli_refuses_*"
        status: pass
    human_judgment: false
  - id: D4
    description: "remediation_needed and lora04_gate are pure functions of the manifest's dev metrics"
    requirement: LORA-04
    verification:
      - kind: unit
        ref: "tests/test_qwen_lora_train.py#test_remediation_* / test_lora04_gate_*"
        status: pass
    human_judgment: false
  - id: D5
    description: "run_train.sh preflight is read-only and gated, launch writes progress log and done marker (3, 5, resume append), poll reports done-ok / done-failed / dead; checks probe, artifacts, smoke, viva behave per spec"
    requirement: LORA-04
    verification:
      - kind: integration
        ref: "tests/test_qwen_lora_checks.py"
        status: pass
    human_judgment: false

duration: not measured (two sessions)
completed: 2026-10-22
status: complete
---

# Phase 3 Plan 03: Trainer CLI, Resume, Fail-Closed Exits, Launcher and Checks Summary

**The LoRA trainer is now a complete, restart-proof command line that proves its own output by serving the kept adapter through `HFLocalProvider` (tuned and base), carries the pre-registered dev-only remediation and LORA-04 rules as code, and ships a read-only GPU launcher and four mechanical check commands for plan 03-04.**

## Accomplishments

- **Task 1 (tracer).** `python -m src.qwen_lora.train` on the tiny CPU model prints `device`, `step`, `eval step`, `final_dev_full`, `gen_check` and `DONE` lines; the manifest has exactly `MANIFEST_KEYS` in order (packages, tokenizer, render, dataset hashes and class counts, LoRA parameter counts, peak VRAM, selection block with `remediation_rule` and `lora04_gate`, `best`, `last_eval`, `final_dev_full`, `gen_check`, `run_notes` with the D-01 disclosure). The kept adapter is evaluated on the whole dev file, then loaded through `HFLocalProvider` (all stamp checks) and greedily asked for the first action of `gen_check_n` dev prompts, as is the base model. A planted environment variable never reaches the manifest.
- **Task 2.** `qwen_lora.resume.tmp` is written atomically after every evaluation of an unfinished run and read back with `weights_only=True`; the killed-then-resumed tiny run matches the uninterrupted one in steps, best step, tokens, log records (rel 1e-4) and adapter tensors (atol 1e-5). `--resume` refuses naming dotted identity fields (`dev_sha256`, `train.max_steps`, `base_model.revision`, ...) or the missing file. `FAILED non_finite_loss|non_finite_dev_loss` exit 3; `FAILED vram_budget` and `FAILED cuda_oom` exit 4; every failure removes adapter, manifest, in-run adapter and resume file (the launcher's `train_progress.log`, `train.done`, `gpu_preflight.log` are never touched). `remediation_needed` and `lora04_gate` read the manifest only.
- **Task 3.** `run_train.sh` (LF, `bash -n` clean, no kill / ollama stop in any code line) and `checks.py` (`probe`, `artifacts`, `smoke`, `viva`), tested with a fake `nvidia-smi`, a fake `ollama` that records only `ps`, and a stub trainer. `.gitignore` gains `models/qwen-agentlora-pilot/`.

## Final CLI flags and progress lines

Flags exactly as in `<interfaces>`: `--train --dev --out-dir --base-model --seed --max-steps --micro-batch --accum --lr --warmup --eval-every --log-every --patience --n-train-used --dev-eval-n --final-dev-n --gen-check-n --gen-max-new-tokens --device {auto,cpu,cuda} --resume --run-note`. Defaults are the D-05 recipe. Progress-line grammar is unchanged from the plan (`device <d> trainable <n> total <m>`, `step <s>/<max> loss <x> lr <y> tok/s <z> elapsed <t>s vram <alloc>/<reserved>MiB rss <r>MB`, `eval step ...`, `final_dev_full n <n> dev_action_exact <e> dev_loss_action <a>`, `gen_check tuned parse <k>/<n> gold <g>/<n> base parse <k>/<n> gold <g>/<n>`, `DONE best_step ...`, `RESUMED from step <s>`, `FAILED ...`). Exit codes 0 / 2 (argparse) / 3 / 4.

## Helper usage as built

- `bash src/qwen_lora/run_train.sh preflight OUT | launch OUT [train args...] | poll OUT [MAX_WAIT_S]`; env `PYTHON`, `MIN_FREE_MIB` (3584), `MIN_AVAIL_RAM_MIB` (3000), `PREFLIGHT_RECHECKS` (6), `PREFLIGHT_WAIT_S` (300). Done-marker codes 0 / 3 / 4 / 5 / other = crash. `poll` prints the last 3 progress lines, `step lines: N`, `FAILED line: yes|no` and `STATUS running|done-ok|done-failed <code>|dead`.
- `python -m src.qwen_lora.checks probe --model NAME [--adapter-dir DIR] [--kind benign|dh|ds]`, `artifacts --out-dir DIR [--data-manifest FILE] [--require-cuda]`, `smoke --results DIR --model NAME [--expect-rows 26]`, `viva --page PAGE (--manifest FILE [--readme FILE] | --structure-only)`; hidden `--config` (probe, smoke) points at another models file for tests. Ok lines: `probe-ok`, `artifacts-ok`, `smoke-plumbing-ok`, `part-c-ok` / `viva-structure-ok`; failures print `FAIL <cmd>: <reason>` and exit 1.

## Exact REMEDIATION_RULE text (as stored in every manifest)

> Pre-registered before the GPU run, applied at most once, from this manifest's dev metrics only (never from benchmark or smoke outcomes): if final_dev_full.dev_action_exact < 0.90 and the run was still improving when it ended (stop_reason max_steps and best_step >= steps - 2 x eval_every, the last two evaluations as in Phase 2), train once more from scratch with the same data, seed and settings and max_steps doubled; a run that converged (stop_reason early_stop) below 0.90 gets no retry and is recorded as a finding. The 60-step research run already reached 0.899 on 200 dev episodes, so the threshold is reachable.

`LORA04_GATE_RULE`: pass when `gen_check.tuned.parse_rate >= 0.90` on the `gen_check_n` first-turn dev prompts of the kept adapter; a fail blocks LORA-04 and is a verification gap, never a reason to tune or retrain (full text in `src/qwen_lora/train.py`).

## Test runtimes

| Run | Result | Wall |
|-----|--------|------|
| `tests/test_qwen_lora_train.py` alone | 35 passed | 189 s |
| `tests/test_qwen_lora_checks.py` alone | 41 passed | 83 s |
| plan verification call 1: `test_hf_local_provider.py` + `test_qwen_lora_data.py` | 57 passed, 1 skipped (QWEN_LIVE probe) | 82 s |
| plan verification call 2: `test_qwen_lora_train.py` + `test_qwen_lora_checks.py` | 76 passed | 163 s |
| rest of the suite (everything except those four files) | 901 passed | 190 s |
| **whole suite (3 calls)** | **1034 passed, 1 skipped** (baseline 972 + 1, +62 tests: 21 train, 41 checks) | 435 s |

`git status --porcelain models data/local` printed nothing. Shell tests ran under WSL (none skipped).

## Task Commits

1. **Task 1 (tracer): CLI with full manifest, final dev pass, gen_check** - `41aee71` (feat)
2. **Task 2: resumable runs, fail-closed exits, remediation rule** - `430c8c7` (feat)
3. **Task 3: GPU job launcher and check commands** - `f57266f` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] viva value check matched substrings**
- **Found during:** Task 3 (test with `steps` = 4 found inside other numbers)
- **Issue:** a plain `value in page` check lets a short value such as `4` be "quoted" by `Q4` or `40`.
- **Fix:** `_quotes` requires a whole token (not inside a longer number or word).
- **Files modified:** src/qwen_lora/checks.py, tests/test_qwen_lora_checks.py
- **Commit:** f57266f

**2. [Rule 3 - Blocking] Two Plan 03-02 assertions changed by the new behaviour**
- `test_tracer_...` compared `base_model` to an exact two-key dict (now also `model_safetensors_bytes`); `test_selection_keeps_...` counted 4 evaluate calls (the final full-dev pass is a 5th and must see the kept weights, now asserted). **Commit:** 41aee71.

### Interpretation choices (no behaviour change against the plan)

- "First n of the stratified dev order" is implemented as a seeded stratified sample by (class, split) (`_dev_pick`), whole file for n = 0.
- `poll` also treats a live `run_train.sh launch OUT` process as running, so a launcher waiting between preflight re-checks is not reported `dead`.
- `checks.py` reads `config/models.yaml` with `yaml` directly instead of the registry, and the `artifacts` target-module check compares leaf names (PEFT stores full module names).
- A CUDA out-of-memory returned as an error value by the provider during gen_check is re-raised as `torch.cuda.OutOfMemoryError` so it exits 4 too.

**Total deviations:** 1 bug fix, 1 test adaptation, plus the interpretation notes above. No file outside the planned set changed.

## Issues Encountered

A first attempt to append tests with a shell heredoc failed on quoting (nothing was written); the tests were then written with the Write tool. The session was interrupted once by the usage limit between Task 2 and Task 3; state was verified from git before continuing.

## Known Stubs

None.

## Threat Flags

None. T-03-12 to T-03-17 are implemented as planned (weights_only resume load plus identity refusal; resolve_under_root and refuse_foreign_manifest before deletion; read-only preflight with no kill / ollama stop, grep-checked; exit codes 3/4/5 and cleanup; remediation reads dev metrics only; fixed manifest keys and viva secret scan); T-03-SC accepted (no installs).

## Self-Check: PASSED

- Files exist: `src/qwen_lora/train.py`, `src/qwen_lora/run_train.sh`, `src/qwen_lora/checks.py`, `tests/test_qwen_lora_train.py`, `tests/test_qwen_lora_checks.py`.
- Commits `41aee71`, `430c8c7`, `f57266f` are ancestors of HEAD; `git rev-list --count f6b21fd..HEAD` gave 3 before this SUMMARY commit.
- Acceptance greps: `assert tuple(manifest) == MANIFEST_KEYS` 1, `HFLocalProvider` 5, both 0.90 thresholds 1 each, `os.environ[` 0, `weights_only=True` 1, `EXIT_VRAM` 2, `def remediation_needed` 1, train test functions 35 (>= 18), kill / ollama stop in code lines of run_train.sh 0, `train.done` 8, `expandable_segments:True` 1, `.gitignore` pilot rule 1, `git ls-files scripts` unchanged (3 files), `git ls-files --eol` of run_train.sh `i/lf w/lf`.
