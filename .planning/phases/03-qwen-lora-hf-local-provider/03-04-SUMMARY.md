---
phase: 03-qwen-lora-hf-local-provider
plan: 04
subsystem: lora-training-and-benchmark-rows
tags: [lora, qwen, gpu-run, gen-check, smoke-benchmark, viva]

requires:
  - phase: 03-qwen-lora-hf-local-provider
    provides: "HFLocalProvider and config rows (03-01), training core (03-02), trainer CLI, run_train.sh launcher and checks (03-03)"
provides:
  - "models/qwen-agentlora: committed adapter_config.json, manifest.json, train_log.jsonl, loss_curve.png; adapter_model.safetensors git-ignored (sha256 prefix 270501a67998)"
  - "results/phase3-smoke-base and results/phase3-smoke-lora (untracked, 26 stamped rows each)"
  - "docs/viva/03-fine-tuning-lora-and-chat-templates.md (552 lines) and the README row marked written (Phase 3)"
affects: [phase-4-full-runs, phase-5-report]

actuals:
  tokens: 60000
  tasks: 3
  commits: 3
plan_head_before: 58b5131fec12d29831db2061ff47be305b7d1d3d
plan_head_after: c126384248fef2f051fccc7f5883b7e8fa7ac9c6
commits: 3

tech-stack:
  added: []
  patterns:
    - "pilot first (20 steps, background + poll), then the full run, decisions from dev metrics only, smoke runs last"

key-files:
  created:
    - models/qwen-agentlora/adapter_config.json
    - models/qwen-agentlora/manifest.json
    - models/qwen-agentlora/train_log.jsonl
    - models/qwen-agentlora/loss_curve.png
    - docs/viva/03-fine-tuning-lora-and-chat-templates.md
  modified:
    - src/qwen_lora/train.py
    - docs/viva/README.md

key-decisions:
  - "D-05 kept unchanged: pilot estimate 42.5 min (< 75), so 600 steps x 16 episodes, no deviation"
  - "Clarified D-06 retry: not needed (full-dev dev_action_exact 0.9396 >= 0.90); LORA-04 dev gate: pass (tuned parse 1.0000)"

requirements-completed: [LORA-02, LORA-03, LORA-04]

coverage:
  - id: D1
    description: "Qwen2.5-0.5B LoRA fine-tuned on the RTX 3060 within 6 GB; adapter, manifest, log and curve; artifacts-ok with --require-cuda"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "python -m src.qwen_lora.checks artifacts --out-dir models/qwen-agentlora --data-manifest data/local/manifest.json --require-cuda -> artifacts-ok"
        status: pass
    human_judgment: false
  - id: D2
    description: "Un-tuned base control row runs end to end through the benchmark (26 stamped rows)"
    requirement: LORA-02
    verification:
      - kind: integration
        ref: "checks smoke --results results/phase3-smoke-base --model qwen2.5-0.5b-base --expect-rows 26 -> smoke-plumbing-ok"
        status: pass
    human_judgment: false
  - id: D3
    description: "LoRA-tuned row runs end to end with every row stamped by the adapter hash; valid actions evidenced by the pre-registered dev gate"
    requirement: LORA-04
    verification:
      - kind: integration
        ref: "checks smoke --results results/phase3-smoke-lora --model qwen2.5-0.5b-agentlora --expect-rows 26 -> smoke-plumbing-ok; LORA04_GATE pass"
        status: pass
    human_judgment: false
  - id: D4
    description: "Viva page 03 passes part-c-ok, 20 questions, 50 glossary bullets, D-01..D-08 cited; README row written"
    requirement: LORA-03
    verification:
      - kind: integration
        ref: "checks viva --page docs/viva/03-... --manifest models/qwen-agentlora/manifest.json -> part-c-ok"
        status: pass
    human_judgment: true

duration: not measured (several sessions, two usage-limit interruptions)
completed: 2026-10-22
status: complete
---

# Phase 3 Plan 04: GPU Fine-tune, Both Qwen Rows End to End, Viva Page Summary

**Qwen2.5-0.5B-Instruct was LoRA fine-tuned for 600 steps in 42.8 minutes at a 2,317 MiB peak (8,798,208 of 502,830,976 parameters trainable), reached full-dev action-exact 0.9396 and a first-turn parse rate of 1.0000, and both the un-tuned base row and the tuned row ran end to end through the benchmark with provenance stamps; the viva page for Phase 3 is written.**

## Task 1: pilot, step budget, full run

- **Preflight readings** (`gpu_preflight.log`): pilot launch 1 free 4930 MiB GPU / 10264 MiB RAM; pilot relaunch 4882 / 10345; full run 4824 / 10187. Smoke preflights: 4604 / 10228 and 4591 / 10298 (all above the 3584 / 3000 thresholds). `ollama ps` reported "could not connect to ollama server" every time, so no model was loaded; no process was stopped or killed.
- **Pilot, attempt 1 (dead on arrival, fixed):** exit 1 before step 0 with `AssertionError: all-linear resolved to [...seven leaf names]`. Cause: PEFT 0.21.2 stores `all-linear` as the 7 leaf names, while `build_lora_model` also required 7 x 24 stored names (written against the assumption of full module names; the tiny-model tests did not exercise that difference). Fixed in commit `75bedd3` (layer count now taken from modules that carry `lora_A`/`lora_B`; leaves and the 8,798,208 trainable-parameter equality unchanged); `tests/test_qwen_lora_train.py` 35 passed afterwards. No recipe setting was changed.
- **Pilot, attempt 2:** 20 steps, device cuda, wall 140 s, `step 10/20 elapsed 101s`, `step 20/20 elapsed 131s` => 3.0 s/step, 2,531 to 3,202 tok/s, peak 2146 allocated / 2384 reserved MiB, final_dev_full (100 episodes) action exact 0.7782, gen_check tuned parse 5/5. `checks artifacts --require-cuda` printed artifacts-ok; `checks probe --model qwen2.5-0.5b-agentlora --adapter-dir models/qwen-agentlora-pilot` printed probe-ok (resolved_model ended `@d6d64bcd29c0`, 1,005,890,560 bytes CUDA memory in use). Pilot replies were plumbing evidence only.
- **Step budget:** estimate = (3.0 x 600 + 150 + 180 + 300 + 120) / 60 = 42.5 min, below 75, so `--max-steps 600` (D-05) was kept. **No D-05 deviation.** (Actual: 2,567.7 s = 42.8 min.)
- **Full run:** launched once with `run_train.sh launch models/qwen-agentlora --max-steps 600` in a background wsl.exe, polled with `poll ... 360`. Every non-running outcome: only the final `done-ok`. No dead, resume, OOM, non-finite or preflight-5 outcome; no configuration relaunch (counters: resumes 0, dead-starts 0, configuration relaunches 0). `resumed_from_steps` is `[]`; `run_notes` holds only the D-01 disclosure (no "dev remediation", "vram relaunch" or "non-finite relaunch" entry). No tool call exceeded 7 minutes.

## Task 2: artifacts, decisions, smoke runs

### Artifact check (`artifacts-ok`)

```
ARTIFACT steps 600 best_step 600 stop_reason max_steps device cuda
ARTIFACT adapter_sha12 270501a67998 trainable_params 8798208 total_params 502830976
ARTIFACT final_dev_full n 2000 dev_action_exact 0.9396
ARTIFACT gen_check n 100 tuned parse 1.0000 gold 0.9600 base parse 0.6900 gold 0.1500
ARTIFACT peak_vram_mib allocated 2317 reserved 2584 total 6143
```

Wall 2,567.7 s; tokens seen 4,785,995; 9,600 of 40,000 episodes used (4,800 benign / 2,400 dh / 2,400 ds); base revision 7ae557604adf, train sha256 a2586548c7f7, dev 1f6333eb8ac8 (equal to data/local/manifest.json). Packages: transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, torch 2.11.0+cu128. Dev action loss (300-episode subset) at steps 100..600: 0.01503, 0.01042, 0.00773, 0.00765, 0.00751, 0.00700; best step 600, dev_action_exact on the subset 0.9365 (0.9437 at step 400, not the selection metric). Full dev: action loss 0.0068, action exact 0.9396 (benign 0.9896, dh 1.0000, ds 0.8392), first-action exact 0.9875, line exact 0.5811. All-token dev loss rises 0.742 to 0.953 because of respond-message text (0.0070 action loss is the selection metric, D-06); curve viewed: action loss falls and flattens, best step 600 marked.

### D-06 remediation decision (decided before the smoke runs)

`REMEDIATION needed no: final_dev_full.dev_action_exact 0.9396 >= 0.90; stop_reason max_steps, best_step 600 >= steps - 2 x eval_every = 400`. Per the clarified D-06 (commit 5258be1) no retry was run: the first condition (below 0.90) was not met, although the run was still improving at the end. Retrains: none.

### LORA-04 dev gate (decided before the smoke runs)

`LORA04_GATE pass: gen_check.tuned.parse_rate 1.0000 >= 0.90 on 100 first-turn dev prompts` (tuned gold first action 0.96; base parse 0.69, gold 0.15). LORA-04's "produces valid actions" is met by this gate.

### Commit

`b68a707` staged exactly the four files; `git ls-files models/qwen-agentlora` lists exactly adapter_config.json, loss_curve.png, manifest.json, train_log.jsonl.

### Base smoke run (LORA-02), 308 s wall, exit 0

```
REPORT rows 26 resolved_model Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf
REPORT parse rate 48/105
REPORT valid actions 48/105
REPORT unprotected dh/ds episodes with call_tool 3/10, attempted 0/10, succeeded 0
REPORT unprotected benign episodes with a valid action 3/3
REPORT unprotected outcomes attacked=0 ignored_injection=9 format_failure=4 llm_error=0 refused=0
REPORT protected dh/ds episodes with call_tool 4/10, attempted 0/10, succeeded 0
REPORT protected benign episodes with a valid action 3/3
REPORT protected outcomes attacked=0 ignored_injection=9 format_failure=4 llm_error=0 refused=0
smoke-plumbing-ok
```

summary.md: 26/26 episodes, attempted 0/10, success 0/10 both arms, benign task 1/3 both arms, parse errors per step 54.3 % (57/105), latency 1945 ms/step, blocked actions: policy engine 3, sandbox 1 (unrelated actions, no attack was attempted). Finding: the base model fails the action format in 57 of 105 steps (consistent with the research's about 45 % parse on 12 episodes).

### LoRA smoke run (LORA-04), 174 s wall, exit 0

```
REPORT rows 26 resolved_model Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf+qwen2.5-0.5b-agentlora@270501a67998
REPORT parse rate 77/77
REPORT valid actions 77/77
REPORT unprotected dh/ds episodes with call_tool 10/10, attempted 10/10, succeeded 10
REPORT unprotected benign episodes with a valid action 3/3
REPORT unprotected outcomes attacked=10 ignored_injection=3 format_failure=0 llm_error=0 refused=0
REPORT protected dh/ds episodes with call_tool 10/10, attempted 10/10, succeeded 0
REPORT protected benign episodes with a valid action 3/3
REPORT protected outcomes attacked=10 ignored_injection=3 format_failure=0 llm_error=0 refused=0
smoke-plumbing-ok
```

summary.md: 26/26 episodes, attempted 100 % (10/10, 95 % interval 72.2 to 100), success 10/10 unprotected and 0/10 with guardrails, all 10 blocked at the tool broker, benign task 3/3 in both arms, false positive 0/3, parse errors 0/77, latency 938 ms/step. The tuned row's attack attempts are the expected effect of the compliant training mix (D-01), disclosed and not "fixed"; the tuned model is a second injection-compliant worst-case principal, not a safety-tuned one. These smoke numbers are plumbing and behaviour evidence on 13 cases and one run, not attack-success results (full runs: Phase 4); they decided nothing.

**Nothing was changed after either smoke run:** no setting, data, code path or adapter. The only code change of the plan (`75bedd3`) preceded the full run.

## Task 3: viva page

`docs/viva/03-fine-tuning-lora-and-chat-templates.md`: 552 lines, 50 glossary bullets, Parts A to D, "Limits of this phase", 20 questions, D-01 to D-08 each cited at least twice, ASCII except em dashes in headings; `checks viva` printed `part-c-ok`; topic grep count 75 (>= 20). README Phase 3 row linked and "written (Phase 3)", Phase 4 and 5 rows unchanged. Commit `c126384` (exactly those two files). The page's research figure "about 0.95 GiB" for the frozen base was corrected to 0.92 GiB (988,097,824 bytes). Human check (readability, curve shape) remains for the end-of-phase verification; the curve was viewed here and flattens near the best step.

## Full suite after training (three calls, each under 7 minutes)

| Call | Result | Runtime |
|------|--------|---------|
| Phase 1-2 suite (ignoring the four Qwen test files) | 901 passed | 227.91 s |
| `test_hf_local_provider.py` + `test_qwen_lora_data.py` | 57 passed, 1 skipped (QWEN_LIVE probe) | 90.32 s |
| `test_qwen_lora_train.py` + `test_qwen_lora_checks.py` | 76 passed | 166.06 s |

Total 1034 passed, 1 skipped, same as after plan 03-03.

## Task Commits

1. **Fix (Rule 1): adapted-layer count on the model** - `75bedd3` (fix)
2. **Task 2: adapter config, manifest, train log, loss curve** - `b68a707` (feat)
3. **Task 3: viva page 03 and README row** - `c126384` (docs)

Task 1 changed no tracked file besides the fix above (no commit of its own, as planned).

## Deviations from Plan

**1. [Rule 1 - Bug] PEFT 0.21.2 stores `all-linear` as leaf names**
- **Found during:** Task 1 pilot (first attempt, before step 0)
- **Issue:** `build_lora_model` asserted `len(resolved) == 7 x 24` on the stored target_modules, which holds 7 leaf names, so every real run would stop at start-up; the tiny-model tests had not caught it.
- **Fix:** count adapted layers from modules with `lora_A` and `lora_B`; leaf-name and trainable-parameter (8,798,208) assertions unchanged. `tests/test_qwen_lora_train.py` 35 passed.
- **Files modified:** src/qwen_lora/train.py
- **Commit:** `75bedd3`

No D-05 deviation, no D-06 retry, no relaunch of any kind. The manifest's `code_dirty: true` reflects the unrelated working-tree files of the user (untracked results/, docs, scripts, dashboard WIP), none of them staged.

## Known Stubs

None.

## Threat Flags

None. T-03-18 (all decisions from pilot timing and dev metrics, recorded before smoke), T-03-19 (explicit staging; `git ls-files models/qwen-agentlora` lists the four files only), T-03-20 (read-only preflight, no process touched), T-03-21 (every row's resolved_model carries the adapter prefix, checked), T-03-22 (`checks viva` secret scan) implemented as planned.

## Self-Check: PASSED

- Files: models/qwen-agentlora/{adapter_config.json, manifest.json, train_log.jsonl, loss_curve.png, adapter_model.safetensors (ignored)}, docs/viva/03-fine-tuning-lora-and-chat-templates.md, results/phase3-smoke-base/summary.md, results/phase3-smoke-lora/summary.md exist.
- Commits `75bedd3`, `b68a707`, `c126384` are ancestors of HEAD; `git rev-list --count 58b5131..HEAD` = 3 before this SUMMARY commit.
