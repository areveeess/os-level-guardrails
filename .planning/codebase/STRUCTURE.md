---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# Codebase Structure

**Analysis Date:** 2026-10-20

## Directory Layout

```
os_level_guardrails/
├── run_agent.py              # Unprotected agent entry point
├── run_sandboxed.py          # Sandboxed agent entry point
├── run_attack_simulation.py  # Scripted attack scenarios (no LLM)
├── run_benchmark.py          # Multi-model benchmark orchestrator
├── run_demo.sh               # Scripted demo (unsandboxed vs sandboxed)
├── setup.sh                  # Environment setup
├── requirements.txt          # Python dependencies
├── config/                   # Model registry (models.yaml)
├── src/                      # All application code (Python packages)
│   ├── agent_harness/        # Agent loop, executors, prompting, tools
│   ├── policy_engine/        # YAML access-matrix evaluator
│   │   └── policies/         # default_policy.yaml
│   ├── sandbox/              # Linux namespace + seccomp isolation
│   ├── llm/                  # Provider abstraction and registry
│   ├── benchmark/            # InjecAgent runner, scoring, reporting
│   ├── dashboard/            # Flask log viewer
│   │   └── templates/        # index.html
│   └── scratch_llm/          # From-scratch transformer (train/format/model)
├── benchmarks/
│   └── injecagent/           # JSONL cases, tools.json, SOURCE.md
├── tests/                    # pytest suite
├── scripts/                  # Maintenance and deck/doc generators
├── docs/                     # SDD, architecture, planning, review decks
│   ├── build/                # Generated SDD HTML and diagram PNGs
│   └── images/               # Figures used in docs
├── results/                  # Benchmark output (timestamped run dirs)
├── logs/                     # guardrails.jsonl decision log (+ .gitkeep)
├── papers/                   # Reference PDFs and notes (not analyzed)
├── .agents/skills/           # Agent design skills (not part of runtime)
├── .impeccable/              # Local design tooling config
├── .planning/codebase/       # Generated codebase maps (this document set)
└── venv/                     # Local virtualenv (not analyzed)
```

## Directory Purposes

**`src/agent_harness/`:**
- Purpose: Agent loop and the action execution abstractions
- Contains: `agent.py`, `loop.py`, `executors.py`, `tool_broker.py`, `prompting.py`, `environment.py`, `tools.py`, `__init__.py`
- Key files: `src/agent_harness/loop.py` (`AgentLoop`), `src/agent_harness/executors.py` (`Executor`, `GuardedExecutor`, `MockExecutor`)

**`src/policy_engine/`:**
- Purpose: Access-matrix policy evaluation and decision logging
- Contains: `engine.py`, `__init__.py`, `policies/default_policy.yaml`
- Key files: `src/policy_engine/engine.py`, `src/policy_engine/policies/default_policy.yaml`

**`src/sandbox/`:**
- Purpose: OS-level isolation for permitted actions
- Contains: `executor.py` (`SandboxExecutor`), `sandbox.py` (`Sandbox` orchestrator with `__main__`), `seccomp_filter.c`, `Makefile`, `seccomp_filter` (compiled binary, present only after `make`), `__init__.py`
- Key files: `src/sandbox/executor.py`, `src/sandbox/sandbox.py`, `src/sandbox/seccomp_filter.c`

**`src/llm/`:**
- Purpose: Model provider interface and registry
- Contains: `base.py`, `registry.py`, `openai_compat.py`, `scratch.py`, `__init__.py`
- Key files: `src/llm/base.py`, `src/llm/registry.py`

**`src/benchmark/`:**
- Purpose: Injection benchmark: cases, runner, classifier, scoring, reports
- Contains: `injecagent.py`, `runner.py`, `scoring.py`, `classify.py`, `report.py`, `__init__.py` (empty)
- Key files: `src/benchmark/runner.py`, `src/benchmark/report.py`

**`src/dashboard/`:**
- Purpose: Web view of the decision log
- Contains: `app.py`, `templates/index.html`, `__init__.py`
- Key files: `src/dashboard/app.py`, `src/dashboard/templates/index.html`

**`src/scratch_llm/`:**
- Purpose: A small transformer implemented without pretrained weights; used as a naive target model
- Contains: `model.py` (`GPTConfig`, ScratchGPT), `dataset.py`, `format.py`
- Key files: `src/scratch_llm/model.py`

**`config/`:**
- Purpose: Model definitions (provider, base URL, env var name for key, benchmark inclusion, timeouts)
- Contains: `models.yaml`
- Key files: `config/models.yaml` (`default: gemini-flash-lite`)

**`benchmarks/injecagent/`:**
- Purpose: Source data for the InjecAgent-style benchmark
- Contains: `attacker_cases_dh.jsonl`, `attacker_cases_ds.jsonl`, `user_cases.jsonl`, `tools.json`, `SOURCE.md`
- Generated: No. Committed: Yes.

**`tests/`:**
- Purpose: pytest tests, one module per subsystem
- Contains: `conftest.py` (autouse fixture redirecting the policy log to `tmp_path`), `test_agent_loop.py`, `test_benchmark.py`, `test_dashboard_clear.py`, `test_llm_layer.py`, `test_policy_engine.py`, `test_sandbox_detection.py`, `test_sandbox_executor.py`, `__init__.py`

**`scripts/`:**
- Purpose: Helper scripts outside the runtime
- Contains: `download_papers.py`, `export_sdd.py`, `generate_presentation.py`, `modify_template.py`, `polish_current_deck.py`, `run_full_benchmark.sh`
- Key files: `scripts/run_full_benchmark.sh`, `scripts/export_sdd.py`

**`docs/`:**
- Purpose: Design and project documentation
- Contains: `architecture.md`, `SDD.md`, `SDD.docx`, `SDD.pdf`, `planning.md`, `project_state_and_plan.md`, `project_review_rubrics.md`, `review_1_presentation.md`, `review_2_presentation.md`, presentation files (`.pptx`, `.pdf`, `template.pptx`), `create_google_slides.gs`
- Generated: `docs/build/` and presentation outputs are generated. Committed: Yes.

**`results/`** (layout only, contents not analyzed):
- Purpose: Benchmark outputs
- Layout: `results/<run-name>/` where each run directory holds `summary.csv`, `summary.md`, `charts/`, and per-model subdirectories (`<model-name>/`). Some runs also hold `raw.jsonl` and `decisions.jsonl`. Current runs: `final-20261011`, `smoke-2`, `smoke-3`, `smoke-merged`.
- Generated: Yes (by `run_benchmark.py`). Committed: Not verified; currently shows as untracked in git status.

**`logs/`:**
- Purpose: Runtime decision log
- Contains: `guardrails.jsonl` (appended by `PolicyEngine`), `.gitkeep`
- Generated: Yes. Committed: directory kept via `.gitkeep`.

## Key File Locations

**Entry Points:**
- `run_agent.py`: unprotected agent
- `run_sandboxed.py`: sandboxed agent
- `run_attack_simulation.py`: scripted attacks
- `run_benchmark.py`: benchmark orchestration
- `src/dashboard/app.py`: dashboard server
- `src/sandbox/sandbox.py`: sandbox `__main__`

**Configuration:**
- `config/models.yaml`: LLM models and providers
- `src/policy_engine/policies/default_policy.yaml`: access-matrix rules
- `requirements.txt`: Python dependencies
- `.env.example`: template for required environment variables (the `.env` file is not documented here)
- `.gitignore`, `.gitattributes`: repository-level settings
- `skills-lock.json`: lock file for agent skills

**Core Logic:**
- `src/agent_harness/loop.py`: agent loop
- `src/agent_harness/executors.py`: executor chain and command matchers
- `src/agent_harness/prompting.py`: action parsing
- `src/policy_engine/engine.py`: policy evaluation
- `src/sandbox/executor.py`: namespace execution
- `src/llm/openai_compat.py`: hosted and local model client
- `src/benchmark/runner.py`: benchmark episodes

**Testing:**
- `tests/conftest.py`: shared fixtures
- `tests/test_*.py`: module tests
- `.pytest_cache/`: pytest cache (generated, not source)

**Documentation:**
- `README.md`: project overview
- `PRODUCT.md`: product notes
- `docs/architecture.md`: architecture summary

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (`tool_broker.py`, `injecagent.py`)
- Entry scripts at repo root: `run_<purpose>.py` (`run_benchmark.py`)
- Test modules: `test_<subsystem>.py` under `tests/`
- Shell scripts: `snake_case.sh` or `run_*.sh`
- Data files: `*.jsonl` for case sets and logs, `*.yaml` for config and policy
- Result directories: `<run-name>/` or timestamped `YYYYMMDD-HHMM` (for example `results/20261011-1500`)

**Directories:**
- Packages under `src/` are `snake_case`, singular subject (`agent_harness`, `policy_engine`, `sandbox`, `llm`, `benchmark`, `dashboard`)
- Each package has an `__init__.py`. `src/` itself has `__init__.py`.
- Model names used as directory keys in results match `config/models.yaml` keys (`gemini-flash-lite`, `ollama-llama3.2-3b`)

## Where to Add New Code

**New agent action type:**
- Handler: add the branch to the executor that owns the action (`src/agent_harness/tools.py` for real tools, `src/agent_harness/executors.py::MockExecutor` for the emulated benchmark)
- Policy rule: add an entry to `src/policy_engine/policies/default_policy.yaml` and handle the category in `PolicyEngine.check_detailed` (`src/policy_engine/engine.py`)
- Tests: `tests/test_policy_engine.py`, `tests/test_agent_loop.py`

**New policy category or check:**
- Implementation: `src/policy_engine/engine.py` (add a `_check_<category>` method following `_check_network`)
- Tests: `tests/test_policy_engine.py`

**New LLM provider:**
- Implementation: a new module in `src/llm/` subclassing `LLMProvider` from `src/llm/base.py`
- Registration: add a branch in `get_provider` in `src/llm/registry.py` and an entry in `config/models.yaml`
- Tests: `tests/test_llm_layer.py`

**New OpenAI-compatible model:**
- Configuration only: add a block under `models:` in `config/models.yaml` (`provider: openai_compat`, `base_url`, `api_key_env`, `model`)

**New executor or isolation mechanism:**
- Implementation: a new `Executor` subclass in `src/agent_harness/executors.py` (general) or `src/sandbox/` (OS-level)
- Wire into: `Sandbox._build_executor` in `src/sandbox/sandbox.py` and/or `build_executor` in `src/benchmark/runner.py`
- Tests: `tests/test_sandbox_executor.py`, `tests/test_sandbox_detection.py`

**New benchmark case set or arm:**
- Data: `benchmarks/injecagent/*.jsonl`
- Loader: `src/benchmark/injecagent.py`
- Arms: `ARMS` in `src/benchmark/runner.py`
- Reporting: `src/benchmark/report.py`
- Tests: `tests/test_benchmark.py`

**New dashboard view or API route:**
- Route: `src/dashboard/app.py` (add `@app.route`)
- Template: `src/dashboard/templates/index.html`
- Tests: `tests/test_dashboard_clear.py`

**New utility or shared helper:**
- Shared harness helpers: `src/agent_harness/` (for example `prompting.py` or `environment.py`)
- Policy helpers: `src/policy_engine/engine.py`
- Do not place helpers in the repo root. Root holds only entry scripts.

**New documentation or deck generator:**
- Script: `scripts/`
- Output: `docs/`

## Special Directories

**`venv/`:**
- Purpose: Local Python virtual environment
- Generated: Yes
- Committed: No

**`papers/`:**
- Purpose: Reference papers and extracted notes for the literature review
- Generated: No
- Committed: Yes

**`results/`:**
- Purpose: Benchmark run outputs
- Generated: Yes
- Committed: Not verified; currently shows as untracked in git status.

**`logs/`:**
- Purpose: Decision log consumed by the dashboard
- Generated: Yes
- Committed: Only `.gitkeep`

**`src/sandbox/seccomp_filter`:**
- Purpose: Compiled seccomp wrapper binary
- Generated: Yes (`make` in `src/sandbox/`)
- Committed: Present in repo listing; treat as a build artifact

**`__pycache__/` and `.pytest_cache/`:**
- Purpose: Python and pytest caches
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-10-20*
