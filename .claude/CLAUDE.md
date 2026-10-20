<!-- GSD:project-start source:PROJECT.md -->

## Project

**OS-Level Guardrails for Autonomous AI Agents — Multi-LLM & Local-Model Expansion**

A capability-based reference monitor (policy engine + Linux namespaces + seccomp-bpf) that mediates every action an LLM-driven agent takes, evaluated on the InjecAgent indirect-prompt-injection benchmark. This expansion answers the reviewer's remark — *"use this on a locally made LLM and also compare with other LLMs other than Gemini"* — by training a language model locally (from scratch, and by local fine-tuning), running the same guardrails on it, and comparing it against several non-Gemini models. Built by Raj Vardhan Singh (24BDS0200) for the BCSE303P Operating Systems Lab; the audience is the review panel and the student's own viva preparation.

**Core Value:** The guardrails must be shown to hold **independently of which LLM drives the agent** — including a weak, unaligned, locally made model — backed by a fair, reproducible comparison table the student can defend in a viva.

### Constraints

- **Timeline**: next review/viva within one week — favour a working end-to-end comparison over polish
- **Hardware**: RTX 3060 Laptop 6 GB VRAM, 7 GB WSL RAM — model sizes and batch sizes must fit
- **Platform**: sandbox/benchmark only run under Linux (WSL2); Windows side is for editing
- **Budget**: free API tiers only; hosted runs are rate/quota limited and resumable
- **Fairness**: every model must receive the same system prompt, cases, seeds, and arms (`src/agent_harness/prompting.py` is the single source of truth); the scratch model uses a compact rendering of the same conversation
- **Evaluation hygiene**: InjecAgent tools used in the benchmark are held out of all local training data

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.10+ - All application code: agent harness (`src/agent_harness/`), policy engine (`src/policy_engine/`), LLM layer (`src/llm/`), benchmark (`src/benchmark/`), sandbox wrapper (`src/sandbox/sandbox.py`, `src/sandbox/executor.py`), dashboard (`src/dashboard/app.py`), from-scratch model (`src/scratch_llm/`), and entry points (`run_agent.py`, `run_sandboxed.py`, `run_benchmark.py`, `run_attack_simulation.py`)
- C - Seccomp-BPF syscall filter: `src/sandbox/seccomp_filter.c` (built via `src/sandbox/Makefile`, links `-lseccomp`)
- Bash - Setup and demo scripts: `setup.sh`, `run_demo.sh`, `scripts/run_full_benchmark.sh`
- HTML/CSS/JS - Dashboard UI: `src/dashboard/templates/index.html`
- YAML - Model registry: `config/models.yaml`
- JSON / JSONL - Logs and benchmark data: `logs/guardrails.jsonl`, `benchmarks/injecagent/*.jsonl`, `benchmarks/injecagent/tools.json`
- Google Apps Script - Slide export helper: `docs/create_google_slides.gs` (not part of the runtime)

## Runtime

- CPython 3.10+ (documented in `README.md` and `PRODUCT.md`; no `.python-version`, `pyproject.toml`, or `setup.py` present)
- Target OS: Linux, or WSL2 Ubuntu. The sandbox requires Linux user/mount/PID/net namespaces (`unshare`) and libseccomp, so it does not run natively on Windows.
- GCC for building the seccomp filter
- pip with a flat requirements file: `requirements.txt`
- Lockfile: Missing (no `requirements.lock`, `poetry.lock`, or `Pipfile.lock`; all dependencies use `>=` lower bounds)
- A local virtual environment is expected at `venv/` (created by `setup.sh`, git-ignored)

## Frameworks

- Flask >=3.0.0 - Real-time web dashboard, served on `0.0.0.0:5000` with `debug=True` (`src/dashboard/app.py`, template `src/dashboard/templates/index.html`)
- OpenAI Python SDK >=1.40.0 - Single client for every hosted and local model through the OpenAI-compatible chat-completions API (`src/llm/openai_compat.py`)
- PyTorch (version not pinned; installed separately, CUDA build recommended) - Used only by the from-scratch model in `src/scratch_llm/model.py`, `src/scratch_llm/dataset.py`, `src/scratch_llm/format.py`, and `src/llm/scratch.py`
- pytest >=8.0.0 - Test runner. Tests live in `tests/` (`conftest.py`, `test_agent_loop.py`, `test_benchmark.py`, `test_dashboard_clear.py`, `test_llm_layer.py`, `test_policy_engine.py`, `test_sandbox_detection.py`, `test_sandbox_executor.py`). The cache is at `.pytest_cache/`.
- GCC with `-Wall -Wextra -O2` - Compiles the seccomp filter (`src/sandbox/Makefile`)
- Linux `unshare` (util-linux) - Creates user, mount, PID, and network namespaces (`src/sandbox/executor.py`)
- No linter, formatter, or type checker configuration detected
- python-pptx (not in `requirements.txt`; imported by `scripts/generate_presentation.py`, `scripts/modify_template.py`, `scripts/polish_current_deck.py`) - Generates the review slide decks in `docs/`
- marked 12.0.2 and mermaid 11.4.1 loaded from the jsDelivr CDN by `scripts/export_sdd.py` - Renders the SDD HTML export in `docs/build/`

## Key Dependencies

- `openai` >=1.40.0 - Only network client for LLM calls. Configured with `max_retries=2` and a per-model timeout (default 60 s, `timeout` field in `config/models.yaml`, 180 s for `gemma-4-31b`).
- `python-dotenv` >=1.0.0 - Loads `.env` from the project root in `src/llm/registry.py` and `run_*.py` entry points
- `pyyaml` >=6.0 - Parses `config/models.yaml` in `src/llm/registry.py`
- `flask` >=3.0.0 - Dashboard server
- `matplotlib` >=3.8.0 - Benchmark charts (imported in the benchmark report code under `src/benchmark/report.py`)
- `torch` (unpinned) - From-scratch model training and inference
- `libseccomp` / `libseccomp-dev` (system package, apt) - Required to build and load `src/sandbox/seccomp_filter`
- Standard library heavily used: `urllib.request` (HTTP tool in `src/agent_harness/tools.py`, paper download in `scripts/download_papers.py`), `subprocess` (shell tool and sandbox execution), `json`, `dataclasses`, `secrets` (canary tokens), `tempfile`, `shlex`

## Configuration

- Secrets are read from environment variables, loaded from `.env` by `python-dotenv`. The `.env` file exists at the repo root and is git-ignored. Do not read or copy its values.
- Variable names documented in `.env.example`:
- Ollama and the scratch model need no key. `OpenAICompatibleProvider` substitutes the placeholder `not-needed` when `api_key_env` is unset.
- Model registry: `config/models.yaml` (fields: `provider`, `base_url`, `api_key_env`, `model`, `timeout`, `benchmark`, `checkpoint`). Default model: `gemini-flash-lite`.
- Dashboard log file: `logs/guardrails.jsonl` (path set in `src/dashboard/app.py`)
- Ignore rules: `.gitignore` excludes `.env`, `.env.*.local`, `venv/`, `.venv/`, `env/`
- `requirements.txt` - Python dependencies
- `setup.sh` - Installs apt packages (`python3-venv`, `gcc`, `libseccomp-dev`, `libseccomp2`), creates `venv/`, installs requirements, compiles the seccomp filter, and creates `logs/`
- `src/sandbox/Makefile` - Builds `seccomp_filter`
- `.gitattributes` - Forces LF line endings for `*.sh`
- No `Dockerfile`, `docker-compose*`, CI workflow (`.github/workflows/` absent), or `pyproject.toml` detected

## Platform Requirements

- Linux or WSL2 Ubuntu (required for the sandbox path; the policy engine, benchmark, and dashboard can run on other platforms)
- Python 3.10+, pip, GCC, `libseccomp-dev`, util-linux `unshare`
- Optional NVIDIA GPU with a CUDA PyTorch build for the scratch model
- Optional local Ollama server on `http://localhost:11434` (model `llama3.2:3b`)
- Not applicable. This is a research prototype and evaluation harness. The dashboard binds to `0.0.0.0:5000` in debug mode and is intended for local demos only.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Python modules are `snake_case.py`, grouped into packages under `src/` (for example `src/agent_harness/loop.py`, `src/llm/openai_compat.py`, `src/policy_engine/engine.py`).
- Each package has an `__init__.py` (for example `src/sandbox/__init__.py`, `src/llm/__init__.py`).
- Top-level entry points are `run_*.py` scripts at the repo root (`run_agent.py`, `run_benchmark.py`, `run_sandboxed.py`, `run_attack_simulation.py`).
- Tests are `tests/test_<area>.py` (for example `tests/test_policy_engine.py`, `tests/test_llm_layer.py`).
- Policy and data config are YAML: `src/policy_engine/policies/default_policy.yaml`, `config/models.yaml`.
- `snake_case` for all functions, including private helpers prefixed with a single underscore (for example `_log_path` in `src/policy_engine/engine.py`, `_check_shell_command`, `_check_filesystem`).
- Predicate helpers are named as questions or states: `is_linux()` in `src/sandbox/executor.py`, `get_seccomp_filter_path()`, `detect_enforcement()` in `src/sandbox/sandbox.py`.
- Verb-first names for actions: `build_messages`, `parse_action`, `action_target`, `classify_action`, `score_episode`, `generate_report`.
- `snake_case` locals. Module-level constants are `UPPER_SNAKE_CASE`: `LOG_FILE`, `PROJECT_ROOT`, `DEFAULT_POLICY` (`src/policy_engine/engine.py`), `ACTION_TIMEOUT_SECONDS` (`src/sandbox/executor.py`), `MAX_ASSISTANT_CHARS` (`src/agent_harness/loop.py`), `VALID_ACTIONS` and `SYSTEM_PROMPT` (`src/agent_harness/prompting.py`).
- Module-level mutable state is avoided where possible, but the dashboard caches in `_log_cache` (`src/dashboard/app.py`).
- `PascalCase` classes: `PolicyEngine`, `AgentLoop`, `LLMProvider`, `LLMResponse`, `ScriptedProvider` (in tests), `ActionParseError`, `QuotaExhausted`.
- Data holders use `@dataclass`: `Step`, `Episode` (`src/agent_harness/loop.py`), `LLMResponse` (`src/llm/base.py`).
- Custom exceptions subclass `ValueError` or `Exception` and carry a clear name ending in `Error` or a state name (`ActionParseError` in `src/agent_harness/prompting.py`, `QuotaExhausted` in `src/benchmark/runner.py`).
- Abstract interfaces use `abc.ABC` with `@abstractmethod`: `LLMProvider` (`src/llm/base.py`) and `Executor` (`src/agent_harness/executors.py`).

## Code Style

- No formatter config is committed (no `pyproject.toml`, `.prettierrc`, `.editorconfig`, or `black` config). Follow the prevailing style: 4-space indentation, double quotes in most library code, single quotes in `src/policy_engine/engine.py` and `src/dashboard/app.py`. Match the surrounding file rather than reformatting.
- Lines are generally kept under about 120 characters, though some lines in `src/agent_harness/prompting.py` and `src/policy_engine/engine.py` run longer.
- Long argument lists wrap with aligned continuation lines, as in `AgentLoop.__init__` in `src/agent_harness/loop.py` and `PolicyEngine.__init__` in `src/policy_engine/engine.py`.
- No linter is configured (no `.flake8`, `ruff.toml`, or `.pre-commit-config.yaml`). Keep code clean under `pyflakes`-level rules: no unused imports, no bare `except:`.
- Type hints are used on public signatures but inconsistently on helpers. Common forms: `-> str`, `-> tuple`, `-> dict`, `list = None` for optional lists (`src/agent_harness/loop.py`). `Optional[...]` from `typing` is the style for dataclass fields (`Optional[dict]` in `Step`). Prefer `Optional[X]` over `X | None`, matching the existing code.
- Use `from __future__` only if needed; it is not used in existing modules.
- Module-level docstrings are mandatory in practice. Every module opens with a `"""Title ... ====="""` block that explains the role and, where relevant, the OS-security mapping. Examples: `src/policy_engine/engine.py`, `src/sandbox/executor.py`, `src/llm/base.py`, `src/benchmark/scoring.py`.
- Section dividers inside test files use `# ── Name ──` comment lines (`tests/test_agent_loop.py`, `tests/test_llm_layer.py`). Source files use the same box-drawing style in comments sparingly.
- Non-ASCII characters appear in strings and comments (for example `✗`, `✓`, `—`). This is acceptable.

## Import Organization

- None. Imports are absolute from the repo root, `src.<package>.<module>`.
- Tests and entry scripts that run from outside the package insert the repo root into `sys.path` (see the `sys.path.insert(0, ...)` blocks at the top of `tests/test_policy_engine.py`, `tests/test_sandbox_detection.py`, and `tests/conftest.py`). Do not copy this pattern into `src/` modules.

## Error Handling

- LLM providers never raise on API failure. They return `LLMResponse` with `error` set, and callers check `response.ok` (`src/llm/base.py`). Implementations in `src/llm/openai_compat.py` catch `Exception` at the call site and convert it to an error response.
- Parse failures raise a dedicated exception (`ActionParseError` in `src/agent_harness/prompting.py`). The agent loop catches it and feeds the message back to the model as an observation (`src/agent_harness/loop.py`). Errors that the model can recover from are returned to the model, not raised.
- Executors return `Observation` objects with a status instead of raising (`src/agent_harness/executors.py`). Blocked actions produce an `Observation` whose output begins with `BLOCKED by policy_engine: ...`.
- Subprocess timeouts are caught explicitly: `except subprocess.TimeoutExpired:` in `src/sandbox/executor.py`, bounded by `ACTION_TIMEOUT_SECONDS = 30`.
- Quota exhaustion stops a run via `QuotaExhausted` (`src/benchmark/runner.py`). Episodes that errored are retried on resume.
- Fatal configuration errors print a message and call `sys.exit(1)`: `PolicyEngine.__init__` does this when the policy file is missing (`src/policy_engine/engine.py`).
- Missing API keys raise an error at provider construction, and tests assert this (`test_provider_requires_api_key` in `tests/test_llm_layer.py`).
- Do not raise from `LLMProvider.generate`. Return an `LLMResponse` with `error`.
- Model-facing errors go in an `Observation` or a feedback message, not an exception.
- Catch specific exceptions where possible. Broad `except Exception` is used only at the API boundary (`src/llm/openai_compat.py`).

## Logging

- Bracketed component tags prefix console output: `[POLICY ENGINE] ...` (`src/policy_engine/engine.py`), `[BUILD] ...` (`src/sandbox/Makefile`).
- Verbose behavior is a constructor flag, not a global level: `PolicyEngine(verbose=True)`, `AgentLoop(verbose=False)`. Use `self._say(...)` in `AgentLoop` for conditional output.
- Structured decision records are written as JSON Lines to `logs/guardrails.jsonl` (`LOG_FILE` in `src/policy_engine/engine.py`), which the dashboard reads. Do not log secrets or canary values to this file in new code beyond what the benchmark already records.
- Benchmark output is written to JSONL under `results/` and the runner de-duplicates episodes on resume.

## Comments

- Explain the security or OS-level reason for a rule, not the mechanics. Example: the comment "Text-only responses are always allowed (no system interaction)" in `src/policy_engine/engine.py`.
- Note ordering constraints that would be easy to break. Example: mount setup happens before the seccomp filter because the filter forbids `mount()` (module docstring of `src/sandbox/executor.py`).
- Mark design phases or assumptions inline when they affect behavior.
- Not applicable. Python docstrings follow a Google-like style with an `Args:` and `Returns:` block on public methods. Example: `PolicyEngine.__init__` and `PolicyEngine.check` in `src/policy_engine/engine.py`.

## Function Design

- Positional for the primary subject (`check(action_type, target)`), keyword for options with defaults (`max_steps=5`, `temperature=0.0`, `verbose=False`).
- Use `None` defaults and resolve inside the function (`policy_path = policy_path or DEFAULT_POLICY`).
- Return simple tuples for paired results: `check_detailed` returns `(decision, reason)` (`src/policy_engine/engine.py`). Expose the simple form as a wrapper (`check` returns `check_detailed(...)[0]`).
- Decisions are string constants: `'ALLOW'`, `'BLOCK'`, `'FLAG'`. Do not introduce booleans for policy decisions.
- Return dataclasses for structured results (`Episode`, `Observation`, `LLMResponse`).

## Module Design

- Public names are plain (no leading underscore). Helpers are underscore-prefixed.
- Each package's `__init__.py` re-exports the main entry points. Keep package `__init__` files light.
- Used in `src/` packages for convenience. Import from the concrete module in new code (`from src.llm.registry import get_provider`), which is how the tests and runner import things.

## Configuration and Secrets

- API keys come from environment variables loaded by `python-dotenv`. Never hard-code keys. Model settings live in `config/models.yaml`, loaded by `src/llm/registry.py` (`load_model_config`).
- `.env` holds real secrets and is git-ignored. Do not read or quote `.env` contents in code, docs, or tests. Use `.env.example` for documenting variables.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Pattern Overview

- The LLM is treated as an untrusted principal. It only produces text; `src/agent_harness/prompting.py` parses that text into an action dict. Actions are the only way effects reach the host.
- Decorator-style composition: `GuardedExecutor` (`src/agent_harness/executors.py`) wraps any `Executor` subclass, so policy enforcement is independent of how the action is ultimately carried out.
- Two-layer enforcement in the real sandbox: an application-level policy check (`src/policy_engine/engine.py`) followed by kernel-level isolation (`src/sandbox/executor.py`: user/mount/pid/net namespaces and a seccomp-BPF filter from `src/sandbox/seccomp_filter.c`).
- Policy is data, not code: rules live in `src/policy_engine/policies/default_policy.yaml`, evaluated first-match-wins with a default action.
- Same agent loop is reused for the unprotected demo, the sandboxed run, and the benchmark. Only the executor and provider differ.
- Platform-gated: kernel isolation only exists on Linux (`is_linux()` in `src/sandbox/executor.py`). On other platforms `Sandbox._build_executor` falls back to policy-only mode with `DirectExecutor`.

## Layers

- Purpose: Parse arguments, print banners, wire components together.
- Location: `run_agent.py`, `run_sandboxed.py`, `run_attack_simulation.py`, `run_benchmark.py`, `run_demo.sh`, `src/sandbox/sandbox.py` (`__main__` block), `src/dashboard/app.py` (`__main__` via `python -m`).
- Contains: argparse setup, `sys.path` insertion of project root, orchestration only.
- Depends on: agent harness, sandbox, policy engine, benchmark, LLM registry.
- Used by: end users and shell scripts (`run_demo.sh`, `scripts/run_full_benchmark.sh`).
- Purpose: Drive the think→act→observe loop for a single prompt.
- Location: `src/agent_harness/`
- Contains:
- Depends on: `src/llm/base.py` (provider interface), `src/policy_engine/` (indirectly through `GuardedExecutor`).
- Used by: `src/sandbox/sandbox.py`, `src/benchmark/runner.py`, `run_agent.py`.
- Purpose: Decide ALLOW / FLAG / BLOCK for an action type and target.
- Location: `src/policy_engine/engine.py`, `src/policy_engine/policies/default_policy.yaml`
- Contains: `PolicyEngine` with `check`, `check_detailed`, and per-category helpers `_check_shell_command`, `_check_command_file_args`, `_check_filesystem`, `_check_network`; `_log_decision` writes JSON lines to `logs/guardrails.jsonl`.
- Depends on: PyYAML, `shlex`, `fnmatch`.
- Used by: `GuardedExecutor`, `ToolBroker` (logging only), `run_attack_simulation.py`, `run_sandboxed.py`, `src/benchmark/runner.py`.
- Purpose: Execute permitted shell commands in a private view of the filesystem with syscall filtering.
- Location: `src/sandbox/executor.py` (`SandboxExecutor`, `detect_enforcement`, `get_seccomp_filter_path`), `src/sandbox/seccomp_filter.c` plus `src/sandbox/Makefile` (builds the `seccomp_filter` binary, links `-lseccomp`), `src/sandbox/__init__.py`.
- Contains: per-episode staging directory (`tempfile.mkdtemp(prefix="guardrails_sbx_")`), fake `/home`, `/root`, `/tmp`, canary `/etc/shadow` copy, and `unshare --user --map-root-user --mount --pid --fork --net` invocation with a 30-second action timeout (`ACTION_TIMEOUT_SECONDS`).
- Depends on: `src/agent_harness/environment.py`, `src/agent_harness/executors.py`.
- Used by: `Sandbox._build_executor` in `src/sandbox/sandbox.py` (Linux only), `src/benchmark/runner.py` (`build_executor`, lazy import, arm "guarded").
- Purpose: Turn a message list into text, normalized as `LLMResponse`.
- Location: `src/llm/`
- Contains:
- Depends on: `config/models.yaml`, API keys named by `api_key_env` (read from environment; see Cross-Cutting Concerns).
- Used by: `AIAgent`, `Sandbox`, `src/benchmark/runner.py`.
- Purpose: A small from-scratch decoder-only transformer used as an intentionally naive, unaligned target.
- Location: `src/scratch_llm/` (`model.py` ScratchGPT, `dataset.py`, `format.py`)
- Depends on: PyTorch, `src/agent_harness/prompting.py` (`parse_action`), `src/benchmark/injecagent.py` (`DATA_DIR`).
- Used by: `src/llm/scratch.py` through a checkpoint file. Not imported by the runtime path directly.
- Purpose: Run InjecAgent-style indirect prompt-injection cases against each model with and without guardrails, then score and report.
- Location: `src/benchmark/`, `benchmarks/injecagent/` (JSONL cases and `tools.json`), `run_benchmark.py`.
- Contains: `injecagent.py` (case loading, canaries, `BenchmarkCase`), `runner.py` (`ARMS`, `run_episode`, backoff for rate limits, `build_executor` per arm), `scoring.py` (`score_episode`), `classify.py` (`classify_action`), `report.py` (`generate_report`, CSV/Markdown/charts).
- Depends on: agent harness, policy engine, LLM registry, sandbox (lazy import).
- Used by: `run_benchmark.py`, `scripts/run_full_benchmark.sh`.
- Purpose: Live read-only view of decisions, with a clear-log action.
- Location: `src/dashboard/app.py` (Flask app), `src/dashboard/templates/index.html`.
- Contains: routes `/`, `/api/info`, `/api/logs`, `/api/stats`, `/api/clear` (POST). `_read_logs_cached` caches parsed `logs/guardrails.jsonl` by mtime.
- Depends on: `logs/guardrails.jsonl` written by `PolicyEngine._log_decision`; reads `results/` directory path constant.
- Used by: end users in a browser at `http://localhost:5000`.

## Data Flow

### Primary Request Path (sandboxed run)

### Benchmark Path

### Unprotected Path

- Episode state lives in `Episode` and `Step` objects (`src/agent_harness/loop.py`) and in the `messages` list local to `AgentLoop.run`. Nothing persists between prompts except the decision log.
- Sandbox filesystem state is per-episode: `SandboxExecutor._stage_environment` creates the staging tree and `executor.close()` releases it, so writes persist across steps within one episode only.
- Dashboard state is a module-level `_log_cache` dict in `src/dashboard/app.py`, invalidated by file mtime.

## Key Abstractions

- Purpose: The single interface through which every action is performed. `Observation` carries `status` (`success`, `error`, `blocked`), `output`, `reason`, `layer`, and `flagged`.
- Examples: `src/agent_harness/executors.py` (`Executor`, `Observation`, `DirectExecutor`, `GuardedExecutor`, `MockExecutor`), `src/agent_harness/tool_broker.py` (`ToolBroker`), `src/sandbox/executor.py` (`SandboxExecutor`).
- Pattern: Strategy plus decorator. `GuardedExecutor` and `ToolBroker` hold an `inner: Executor`.
- Purpose: Uniform model access. Every provider returns `LLMResponse` and never raises for API failures; errors are returned as `ok=False`.
- Examples: `src/llm/openai_compat.py::OpenAICompatibleProvider`, `src/llm/scratch.py::LocalScratchProvider`.
- Pattern: Factory via `get_provider(name)` in `src/llm/registry.py`, driven by `config/models.yaml`.
- Purpose: Evaluate `(action_type, target)` against ordered YAML rules.
- Examples: `src/policy_engine/engine.py`, `src/policy_engine/policies/default_policy.yaml`.
- Pattern: First-match-wins with a default action, so rule order matters.
- Purpose: Declarative description of files, web content, and canary tokens to place into a sandbox or emulated OS.
- Examples: `src/agent_harness/environment.py` (`Environment`, `build_environment`, `validate_stageable`).
- Pattern: Dataclass plus factory function.
- Purpose: Describe one injection scenario and the tools a task may call. A `None` capability list means no restriction.
- Examples: `src/benchmark/injecagent.py`, `src/agent_harness/tool_broker.py`.

## Entry Points

- Location: `run_agent.py`
- Triggers: `python run_agent.py [--model NAME] '<prompt>'`
- Responsibilities: Run the unprotected agent with `AIAgent(model=...)` (no policy engine).
- Location: `run_sandboxed.py`
- Triggers: `python run_sandboxed.py '<prompt>'`
- Responsibilities: Run the agent with `PolicyEngine` and `Sandbox`.
- Location: `run_attack_simulation.py`
- Triggers: `python run_attack_simulation.py`
- Responsibilities: Scripted attack scenarios that call `ToolExecutor` and `PolicyEngine` directly, without an LLM.
- Location: `run_benchmark.py`
- Triggers: `python run_benchmark.py --models all | --resume DIR | --report-only DIR [--merge ...]`
- Responsibilities: Multi-model benchmark orchestration and report generation.
- Location: `src/sandbox/sandbox.py`
- Triggers: `python -m src.sandbox.sandbox '<prompt>'` (or direct run)
- Responsibilities: Same as `run_sandboxed.py` with a built-in usage message.
- Location: `src/dashboard/app.py`
- Triggers: `python -m src.dashboard.app`, then open `http://localhost:5000`
- Responsibilities: Serve `index.html` and JSON APIs over `logs/guardrails.jsonl`.

## Architectural Constraints

- **Threading:** Single-threaded, synchronous. `AgentLoop` runs steps sequentially. Each `SandboxExecutor` action is a blocking `subprocess` call with a 30-second timeout. No worker threads or async I/O are used in `src/`.
- **Global state:** `LOG_FILE` and `DEFAULT_POLICY` are module-level constants in `src/policy_engine/engine.py`. `_log_cache` is module-level in `src/dashboard/app.py`. `tests/conftest.py` monkeypatches `policy_engine_module.LOG_FILE` so tests do not write the real log.
- **Platform:** Kernel isolation requires Linux (WSL is the documented route on Windows, per the `run_benchmark.py` docstring). `SandboxExecutor.__init__` raises `RuntimeError` on non-Linux.
- **Import path:** Scripts and `AIAgent` insert the project root into `sys.path` at import time. Modules use absolute `src.` imports, so the repo root must be the working directory or on `PYTHONPATH`.
- **Lazy import to avoid a Linux-only dependency:** `src/benchmark/runner.py::build_executor` imports `src.sandbox.executor` inside the function body.
- **Dependency direction:** `src/agent_harness/` is the lowest layer and must not import `src/sandbox/` or `src/policy_engine/`. The sandbox and policy engine depend on the harness. `src/scratch_llm/dataset.py` imports `src/benchmark/injecagent.py`, which means the training code depends on the benchmark package.
- **Secrets:** Model API keys are read from environment variables named in `config/models.yaml` (`api_key_env`). A `.env` file exists at the repo root; its contents are not part of this analysis.

## Anti-Patterns

### Policy check and execution happen in separate objects

### Private method called across module boundary

### Module-level log path as hidden dependency

### Sys.path mutation in every entry script

## Error Handling

- LLM failures: providers return `LLMResponse(ok=False, error=...)` (`src/llm/openai_compat.py::_failed_generation`). `AgentLoop.run` sets `stop_reason="llm_error"`.
- Parse failures: `ActionParseError` (`src/agent_harness/prompting.py`) is caught in `AgentLoop.run`, recorded on the step, and reported back to the model with `INVALID_REPLY_TEMPLATE`.
- Tool failures: `DirectExecutor` maps non-success results to `Observation("error", ...)`.
- Policy blocks: `Observation("blocked", layer=...)`, never an exception.
- Setup failures: `Sandbox.run_agent` prints and calls `sys.exit(1)` on `RuntimeError`/`KeyError` from `get_provider`. `SandboxExecutor.__init__` raises `RuntimeError` on non-Linux.
- Rate limits in the benchmark: `src/benchmark/runner.py::_run_with_backoff` parses wait hints from error text and retries.

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| brandkit | Premium brand-kit image generation skill for creating high-end brand-guidelines boards, logo systems, identity decks, and visual-world presentations. Trained for minimalist, cinematic, editorial, dark-tech, luxury, cultural, security, gaming, developer-tool, and consumer-app brand systems. Optimized for intentional logo concepting, refined composition, sparse typography, strong symbolic meaning, premium mockups, art-directed imagery, and flexible grid layouts. | `.agents/skills/brandkit/SKILL.md` |
| design-taste-frontend | Anti-slop frontend skill for landing pages, portfolios, and redesigns. The agent reads the brief, infers the right design direction, and ships interfaces that do not look templated. Real design systems when applicable, audit-first on redesigns, strict pre-flight check. | `.agents/skills/design-taste-frontend/SKILL.md` |
| design-taste-frontend-v1 | The original v1 taste-skill, preserved for projects depending on its exact behavior. The current default is `design-taste-frontend` (v2 experimental), which is a substantial rewrite. Use this v1 install name only if you need exact backward compatibility. | `.agents/skills/design-taste-frontend-v1/SKILL.md` |
| full-output-enforcement | Overrides default LLM truncation behavior. Enforces complete code generation, bans placeholder patterns, and handles token-limit splits cleanly. Apply to any task requiring exhaustive, unabridged output. | `.agents/skills/full-output-enforcement/SKILL.md` |
| gpt-taste | Elite UX/UI & Advanced GSAP Motion Engineer. Enforces Python-driven true randomization for layout variance, strict AIDA page structure, wide editorial typography (bans 6-line wraps), gapless bento grids, strict GSAP ScrollTriggers (pinning, stacking, scrubbing), inline micro-images, and massive section spacing. | `.agents/skills/gpt-taste/SKILL.md` |
| high-end-visual-design | Teaches the AI to design like a high-end agency. Defines the exact fonts, spacing, shadows, card structures, and animations that make a website feel expensive. Blocks all the common defaults that make AI designs look cheap or generic. | `.agents/skills/high-end-visual-design/SKILL.md` |
| image-to-code | Elite website image-to-code skill for Codex. For visually important web tasks, it must first generate the design image(s) itself, deeply analyze them, then implement the website to match them as closely as possible. In Codex, it must prefer large, readable, section-specific images instead of tiny compressed boards, generate fresh standalone images for sections or detail views instead of cropping old ones, avoid lazy under-generation, avoid cards-inside-cards-inside-cards UI, and keep the hero clean, spacious, readable, and visible on a small laptop. | `.agents/skills/image-to-code/SKILL.md` |
| imagegen-frontend-mobile | Elite mobile app image-generation skill for creating premium, app-native screen concepts and flows. Designed for iOS, Android, and cross-platform mobile products. Prioritizes clean hierarchy, comfortably readable text, strong multi-screen consistency, controlled color palettes, non-generic creative direction, textured surfaces, image-led composition, tasteful custom iconography, and clean phone mockup framing. By default, screens should be shown inside a subtle premium iPhone or similar phone mockup with a visible frame, while the main focus stays on the app content itself. This skill generates images only. It does not write code. | `.agents/skills/imagegen-frontend-mobile/SKILL.md` |
| imagegen-frontend-web | Elite frontend image-direction skill for generating premium, conversion-aware website design references. CRITICAL OUTPUT RULE — generate ONE separate horizontal image FOR EVERY section. A landing page with 8 sections produces 8 images. Never compress multiple sections into one image. Enforces composition variety (not always left-text / right-image), background-image freedom, varied CTAs, varied hero scales (giant / mid / mini minimalist), narrative concept spine, second-read moments, and a single consistent palette across all images. Optimized for landing pages, marketing sites, and product comps that developers or coding models can accurately recreate. | `.agents/skills/imagegen-frontend-web/SKILL.md` |
| industrial-brutalist-ui | Raw mechanical interfaces fusing Swiss typographic print with military terminal aesthetics. Rigid grids, extreme type scale contrast, utilitarian color, analog degradation effects. For data-heavy dashboards, portfolios, or editorial sites that need to feel like declassified blueprints. | `.agents/skills/industrial-brutalist-ui/SKILL.md` |
| minimalist-ui | Clean editorial-style interfaces. Warm monochrome palette, typographic contrast, flat bento grids, muted pastels. No gradients, no heavy shadows. | `.agents/skills/minimalist-ui/SKILL.md` |
| redesign-existing-projects | Upgrades existing websites and apps to premium quality. Audits current design, identifies generic AI patterns, and applies high-end design standards without breaking functionality. Works with any CSS framework or vanilla CSS. | `.agents/skills/redesign-existing-projects/SKILL.md` |
| stitch-design-taste | Semantic Design System Skill for Google Stitch. Generates agent-friendly DESIGN.md files that enforce premium, anti-generic UI standards — strict typography, calibrated color, asymmetric layouts, perpetual micro-motion, and hardware-accelerated performance. | `.agents/skills/stitch-design-taste/SKILL.md` |
| ui-ux-pro-max | "UI/UX design intelligence for web, mobile, and desktop. This skill should be used when designing, building, reviewing, or fixing interfaces, including pages, components, design systems, accessibility, interaction, responsive layout, typography, color, charts, and stack-specific UI implementation. Searchable local data: 79 searchable styles (50 active), 192 product palettes and reasoning profiles, 74 font pairings, 119 UX guidelines, 105 icons, 17 GSAP presets, 25 chart types, and 22 stacks." | `.agents/skills/ui-ux-pro-max/SKILL.md` |
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-fast` for a trivial task inline, with no subagents and no PLAN.md
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
