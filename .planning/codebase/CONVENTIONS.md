---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# Coding Conventions

**Analysis Date:** 2026-10-20

## Naming Patterns

**Files:**
- Python modules are `snake_case.py`, grouped into packages under `src/` (for example `src/agent_harness/loop.py`, `src/llm/openai_compat.py`, `src/policy_engine/engine.py`).
- Each package has an `__init__.py` (for example `src/sandbox/__init__.py`, `src/llm/__init__.py`).
- Top-level entry points are `run_*.py` scripts at the repo root (`run_agent.py`, `run_benchmark.py`, `run_sandboxed.py`, `run_attack_simulation.py`).
- Tests are `tests/test_<area>.py` (for example `tests/test_policy_engine.py`, `tests/test_llm_layer.py`).
- Policy and data config are YAML: `src/policy_engine/policies/default_policy.yaml`, `config/models.yaml`.

**Functions:**
- `snake_case` for all functions, including private helpers prefixed with a single underscore (for example `_log_path` in `src/policy_engine/engine.py`, `_check_shell_command`, `_check_filesystem`).
- Predicate helpers are named as questions or states: `is_linux()` in `src/sandbox/executor.py`, `get_seccomp_filter_path()`, `detect_enforcement()` in `src/sandbox/sandbox.py`.
- Verb-first names for actions: `build_messages`, `parse_action`, `action_target`, `classify_action`, `score_episode`, `generate_report`.

**Variables:**
- `snake_case` locals. Module-level constants are `UPPER_SNAKE_CASE`: `LOG_FILE`, `PROJECT_ROOT`, `DEFAULT_POLICY` (`src/policy_engine/engine.py`), `ACTION_TIMEOUT_SECONDS` (`src/sandbox/executor.py`), `MAX_ASSISTANT_CHARS` (`src/agent_harness/loop.py`), `VALID_ACTIONS` and `SYSTEM_PROMPT` (`src/agent_harness/prompting.py`).
- Module-level mutable state is avoided where possible, but the dashboard caches in `_log_cache` (`src/dashboard/app.py`).

**Types and Classes:**
- `PascalCase` classes: `PolicyEngine`, `AgentLoop`, `LLMProvider`, `LLMResponse`, `ScriptedProvider` (in tests), `ActionParseError`, `QuotaExhausted`.
- Data holders use `@dataclass`: `Step`, `Episode` (`src/agent_harness/loop.py`), `LLMResponse` (`src/llm/base.py`).
- Custom exceptions subclass `ValueError` or `Exception` and carry a clear name ending in `Error` or a state name (`ActionParseError` in `src/agent_harness/prompting.py`, `QuotaExhausted` in `src/benchmark/runner.py`).
- Abstract interfaces use `abc.ABC` with `@abstractmethod`: `LLMProvider` (`src/llm/base.py`) and `Executor` (`src/agent_harness/executors.py`).

## Code Style

**Formatting:**
- No formatter config is committed (no `pyproject.toml`, `.prettierrc`, `.editorconfig`, or `black` config). Follow the prevailing style: 4-space indentation, double quotes in most library code, single quotes in `src/policy_engine/engine.py` and `src/dashboard/app.py`. Match the surrounding file rather than reformatting.
- Lines are generally kept under about 120 characters, though some lines in `src/agent_harness/prompting.py` and `src/policy_engine/engine.py` run longer.
- Long argument lists wrap with aligned continuation lines, as in `AgentLoop.__init__` in `src/agent_harness/loop.py` and `PolicyEngine.__init__` in `src/policy_engine/engine.py`.

**Linting:**
- No linter is configured (no `.flake8`, `ruff.toml`, or `.pre-commit-config.yaml`). Keep code clean under `pyflakes`-level rules: no unused imports, no bare `except:`.
- Type hints are used on public signatures but inconsistently on helpers. Common forms: `-> str`, `-> tuple`, `-> dict`, `list = None` for optional lists (`src/agent_harness/loop.py`). `Optional[...]` from `typing` is the style for dataclass fields (`Optional[dict]` in `Step`). Prefer `Optional[X]` over `X | None`, matching the existing code.
- Use `from __future__` only if needed; it is not used in existing modules.

**Other style rules:**
- Module-level docstrings are mandatory in practice. Every module opens with a `"""Title ... ====="""` block that explains the role and, where relevant, the OS-security mapping. Examples: `src/policy_engine/engine.py`, `src/sandbox/executor.py`, `src/llm/base.py`, `src/benchmark/scoring.py`.
- Section dividers inside test files use `# ── Name ──` comment lines (`tests/test_agent_loop.py`, `tests/test_llm_layer.py`). Source files use the same box-drawing style in comments sparingly.
- Non-ASCII characters appear in strings and comments (for example `✗`, `✓`, `—`). This is acceptable.

## Import Organization

**Order:**
1. Standard library (`os`, `sys`, `json`, `subprocess`, `dataclasses`, `typing`).
2. Third-party (`yaml`, `pytest`, `flask`, `openai`).
3. Project imports using the absolute `src.` package path (`from src.llm.base import LLMProvider`).

Separate groups with a blank line. Example from `src/benchmark/runner.py`: stdlib imports, blank line, then `from src.agent_harness...` and `from src.llm...` imports.

**Path Aliases:**
- None. Imports are absolute from the repo root, `src.<package>.<module>`.
- Tests and entry scripts that run from outside the package insert the repo root into `sys.path` (see the `sys.path.insert(0, ...)` blocks at the top of `tests/test_policy_engine.py`, `tests/test_sandbox_detection.py`, and `tests/conftest.py`). Do not copy this pattern into `src/` modules.

## Error Handling

**Patterns:**
- LLM providers never raise on API failure. They return `LLMResponse` with `error` set, and callers check `response.ok` (`src/llm/base.py`). Implementations in `src/llm/openai_compat.py` catch `Exception` at the call site and convert it to an error response.
- Parse failures raise a dedicated exception (`ActionParseError` in `src/agent_harness/prompting.py`). The agent loop catches it and feeds the message back to the model as an observation (`src/agent_harness/loop.py`). Errors that the model can recover from are returned to the model, not raised.
- Executors return `Observation` objects with a status instead of raising (`src/agent_harness/executors.py`). Blocked actions produce an `Observation` whose output begins with `BLOCKED by policy_engine: ...`.
- Subprocess timeouts are caught explicitly: `except subprocess.TimeoutExpired:` in `src/sandbox/executor.py`, bounded by `ACTION_TIMEOUT_SECONDS = 30`.
- Quota exhaustion stops a run via `QuotaExhausted` (`src/benchmark/runner.py`). Episodes that errored are retried on resume.
- Fatal configuration errors print a message and call `sys.exit(1)`: `PolicyEngine.__init__` does this when the policy file is missing (`src/policy_engine/engine.py`).
- Missing API keys raise an error at provider construction, and tests assert this (`test_provider_requires_api_key` in `tests/test_llm_layer.py`).

**Guidance for new code:**
- Do not raise from `LLMProvider.generate`. Return an `LLMResponse` with `error`.
- Model-facing errors go in an `Observation` or a feedback message, not an exception.
- Catch specific exceptions where possible. Broad `except Exception` is used only at the API boundary (`src/llm/openai_compat.py`).

## Logging

**Framework:** `print()` to stdout. The `logging` module is not used anywhere in `src/`.

**Patterns:**
- Bracketed component tags prefix console output: `[POLICY ENGINE] ...` (`src/policy_engine/engine.py`), `[BUILD] ...` (`src/sandbox/Makefile`).
- Verbose behavior is a constructor flag, not a global level: `PolicyEngine(verbose=True)`, `AgentLoop(verbose=False)`. Use `self._say(...)` in `AgentLoop` for conditional output.
- Structured decision records are written as JSON Lines to `logs/guardrails.jsonl` (`LOG_FILE` in `src/policy_engine/engine.py`), which the dashboard reads. Do not log secrets or canary values to this file in new code beyond what the benchmark already records.
- Benchmark output is written to JSONL under `results/` and the runner de-duplicates episodes on resume.

## Comments

**When to Comment:**
- Explain the security or OS-level reason for a rule, not the mechanics. Example: the comment "Text-only responses are always allowed (no system interaction)" in `src/policy_engine/engine.py`.
- Note ordering constraints that would be easy to break. Example: mount setup happens before the seccomp filter because the filter forbids `mount()` (module docstring of `src/sandbox/executor.py`).
- Mark design phases or assumptions inline when they affect behavior.

**JSDoc/TSDoc:**
- Not applicable. Python docstrings follow a Google-like style with an `Args:` and `Returns:` block on public methods. Example: `PolicyEngine.__init__` and `PolicyEngine.check` in `src/policy_engine/engine.py`.

## Function Design

**Size:** Functions are typically under 40 lines. The `AgentLoop.run` method in `src/agent_harness/loop.py` is the longest core flow and still stays readable. Split new logic into helpers that start with `_`.

**Parameters:**
- Positional for the primary subject (`check(action_type, target)`), keyword for options with defaults (`max_steps=5`, `temperature=0.0`, `verbose=False`).
- Use `None` defaults and resolve inside the function (`policy_path = policy_path or DEFAULT_POLICY`).

**Return Values:**
- Return simple tuples for paired results: `check_detailed` returns `(decision, reason)` (`src/policy_engine/engine.py`). Expose the simple form as a wrapper (`check` returns `check_detailed(...)[0]`).
- Decisions are string constants: `'ALLOW'`, `'BLOCK'`, `'FLAG'`. Do not introduce booleans for policy decisions.
- Return dataclasses for structured results (`Episode`, `Observation`, `LLMResponse`).

## Module Design

**Exports:**
- Public names are plain (no leading underscore). Helpers are underscore-prefixed.
- Each package's `__init__.py` re-exports the main entry points. Keep package `__init__` files light.

**Barrel Files:**
- Used in `src/` packages for convenience. Import from the concrete module in new code (`from src.llm.registry import get_provider`), which is how the tests and runner import things.

## Configuration and Secrets

- API keys come from environment variables loaded by `python-dotenv`. Never hard-code keys. Model settings live in `config/models.yaml`, loaded by `src/llm/registry.py` (`load_model_config`).
- `.env` holds real secrets and is git-ignored. Do not read or quote `.env` contents in code, docs, or tests. Use `.env.example` for documenting variables.

---

*Convention analysis: 2026-10-20*
