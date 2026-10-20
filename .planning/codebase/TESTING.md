---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# Testing Patterns

**Analysis Date:** 2026-10-20

## Test Framework

**Runner:**
- pytest `>=8.0.0` (declared in `requirements.txt`).
- Config: no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini`. Defaults apply. The root `conftest.py` is `tests/conftest.py`.
- Test discovery runs over `tests/`. A stale `.pytest_cache/` exists at the repo root.

**Assertion Library:**
- Plain `assert` statements (pytest rewriting). No third-party assertion library.
- `pytest.raises` for expected exceptions (`tests/test_llm_layer.py`).

**Execution environment:**
- Tests run under WSL, not Windows. Use the project venv:
  ```bash
  venv/bin/python -m pytest tests/ -v
  ```
- Do not run the suite from Windows PowerShell or Git Bash. Some tests depend on Linux namespaces (`unshare`) and the compiled seccomp filter.

**Run Commands:**

```bash
venv/bin/python -m pytest tests/ -v                       # Run all tests (inside WSL)
venv/bin/python -m pytest tests/test_policy_engine.py -v  # Run one module
venv/bin/python -m pytest tests/ -k "seccomp" -v          # Run by keyword
venv/bin/python -m pytest tests/ -x                       # Stop at first failure
```

- No watch mode and no coverage command are configured. Coverage is not measured.

## Test File Organization

**Location:**
- All tests live in a separate top-level `tests/` directory. They are not co-located with source.
- Tests under `.agents/skills/ui-ux-pro-max/scripts/tests/` belong to a vendored skill. They are not part of the project suite. Do not add project tests there.

**Naming:**
- Files: `tests/test_<area>.py`, matching a source area.
- Functions: `test_<behavior>` with a descriptive, sentence-like name, for example `test_block_etc_shadow`, `test_loop_feeds_back_parse_errors_and_continues`, `test_parse_strips_code_fence`.
- Classes: `Test<Area>` groups in `tests/test_policy_engine.py` (`TestFilesystemReadRules`, `TestFilesystemWriteRules`, `TestProcessRules`).

**Current inventory (68 test functions):**

| File | Test functions | Covers |
|------|----------------|--------|
| `tests/test_policy_engine.py` | 21 | Access-matrix rules, read/write/process decisions |
| `tests/test_llm_layer.py` | 14 | `parse_action`, registry, OpenAI-compatible adapter (fake client) |
| `tests/test_agent_loop.py` | 10 | `AgentLoop`, executors, emulated environment, egress detection |
| `tests/test_benchmark.py` | 10 | Runner end-to-end, retry and quota logic, scoring, report |
| `tests/test_sandbox_detection.py` | 6 | `detect_enforcement` attribution (seccomp vs sandbox) |
| `tests/test_sandbox_executor.py` | 6 | `SandboxExecutor` (Linux and seccomp only) |
| `tests/test_dashboard_clear.py` | 1 | Dashboard clear flow |

**Structure:**

```
tests/
├── __init__.py
├── conftest.py                 # Shared autouse fixture, sys.path setup
├── test_agent_loop.py
├── test_benchmark.py
├── test_dashboard_clear.py
├── test_llm_layer.py
├── test_policy_engine.py
├── test_sandbox_detection.py
└── test_sandbox_executor.py
```

## Test Structure

**Suite Organization:**
Module docstring first, then imports, then fixtures or helper classes, then grouped tests with `# ── Section ──` comments or `class Test...` groupings.

```python
"""
Tests for the multi-step agent loop, the executors, and the emulated environment.
"""

import pytest

from src.agent_harness.loop import AgentLoop
from src.llm.base import LLMProvider, LLMResponse

class ScriptedProvider(LLMProvider):
    """Replays fixed replies and records the conversations it was shown."""

    def __init__(self, replies, name="scripted"):
        self.name = name
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, temperature=0.0, max_tokens=512):
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            return LLMResponse(text="", latency_ms=1.0, model=self.name, error="no more replies")
        return LLMResponse(text=self.replies.pop(0), latency_ms=1.0, model=self.name)

# ── AgentLoop ──

def test_loop_runs_actions_until_respond():
    provider = ScriptedProvider(['{"action": "run_shell", "command": "ls"}',
                                 '{"action": "respond", "message": "done"}'])
    executor = RecordingExecutor()
    episode = AgentLoop(provider, executor).run("list files")
    assert episode.stop_reason == "responded" and episode.final_message == "done"
```

**Patterns:**
- Setup: a fixture returns a fresh object per test. Example: `engine()` returns `PolicyEngine()` in `tests/test_policy_engine.py`.
- Parametrize for input tables. Example: `@pytest.mark.parametrize("raw", ["", "I cannot help with that.", ...])` in `tests/test_llm_layer.py`, and the `command,is_egress` table in `tests/test_agent_loop.py`.
- Teardown: rely on `tmp_path` and `monkeypatch`, which pytest reverts automatically. No explicit teardown is used.
- Assertions often combine checks with `and` on one line (`assert episode.stop_reason == "responded" and episode.final_message == "done"`). Keep this style for small related checks, but split assertions when a failure would be hard to read.

**Fixtures in `tests/conftest.py`:**
- `isolated_decision_log` is `autouse=True`. It redirects `policy_engine_module.LOG_FILE` to `tmp_path / "guardrails.jsonl"` so tests never write to `logs/guardrails.jsonl`. Every new test inherits this. Do not remove it.
- `conftest.py` also inserts the repo root into `sys.path`.

## Mocking

**Framework:** pytest `monkeypatch` (standard library `unittest.mock` is not used in the project suite).

**Patterns:**

Replace a module attribute to redirect a path or a dependency:

```python

# tests/test_dashboard_clear.py

monkeypatch.setattr(dashboard_app, 'LOG_FILE', LOG_FILE)
monkeypatch.setattr(dashboard_app, '_log_cache', {...})
```

Replace a provider factory so the runner uses a scripted model:

```python

# tests/test_benchmark.py

monkeypatch.setattr(runner_module, "get_provider", scripted)
```

Block a system call to prove code does not run it:

```python

# tests/test_agent_loop.py

@pytest.fixture
def no_subprocess(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run for this action")
    monkeypatch.setattr(subprocess, "run", boom)
```

Fake a network client rather than calling a real API (`tests/test_llm_layer.py` uses `SimpleNamespace` to shape a fake OpenAI response). Remove the real key with `monkeypatch.delenv(..., raising=False)`.

**Test doubles:**
- Subclass the real interface for doubles: `ScriptedProvider(LLMProvider)` and `RecordingExecutor(Executor)` in `tests/test_agent_loop.py`. This keeps the double type-correct against the ABC.

**What to Mock:**
- LLM providers (always). Use `ScriptedProvider` or a fake client. Tests must make no network calls.
- Filesystem locations that the code reads or writes at module load (`LOG_FILE`, dashboard log path). Redirect to `tmp_path`.
- Subprocess execution, when a test asserts that an action is blocked before execution.
- Provider factories in the benchmark runner.

**What NOT to Mock:**
- The policy engine. Tests call the real `PolicyEngine` against the real `default_policy.yaml`. The point of these tests is the rules.
- `parse_action` and the parsing layer. Test them with real strings, including code fences and chatter around JSON.
- The sandbox detection logic. Feed real stderr and exit-code strings to `detect_enforcement`.

## Fixtures and Factories

**Test Data:**
- Inline literals are the norm. Action dictionaries and raw model replies are written directly in each test (`'{"action": "run_shell", "command": "ls"}'`).
- Benchmark specs come from the vendored `benchmarks/injecagent/tools.json`. Load them through `load_tool_specs` (`src/benchmark/injecagent.py`) and pass them to tests via a `specs` fixture in `tests/test_benchmark.py`.

**Location:**
- Shared fixtures: `tests/conftest.py`.
- Module-local fixtures: defined at the top of each test module.
- Benchmark fixture data: `benchmarks/injecagent/`.

## Coverage

**Requirements:** None enforced. No `--cov` flags, no `.coveragerc`, and no CI threshold.

**View Coverage:**

```bash
venv/bin/python -m pip install pytest-cov
venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing
```

Coverage is optional. The package is not in `requirements.txt`, so install it only when asked.

## Platform-Dependent Tests

- Linux-only tests are gated with `skipif`:
  - `tests/test_sandbox_executor.py` uses a module-level `pytestmark = pytest.mark.skipif(...)`, and a per-test skip when the seccomp binary is missing: `@pytest.mark.skipif(not os.path.isfile(get_seccomp_filter_path()), reason="seccomp filter not compiled")`.
  - `tests/test_benchmark.py` defines `linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("unshare") is None, ...)`.
- Before relying on sandbox tests, build the filter in WSL: `make -C src/sandbox` (requires `gcc` and `libseccomp-dev`). Otherwise those tests skip, which is expected.
- Keep platform gates in the same style: a named `skipif` marker with a clear `reason=`.

## Common Patterns

**Async Testing:**
- Not applicable. The codebase is synchronous.

**Error Testing:**

```python

# tests/test_llm_layer.py

@pytest.mark.parametrize("raw", ["", "I cannot help with that.", '{"action": "run_shell", "command": ', "[1, 2]"])
def test_parse_rejects_non_actions(raw):
    with pytest.raises(ActionParseError):
        parse_action(raw)
```

For behavior that should produce an error observation rather than an exception, assert the field instead:

```python
assert episode.steps[0].parse_error and episode.stop_reason == "responded"
assert "not a valid action" in provider.seen[1][-1]["content"]
```

**Policy decisions:**

```python
assert engine.check("read_file", "/etc/shadow") == "BLOCK"
```

Use the exact string constants `'ALLOW'`, `'BLOCK'`, `'FLAG'`.

**File output:**

```python
out = tmp_path / "results"
```

Always write test output under `tmp_path`. Never write to `results/` or `logs/` from tests.

## Guidance for New Tests

1. Add the test to the matching `tests/test_<area>.py`, or create a new module with a docstring that names the area.
2. Start the file with the module docstring and the same `sys.path` block as the existing modules if the file runs outside the package.
3. Use `ScriptedProvider`-style doubles for any LLM call. Never call a real API.
4. Redirect any file path with `monkeypatch` or `tmp_path`. The autouse fixture already protects the policy log.
5. Use `@pytest.mark.parametrize` for tables of inputs, and `pytest.raises` for exceptions.
6. Gate Linux or binary dependencies with a named `skipif` marker and a `reason=`.
7. Run with `venv/bin/python -m pytest tests/ -v` inside WSL.

## Known Gaps

- No coverage measurement and no coverage target.
- No CI configuration is present in the repo to run the suite automatically.
- `src/dashboard/app.py` has a single test (`tests/test_dashboard_clear.py`). Routes beyond the clear flow are untested.
- `src/scratch_llm/` (dataset, format, model) has no test module.
- `src/agent_harness/tools.py` and `src/agent_harness/tool_broker.py` have no dedicated test module. Their behavior is exercised only indirectly through the benchmark tests.

---

*Testing analysis: 2026-10-20*
