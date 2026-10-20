---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
<!-- refreshed: 2026-10-20 -->

# Architecture

**Analysis Date:** 2026-10-20

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                 Entry points (CLI scripts / dashboard)               │
│ run_agent.py  run_sandboxed.py  run_attack_simulation.py             │
│ run_benchmark.py  src/sandbox/sandbox.py (__main__)  src/dashboard/  │
└───────────┬───────────────────────┬──────────────────────┬───────────┘
            │                       │                      │
            ▼                       ▼                      ▼
┌──────────────────────┐ ┌────────────────────────┐ ┌──────────────────┐
│ Agent harness        │ │ Sandbox orchestration  │ │ Benchmark        │
│ src/agent_harness/   │ │ src/sandbox/sandbox.py │ │ src/benchmark/   │
│ AIAgent → AgentLoop  │ │ Sandbox.run_agent      │ │ runner/scoring/  │
└──────┬───────────────┘ └──────────┬─────────────┘ │ report/injecagent│
       │ actions                    │ wraps          └────────┬─────────┘
       ▼                            ▼                         │
┌─────────────────────────────────────────────────────────────┴────────┐
│  Executor chain (src/agent_harness/executors.py, tool_broker.py)     │
│  GuardedExecutor ─► policy check ─► inner Executor                   │
│     inner: DirectExecutor | SandboxExecutor | ToolBroker | MockExec. │
└───────┬─────────────────────────┬───────────────────────┬────────────┘
        │                         │                       │
        ▼                         ▼                       ▼
┌────────────────────┐ ┌──────────────────────────┐ ┌───────────────────┐
│ PolicyEngine       │ │ OS isolation             │ │ Emulated OS       │
│ src/policy_engine/ │ │ src/sandbox/executor.py  │ │ MockExecutor      │
│ YAML access matrix │ │ unshare ns + seccomp_BPF │ │ (benchmark only)  │
└─────────┬──────────┘ └──────────────────────────┘ └───────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Decision log `logs/guardrails.jsonl` ──► Flask dashboard (poll)     │
│ src/dashboard/app.py                                                 │
└─────────────────────────────────────────────────────────────────────┘

LLM layer (independent of the above, called by AgentLoop):
  src/llm/registry.py ─► src/llm/openai_compat.py | src/llm/scratch.py
  config/models.yaml defines the models; src/scratch_llm/ is the local
  from-scratch transformer used by the `scratch` provider.
```

## Pattern Overview

**Overall:** Layered reference-monitor pipeline with a pluggable executor chain. An LLM proposes JSON actions; each action passes through a policy check and then an executor that either runs it on the host, inside a Linux sandbox, or against an emulated OS.

**Key Characteristics:**
- The LLM is treated as an untrusted principal. It only produces text; `src/agent_harness/prompting.py` parses that text into an action dict. Actions are the only way effects reach the host.
- Decorator-style composition: `GuardedExecutor` (`src/agent_harness/executors.py`) wraps any `Executor` subclass, so policy enforcement is independent of how the action is ultimately carried out.
- Two-layer enforcement in the real sandbox: an application-level policy check (`src/policy_engine/engine.py`) followed by kernel-level isolation (`src/sandbox/executor.py`: user/mount/pid/net namespaces and a seccomp-BPF filter from `src/sandbox/seccomp_filter.c`).
- Policy is data, not code: rules live in `src/policy_engine/policies/default_policy.yaml`, evaluated first-match-wins with a default action.
- Same agent loop is reused for the unprotected demo, the sandboxed run, and the benchmark. Only the executor and provider differ.
- Platform-gated: kernel isolation only exists on Linux (`is_linux()` in `src/sandbox/executor.py`). On other platforms `Sandbox._build_executor` falls back to policy-only mode with `DirectExecutor`.

## Layers

**Entry / CLI layer:**
- Purpose: Parse arguments, print banners, wire components together.
- Location: `run_agent.py`, `run_sandboxed.py`, `run_attack_simulation.py`, `run_benchmark.py`, `run_demo.sh`, `src/sandbox/sandbox.py` (`__main__` block), `src/dashboard/app.py` (`__main__` via `python -m`).
- Contains: argparse setup, `sys.path` insertion of project root, orchestration only.
- Depends on: agent harness, sandbox, policy engine, benchmark, LLM registry.
- Used by: end users and shell scripts (`run_demo.sh`, `scripts/run_full_benchmark.sh`).

**Agent harness layer:**
- Purpose: Drive the think→act→observe loop for a single prompt.
- Location: `src/agent_harness/`
- Contains:
  - `agent.py`: `AIAgent` facade; builds an executor and provider, runs `AgentLoop`.
  - `loop.py`: `AgentLoop` (max-step loop), `Step`, `Episode` dataclasses.
  - `prompting.py`: message construction, tool-spec formatting, `parse_action`, `action_target`.
  - `environment.py`: `Environment` dataclass plus `build_environment` (canary files, fake home, seed token).
  - `tools.py`: `ToolExecutor`, real implementations of shell/file/http tools used by `DirectExecutor`.
  - `executors.py`: `Executor` base, `Observation`, `DirectExecutor`, `GuardedExecutor`, `MockExecutor` (emulated OS for the benchmark), and the regex matchers `EGRESS_CMD`, `DESTRUCTIVE_CMD`, `ESCALATION_CMD`, `URL`.
  - `tool_broker.py`: `ToolBroker`, handles `call_tool` actions against a simulated tool set and a per-task capability list.
- Depends on: `src/llm/base.py` (provider interface), `src/policy_engine/` (indirectly through `GuardedExecutor`).
- Used by: `src/sandbox/sandbox.py`, `src/benchmark/runner.py`, `run_agent.py`.

**Policy layer:**
- Purpose: Decide ALLOW / FLAG / BLOCK for an action type and target.
- Location: `src/policy_engine/engine.py`, `src/policy_engine/policies/default_policy.yaml`
- Contains: `PolicyEngine` with `check`, `check_detailed`, and per-category helpers `_check_shell_command`, `_check_command_file_args`, `_check_filesystem`, `_check_network`; `_log_decision` writes JSON lines to `logs/guardrails.jsonl`.
- Depends on: PyYAML, `shlex`, `fnmatch`.
- Used by: `GuardedExecutor`, `ToolBroker` (logging only), `run_attack_simulation.py`, `run_sandboxed.py`, `src/benchmark/runner.py`.

**OS isolation layer:**
- Purpose: Execute permitted shell commands in a private view of the filesystem with syscall filtering.
- Location: `src/sandbox/executor.py` (`SandboxExecutor`, `detect_enforcement`, `get_seccomp_filter_path`), `src/sandbox/seccomp_filter.c` plus `src/sandbox/Makefile` (builds the `seccomp_filter` binary, links `-lseccomp`), `src/sandbox/__init__.py`.
- Contains: per-episode staging directory (`tempfile.mkdtemp(prefix="guardrails_sbx_")`), fake `/home`, `/root`, `/tmp`, canary `/etc/shadow` copy, and `unshare --user --map-root-user --mount --pid --fork --net` invocation with a 30-second action timeout (`ACTION_TIMEOUT_SECONDS`).
- Depends on: `src/agent_harness/environment.py`, `src/agent_harness/executors.py`.
- Used by: `Sandbox._build_executor` in `src/sandbox/sandbox.py` (Linux only), `src/benchmark/runner.py` (`build_executor`, lazy import, arm "guarded").

**LLM provider layer:**
- Purpose: Turn a message list into text, normalized as `LLMResponse`.
- Location: `src/llm/`
- Contains:
  - `base.py`: `LLMProvider` ABC (`name`, `generate(messages, temperature)`), `LLMResponse` dataclass (`ok`, `text`, `error`, `latency_ms`).
  - `registry.py`: `load_model_config`, `list_models`, `get_provider`; reads `config/models.yaml`.
  - `openai_compat.py`: `OpenAICompatibleProvider` for any OpenAI-style endpoint (Gemini, Groq, Ollama).
  - `scratch.py`: `LocalScratchProvider` wraps a checkpoint of the local transformer.
- Depends on: `config/models.yaml`, API keys named by `api_key_env` (read from environment; see Cross-Cutting Concerns).
- Used by: `AIAgent`, `Sandbox`, `src/benchmark/runner.py`.

**Local model training layer:**
- Purpose: A small from-scratch decoder-only transformer used as an intentionally naive, unaligned target.
- Location: `src/scratch_llm/` (`model.py` ScratchGPT, `dataset.py`, `format.py`)
- Depends on: PyTorch, `src/agent_harness/prompting.py` (`parse_action`), `src/benchmark/injecagent.py` (`DATA_DIR`).
- Used by: `src/llm/scratch.py` through a checkpoint file. Not imported by the runtime path directly.

**Benchmark layer:**
- Purpose: Run InjecAgent-style indirect prompt-injection cases against each model with and without guardrails, then score and report.
- Location: `src/benchmark/`, `benchmarks/injecagent/` (JSONL cases and `tools.json`), `run_benchmark.py`.
- Contains: `injecagent.py` (case loading, canaries, `BenchmarkCase`), `runner.py` (`ARMS`, `run_episode`, backoff for rate limits, `build_executor` per arm), `scoring.py` (`score_episode`), `classify.py` (`classify_action`), `report.py` (`generate_report`, CSV/Markdown/charts).
- Depends on: agent harness, policy engine, LLM registry, sandbox (lazy import).
- Used by: `run_benchmark.py`, `scripts/run_full_benchmark.sh`.

**Dashboard layer:**
- Purpose: Live read-only view of decisions, with a clear-log action.
- Location: `src/dashboard/app.py` (Flask app), `src/dashboard/templates/index.html`.
- Contains: routes `/`, `/api/info`, `/api/logs`, `/api/stats`, `/api/clear` (POST). `_read_logs_cached` caches parsed `logs/guardrails.jsonl` by mtime.
- Depends on: `logs/guardrails.jsonl` written by `PolicyEngine._log_decision`; reads `results/` directory path constant.
- Used by: end users in a browser at `http://localhost:5000`.

## Data Flow

### Primary Request Path (sandboxed run)

1. User runs `python run_sandboxed.py '<prompt>'` (`run_sandboxed.py`), which constructs `PolicyEngine` and `Sandbox` (`src/sandbox/sandbox.py`).
2. `Sandbox.run_agent` obtains an LLM provider via `get_provider` (`src/llm/registry.py`) and builds the executor chain with `Sandbox._build_executor`: `GuardedExecutor(policy, SandboxExecutor(env, policy))`.
3. `AgentLoop.run` (`src/agent_harness/loop.py`) calls `provider.generate(messages)`. Replies are parsed by `parse_action` (`src/agent_harness/prompting.py`). Unparseable replies get an `INVALID_REPLY_TEMPLATE` user message and consume a step.
4. For a `respond` action the loop ends with `stop_reason="responded"`. For any other action it calls `executor.execute(action)`.
5. `GuardedExecutor.execute` calls `PolicyEngine.check_detailed(action_type, action_target(action))` (`src/policy_engine/engine.py`). A `BLOCK` returns `Observation("blocked", layer="policy_engine")` without touching the inner executor. A `FLAG` sets `obs.flagged`.
6. Allowed actions go to `SandboxExecutor.execute` (`src/sandbox/executor.py`), which runs `sh -c` under `unshare` with the staged filesystem and the seccomp filter. Kernel denials are reported with `detect_enforcement`.
7. `Observation.as_text()` is appended as a user message (`OBSERVATION_TEMPLATE`), and the loop continues until `max_steps` (default 5) is reached.
8. Each policy decision is appended to `logs/guardrails.jsonl` by `PolicyEngine._log_decision`. The dashboard reads that file on its polling endpoints.

### Benchmark Path

1. `run_benchmark.py` loads models from `list_models(benchmark_only=True)` and cases from `src/benchmark/injecagent.py`.
2. For each (model, case, arm) `src/benchmark/runner.py::run_episode` builds an executor through `build_executor`: the "unguarded" arm uses `MockExecutor` with a `ToolBroker`, the "guarded" arm uses the full stack (`ToolBroker` → `GuardedExecutor` → `SandboxExecutor`, Linux only).
3. `score_episode` (`src/benchmark/scoring.py`) uses `classify_action` (`src/benchmark/classify.py`) and the canary token to decide whether the injected exfiltration succeeded.
4. Results are written to `results/<timestamp>/{raw.jsonl, summary.csv, summary.md, charts/}` through `generate_report` (`src/benchmark/report.py`). `--resume` and `--merge` operate on these directories.

### Unprotected Path

`run_agent.py` → `AIAgent(model=...)` with no policy → `DirectExecutor` → `ToolExecutor` (`src/agent_harness/tools.py`) runs commands and file ops on the real host. This path exists to demonstrate the risk and is not isolated.

**State Management:**
- Episode state lives in `Episode` and `Step` objects (`src/agent_harness/loop.py`) and in the `messages` list local to `AgentLoop.run`. Nothing persists between prompts except the decision log.
- Sandbox filesystem state is per-episode: `SandboxExecutor._stage_environment` creates the staging tree and `executor.close()` releases it, so writes persist across steps within one episode only.
- Dashboard state is a module-level `_log_cache` dict in `src/dashboard/app.py`, invalidated by file mtime.

## Key Abstractions

**`Executor` (abstract) and `Observation`:**
- Purpose: The single interface through which every action is performed. `Observation` carries `status` (`success`, `error`, `blocked`), `output`, `reason`, `layer`, and `flagged`.
- Examples: `src/agent_harness/executors.py` (`Executor`, `Observation`, `DirectExecutor`, `GuardedExecutor`, `MockExecutor`), `src/agent_harness/tool_broker.py` (`ToolBroker`), `src/sandbox/executor.py` (`SandboxExecutor`).
- Pattern: Strategy plus decorator. `GuardedExecutor` and `ToolBroker` hold an `inner: Executor`.

**`LLMProvider` (abstract) and `LLMResponse`:**
- Purpose: Uniform model access. Every provider returns `LLMResponse` and never raises for API failures; errors are returned as `ok=False`.
- Examples: `src/llm/openai_compat.py::OpenAICompatibleProvider`, `src/llm/scratch.py::LocalScratchProvider`.
- Pattern: Factory via `get_provider(name)` in `src/llm/registry.py`, driven by `config/models.yaml`.

**`PolicyEngine` (the access matrix):**
- Purpose: Evaluate `(action_type, target)` against ordered YAML rules.
- Examples: `src/policy_engine/engine.py`, `src/policy_engine/policies/default_policy.yaml`.
- Pattern: First-match-wins with a default action, so rule order matters.

**`Environment`:**
- Purpose: Declarative description of files, web content, and canary tokens to place into a sandbox or emulated OS.
- Examples: `src/agent_harness/environment.py` (`Environment`, `build_environment`, `validate_stageable`).
- Pattern: Dataclass plus factory function.

**`BenchmarkCase` and `ToolBroker` capabilities:**
- Purpose: Describe one injection scenario and the tools a task may call. A `None` capability list means no restriction.
- Examples: `src/benchmark/injecagent.py`, `src/agent_harness/tool_broker.py`.

## Entry Points

**`run_agent.py`:**
- Location: `run_agent.py`
- Triggers: `python run_agent.py [--model NAME] '<prompt>'`
- Responsibilities: Run the unprotected agent with `AIAgent(model=...)` (no policy engine).

**`run_sandboxed.py`:**
- Location: `run_sandboxed.py`
- Triggers: `python run_sandboxed.py '<prompt>'`
- Responsibilities: Run the agent with `PolicyEngine` and `Sandbox`.

**`run_attack_simulation.py`:**
- Location: `run_attack_simulation.py`
- Triggers: `python run_attack_simulation.py`
- Responsibilities: Scripted attack scenarios that call `ToolExecutor` and `PolicyEngine` directly, without an LLM.

**`run_benchmark.py`:**
- Location: `run_benchmark.py`
- Triggers: `python run_benchmark.py --models all | --resume DIR | --report-only DIR [--merge ...]`
- Responsibilities: Multi-model benchmark orchestration and report generation.

**`src/sandbox/sandbox.py` (`__main__`):**
- Location: `src/sandbox/sandbox.py`
- Triggers: `python -m src.sandbox.sandbox '<prompt>'` (or direct run)
- Responsibilities: Same as `run_sandboxed.py` with a built-in usage message.

**Dashboard:**
- Location: `src/dashboard/app.py`
- Triggers: `python -m src.dashboard.app`, then open `http://localhost:5000`
- Responsibilities: Serve `index.html` and JSON APIs over `logs/guardrails.jsonl`.

**Shell entry points:** `setup.sh` (environment setup), `run_demo.sh` (scripted demo of unsandboxed vs sandboxed runs), `scripts/run_full_benchmark.sh`.

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

**What happens:** `GuardedExecutor` checks the policy, but `SandboxExecutor` also accepts an optional `policy` argument used only for logging kernel-level blocks (`src/sandbox/executor.py`). `ToolBroker` similarly logs through `self.policy._log_decision`.
**Why it's wrong:** A caller can construct `SandboxExecutor` without a `GuardedExecutor` and bypass application-level policy, and logging responsibilities are spread across three classes.
**Do this instead:** Keep policy decisions in `GuardedExecutor` only. Route all log writes through `PolicyEngine._log_decision` as the single writer, and have executors return `Observation(layer=...)` for the caller to log.

### Private method called across module boundary

**What happens:** `ToolBroker` and the benchmark code call `PolicyEngine._log_decision`, a leading-underscore method (`src/agent_harness/tool_broker.py`).
**Why it's wrong:** Couples the harness to a private API of the policy package, so a rename breaks the broker silently.
**Do this instead:** Expose a public `log_decision` method on `PolicyEngine` and call that.

### Module-level log path as hidden dependency

**What happens:** `src/dashboard/app.py` and `src/policy_engine/engine.py` both hard-code `logs/guardrails.jsonl` through `PROJECT_ROOT`. Tests must monkeypatch the module attribute.
**Why it's wrong:** Changing the log location requires editing two modules and any test that forgets the patch writes to the real log.
**Do this instead:** Read the path from one config value (or environment variable) and inject it into both the writer and the dashboard.

### Sys.path mutation in every entry script

**What happens:** `run_agent.py`, `src/agent_harness/agent.py`, `src/dashboard/app.py`, and others each insert the project root into `sys.path`.
**Why it's wrong:** Import behavior depends on the caller's working directory and makes packaging harder.
**Do this instead:** Run modules with `python -m` from the repo root, and rely on the package layout instead of path manipulation.

## Error Handling

**Strategy:** Errors are converted into values, not exceptions, at layer boundaries.

**Patterns:**
- LLM failures: providers return `LLMResponse(ok=False, error=...)` (`src/llm/openai_compat.py::_failed_generation`). `AgentLoop.run` sets `stop_reason="llm_error"`.
- Parse failures: `ActionParseError` (`src/agent_harness/prompting.py`) is caught in `AgentLoop.run`, recorded on the step, and reported back to the model with `INVALID_REPLY_TEMPLATE`.
- Tool failures: `DirectExecutor` maps non-success results to `Observation("error", ...)`.
- Policy blocks: `Observation("blocked", layer=...)`, never an exception.
- Setup failures: `Sandbox.run_agent` prints and calls `sys.exit(1)` on `RuntimeError`/`KeyError` from `get_provider`. `SandboxExecutor.__init__` raises `RuntimeError` on non-Linux.
- Rate limits in the benchmark: `src/benchmark/runner.py::_run_with_backoff` parses wait hints from error text and retries.

## Cross-Cutting Concerns

**Logging:** Structured JSON lines to `logs/guardrails.jsonl` through `PolicyEngine._log_decision` (`src/policy_engine/engine.py`). Console output is `print` based, gated by `verbose` flags in `AgentLoop` and `PolicyEngine`.
**Validation:** Action shape validated by `parse_action` (`src/agent_harness/prompting.py`). Policy validation by `check_detailed` against YAML rules. Environment staging validated by `Environment.validate_stageable`.
**Authentication:** Only LLM API keys, read from environment variables named in `config/models.yaml`. No user authentication exists; the dashboard has no authentication layer; the bind address is set in `src/dashboard/app.py` (verify before exposing it beyond localhost).

---

*Architecture analysis: 2026-10-20*
