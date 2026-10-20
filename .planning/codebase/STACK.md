---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# Technology Stack

**Analysis Date:** 2026-10-20

## Languages

**Primary:**
- Python 3.10+ - All application code: agent harness (`src/agent_harness/`), policy engine (`src/policy_engine/`), LLM layer (`src/llm/`), benchmark (`src/benchmark/`), sandbox wrapper (`src/sandbox/sandbox.py`, `src/sandbox/executor.py`), dashboard (`src/dashboard/app.py`), from-scratch model (`src/scratch_llm/`), and entry points (`run_agent.py`, `run_sandboxed.py`, `run_benchmark.py`, `run_attack_simulation.py`)

**Secondary:**
- C - Seccomp-BPF syscall filter: `src/sandbox/seccomp_filter.c` (built via `src/sandbox/Makefile`, links `-lseccomp`)
- Bash - Setup and demo scripts: `setup.sh`, `run_demo.sh`, `scripts/run_full_benchmark.sh`
- HTML/CSS/JS - Dashboard UI: `src/dashboard/templates/index.html`
- YAML - Model registry: `config/models.yaml`
- JSON / JSONL - Logs and benchmark data: `logs/guardrails.jsonl`, `benchmarks/injecagent/*.jsonl`, `benchmarks/injecagent/tools.json`
- Google Apps Script - Slide export helper: `docs/create_google_slides.gs` (not part of the runtime)

## Runtime

**Environment:**
- CPython 3.10+ (documented in `README.md` and `PRODUCT.md`; no `.python-version`, `pyproject.toml`, or `setup.py` present)
- Target OS: Linux, or WSL2 Ubuntu. The sandbox requires Linux user/mount/PID/net namespaces (`unshare`) and libseccomp, so it does not run natively on Windows.
- GCC for building the seccomp filter

**Package Manager:**
- pip with a flat requirements file: `requirements.txt`
- Lockfile: Missing (no `requirements.lock`, `poetry.lock`, or `Pipfile.lock`; all dependencies use `>=` lower bounds)
- A local virtual environment is expected at `venv/` (created by `setup.sh`, git-ignored)

## Frameworks

**Core:**
- Flask >=3.0.0 - Real-time web dashboard, served on `0.0.0.0:5000` with `debug=True` (`src/dashboard/app.py`, template `src/dashboard/templates/index.html`)
- OpenAI Python SDK >=1.40.0 - Single client for every hosted and local model through the OpenAI-compatible chat-completions API (`src/llm/openai_compat.py`)
- PyTorch (version not pinned; installed separately, CUDA build recommended) - Used only by the from-scratch model in `src/scratch_llm/model.py`, `src/scratch_llm/dataset.py`, `src/scratch_llm/format.py`, and `src/llm/scratch.py`

**Testing:**
- pytest >=8.0.0 - Test runner. Tests live in `tests/` (`conftest.py`, `test_agent_loop.py`, `test_benchmark.py`, `test_dashboard_clear.py`, `test_llm_layer.py`, `test_policy_engine.py`, `test_sandbox_detection.py`, `test_sandbox_executor.py`). The cache is at `.pytest_cache/`.

**Build/Dev:**
- GCC with `-Wall -Wextra -O2` - Compiles the seccomp filter (`src/sandbox/Makefile`)
- Linux `unshare` (util-linux) - Creates user, mount, PID, and network namespaces (`src/sandbox/executor.py`)
- No linter, formatter, or type checker configuration detected

**Document tooling (not runtime):**
- python-pptx (not in `requirements.txt`; imported by `scripts/generate_presentation.py`, `scripts/modify_template.py`, `scripts/polish_current_deck.py`) - Generates the review slide decks in `docs/`
- marked 12.0.2 and mermaid 11.4.1 loaded from the jsDelivr CDN by `scripts/export_sdd.py` - Renders the SDD HTML export in `docs/build/`

## Key Dependencies

**Critical:**
- `openai` >=1.40.0 - Only network client for LLM calls. Configured with `max_retries=2` and a per-model timeout (default 60 s, `timeout` field in `config/models.yaml`, 180 s for `gemma-4-31b`).
- `python-dotenv` >=1.0.0 - Loads `.env` from the project root in `src/llm/registry.py` and `run_*.py` entry points
- `pyyaml` >=6.0 - Parses `config/models.yaml` in `src/llm/registry.py`
- `flask` >=3.0.0 - Dashboard server

**Infrastructure:**
- `matplotlib` >=3.8.0 - Benchmark charts (imported in the benchmark report code under `src/benchmark/report.py`)
- `torch` (unpinned) - From-scratch model training and inference
- `libseccomp` / `libseccomp-dev` (system package, apt) - Required to build and load `src/sandbox/seccomp_filter`
- Standard library heavily used: `urllib.request` (HTTP tool in `src/agent_harness/tools.py`, paper download in `scripts/download_papers.py`), `subprocess` (shell tool and sandbox execution), `json`, `dataclasses`, `secrets` (canary tokens), `tempfile`, `shlex`

## Configuration

**Environment:**
- Secrets are read from environment variables, loaded from `.env` by `python-dotenv`. The `.env` file exists at the repo root and is git-ignored. Do not read or copy its values.
- Variable names documented in `.env.example`:
  - `GEMINI_API_KEY` - Google AI Studio key, needed by `gemini-flash-lite` and `gemma-4-31b`
  - `GROQ_API_KEY` - Groq console key, needed by the three `groq-*` models
  - `OPENROUTER_API_KEY` - Optional; listed in `.env.example` but no model in `config/models.yaml` currently references it
- Ollama and the scratch model need no key. `OpenAICompatibleProvider` substitutes the placeholder `not-needed` when `api_key_env` is unset.
- Model registry: `config/models.yaml` (fields: `provider`, `base_url`, `api_key_env`, `model`, `timeout`, `benchmark`, `checkpoint`). Default model: `gemini-flash-lite`.
- Dashboard log file: `logs/guardrails.jsonl` (path set in `src/dashboard/app.py`)
- Ignore rules: `.gitignore` excludes `.env`, `.env.*.local`, `venv/`, `.venv/`, `env/`

**Build:**
- `requirements.txt` - Python dependencies
- `setup.sh` - Installs apt packages (`python3-venv`, `gcc`, `libseccomp-dev`, `libseccomp2`), creates `venv/`, installs requirements, compiles the seccomp filter, and creates `logs/`
- `src/sandbox/Makefile` - Builds `seccomp_filter`
- `.gitattributes` - Forces LF line endings for `*.sh`
- No `Dockerfile`, `docker-compose*`, CI workflow (`.github/workflows/` absent), or `pyproject.toml` detected

## Platform Requirements

**Development:**
- Linux or WSL2 Ubuntu (required for the sandbox path; the policy engine, benchmark, and dashboard can run on other platforms)
- Python 3.10+, pip, GCC, `libseccomp-dev`, util-linux `unshare`
- Optional NVIDIA GPU with a CUDA PyTorch build for the scratch model
- Optional local Ollama server on `http://localhost:11434` (model `llama3.2:3b`)

**Production:**
- Not applicable. This is a research prototype and evaluation harness. The dashboard binds to `0.0.0.0:5000` in debug mode and is intended for local demos only.

---

*Stack analysis: 2026-10-20*
