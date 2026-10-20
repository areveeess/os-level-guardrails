---
last_mapped_commit: 94cffd318d559d0d6afc2bec044c79d6fd0533d3
last_mapped_at: 2026-10-20
---
# External Integrations

**Analysis Date:** 2026-10-20

## APIs & External Services

**LLM providers (all via the OpenAI-compatible chat-completions API):**
- Google Gemini API (OpenAI-compatible endpoint) - Hosts `gemini-3.5-flash-lite` (`gemini-flash-lite`, default) and `gemma-4-31b-it` (`gemma-4-31b`)
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/` (`config/models.yaml`)
  - SDK/Client: `openai` package, `OpenAI` class in `src/llm/openai_compat.py`
  - Auth: `GEMINI_API_KEY` environment variable
- Groq API - Hosts `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` (`groq-gpt-oss-120b`, `groq-gpt-oss-20b`, `groq-qwen3.8-27b`)
  - Base URL: `https://api.groq.com/openai/v1` (`config/models.yaml`)
  - SDK/Client: `openai` package, `src/llm/openai_compat.py`
  - Auth: `GROQ_API_KEY` environment variable
  - Note: `src/llm/openai_compat.py` special-cases Groq `tool_use_failed` / `output_parse_failed` 400 errors and scores the `failed_generation` text as the model reply
- Ollama (local) - Serves `llama3.2:3b` (`ollama-llama3.2-3b`)
  - Base URL: `http://localhost:11434/v1` (`config/models.yaml`)
  - Auth: none; the placeholder key `not-needed` is used
- OpenRouter - Listed as optional in `.env.example` (`OPENROUTER_API_KEY`, sign-up at openrouter.ai/keys). No model entry in `config/models.yaml` uses it, and no code path reads it.
- Local from-scratch model (not a network service) - `scratch-gpt`, checkpoint `models/scratch/scratch_gpt.pt`, loaded by `src/llm/scratch.py`. The checkpoint path does not exist in the repo and must be produced by `python -m src.scratch_llm.train`.

**Arbitrary HTTP access (intentional attack surface, not a managed integration):**
- `src/agent_harness/tools.py` - `http_request` tool performs unrestricted `urllib.request.urlopen` GET calls with a 10 s timeout. The policy engine (`src/policy_engine/engine.py`) and the sandbox (`src/sandbox/executor.py`, network namespace) are the controls for this path.
- `run_attack_simulation.py` - Uses fictional hosts (`http://evil.example`, `http://malware.example`) as attack payloads. No real requests are expected to succeed.

**Academic and dataset sources (build-time only):**
- arXiv API - `http://export.arxiv.org/api/query` and `https://arxiv.org/pdf/...` used by `scripts/download_papers.py` to fetch PDFs into `papers/`
- InjecAgent - Vendored data in `benchmarks/injecagent/` (pinned to upstream commit `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`, see `benchmarks/injecagent/SOURCE.md`). No runtime fetch.

**CDN resources (browser-side, build/export only):**
- jsDelivr - `marked@12.0.2` and `mermaid@11.4.1` in `scripts/export_sdd.py` (SDD HTML export)
- Google Fonts - `Fira Code` and `Fira Sans` loaded by `src/dashboard/templates/index.html`

## Data Storage

**Databases:**
- None. No database client, ORM, or database connection string is present.

**File Storage:**
- Local filesystem only
  - Audit and event log: `logs/guardrails.jsonl` (written by `src/dashboard/app.py` and the agent loop; cleared by `run_demo.sh` and the dashboard clear action)
  - Benchmark results: `results/` (untracked in git; contents not inspected)
  - Canary and fake-home files: created in per-run temp directories by `src/agent_harness/environment.py` and `src/sandbox/executor.py`
  - Scratch model checkpoint: `models/scratch/scratch_gpt.pt` (expected location, not committed)
  - Documents: `docs/SDD.md`, `docs/build/SDD.html`, `docs/SDD.pdf`, `docs/SDD.docx`

**Caching:**
- None. `.pytest_cache/` is a test artifact only.

## Authentication & Identity

**Auth Provider:**
- None for end users. The dashboard has no login.
- Provider API keys only, read from environment variables loaded from `.env` by `python-dotenv` (`src/llm/registry.py`). Keys are passed to the `OpenAI` client constructor in `src/llm/openai_compat.py`; a missing key raises `RuntimeError` naming the variable.
- Agent identity inside the sandbox is simulated: fake user home, fake `/etc/shadow`, and canary tokens generated with `secrets` (`src/agent_harness/environment.py`).

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, or similar SDK).

**Logs:**
- Structured JSON Lines written to `logs/guardrails.jsonl` for every policy decision
- Dashboard reads that file through JSON endpoints in `src/dashboard/app.py` (`/api/info`, `/api/logs`, `/api/stats`, POST `/api/clear`); the page at `/` (`src/dashboard/templates/index.html`) fetches these endpoints
- Benchmark runs write per-episode `raw.jsonl` and report files under `results/` (`src/benchmark/runner.py`, `src/benchmark/report.py`)
- Console output via `print` in CLI entry points

## CI/CD & Deployment

**Hosting:**
- Not applicable. Runs locally from the developer's machine or WSL2.

**CI Pipeline:**
- None detected (no `.github/workflows/`, `.gitlab-ci.yml`, or similar)

## Environment Configuration

**Required env vars (by model choice):**
- `GEMINI_API_KEY` - required for the default model `gemini-flash-lite` and `gemma-4-31b`
- `GROQ_API_KEY` - required for the `groq-*` models
- `OPENROUTER_API_KEY` - optional, unused by current code

**Optional/none:**
- Ollama and `scratch-gpt` need no key.

**Secrets location:**
- `.env` at the repo root (git-ignored via `.gitignore`). Template: `.env.example`, which is safe to commit. Setup instructions: `README.md`.
- No secret manager or vault integration.

## Webhooks & Callbacks

**Incoming:**
- None. The Flask dashboard exposes only its own UI and JSON endpoints on port 5000; it does not accept webhooks.

**Outgoing:**
- None. The only outbound traffic is LLM API calls and the agent's own `http_request` tool, which is the subject of the security evaluation.

---

*Integration audit: 2026-10-20*
