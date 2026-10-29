"""
Deliverables — one command from the Phase 4 runs to every Phase 5 result number
===============================================================================
The Phase 5 report, charts, slide deck, README, SDD and viva pages all quote results.
This package makes them quote the same results:

  freeze   copies the nine Phase 4 directories and the two legacy October directories,
           read-only and complete lines only, into results/phase5-final/sources/ and
           records the snapshot time, the data state and each copy's sha256 (D-01)
  report   regenerates summary.md / summary.csv / layer_blocks.csv through
           src/benchmark/report.py from the frozen copies only (D-04)
  numbers  writes numbers.json, the one source of numbers every document reads (D-03)
  blocks   writes generated markdown between marker lines in the tracked docs
  check    fails when a document carries a number that is not in numbers.json, or
           when any deliverable carries key-shaped text

Why one command: the Groq re-run keeps appending until the D-09 freeze, so every
deliverable must be regenerable from a fresh snapshot, and a number typed by hand
would silently drift from the data. The legacy October rows were produced by an older
engine; they are frozen for their own labelled table only and are never ranked,
pooled or charted (D-02).

This package never imports src.llm, src.benchmark.jobs or src.benchmark.run_checks:
each of those loads the provider registry (and with it the API keys), which the
deliverables do not need.
"""

import os

RESULTS_ROOT = "results"
FROZEN_DIR = os.path.join(RESULTS_ROOT, "phase5-final")

# Mirrors src.benchmark.jobs.FREEZE_UTC (pinned by a test that reads the constant from the source)
FREEZE_UTC = "2026-10-29T15:30:00+00:00"

# Mirror src.benchmark.jobs.HOSTED_COMPLETE and HOSTED_TERMINAL (pinned by a test that reads the source): a hosted
# model in a terminal state is never relaunched, so its directory gets no new row (CR-03)
HOSTED_COMPLETE = ("complete", "complete_with_problems")
HOSTED_TERMINAL = HOSTED_COMPLETE + ("stopped_by_quota", "stopped_at_freeze", "stalled", "unavailable")
HOSTED_STATE_FILE = "hosted_state.json"

# Mirrors src.benchmark.run_checks.HOSTED_DS_EXPECTED: 20 ds cases x 3 runs x 2 arms (pinned by a test)
DS_EXPECTED = 120

# The current-engine Phase 4 directories, in report order; the scripted control last
CURRENT_SOURCES = (
    "phase4-qwen2.5-0.5b-agentlora",
    "phase4-scratch-gpt",
    "phase4-qwen2.5-0.5b-base",
    "phase4-ollama-qwen2.5-3b",
    "phase4-ollama-llama3.2-3b",
    "phase4-groq-gpt-oss-120b",
    "phase4-groq-gpt-oss-20b",
    "phase4-groq-qwen3.8-27b",
    "phase4-control",
)

# (frozen name, origin under the results root): October rows of an older engine, legacy table only
LEGACY_SOURCES = (
    ("legacy-gemini-flash-lite", "final-20261011/gemini-flash-lite"),
    ("legacy-gemma-4-31b", "final-20261011/gemma-4-31b"),
)

# Fixed headline order (never a ranking): five local models, then the three hosted ones
MODEL_ORDER = (
    "qwen2.5-0.5b-agentlora",
    "scratch-gpt",
    "qwen2.5-0.5b-base",
    "ollama-qwen2.5-3b",
    "ollama-llama3.2-3b",
    "groq-gpt-oss-120b",
    "groq-gpt-oss-20b",
    "groq-qwen3.8-27b",
)
LOCAL_MODELS = MODEL_ORDER[:5]
HOSTED_MODELS = MODEL_ORDER[5:]

CONTROL_MODEL = "control-always-comply"

# Files copied from each source directory when present (raw.jsonl is required)
COPY_FILES = ("raw.jsonl", "manifest.json", "hosted_state.json", "ollama_show.txt")
