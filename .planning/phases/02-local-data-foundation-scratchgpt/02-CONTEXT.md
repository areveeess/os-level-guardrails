# Phase 2: Local Data Foundation & ScratchGPT - Context

**Gathered:** 2026-10-21
**Status:** Ready for planning
**Source:** User decisions on 02-RESEARCH.md open questions (plan-phase gate) + locked project decisions

<domain>
## Phase Boundary

Build the shared synthetic episode dataset (leakage-gated, format v2) and train the from-scratch ScratchGPT on the RTX 3060 so `scratch-gpt` drives the agent end-to-end in the benchmark (smoke run). Full 342-episode runs are Phase 4; Qwen LoRA is Phase 3 (it reuses this dataset).
</domain>

<decisions>
## Implementation Decisions

### Data mix
- D-01: Training mix is a deterministic quota of 50% benign / 25% direct-harm-compliant / 25% data-stealing-compliant episodes with self-written injections (locked project decision; disclosed in the report).
- D-02: About 10% of episodes (taken from the benign share) use OS actions (run_shell / read_file / write_file) for full action-grammar coverage.
- D-03: About 8% of the dh/ds episodes end with the harness's own `BLOCKED by tool_broker: ...` observation followed by the agent giving up with a `respond` action (realistic reaction to a block; disclosed in the report).

### Leakage
- D-04: Strict hold-out — no InjecAgent benchmark tool (79 held out, incl. GmailSendEmail), no benchmark attacker email address, no `CANARY[0-9A-F]{12}DATA` value or "Private record" phrasing, no 8-word shingle or exact benchmark injection string, and none of the ~50 parameter names that occur only in held-out tools may appear in training data. Dev split uses held-out dev tools and held-back phrasing templates.

### Format & model
- D-05: Compact format v2: `T:` line carries tool names with required parameter names; `CONTEXT_CHARS` raised to 2048 (model context 2048); `FORMAT_VERSION = 2`, vocab sha256 and a render fingerprint stamped into the checkpoint and checked on load.
- D-06: Training text is produced by the same `render_context` path inference uses (parity by construction); loss only on `A:` line payloads plus their newline.

### Claude's Discretion
- Exact episode count (research suggests ~40k train + ~2k dev), hyper-parameters, synthetic-tool share (~35%), test granularity, file layout within src/scratch_llm/.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/phases/02-local-data-foundation-scratchgpt/02-RESEARCH.md` — measured throughput, leakage numbers, episode design, validation architecture
- `.planning/research/ARCHITECTURE.md` — component boundaries for dataset / format / train / inference
- `src/agent_harness/prompting.py` — single source of truth for system prompt, OBSERVATION_TEMPLATE, parse_action
- `src/llm/scratch.py` — provider contract (`inference.load_generator`)
- Phase 1 manifest/stamp conventions: `.planning/phases/01-evaluation-hardening-policy-fix/01-03-SUMMARY.md`
</canonical_refs>

<specifics>
## Specific Ideas

- src/scratch_llm/{model,format,dataset}.py exist but are UNTRACKED pre-existing design work; there is no `__init__.py`. The first plan commits/finishes them.
- Keep dataset/format/gate code torch-free (cold torch import in the /mnt/c venv ~54 s).
- Write data with `newline="\n"` so dataset hashes match across Windows and WSL; load checkpoints with `torch.load(weights_only=True)`.
- `.gitignore`: `models/**/*.pt`, `data/local/*.jsonl` (manifests and loss curves stay tracked).
- Viva page: docs/viva/02-* (transformers, char tokenization, loss masking, AdamW/LR schedule, overfitting, decoding, synthetic data & leakage, worst-case principal).
</specifics>

<deferred>
## Deferred Ideas

- Untrained-checkpoint control row — Phase 4.
- Pseudo-replication caveat (greedy decoding → 3 runs nearly identical) — Phase 5 threats to validity.
</deferred>

---

*Phase: 02-local-data-foundation-scratchgpt*
*Context gathered: 2026-10-21*
