# Phase 3: Qwen LoRA & hf_local Provider - Context

**Gathered:** 2026-10-22
**Status:** Ready for planning
**Source:** Locked project decisions + 03-RESEARCH.md measured day-0 smoke test (YOLO defaults; no discuss-phase)

<domain>
## Phase Boundary

Add an in-process `hf_local` provider, benchmark un-tuned Qwen2.5-0.5B-Instruct as a base control row (smoke run), LoRA fine-tune it locally on the Phase 2 synthetic dataset within 6 GB VRAM, and run the tuned model through the benchmark (smoke run). Full 342-episode runs are Phase 4.
</domain>

<decisions>
## Implementation Decisions

### Data & disclosure
- **D-01:** Reuse the Phase 2 dataset unchanged (train sha a2586548c7f7…, dev 1f6333eb8ac8…) — same 50/25/25 compliance mix; the LoRA model is disclosed as a second injection-compliant worst-case principal (it attacked 8/12 in the probe), alongside the un-tuned base control.
- **D-02:** Leakage gate (`assert_clean`) runs over the data before training, exactly as in Phase 2.

### Rendering & loss
- **D-03:** One Qwen chat sequence per episode built from `prompting.build_messages` (project SYSTEM_PROMPT always supplied), assistant turns = compact JSON actions, observations = user turns via `OBSERVATION_TEMPLATE`, rendered with `apply_chat_template`; serving uses the same path (prefix-stable, tested).
- **D-04:** Loss only on assistant content + `<|im_end|>`, via a character-offset mask (never by token id); LM head computed only on supervised positions (measured 2.4 GiB peak vs 8.3 GiB with HF `labels=`). `max_len` 1024 with an assertion, no truncation.

### Training
- **D-05:** LoRA r=16, alpha=32, all-linear, bf16, gradient checkpointing; lr 2e-4 cosine to 2e-5, 20 warmup steps, grad-clip 1.0; ~600 optimizer steps × 16 episodes (~41 min) on a seeded subset of the train split.
- **D-06:** Select / early-stop on `dev_loss_action` (respond text excluded — Phase 2 CR-01 lesson); pre-registered one-retry rule if full-dev `dev_action_exact` < 0.90, gated on dev metrics only, applied before any smoke run; resumable checkpoints after every eval (Phase 2 pattern) and GPU/free-memory preflight; never touch the other project's processes sharing WSL.

### Serving
- **D-07:** `hf_local` provider: lazy imports, fully offline (`HF_HUB_OFFLINE`, `local_files_only`), explicit greedy kwargs (`do_sample=False`, `repetition_penalty=1.0`) overriding Qwen's sampling `generation_config`, stop on `<|im_end|>`/`<|endoftext|>`, merge the adapter at load (identical outputs, 2.3× faster), stamps `finish_reason` / token counts / `resolved_model = base@rev12+name@adapter_sha12`, context-overflow → recorded error, bad state → `RuntimeError` → runner `[skip]`.
- **D-08:** Config entries `qwen2.5-0.5b-base` and `qwen2.5-0.5b-agentlora`; HF packages pinned in `requirements-train.txt` (transformers 5.20.2, peft 0.21.2, accelerate 1.15.0 — already installed, user-approved).

### Claude's Discretion
- Exact subset selection (first-N vs seeded sample — prefer seeded sample for class balance), eval subset size, test granularity, file layout (e.g. `src/qwen_lora/`), smoke-run args (`--runs 1 --dh 5 --ds 5 --benign 3`, matching Phase 2).
</decisions>

<canonical_refs>
## Canonical References

- `.planning/phases/03-qwen-lora-hf-local-provider/03-RESEARCH.md` — measured facts, load-bearing code, validation architecture
- `.planning/phases/02-local-data-foundation-scratchgpt/02-REVIEW-FIX.md` — selection metric lesson, resumable training, WR-01 scoring
- `src/scratch_llm/{train,provenance}.py`, `src/llm/scratch.py` — patterns to mirror
- `src/agent_harness/prompting.py` — single source of truth for prompts and parse_action
</canonical_refs>

<specifics>
## Specific Ideas

- HF import costs ~65–75 s cold on the /mnt/c venv: keep HF imports lazy and HF-touching tests few (a `STRICT_HF=1` conftest switch, like `STRICT_TORCH`).
- Base model parse rate ~45% in the probe — a reported finding (format_failure rows), not something to fix.
- Viva page: docs/viva/03-* (pretraining vs fine-tuning, LoRA W+BA math with this model's numbers: 8,798,208 / 502,830,976 = 1.75% trainable, chat templates & special tokens, loss masking, bf16, gradient checkpointing, greedy decoding, base-vs-tuned comparison).
</specifics>

<deferred>
## Deferred Ideas

- GGUF/Ollama export of the tuned model — v2 (SERVE-V2-01).
- Full 342-episode runs of both Qwen rows — Phase 4.
</deferred>

---

*Phase: 03-qwen-lora-hf-local-provider*
*Context gathered: 2026-10-22*
