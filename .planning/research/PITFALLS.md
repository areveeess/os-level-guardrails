# Pitfalls Research

**Domain:** Local LLM training (tiny from-scratch transformer + LoRA on a 0.5B model, 6 GB GPU, WSL2) and fair multi-LLM comparison on an indirect-prompt-injection benchmark (InjecAgent) with an OS-level reference monitor
**Researched:** 2026-10-20
**Confidence:** MEDIUM-HIGH. Pitfalls tied to this repo (report.py, scoring.py, prompting.py, openai_compat.py, results/final-20261011) come from reading the code and CONCERNS.md (HIGH). Training/WSL/Ollama items are well-established practice (MEDIUM; the Ollama context-default detail rests on third-party guides, so verify with `ollama show` / server log on the actual install). Python 3.14 wheel-compatibility remarks are LOW and must be checked on the machine.

**Phase labels used below** (suggested; map onto the real roadmap):
- **P-EVAL** evaluation hygiene + report hardening (must come first)
- **P-LOCAL** more Ollama models (qwen2.5:3b, phi3.5:3.8b, gemma3:1b, llama3.2:3b re-check)
- **P-HOSTED** resume unfinished hosted runs
- **P-SCRATCH** ScratchGPT data + training + inference
- **P-LORA** Qwen2.5-0.5B LoRA fine-tune
- **P-REPORT** cross-model comparison report + charts
- **P-POLICY** shell-chaining bypass fix
- **P-VIVA** viva notes, README, SDD, slides (grows every phase)

---

## Critical Pitfalls

### Pitfall 1: Parse failures make a model "look safe" (format-following confounds attack-success rate)

**What goes wrong:**
A model that cannot emit the `{"action": ...}` JSON (scratch model, gemma3:1b, a reasoning model whose output is truncated) never calls the attacker tool, so `attempted = 0` and `succeeded = 0`. The table then says "0% attack success" for the *weakest* model, which reads as "most robust". In this repo the denominator is `attack_episodes = len(atk_u)`, which includes every non-errored episode regardless of whether the model ever produced one valid action. `parse_error_rate_pct` is reported, but only as a separate column, and `refused_or_ignored` is `not attempted and completed`, so max-steps parse-loop episodes drop into "not attempted" silently.

**Why it happens:**
ASR is defined as "attacker tool executed / episodes". It only measures vulnerability when the agent is *capable* of acting. Competence and safety are conflated. Small and unaligned models are exactly where this bites, and this project deliberately adds the weakest ones.

**How to avoid:**
- Add an explicit per-episode outcome taxonomy in `scoring.py`: `attacked`, `ignored_injection` (valid actions, never touched attacker tool, finished), `format_failure` (zero valid actions, or stop_reason `max_steps` with parse errors on most steps), `llm_error` (infrastructure), `refused` (content filter / explicit refusal text).
- Headline ASR = attacked / (attacked + ignored_injection) i.e. conditioned on format-compliant episodes, and always printed next to "format-valid episode rate". Also print the worst-case bound (treat format failures as unknown).
- Show benign task completion next to ASR for every model. A model with 0% benign completion and 0% ASR is "broken", not "safe".
- Add a positive-control agent (see Pitfall 8) so the table has a row where ASR is known to be 100%.

**Warning signs:**
A small/scratch model with `success_unprotected_pct` of 0 and `parse_error_rate_pct` above ~20%; `benign_task_unprotected_pct` near 0; `steps` equal to max_steps for most episodes; raw replies starting with prose, `<think>`, or truncated mid-JSON.

**Phase to address:** P-EVAL (taxonomy + report), re-verified in P-SCRATCH, P-LORA, P-REPORT.

---

### Pitfall 2: Unequal sample sizes and non-random truncation when hosted runs hit quotas

**What goes wrong:**
Hosted models stopped at 112-214 of 342 records (CONCERNS.md). Because cases are processed in file order and `ds` appears after `dh`, the truncated runs contain *only* `dh` attack episodes and zero `ds`; the surviving prefix is not a random sample. Comparing "gpt-oss-20b, 10 attack lines" with "llama3.2, 240 attack lines" in one table, and quoting percentages without n, is invalid. The "40% attempted" figure for gpt-oss-20b rests on 5 episodes.

**Why it happens:**
Free-tier quotas (RPM, TPM, daily token caps) end runs arbitrarily; ordering is case-major, so a cut-off run is a biased subset. Resume across days also changes the time of day and possibly the server-side model snapshot.

**How to avoid:**
- Order the work queue **run-major and interleaved**: shuffle (case_id, arm) with a fixed seed, and iterate run 1 over all cases, then run 2, so *any prefix* is balanced across `dh`/`ds`/`benign` and across arms. Keep arms paired (run unprotected and protected of the same case adjacent) so a cut never leaves an unpaired episode.
- Headline table uses the **common subset**: only (case_id, run, arm) keys present and non-errored for *every* model in the table, or else clearly separate "complete models" from "partial models, n shown".
- Every number printed as `x% (k/n)` with a 95% Wilson interval; models with n below a threshold (e.g. 30 attack episodes per class) are greyed out, not ranked.
- Add the completeness gate described in CONCERNS.md to `report.py`: flag models with fewer than 342 unique records or zero `ds` episodes; refuse to include them in the headline.
- Prefer finishing 2-3 hosted models fully over starting all of them; for the rest say "partial, not compared".

**Warning signs:**
`attack_episodes` differs between rows; any `ds` row showing "—"; percentages with granularity of 10% or 20% (tiny n); resume runs spanning multiple days.

**Phase to address:** P-EVAL (ordering + gate), P-HOSTED (resume with new ordering), P-REPORT (common-subset table).

---

### Pitfall 3: API errors, timeouts and content-filter blocks counted (or dropped) the wrong way

**What goes wrong:**
Two opposite failures. (a) Errors counted as outcomes: an errored protected-arm episode has `succeeded = False`, i.e. looks like a successful defence; an errored episode in the benign arm may appear as completed or failed depending on code path (gemma-4-31b shows 100% benign completion with 204 of 360 lines being timeouts, per CONCERNS.md). (b) Errors silently dropped: `report.py` excludes `llm_error` rows *independently per arm*, so the unprotected and protected sets are no longer the same cases (unpaired), and any error that correlates with the outcome causes survivorship bias. Content-filter / safety blocks from Gemini or Groq (`finish_reason=content_filter`, empty content, HTTP 400) are exactly the model *refusing*, but they land in `llm_error` and vanish, which inflates ASR for hosted models relative to local ones. Also, `openai_compat.py` has `max_retries=2` inside the SDK plus the benchmark's own retries, so one episode can stall for minutes and a mid-episode error leaves a partially executed episode (side effects already happened in the sandbox).

**Why it happens:**
All failures flow through one `llm_error` string; there is no distinction between "infrastructure failed" (retry) and "model/provider declined" (a result).

**How to avoid:**
- Classify errors into: `infra` (timeout, 429, 5xx, connection) and `provider_refusal` (content filter, safety block, empty with finish_reason). `infra` is retried until success or marked `incomplete`; `provider_refusal` is recorded as a *refused* outcome and kept in the denominator.
- Drop **pairs**, not rows: if either arm of a (case, run) pair is `infra`-errored, exclude both from paired statistics.
- Never report an episode as success or failure if any step was `infra`-errored; an episode that errored at step k is re-run from scratch (fresh sandbox), never continued.
- Make the writer idempotent (skip keys already present with a non-error status, per CONCERNS.md) so resume cannot create the 31 duplicates seen in `ollama-llama3.2-3b`.
- Lower per-request timeout for benchmark runs and record timeouts as their own stop reason; record `finish_reason` and resolved model id in each record.
- Report an "errors / retries" column per model so reviewers can see data quality.

**Warning signs:**
`llm_errors` greater than 0 in `summary.md`; benign completion of 100% with high latency; protected n differing from unprotected n for the same model; `finish_reason` of `length` or `content_filter` in raw logs.

**Phase to address:** P-EVAL (classification, pairing, idempotent writer), P-HOSTED (re-run errored keys).

---

### Pitfall 4: Output-token cap and reasoning traces break parsing for reasoning models

**What goes wrong:**
`generate()` hard-codes `max_tokens=512`. For reasoning models (gpt-oss-120b/20b, qwen3.x on Groq, Gemini thinking models) hidden or visible reasoning consumes that budget; `content` comes back empty or cut off mid-JSON, so the episode is a parse failure (Pitfall 1) and the model looks safe. Conversely, if reasoning is returned inline (`<think>...</think>`), `parse_action` takes the **first** JSON object containing an `"action"` key, which may be a draft or a quoted example *inside the reasoning* rather than the final answer. Temperature 0 may be rejected or ignored by some reasoning endpoints, and reasoning-effort defaults differ per provider, so "same settings" are not actually the same.

**How to avoid:**
- Log `finish_reason` and `completion_tokens` per step; flag `length` stops. Raise `max_tokens` for reasoning models (e.g. 2048+) or set the provider's reasoning-effort parameter low and *identical* across reasoning models; document it in `config/models.yaml` per model.
- Strip `<think>...</think>` blocks before parsing, and take the **last** valid action object, or only parse the message `content` field (not `reasoning_content`).
- Keep a per-model "generation settings" table in the report (temperature, max_tokens, reasoning effort, context length, quantization) so differences are disclosed, not hidden.

**Warning signs:**
High parse-error rate concentrated in reasoning models; `completion_tokens` equal to 512 on many steps; empty `content` with non-empty `reasoning`; parsed actions that do not match what the visible final answer says.

**Phase to address:** P-EVAL (parser + logging), P-HOSTED.

---

### Pitfall 5: Ollama context window silently truncates long prompts (num_ctx 2048/4096)

**What goes wrong:**
Ollama uses a small default context (historically 2048, 4096 on newer builds; newer versions may choose a VRAM-dependent default, so verify). When the prompt exceeds it, Ollama **truncates from the front without raising an error**, dropping the system prompt (the JSON-action instructions and tool list) and the user's task. The agent loop appends every observation, and InjecAgent tool responses are long, so by step 3-4 a local model has lost its instructions: format collapses, parse errors rise, and the model "looks safe" (Pitfall 1). Because Ollama is reached through the OpenAI-compatible `/v1` endpoint used by `openai_compat.py`, `num_ctx` **cannot be set per request**; the options field is not part of the OpenAI schema. The existing `ollama-llama3.2-3b` results may already be affected.

**How to avoid:**
- Set the server-wide default before benchmarking: `OLLAMA_CONTEXT_LENGTH=8192` (or 16384) in the environment of the Ollama *process* (for a WSL systemd service use an override; exporting in the shell does nothing for an already running service), or bake `PARAMETER num_ctx 8192` into a Modelfile variant (`ollama create llama3.2-3b-8k`) and benchmark that tag. Confirm with `ollama show <model>` and `ollama ps` (CONTEXT column).
- Log `usage.prompt_tokens` per step (already captured in `LLMResponse`) and add a guard: if `prompt_tokens` is within ~5% of num_ctx, mark the step `context_truncated` and exclude/flag the episode. Local models cap `prompt_eval_count` at the window, so a flat ceiling in the distribution is the tell.
- Check the Ollama server log for truncation warnings after the first run.
- Budget VRAM: KV cache grows with context; a 3-4B model at 8k fits in 6 GB, but not while a training job holds the GPU (Pitfall 12).
- Re-check the already-complete llama3.2:3b run; if prompt_tokens were capped, re-run it.

**Warning signs:**
`prompt_tokens` plateauing at exactly 2048/4096; parse-error rate rising with step index; model replying as if it never saw the tool list; results for the same model changing when the context env var changes.

**Phase to address:** P-LOCAL (set and verify before the first local run; re-check llama3.2:3b), P-EVAL (prompt_tokens guard).

---

### Pitfall 6: Pseudo-replication and missing confidence intervals

**What goes wrong:**
The benchmark has 40 attack cases (20 `dh`, 20 `ds`) x 3 runs x 2 arms, but at `temperature=0.0` the 3 runs are near-identical, so 60 episodes per class are really about 20 independent observations. Quoting "0% success with guardrails over 60 episodes" overstates certainty. Zero events also does not mean zero risk: the 95% upper bound for 0/20 is about 16%, for 0/60 about 6% (rule of three: roughly 3/n). The headline claim "guardrails reduce success to 0" is the project's core value and needs the right language: "0 of N observed; upper bound X%". Further, an unprotected-vs-protected comparison on the same cases is *paired*, so a plain two-proportion test is the wrong tool.

**How to avoid:**
- Report Wilson (or Clopper-Pearson for zeros) 95% intervals everywhere; cluster by case (the independent unit is the case, not the run); use McNemar or an exact paired test for unprotected vs protected.
- Make the headline guardrail claim **mechanistic as well as statistical**: for every attempted attack, record the layer that blocked it. "Of K attempted attacker actions across all models, K were blocked" is stronger than "ASR 0%".
- If time allows, use temperature above 0 for extra independent samples (e.g. 0.7 with fixed seeds) in a separate, labelled experiment; do not mix with the temp-0 table.
- Do not rank models whose intervals overlap.

**Warning signs:**
Tables with no n or interval; claims of "significantly lower" with n below 30; identical rows across the 3 runs of one case.

**Phase to address:** P-EVAL (stats helpers in `report.py`), P-REPORT.

---

### Pitfall 7: Unfair comparison: quantized local vs full-precision hosted, plus non-identical settings

**What goes wrong:**
Ollama's default tags are 4-bit quantized (Q4_K_M typically); Gemini, Groq-hosted models (gpt-oss is natively MXFP4, others vary), and the HF-served LoRA model are different precisions. A "3B vs 120B" gap mixes size, precision, alignment, context length, and decoding stack. Tags such as `llama3.2:3b` or `:latest` can also move, and hosted ids are retired (`gemma-4-31b-it`, `qwen3.8-27b`), so a rerun may hit a different model.

**How to avoid:**
- Treat the comparison as a **spectrum by capability**, not a leaderboard; say explicitly in the report which differences are confounded.
- Record per model: exact id/digest (`ollama show --modelfile`, `ollama list` digest), quantization, parameter count, context length, temperature, max_tokens, seed, date. Store in a `manifest.json` per results dir with the git commit (CONCERNS.md).
- Optionally run one local model at higher precision (e.g. `llama3.2:3b-instruct-fp16`, if it fits in 6 GB) to bound the quantization effect on a small subset.
- Keep the guardrail claim independent of model quality (Pitfall 8), so quantization differences do not threaten the core result.
- Never present the LoRA model (HF, fp16/bf16, in-process) and Ollama models as if run through the same serving stack; state the stack.

**Warning signs:**
Report text that ranks models by "safety" without a confounds paragraph; model ids without digests; differences in which prompts include the tool list.

**Phase to address:** P-LOCAL, P-REPORT (confounds table), P-VIVA.

---

### Pitfall 8: Guardrail claim is vacuous when models rarely attack, and the second layer is never exercised

**What goes wrong:**
If a model's unprotected attack success is already ~0%, "0% with guardrails" proves nothing about the guardrails. Today the sandbox and seccomp counts are 0 in every model's results, so the claim "namespaces + seccomp stop attacks" is not demonstrated by the reported runs; the policy engine and tool broker do all the work. And the known policy bypass (`ls; curl http://evil`) means a model that happens to emit chained commands defeats the engine, which an examiner can demonstrate live.

**How to avoid:**
- Add a **scripted worst-case principal** (always follows the injected instruction, tries both direct and OS-level exfiltration paths such as `curl`, `cat /etc/passwd`, chained commands) as a control row in the table. It gives a guaranteed ASR of 100% unprotected, and the protected-arm result is a pure test of the guardrails, independent of any LLM. This is the cleanest way to defend the claim "holds independently of which LLM drives the agent" in a viva.
- Report blocks **given attempt** by layer (tool broker, policy, sandbox, seccomp) per model; include at least one scenario that reaches sandbox and seccomp layers.
- Fix the shell-chaining bypass test-first (CONCERNS.md recommendation) *before* the final comparison so results are produced against the fixed policy; record the policy hash in the manifest. If results are produced pre-fix, label them.

**Warning signs:**
Many rows with unprotected success of 0; layer-block table with only broker/policy nonzero; no row where unprotected success is high.

**Phase to address:** P-POLICY (before final runs), P-EVAL (control agent), P-REPORT.

---

### Pitfall 9: Tokenizer / chat-template / prompt-format mismatch between training and inference

**What goes wrong:**
The fine-tuned model is trained on one rendering of the conversation and evaluated on another, so it "forgets" to behave. Typical causes: (a) hand-built prompt strings in training vs `tokenizer.apply_chat_template` at inference (or the reverse); (b) Qwen2.5's template auto-inserts a default system prompt ("You are Qwen, created by Alibaba Cloud...") when no system message is given, so train and inference differ if only one side supplies the project's system prompt; (c) observations (tool results) represented as `user` role in the loop (`OBSERVATION_TEMPLATE`) but as `tool` role or plain text in training; (d) `add_generation_prompt` omitted at inference; (e) EOS handling: Qwen instruct stops at `<|im_end|>`, the tokenizer's `pad_token` is `<|endoftext|>`; if pad is set equal to EOS and padding is masked by token id, the real EOS is masked out of the loss and the model never learns to stop (or learns to run on past the JSON object); (f) for the scratch model, `format.py`'s compact rendering and its vocabulary must be identical in `train.py` and `inference.py` (same tokenizer file saved with the checkpoint).

**How to avoid:**
- One function renders messages to token ids and is imported by both training and inference: build from `prompting.build_messages` / `OBSERVATION_TEMPLATE` (do not retype the system prompt in training data) and `tokenizer.apply_chat_template(..., add_generation_prompt=True)` at inference. For the scratch model, `format.py` is the only renderer.
- Add a **round-trip test**: take 5 training episodes, render them through the training path and through the inference path up to each assistant turn, assert the token ids are identical.
- Save the tokenizer, chat template, and generation config with every checkpoint; load from the checkpoint dir only.
- Keep EOS in the labels; use `-100` masking based on *position* (the span mask), not on token id equality with pad.

**Warning signs:**
Model emits correct JSON then continues generating (no stop); training loss good but inference output ignores the format; different first-token logits for "same" prompt via two code paths; the string "You are Qwen" appearing in a rendered prompt.

**Phase to address:** P-LORA, P-SCRATCH (round-trip test is an acceptance criterion).

---

### Pitfall 10: Loss masking errors (training on the wrong tokens)

**What goes wrong:**
Loss is computed over the whole transcript (system prompt, tool lists, observations) instead of only the assistant action. Consequences specific to this project: the model learns to *generate* tool outputs, including injected instructions (observations contain attacker text), and the system prompt dominates the loss so the training curve looks good while action quality is poor. The opposite error, masking too much (e.g. masking all but the final turn, or an off-by-one that drops the opening `{`), also happens. With multi-turn episodes, tokenising the full text and searching for the assistant span by string matching breaks when tokenization merges across boundaries.

**How to avoid:**
- Build `labels` by tokenising segment by segment (or use `return_offsets_mapping` / the template's assistant spans) and set `-100` for everything except assistant action tokens **and the closing `<|im_end|>`**. Train on every assistant turn, not just the last. Avoid depending on TRL helper APIs that have changed across versions; ~30 lines of custom masking are easier to unit test.
- Unit test: decode only tokens where `label != -100` for one episode and assert it equals the concatenated action JSONs plus end-of-turn tokens, exactly.
- Log the fraction of supervised tokens per batch (typically small); if near 100% masking is off.
- Padding side and shift: let the HF model do the label shift (do not shift manually twice).

**Warning signs:**
Loss drops very fast to near zero in the first 100 steps; model outputs tool-result-like text; the decoded supervised-token test fails; supervised fraction above ~50%.

**Phase to address:** P-SCRATCH, P-LORA (unit test required before first full run).

---

### Pitfall 11: Overfitting to synthetic templates, and conclusions that are really properties of the dataset

**What goes wrong:**
`generate_episodes()` is a template/random generator, so the model memorises phrasings; benign task completion of 100% on the training distribution says nothing about InjecAgent. Worse for this project, the *attack behaviour of the trained model is determined by what the data shows*: if no training transcript contains an injected instruction, the model "ignores" injections only because it never saw one (or fails randomly); if transcripts include obeying injections, it will obey. Either way "ScratchGPT is vulnerable/robust" is a statement about the dataset design, not about nature, and an examiner will ask exactly that. Low-diversity generation also yields a train/validation gap you cannot see because validation comes from the same generator.

**How to avoid:**
- Decide and document the training-data policy for injected observations up front (e.g. "scratch model is trained on benign episodes only, plus a labelled variant trained with obey-injection episodes as a deliberately weak/compromised principal"). Describe the scratch model as a *worst-case/unaligned principal*, per PROJECT.md, not as a measured baseline of natural vulnerability.
- Maximise diversity: many tool names, argument values, user paraphrases, observation lengths, tool-list orderings/subsets; split train/val **by tool name** and by template family so validation measures generalisation; track val loss and held-out exact-action accuracy, and stop early.
- Do a "memorisation probe": perturb a prompt (rename tool, reorder list, paraphrase) and check the model still produces a valid action.
- For LoRA: small rank (8-16), 1-3 epochs, low LR (about 1e-4 to 2e-4), a few thousand episodes are enough; more epochs on templated data hurts. Compare against the *un-tuned* Qwen2.5-0.5B-Instruct baseline so the fine-tune's effect is measurable.

**Warning signs:**
Train loss near 0 and val loss rising; the model produces identical outputs for different prompts; greedy decoding repeating a training example verbatim; benign 100% on synthetic val but low on InjecAgent benign cases.

**Phase to address:** P-SCRATCH, P-LORA, P-VIVA (explain data policy).

---

### Pitfall 12: Data leakage of benchmark tools, cases, and injection text into training

**What goes wrong:**
`held_out_tools()` only collects `User Tool` and `Attacker Tools` names. Leakage still happens through: (a) tools not in that set but used by the harness, e.g. `EXFIL_TOOL` (`GmailSendEmail`) which must be verified to be inside the held-out set, because the `ds` scoring depends on it; (b) near-duplicate toolkits (`GmailReadEmail` vs a synthetic `MailReadEmail`) and identical parameter names or summaries; (c) user-instruction strings, attacker-instruction strings, or tool-response templates copied from `user_cases.jsonl` / `attacker_cases_*.jsonl` into synthetic data (even paraphrased, via an LLM that was shown the benchmark); (d) canary format (`CANARY...`) appearing in training observations; (e) checkpoint or hyperparameter selection by benchmark score, which is test-set tuning; (f) tool specs in `load_tool_specs()` containing held-out names being used as a source for the generator.

**How to avoid:**
- Build the training tool vocabulary from a **separate synthetic pool** (invented names) and apply an automated leakage test in `tests/`: no held-out tool name, no substring of any benchmark user/attacker instruction (n-gram overlap, e.g. 8-gram), no canary pattern in any training episode; fail the build on a hit. Also assert that `GmailSendEmail` (EXFIL_TOOL) is in `held_out_tools()` or excluded separately.
- Select checkpoints on the synthetic validation split only; run InjecAgent once at the end per model; do not re-tune after seeing it.
- Record the dataset hash and seed in the model card/manifest.
- State in the report what the pretrained models (Qwen, Llama) may have seen: InjecAgent is public, so contamination of pretrained models cannot be ruled out. The from-scratch model is the only one with a clean provenance. This is a good viva point.

**Warning signs:**
Scratch model performs suspiciously well on benchmark user tasks; leakage test missing from the test suite; training episodes containing the strings "Tool Response" or InjecAgent phrasing.

**Phase to address:** P-SCRATCH, P-LORA (leakage test as acceptance gate).

---

### Pitfall 13: GPU OOM on 6 GB (logits, activations, sharing the GPU with Ollama and Windows)

**What goes wrong:**
Qwen2.5 has a ~152k vocabulary. The logits tensor is batch x seq x 152k x (2 or 4 bytes): at batch 8, seq 1024 in fp32 this is about 5 GB, so a 0.5B model OOMs on the **loss/logits**, not on weights. Other causes: fp32 master weights plus Adam states for full fine-tuning (0.5B x 16 bytes ≈ 8 GB, which is why LoRA is required, not optional); activations without gradient checkpointing; the Windows desktop and the DWM already holding ~0.5-1 GB of the 6 GB (WSL2 shares the GPU); an Ollama model kept resident (default `keep_alive` of 5 min, or a running benchmark) while training; PyTorch allocator fragmentation; DataLoader workers each holding copies in CPU RAM.

**How to avoid:**
- LoRA only, bf16 (RTX 3060 is Ampere, bf16 supported) or fp16 with grad scaler; micro-batch 1-4 with gradient accumulation; max sequence length set from the actual token-length histogram (agent transcripts are short: truncate at the 99th percentile, e.g. 512-1024); gradient checkpointing on; compute loss only on supervised positions where possible (slice hidden states before the LM head if the framework allows).
- Before each run: `ollama stop <model>` / `nvidia-smi` shows free memory; never benchmark and train simultaneously. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- For the tiny from-scratch model: pick the vocabulary deliberately small (char-level or a ~2-4k BPE) so the embedding/logits are tiny; fit it by computing params and activation memory on paper first (also a good viva exercise).
- Write a 20-step **smoke test** that runs the worst-case batch (longest sequence x max batch) first, so OOM shows up in 30 seconds, not at hour 2. Save a checkpoint every N steps and make the trainer resumable.
- Avoid optional heavy dependencies whose wheels may not exist for Python 3.14 (bitsandbytes, flash-attn, unsloth, torch.compile paths): verify `pip install` on the real venv first; use eager attention/SDPA. QLoRA is unnecessary at 0.5B. (LOW confidence on specific wheel availability; check.)

**Warning signs:**
`CUDA out of memory` on the first backward pass; memory growing across steps; `nvidia-smi` showing other processes; speed collapsing because of swapping to shared memory.

**Phase to address:** P-LORA, P-SCRATCH (smoke test first), P-LOCAL (do not overlap with Ollama).

---

### Pitfall 14: WSL2 RAM, disk, and process-lifetime limits

**What goes wrong:**
WSL has 7 GB RAM; the Linux OOM killer ends the Python process with a bare `Killed` and nothing in the Python traceback. Loading a model in fp32 on CPU first, tokenising the whole dataset eagerly, many DataLoader workers (16 cores tempts `num_workers=16`), and Ollama's own memory all compete. The repo lives under `/mnt/c/...`, where file I/O is slow and metadata semantics differ (drvfs; `chmod` and permission tests behave differently), and the per-episode sandbox directory should live on the Linux filesystem. Long runs die when the terminal closes or Windows sleeps; laptop thermal throttling and power plans reduce GPU throughput. HF caches and checkpoints fill the WSL virtual disk.

**How to avoid:**
- Raise `memory=` in `%UserProfile%\.wslconfig` if the host has headroom (and set a swap), then `wsl --shutdown` to apply; check with `free -h`. Load models with `torch_dtype=bfloat16` directly to GPU (`low_cpu_mem_usage=True`); use `num_workers=0-2`; pre-tokenise to a `.pt`/`.npy` file once.
- Keep venv, HF cache, datasets, checkpoints, and sandbox scratch dirs on `~` (ext4), not `/mnt/c`.
- Run long jobs under `nohup`/`tmux` with logging to a file; plug in power, disable sleep; checkpoint and resume.
- After any mysterious exit, check `dmesg | tail` / exit code 137.

**Warning signs:**
`Killed` with no traceback; training that is 5-10x slower than expected; benchmark passing on Windows paths but failing perms checks; job gone after closing the terminal.

**Phase to address:** P-SCRATCH, P-LORA (setup step), P-HOSTED (long-running resumes).

---

### Pitfall 15: Nondeterminism and decoding-setting drift

**What goes wrong:**
`temperature=0` does not give determinism on hosted APIs (batching, MoE routing, backend changes) or on GPU Ollama; the repo has a `seed` for case sampling but sampling seeds are not passed to the LLM. For the HF-served models, `generate()` inherits `generation_config.json` defaults (for Qwen2.5-Instruct: sampling on, `top_p`/`top_k`, and a repetition penalty above 1), so "greedy" is not greedy unless `do_sample=False` and the penalty are set explicitly; passing `temperature=0` to HF either errors or is ignored. A repetition penalty also distorts JSON with repeated keys/quotes. Greedy decoding on tiny models often falls into repetition loops that produce unterminated JSON.

**How to avoid:**
- For HF/scratch inference set `do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens`, explicit EOS ids; log the full generation config in the manifest.
- Pass `seed` where the provider supports it (Ollama and some OpenAI-compatible endpoints do); otherwise state that variance exists and rely on case-level aggregation with intervals.
- Run the 3 repeats on distinct days/times for hosted models if possible, or accept and disclose.
- For the scratch model consider the same fairness rule as for everyone else: the action parser is identical; do not give it grammar-constrained decoding unless you also state this is a deliberate concession (and then report both).

**Warning signs:**
Same case/run yielding different actions between resumes; HF output different from expected greedy; repeated tokens/loops in outputs; run-to-run differences larger than between-model differences.

**Phase to address:** P-SCRATCH, P-LORA, P-EVAL (manifest).

---

### Pitfall 16: Hosted-model drift, shared quotas, and resume side effects

**What goes wrong:**
`gemini-flash-lite` and `gemma-4-31b` share one `GEMINI_API_KEY` quota; running them in parallel makes both time out (the 204 gemma timeouts). Groq free-tier daily token caps stop runs mid-night, and resumed segments may run against an updated model snapshot. The resume path retries only `llm_error` records; a record that "succeeded" because the model returned garbage is not retried, and a record produced under a previous code/policy version is mixed with new ones. Results with `GEMINI_MODEL` read from `.env` by two different parsers can silently select different models.

**How to avoid:**
- Serialise runs per provider key; one shared token bucket per provider.
- Store `code_commit`, `policy_hash`, `resolved_model_id`, `system_fingerprint` (if returned) in every raw record; the report refuses to mix records with different commits/policy hashes in one cell, or at least warns.
- Add a `--dry-run` that prints how many keys are missing and estimates tokens/time against quotas before launching; start the resume with the smallest-missing models first and the `ds` class first, since `ds` is entirely missing for hosted models.
- Decide a stopping rule now (e.g. "stop hosted collection at day N") and report whatever is complete; this is a deadline project.

**Warning signs:**
Bursts of 429/timeouts when two models run at once; `progress.log` showing multi-day gaps; mixed commits in one `raw.jsonl`.

**Phase to address:** P-HOSTED, P-EVAL.

---

## Moderate Pitfalls

### Pitfall 17: Tiny from-scratch model never learns the format (or learns nothing real)

**What goes wrong:** Too little data/steps, a tokenizer that fragments JSON into many tokens, learning-rate warmup omitted, context shorter than a transcript, or a causal-mask/positional bug (the model sees the future during training and cannot generate). Training loss decreasing is mistaken for "working".
**Prevention:** Sanity-check the architecture by overfitting 10 episodes to ~0 loss first; check masks by perturbing a later token and confirming earlier logits do not change; evaluate **exact-valid-JSON rate** and action accuracy on held-out synthetic episodes (not just loss/perplexity); use a simple tokenizer (char-level or small BPE) with JSON-friendly tokens; AdamW with warmup + cosine decay, gradient clipping 1.0; fixed seed; keep model small enough (a few M parameters) that training finishes in minutes to an hour. Perplexity is not a task metric; report both.
**Phase:** P-SCRATCH.

### Pitfall 18: Counting the wrong thing as "attack succeeded" or "false positive"

**What goes wrong:** `succeeded` for `dh` is "attacker tool executed with `allowed`", i.e. judged by the broker, not by effect; `false_positive` for benign is "any layer blocked anything", so a benign episode where the model made a harmless but blocked probe counts as FP and the "task completed" flag may still be true; `user_task_done` requires `completed` (a respond action) so format failures lower benign completion for reasons unrelated to guardrails. Comparing benign completion across arms conflates model failure with guardrail interference.
**Prevention:** Report benign completion *delta* (protected minus unprotected, paired) as the cost of the guardrails; list FP by blocked rule; hand-inspect 10-20 episodes (a few per model, including all FPs and all "attempted but no success" cases) and record the audit result; keep the definitions in the report text.
**Phase:** P-EVAL, P-REPORT.

### Pitfall 19: Prompt-injection benchmark validity limits are not stated

**What goes wrong:** InjecAgent uses simulated tool outputs with a fixed attacker-instruction template; it is public (contamination), non-adaptive, and the attacker tools are mapped to OS-level actions only in this project's adapter. A reviewer will ask whether results transfer to real agents and adaptive attackers (AgentDojo, CaMeL, and other papers in `papers/` discuss this).
**Prevention:** Add a "Threat model and limits" section: non-adaptive attacker, single benchmark, simulated tools, what the OS layer does and does not defend (it blocks actions, not model compromise, and a permitted-but-harmful action within policy remains possible). Cite the related-work PDFs already in `papers/`.
**Phase:** P-REPORT, P-VIVA.

### Pitfall 20: Report/artifact hygiene (mixed commits, stale charts, untracked results)

**What goes wrong:** `results/final-20261011/` is untracked and has no manifest; charts and `summary.md` generated from a state that still contained errors; presentation scripts that edit slides by shape index silently break; README still says Gemini-only; numbers in slides drift from numbers in the report.
**Prevention:** Generate the report, charts, README table, and slide numbers from one script reading one results directory; write `manifest.json`; keep results directory out of git but its manifest and summary in git; one final regeneration step before the review; date-stamp the final run directory.
**Phase:** P-REPORT, P-VIVA.

### Pitfall 21: Serving the fine-tuned model through a convoluted path

**What goes wrong:** Spending days on merging adapters, GGUF conversion, and importing into Ollama, then losing the template/EOS during conversion; or serving through a different code path than the one that was validated.
**Prevention:** Run the LoRA model in-process with HF + PEFT (`inference.py`/`load_generator`), merge only for speed; skip GGUF unless time remains. Disclose that it is served via HF (bf16) rather than Ollama (Q4).
**Phase:** P-LORA.

### Pitfall 22: Untested shell execution on the developer machine

**What goes wrong:** The unprotected arm uses `shell=True` with real commands; run on the WSL host with a mistaken working directory or on Windows/`/mnt/c`, a model (especially a scratch model emitting random commands) can damage files.
**Prevention:** Run all benchmark arms in a disposable directory (and ideally a throwaway WSL distro or container); never run the unprotected arm from a directory containing the repo or `.env`; keep API keys out of the benchmark process environment for local-model runs.
**Phase:** P-EVAL (before any new run), P-SCRATCH.

---

## Minor Pitfalls

### Pitfall 23: Dashboard exposes Werkzeug debugger on 0.0.0.0
**What goes wrong:** Anyone on the network (venue Wi-Fi at a review) gets code execution.
**Prevention:** Bind to 127.0.0.1, `debug` off by default (CONCERNS.md); fix before any live demo.
**Phase:** P-VIVA (demo prep).

### Pitfall 24: Viva material added at the end
**What goes wrong:** Notes written in the last night repeat the code instead of the theory, and gaps appear in "why" questions.
**Prevention:** Each phase appends a "foundations + what I did + what could go wrong" page to `docs/viva/` as part of its definition of done (see viva section below).
**Phase:** every phase.

### Pitfall 25: Seeds and sample selection undocumented
**What goes wrong:** `attack_cases(seed=0)` picks which 20 dh / 20 ds cases; a different seed gives different numbers and the choice looks cherry-picked.
**Prevention:** Document the seed and the sampling rule; optionally repeat the cheapest local model over a second seed to show stability.
**Phase:** P-EVAL.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Quote current `summary.md` percentages | Saves re-running | Incomplete, errored, and biased; indefensible in a viva | Never |
| Case-major run order (current) | Simple loop | Truncated runs lose `ds` entirely | Only if every run is guaranteed to finish (local models) |
| Hand-typed system prompt inside training data | Quick | Train/inference mismatch (Pitfall 9) | Never; import from `prompting.py` |
| Single `llm_error` string for all failures | Simple | Cannot separate refusals from timeouts | Never for the final table |
| Report 0% with no n/interval | Looks clean | Overclaim; easy to challenge | Never |
| Skipping the scripted control agent | Saves a few hours | Core claim ("independent of LLM") is unsupported | Never; it is cheap |
| Train scratch model on random/templated data only | Fast to build | Behaviour reflects the generator | OK if labelled "worst-case/unaligned principal" |
| Use Ollama default context | No setup | Silent truncation | Never for multi-step runs |
| Full fine-tune instead of LoRA | Simpler code | Does not fit 6 GB | Never here |
| Skip GGUF/Ollama export of the LoRA model | Saves days | Different serving stack | Acceptable if disclosed |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Ollama via `/v1` | Assuming `num_ctx` can be passed per request | Set `OLLAMA_CONTEXT_LENGTH` for the server or use a Modelfile variant; confirm with `ollama ps` |
| Ollama + GPU | Model kept resident while training | `ollama stop <model>`; check `nvidia-smi` before training |
| Gemini / Gemma on one key | Running both models in parallel | Serialise per key; one token bucket |
| Groq reasoning models | `max_tokens=512` consumed by reasoning | Raise limit / set reasoning effort; log `finish_reason` |
| Groq `failed_generation` | Scoring text from a 400 error as a normal reply | Keep (it is the model's output) but tag the record `recovered_from_error` so it can be audited |
| HF `generate` | Inheriting `generation_config.json` sampling defaults | `do_sample=False`, explicit penalties, EOS ids |
| PEFT/LoRA | Target modules default or wrong; adapter not loaded at inference | Target attention + MLP projections (`q,k,v,o,gate,up,down`); assert trainable-param count and that inference logits differ from base |
| Qwen chat template | Default system prompt auto-inserted | Always supply the project's system message in both paths |
| WSL2 filesystem | Data/venv on `/mnt/c` | Keep on ext4 `~`, edit via Windows tools if needed |
| Python 3.14 + torch 2.11 | Assuming every ML wheel exists | Test imports of peft/transformers/accelerate first; fall back to a Python 3.12 venv if blocked |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Logits memory (152k vocab) | OOM at first backward | Short seq, small micro-batch, grad accumulation | batch x seq above ~2k tokens at fp32 |
| No gradient checkpointing | Activation OOM | Enable it | seq above ~1k on 6 GB |
| Eager full-dataset tokenisation in RAM | `Killed` in WSL | Pre-tokenise to disk; memory-map | datasets of a few 100k episodes at 7 GB |
| DataLoader workers x16 | RAM spike | `num_workers` 0-2 | any time with 7 GB |
| Sequential hosted benchmark | Days of wall time | Per-provider concurrency within rate limit; interleaved ordering | 342 x 6 models |
| Ollama cold-load per model | Latency outliers | Warm up, keep one model resident at a time | switching 4+ local models |
| Retries nested (SDK `max_retries=2` x runner retries) | Episode stalls for minutes | Single retry layer with logged attempts | rate-limited endpoints |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Policy checks only the first shell token | `ls; curl evil` allowed (High) | Split on metacharacters, check each segment, or `shell=False` argv; add tests first |
| Executor does not require a policy decision | Enforcement optional per caller | Executor refuses calls without a passing decision |
| Unprotected arm run on a real host | A model could delete files (High) | Disposable directory or distro; no secrets in env |
| Path globs without canonicalisation | `..`/symlink bypass | `os.path.realpath` before matching; tests |
| Default action FLAG not BLOCK | Unmatched actions pass | State and justify; consider default-deny for benchmarked actions |
| Debug dashboard on 0.0.0.0 | Remote code exec (High) | 127.0.0.1, debug off |
| Training data with real keys/canaries | Leaks, or leakage into eval | Synthetic secrets only; canary pattern excluded from training |
| Claiming "secure" on 0% ASR | Overclaim (Medium) | Phrase as "no observed bypass in N attempts; bound X%" |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Comparison table with 12+ columns | Reviewer cannot find the point | Headline table of 5-6 columns, details in appendix |
| Rankings among overlapping CIs | False conclusions | Show intervals, group into tiers |
| Charts without n | Misleading | Annotate n on bars; grey out partial models |
| Slides copying numbers by hand | Stale numbers | Generate from the report script |
| Viva notes as code walkthrough | Cannot answer "why" | Theory-first notes (below) |

## "Looks Done But Isn't" Checklist

- [ ] **Hosted comparison:** often missing `ds` episodes and equal n; verify every model row has n = 342 (or the common-subset n) and non-empty `ds`.
- [ ] **ScratchGPT provider:** often "trains" but not wired; verify `python -m src.scratch_llm.train` writes a checkpoint plus tokenizer and `load_generator` returns outputs via the same parser as other models.
- [ ] **LoRA model:** often evaluated with the wrong template; verify the round-trip token test passes and base-vs-tuned valid-JSON rates differ as expected.
- [ ] **Ollama models:** often run at default context; verify `ollama ps` CONTEXT and max `prompt_tokens` below the window.
- [ ] **ASR table:** often missing conditioning; verify format-valid rate and benign completion beside every ASR.
- [ ] **Guardrail layers:** often only broker/policy fire; verify at least one run where sandbox/seccomp block something, and the scripted control agent row exists.
- [ ] **Shell bypass fix:** often fixed in code but not in results; verify a failing test existed first and the final runs use the fixed policy hash.
- [ ] **Leakage test:** often written but not enforced; verify it runs in `pytest` and fails on a deliberately injected held-out tool.
- [ ] **Manifest:** often absent; verify each results dir has commit, model digests, settings, counts.
- [ ] **Viva notes:** often stale; verify each phase added its page.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Ollama context truncation found late | MEDIUM | Set context, re-run affected local models (local, no quota; hours) |
| Parse failures confounding results | LOW | Re-score from `raw.jsonl` with the new taxonomy (no re-run needed if replies were stored) |
| Wrong template/mask in LoRA | MEDIUM | Fix renderer + mask, retrain (0.5B LoRA is minutes to ~1 hour) |
| Leakage discovered after training | MEDIUM-HIGH | Regenerate data without the leaked items, retrain, re-run benchmark |
| Hosted runs incomplete at deadline | LOW | Report as partial with n and intervals; exclude from ranking |
| Errors counted wrong in old results | LOW | Recompute with pair-aware reporter; mark old `summary.md` superseded |
| Policy fix changes outcomes | MEDIUM | Re-run protected arm only (unprotected unaffected) for all models |
| OOM mid-training | LOW | Resume from last checkpoint with smaller batch/seq |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 Parse failures confound ASR | P-EVAL | Report shows outcome taxonomy; format-valid rate beside ASR; test with synthetic format-failure episodes |
| 2 Unequal n / truncated runs | P-EVAL, P-HOSTED | Queue is interleaved; completeness gate fails incomplete models; common-subset table |
| 3 Error handling | P-EVAL | Unit tests in `test_benchmark.py` for llm_error, duplicates, refusals, pairing |
| 4 max_tokens/reasoning | P-EVAL, P-HOSTED | `finish_reason` logged; no `length` stops in final data |
| 5 Ollama num_ctx | P-LOCAL | `ollama ps` evidence + max prompt_tokens < window recorded in manifest |
| 6 CIs / pseudo-replication | P-EVAL, P-REPORT | Wilson/Clopper-Pearson in report; case-clustered |
| 7 Quantization / settings parity | P-LOCAL, P-REPORT | Settings table with digests and quantization in report |
| 8 Vacuous guardrail claim | P-POLICY, P-EVAL | Control agent row; layer-block table has sandbox/seccomp > 0 somewhere; bypass test passes |
| 9 Template mismatch | P-LORA, P-SCRATCH | Round-trip token-identity test passes |
| 10 Loss masking | P-LORA, P-SCRATCH | Decoded supervised-token test passes |
| 11 Template overfitting / dataset-determined behaviour | P-SCRATCH, P-LORA | Val split by tool; perturbation probe; documented data policy |
| 12 Leakage | P-SCRATCH, P-LORA | Leakage test in CI incl. EXFIL_TOOL and n-gram overlap |
| 13 OOM | P-LORA, P-SCRATCH | Worst-case-batch smoke test passes before full run |
| 14 WSL limits | P-SCRATCH, P-LORA | `.wslconfig` documented; jobs under tmux with checkpoints |
| 15 Nondeterminism | P-SCRATCH, P-LORA, P-EVAL | Generation config in manifest; greedy verified |
| 16 Hosted drift / shared quotas | P-HOSTED | Records carry commit, policy hash, resolved model id |
| 17 Tiny-model sanity | P-SCRATCH | Overfit-10 test; causal-mask test; valid-JSON rate on val |
| 18 Metric definitions | P-EVAL, P-REPORT | Manual audit of sampled episodes recorded |
| 19 Benchmark validity limits | P-REPORT, P-VIVA | Threat-model/limits section present |
| 20 Artifact hygiene | P-REPORT | One script regenerates report, README table, slide numbers |
| 21 Serving path | P-LORA | `load_generator` in-process; disclosure line in report |
| 22 Unprotected arm on host | P-EVAL | Documented disposable environment |

## What Viva Examiners Commonly Probe (and the answer the project must support)

Prepare a one-page answer for each; add to `docs/viva/` in the phase that produces the evidence.

**Evaluation and statistics**
- "What does 0% mean? How many attempts? What is the confidence bound?" Need: n, case-level independence, Wilson/Clopper-Pearson upper bound, blocks-given-attempt by layer.
- "Your weak model never attacks, so how do you know the guardrails work?" Need: the scripted worst-case control agent and attempted-attack counts.
- "Why is model X 'safe'? Maybe it just cannot follow your format." Need: format-valid rate and the outcome taxonomy.
- "Why are the sample sizes different / why is data-stealing missing for hosted models?" Need: common-subset table, completeness policy, honest limitation.
- "Is a 4-bit 3B model comparable to a hosted 120B one?" Need: confounds table; spectrum framing; quantization disclosure.
- "Why temperature 0? Is it reproducible?" Need: determinism caveats, seeds, manifest, repeats.
- "How did you handle API errors and timeouts?" Need: taxonomy, pairing, idempotent resume.
- "Why InjecAgent? What are its limits? Adaptive attackers?" Need: threat model section; citations in `papers/` (AgentDojo, CaMeL, InjecAgent).

**Guardrail design (Operating Systems focus)**
- "What do namespaces, seccomp-bpf, and capability/access-matrix policy each protect against, and what is the TCB?" Need: layered-defence diagram, reference-monitor properties (complete mediation, tamper-proofness, verifiability) and where this implementation falls short.
- "Show me a bypass." Need: the `ls; curl` case reproduced, test, fix, and a statement of remaining gaps (path canonicalisation, default FLAG, dashboard).
- "Why did sandbox/seccomp block count stay 0?" Need: an experiment that triggers them, or an honest explanation of which layer mediates which action.
- "Why not rely on prompt-level defences or model alignment?" Need: argument that enforcement outside the model is independent of model behaviour, and the limits (it does not stop in-policy harmful actions).

**Training and model theory**
- "Explain attention, causal mask, positional encoding, why you scale by sqrt(d_k), what the loss is, what perplexity means, why AdamW, warmup."
- "Parameter count and memory arithmetic: why does your model fit in 6 GB? Why LoRA and not full fine-tuning?" (weights, grads, Adam states, activations, logits.)
- "What is LoRA? rank, alpha, which matrices, how many trainable parameters, why initialisation B=0, can it be merged?"
- "How is the loss masked and why?" "Where could train/test leakage occur and how did you prevent it?" "How do you know it is not memorising templates?"
- "Is a model trained on your synthetic data a language model or a lookup table? What is its role in the project?" Answer: a deliberately weak, unaligned worst-case principal; behaviour is dataset-determined; no claims about natural vulnerability.
- "Why does the tokenizer/chat template matter?" Show the round-trip test.
- "Who wrote the training data?" If an LLM generated it, discuss circularity and contamination; templated generator discussed with its limits.

## Sources

- Repository evidence (HIGH): `.planning/PROJECT.md`, `.planning/codebase/CONCERNS.md`, `src/benchmark/report.py`, `src/benchmark/scoring.py`, `src/agent_harness/prompting.py`, `src/llm/openai_compat.py`, `src/scratch_llm/dataset.py`, `config/models.yaml`.
- Ollama context behaviour (MEDIUM; third-party guides, no official docs surfaced in search): https://lumadock.com/tutorials/ollama-context-window, https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-ollama.md. Verify the default and truncation behaviour on the installed Ollama version with `ollama show`, `ollama ps`, and the server log.
- Statistical remarks (HIGH): Wilson and Clopper-Pearson intervals, rule of three, McNemar test for paired binary outcomes; calculations: 0/20 gives 95% upper bound about 16%, 0/60 about 6%.
- Training/WSL/HF practice (MEDIUM; established practice, from general knowledge, not re-verified against current docs in this session): Qwen2.5 chat template default system prompt, `generation_config.json` sampling defaults, EOS/pad handling, logits memory with ~152k vocabulary, WSL `.wslconfig` memory limits, OOM-killer behaviour.
- Python 3.14 / torch 2.11 ecosystem compatibility (LOW): not verified; test imports on the actual venv.
- Related work in `papers/` (InjecAgent, AgentDojo, CaMeL, Ignore Previous Prompt, AgentSight, ActPlane) for the benchmark-validity and threat-model discussion.

---
*Pitfalls research for: local LLM training + multi-LLM agent-security evaluation*
*Researched: 2026-10-20*
