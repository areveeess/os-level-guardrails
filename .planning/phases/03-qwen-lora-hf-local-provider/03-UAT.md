---
status: testing
phase: 03-qwen-lora-hf-local-provider
source: [03-VERIFICATION.md]
started: 2026-10-22T15:06:47Z
updated: 2026-10-22T15:06:47Z
---

## Current Test

number: 1
name: Viva page 03 is understandable
expected: |
  docs/viva/03-fine-tuning-lora-and-chat-templates.md (pretraining vs fine-tuning, LoRA W + BA with this model's numbers, chat templates & special tokens, loss masking, bf16, gradient checkpointing, greedy decoding, base vs tuned, and "what the code review found and what we fixed") reads clearly to you; report any term that is not explained.
awaiting: user response

## Tests

### 1. Viva page 03 is understandable
expected: The page reads clearly; any undefined term is reported as an issue.
result: [pending]

### 2. LoRA training result looks sensible
expected: models/qwen-agentlora/loss_curve.png shows train loss falling (0.151 → 0.045) and dev action loss flattening, best step 600; full-dev action exact 0.94 (benign 0.99 / dh 1.00 / ds 0.84); tuned parse 100/100 vs base 69/100.
result: [pending]

### 3. Honest-reporting judgments hold
expected: You accept that (a) the provider never repairs model replies, (b) no hand-built prompts are used, (c) nothing was tuned on smoke/benchmark results, (d) the LoRA model is presented as an injection-compliant worst-case principal, not a safety-tuned model — and that the review fixes WR-01..WR-03 (rate-limit matching, stale done marker, staged promotion) are acceptable as tested.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
