# Viva study notes

These pages prepare the student (Raj Vardhan Singh, 24BDS0200) for the oral examination on the
OS-level guardrails project. Each page is written in plain language: every term is defined, the
reason behind each design choice is explained, and each page ends with likely examiner questions
and model answers. Read a page, close it, and try to explain the page's questions out loud; then
re-read the parts you stumbled on. Numbers quoted in the pages are copied from the generated
reports in `results/`, so re-check them against the report if you re-run the benchmark.

One page is added per phase of the project, so the notes grow with the work.

| Page | Topics | Phase | Status |
|---|---|---|---|
| [01-reference-monitor-and-evaluation.md](01-reference-monitor-and-evaluation.md) | Reference monitor, access matrix, capability lists vs ACLs, least privilege, complete mediation, TOCTOU, shell metacharacter chaining, fail-closed, defense in depth; ASR, ASR-valid, utility, false-positive rate, Wilson interval, rule of three, paired design, the scripted control agent, provenance stamps | Phase 1 | written (Phase 1) |
| [02-llm-foundations-and-synthetic-data.md](02-llm-foundations-and-synthetic-data.md) | Transformers, attention, character tokenization, the training loop (loss mask, AdamW, warmup, overfitting, dev split), greedy decoding; the synthetic-data mix and the train/benchmark leakage gate; this run's real numbers | Phase 2 | written (Phase 2) |
| 03 Fine-tuning, LoRA, chat templates and decoding | Fine-tuning and LoRA, chat templates and loss masking, decoding choices | Phase 3 | planned (Phase 3) |
| 04 Namespaces, seccomp-bpf and local-inference confounds | What each Linux isolation layer protects, quantization and context-window confounds when comparing local and hosted models | Phase 4 | planned (Phase 4) |
| 05 Examiner Q&A and threats to validity | Final question-and-answer sheet, limits of the benchmark, threats to validity | Phase 5 | planned (Phase 5) |
