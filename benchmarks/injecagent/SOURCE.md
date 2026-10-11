# InjecAgent data (vendored)

Source: https://github.com/uiuc-kang-lab/InjecAgent (`data/`), pinned to commit
`f19c9f2c79a41046eb13c03c51a24c567a8ffa07`. Repository license: MIT (per GitHub repository metadata).

Paper: Q. Zhan, Z. Liang, Z. Ying, D. Kang. *InjecAgent: Benchmarking Indirect Prompt
Injections in Tool-Integrated Large Language Model Agents.* Findings of ACL 2024.
(Copy in `papers/`.)

| File | Contents |
|---|---|
| `user_cases.jsonl` | 17 benign user tasks; each user tool's simulated response contains an `<Attacker Instruction>` slot |
| `attacker_cases_dh.jsonl` | 30 direct-harm attacker cases (physical, financial, data-security harm) |
| `attacker_cases_ds.jsonl` | 32 data-stealing cases (read private data with a tool, then send it via `GmailSendEmail`) |
| `tools.json` | Tool specifications (toolkit + tool name, summary, parameters) |

Files are used unmodified; `src/benchmark/injecagent.py` adapts them to this project's
agent (the `call_tool` action, task-scoped tool capabilities, canary-seeded private data).
