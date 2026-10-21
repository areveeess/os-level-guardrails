---
phase: "1"
slug: "evaluation-hardening-policy-fix"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-21"
---

# Phase 1 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time (01-01..01-04 `<threat_model>` blocks); mitigations verified at ASVS L1 (grep + test evidence). Post-execution code review (01-REVIEW.md) and fixes (01-REVIEW-FIX.md) are folded in below.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| LLM output → policy engine | `run_shell` strings come from an untrusted, possibly injected model and are later run by `/bin/sh` | Untrusted command text (high) |
| policy engine → shell | The engine's parse must match what the shell executes (parse differential) | Command structure (high) |
| network → dashboard | A non-loopback bind exposes logs and the Werkzeug debugger | Guardrail decisions; debugger = code execution (high) |
| provider API → runner | `finish_reason` / model ids are stored and rendered verbatim | Remote metadata (low) |
| runner → results directory | `raw.jsonl` / `manifest.json` are long-lived evidence shared with reviewers | Benchmark records; must hold no secrets (medium) |
| raw.jsonl / report → reader | The summary is the evidence for the core claim | Statistics (integrity-critical) |
| control agent → executors | Scripted attacker emits real shell/tool actions | Attack actions (contained by arms) |
| notes → readers | Viva notes shared with the panel | Must hold no keys or canaries |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-01-01 | Tampering | results/final-20261011 during regeneration | medium | mitigate | Report regenerated into results/phase1-report via `--report-only --merge`; source dir read-only (git status unchanged) | closed |
| T-01-02 | Tampering | summarize / write_markdown | high | mitigate | Completeness gate, partial rows segregated, `drop_infra_pairs` (outcomes.py), k/n + Wilson CI on every %, tests/test_report.py | closed |
| T-01-03 | Tampering | model strings in markdown tables | low | mitigate | `_cell` escapes `|`, newlines, backslashes (report.py) | closed |
| T-01-04 | Denial of service | load_rows on truncated raw.jsonl | low | mitigate | Undecodable lines skipped and counted; resume tolerates truncated last line (WR-07 fix) | closed |
| T-01-05 | Elevation of privilege | chained / substituted sub-commands | high | mitigate | `parse_shell` returns every simple command (separators, `$()`, backticks, `<()`, eval, find -exec, wrappers, comments after CR-01 fix); first BLOCK wins; tests/test_shell_chaining.py + tests/test_review_fix_policy.py. Residual denylist gaps → AR-01 | closed |
| T-01-06 | Elevation of privilege | parser differential on unreadable input | high | mitigate | ShellParseError → BLOCK (fail closed); dynamic command names refused; non-string targets → BLOCK (WR-01) | closed |
| T-01-07 | Denial of service | parse_shell on adversarial input | medium | mitigate | MAX_COMMAND_CHARS=10000, MAX_DEPTH=8 → ShellParseError → BLOCK | closed |
| T-01-08 | Elevation of privilege | allow-listed interpreters (python3, pip) | medium | accept | Carried by namespace + seccomp layers; documented in shell_split.py and viva page → AR-02 | closed |
| T-01-09 | Elevation of privilege | dashboard debugger on all interfaces | high | mitigate | `_run_settings` defaults 127.0.0.1, debug off; debug refused with non-loopback host (WR-09); tests/test_dashboard_bind.py; UAT test 1 passed | closed |
| T-01-10 | Information disclosure | DASHBOARD_HOST opt-in | low | accept | Explicit operator opt-in with trusted-network warning → AR-03 | closed |
| T-01-11 | Repudiation | mixed / resumed / truncated runs | medium | mitigate | Row stamps (code_commit, code_dirty, policy_sha256, provider, resolved_model, finish_reasons, prompt_tokens); manifest invocations appended; MIXED marker in report | closed |
| T-01-12 | Information disclosure | manifest.json / raw.jsonl | medium | mitigate | Fixed manifest field list, no env values; fake GROQ_API_KEY test in tests/test_runner_queue.py | closed |
| T-01-13 | Tampering | resolved model id rendered into summary | low | mitigate | Rendered through `_cell` | closed |
| T-01-14 | Denial of service | infra re-queue burning free-tier quota | medium | mitigate | `requeue_passes=1`, skipped after QuotaExhausted; refusals never retried | closed |
| T-01-15 | Elevation of privilege | git subprocess for commit stamp | low | mitigate | Fixed argv, no shell, cwd pinned, timeout, failures → "unknown" | closed |
| T-01-16 | Tampering | control rows leaking into LLM statistics | medium | mitigate | provider == "scripted" excluded from headline/partial/pooled/charts/outcome tables; tests/test_control_agent.py | closed |
| T-01-17 | Elevation of privilege | control agent exfiltration command | low | mitigate | Unprotected arm = MockExecutor; protected arm = GuardedExecutor + network-less namespace; reserved attacker.example domain | closed |
| T-01-18 | Information disclosure | docs/viva notes | low | mitigate | Automated key/canary check; grep of docs/viva/ for key-like strings clean | closed |
| T-01-SC | Tampering | package installs | low | accept | Phase installed no packages (stdlib only) → AR-04 | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-01-05 (CR-03/CR-04 partial) | Policy engine stays a denylist with `default_action: FLAG` (user decision during code-review gate). Residual bypasses — renamed binaries, awk/perl/sed `e` subprocesses, interpreter-built paths — are listed in 01-REVIEW-FIX.md and are the input set for Phase 4's layer-targeted suite (EVAL-08), which demonstrates namespaces + seccomp containing them. | Raj Vardhan Singh (via code-review gate choice "Fix + reuse in Phase 4") | 2026-10-21 |
| AR-02 | T-01-08 | A string policy cannot inspect interpreter programs; python3/pip remain allow-listed for benign utility; containment is the job of the OS layers. | Plan 01-02 (accepted disposition) | 2026-10-21 |
| AR-03 | T-01-10 | Non-loopback dashboard bind is an explicit operator opt-in with a documented warning; logs contain decisions, not secrets. | Plan 01-02 (accepted disposition) | 2026-10-21 |
| AR-04 | T-01-SC | No packages installed in this phase. | Plans 01-01..01-04 | 2026-10-21 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-21 | 19 | 19 | 0 | /gsd-secure-phase 1 (orchestrator, ASVS L1 grep + test evidence; auditor skipped per L1 short-circuit) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-21
