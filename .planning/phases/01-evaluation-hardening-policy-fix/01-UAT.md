---
status: testing
phase: 01-evaluation-hardening-policy-fix
source: [01-VERIFICATION.md]
started: 2026-10-21T02:35:00Z
updated: 2026-10-21T02:35:00Z
---

## Current Test

number: 1
name: Dashboard reachable from the Windows browser while bound to 127.0.0.1 in WSL
expected: |
  In WSL run `venv/bin/python -m src.dashboard.app`; on Windows open http://localhost:5000 — the dashboard loads. (WSL2 forwards localhost; it must NOT be reachable from another machine on the LAN.)
awaiting: user response

## Tests

### 1. Dashboard reachable from the Windows browser while bound to 127.0.0.1 in WSL
expected: http://localhost:5000 loads from Windows; startup log shows 127.0.0.1:5000 and debug off.
result: [pending]

### 2. Student reads the Phase 1 viva page
expected: docs/viva/01-reference-monitor-and-evaluation.md (especially Part D and "What the code review found") is understandable; any undefined term is reported as an issue.
result: [pending]

### 3. (Recommended) Shell-parser fidelity against a real shell
expected: Spot-check a few chained/commented commands from 01-REVIEW-FIX.md against `dash` behaviour; a dash differential test is planned for Phase 4 (EVAL-08 suite).
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
