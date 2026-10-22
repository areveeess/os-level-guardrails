---
phase: "2"
slug: "local-data-foundation-scratchgpt"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-22"
---

# Phase 2 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time (02-01..02-04 `<threat_model>` blocks); mitigations verified at ASVS L1 (grep + test evidence, 901-test suite, independent leakage scan in 02-VERIFICATION.md). Post-execution review fixes (02-REVIEW-FIX.md) are folded in.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| benchmark files → training data | Vendored InjecAgent tools/cases must never leak into synthetic training episodes | Evaluation integrity (high) |
| checkpoint file → inference | A checkpoint on disk is deserialised into the process | Code-execution risk via pickle (medium) |
| CLI arguments → filesystem | Dataset/train CLIs write to caller-supplied paths | Path traversal (low) |
| training run → benchmark | Hyper-parameters must not be tuned against benchmark outcomes | Evaluation integrity (medium) |
| repository → git history | Weights, episodes, results and user files must not be committed by accident | Disclosure / repo bloat (medium) |
| GPU/WSL shared with another project | Training shares VRAM and VM memory with an unrelated workload | Availability (low) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-02-01 | Tampering | generate_episodes / assert_clean (benchmark leakage) | high | mitigate | Programmatic gate (held-out tools incl. GmailSendEmail, all benchmark addresses, case-insensitive canary regex after WR-07, record phrasing, 8-word shingles, exact sentences, held-only param names per clarified D-04) inside generate_episodes; planted-violation tests; per-reason rejection counts with leak cap (WR-08); independent verifier scan found 0 leaks in 42,000 episodes | closed |
| T-02-02 | Tampering | render_context (train/serve skew) | medium | mitigate | Training text = render_context output by construction; prefix + AgentLoop replay parity tests; FORMAT_VERSION / VOCAB_SHA256 / render fingerprint stamped and checked | closed |
| T-02-03 | Information disclosure | synthetic values | low | mitigate | Reserved example domains, benchmark addresses banned, ASCII-only, no real data read | closed |
| T-02-04 | Tampering | dataset reproducibility | medium | mitigate | Integer-seeded RNG, sorted pools, same-seed and cross-PYTHONHASHSEED hash tests | closed |
| T-02-05 | Tampering | attack templates drifting toward benchmark text | high | mitigate | assert_clean on every split/class; template scan; separate dev templates | closed |
| T-02-06 | Tampering | cross-process reproducibility | medium | mitigate | Manifest sha256 checked against files (verifier re-hashed: match) | closed |
| T-02-07 | Tampering | dataset CLI --out-dir traversal | low | mitigate | resolve_under_root + parser.error; manifests protected from cross-overwrite (WR-06) | closed |
| T-02-08 | Denial of service | dataset CLI size | low | mitigate | MAX_EPISODES = 200000 | closed |
| T-02-09 | Information disclosure | data/local/manifest.json | low | mitigate | Fixed field list, repo-relative paths, env-sentinel test | closed |
| T-02-10 | Information disclosure | public-provider synthetic addresses | low | accept | ≤15% numeric public-provider addresses, never sent anywhere → AR-01 | closed |
| T-02-11 | Elevation of privilege | load_generator (pickle) | medium | mitigate | torch.load(weights_only=True), exact key set, checkpoint sha in manifest and every row's resolved_model | closed |
| T-02-12 | Repudiation | stale/mismatched checkpoint | medium | mitigate | Stamp checks raise RuntimeError → runner [skip]; verifier tampered each stamp on the real checkpoint | closed |
| T-02-13 | Denial of service | generation loop | low | mitigate | Bounded generation (cap now = format action limit + END after WR-02); finish_reason "length"; over-long prompts recorded as context_overflow, never cropped (WR-03) | closed |
| T-02-23 | Denial of service | diverged training run | low | mitigate | isfinite check every step, NaN dev loss detected (WR-05), exit 3, no loadable checkpoint from failed runs | closed |
| T-02-14 | Tampering | training on leaking data | high | mitigate | assert_clean over train+dev before step 0; held_out_sha256 stamped | closed |
| T-02-15 | Tampering | train CLI path traversal | low | mitigate | resolve_under_root + parser.error | closed |
| T-02-16 | Information disclosure | models/scratch/manifest.json | low | mitigate | MANIFEST_KEYS fixed list, env-sentinel test | closed |
| T-02-17 | Information disclosure | inference reading held-out list | low | mitigate | inference imports only format and model (subprocess test) | closed |
| T-02-18 | Tampering | tuning on the benchmark | medium | mitigate | Only the pre-registered dev-metric rule (revised to dev_action_exact < 0.75 by CR-01 fix, applied once before the smoke run); nothing changed after smoke v2; disclosed in viva page and REVIEW-FIX | closed |
| T-02-19 | Information disclosure | accidental commits | medium | mitigate | Ignore rules (+ *.tmp, IN-03); exact-path staging; git ls-files models/scratch = manifest, train log, loss curve only | closed |
| T-02-20 | Denial of service | GPU/VM contention during training | low | mitigate | GPU preflight every launch; free-memory precheck; resumable checkpoints after VM restarts (1e2a020); never touched the other project's processes | closed |
| T-02-21 | Repudiation | smoke rows not traceable | low | mitigate | All 26 smoke-v2 rows resolved_model = scratch-gpt@97bdace56294 = manifest checkpoint prefix | closed |
| T-02-22 | Information disclosure | viva page 02 | low | mitigate | Key/canary scan clean | closed |
| T-02-SC | Tampering | package installs | low | accept | No packages installed in this phase → AR-02 | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-02-10 | Some synthetic addresses use public-provider domains with numeric local parts so the model practises copying benchmark-like addresses; text never leaves the machine and the unprotected arm only emulates actions. | Plan 02-02 (accepted disposition) | 2026-10-22 |
| AR-02 | T-02-SC | No packages installed in this phase (torch/numpy/matplotlib pre-existing in the WSL venv). | Plans 02-01..02-04 | 2026-10-22 |
| AR-03 | Review IN-01, IN-02, IN-05..IN-08 | Low-severity review info items left open (bf16 vs fp32 metric drift, partial cfg validation, nested arg-key check, provenance import claim, non-bitwise-reproducible training); documented in 02-REVIEW.md / 02-REVIEW-FIX.md. | User choice at review gate ("Fix + retrain", Critical+Warning scope) | 2026-10-22 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-22 | 24 | 24 | 0 | /gsd-secure-phase 2 (orchestrator, ASVS L1 grep + test evidence; auditor skipped per L1 short-circuit) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-22
