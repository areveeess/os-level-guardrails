# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Academic evaluators, lab reviewers, and security researchers inspecting lab benchmarks, reproducibility, and attack simulation metrics for the BCSE303P Operating Systems Lab project.

## Product Purpose
Provide real-time security observability and verification for autonomous AI agent execution, proving that a multi-layer OS-level defense (policy engine, namespace isolation, and seccomp-bpf syscall filtering) successfully neutralizes agent attacks while allowing legitimate tasks.

## Positioning
Unlike probabilistic app-layer prompt guards or complex multi-author frameworks, this project delivers a lightweight, solo-buildable reference monitor implementing classical access-matrix theory and kernel-level enforcement directly on Linux/WSL2.

## Operating Context
- Environments: Linux / WSL2 Ubuntu, Python 3.10+, GCC with `libseccomp-dev`, Google Gemini API.
- Evaluation workflows: Running comparative attack simulations (`run_attack_simulation.py`), sandboxed agent execution (`run_sandboxed.py`), and real-time telemetry streaming into the Flask dashboard (`http://localhost:5000`).
- Telemetry store: `logs/guardrails.jsonl`.

## Capabilities and Constraints
- Three-layer defense architecture:
  1. Access-Matrix Policy Engine (YAML capability rules: Allow / Block / Flag)
  2. Linux Namespace Isolation (`unshare` for user, PID, and network boundaries)
  3. Syscall Filtering (`seccomp-bpf` compiled filter blocking high-risk syscalls: `chmod`, `mount`, `ptrace`, `setuid`)
- Two-phase execution model: LLM harness runs outside the sandbox, while tool actions execute strictly inside.
- Real-time Flask dashboard displaying live metrics, event tables with color-coded decisions, and log controls.

## Brand Commitments
- Project: OS-Level Guardrails for Autonomous AI Agents
- Author: Raj Vardhan Singh (24BDS0200) | Course: BCSE303P Operating Systems Lab
- Voice: Precise, academic, technically rigorous, and security-conscious.

## Evidence on Hand
- Attack simulation benchmark: 7 attack vectors + 1 benign control. Unprotected, 7/7 attacks reach the OS; with the policy engine, 7/7 are blocked and the benign control is allowed (no false positives).
- 28/28 pytest tests passing (21 policy engine, 6 sandbox block detection, 1 dashboard flow).
- Documented literature review of 12 peer-reviewed papers in `papers/`.
- Working codebase across harness, policy engine, C seccomp filter, and Flask UI.

## Product Principles
1. **Defense-in-Depth Observability**: Every mediation decision (policy engine vs. sandbox vs. syscall filter) must be visibly distinguishable.
2. **Verifiable Reproducibility**: Evaluators must be able to independently verify test benchmarks and attack outcomes against clear logs.
3. **High-Signal Telemetry**: Clear visual separation of normal operations, policy flags, and critical blocked threats.

## Accessibility & Inclusion
- High-contrast status indicators with redundant icons for Allow (green check), Block (red cross), and Flag (amber alert).
- Respects `prefers-reduced-motion` and supports dense, readable tabular data navigation.
