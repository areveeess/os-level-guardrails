# OS-Level Guardrails for Autonomous AI Agents

**Author:** Raj Vardhan Singh | 24BDS0200 | Review 1

## Problem Statement
*   Autonomous AI agents (coding agents, browser agents, tool-using assistants) are increasingly granted real execution authority — shell access, file writes, network calls.
*   The OS has no native concept of "an LLM as an untrusted principal" — every existing security boundary assumes the caller is a human or a fixed program, not a probabilistic, promptable one.
*   Current defenses fall into two weak camps:
    *   Application/prompt-layer guardrails (e.g., prompt-filtering middleware) — enforce policy probabilistically and are bypass able by indirect prompt injection.
    *   Ad-hoc container isolation (Docker, seccomp defaults) — restrict resources, not actions.
*   Real-world evidence: MCP filesystem symlink sandbox escapes and the Antigravity sandbox escape show both approaches keep failing in practice.

## Objectives
*   Design a lightweight, OS-level capability-based reference monitor that mediates every action (file, network, process) requested by an autonomous AI agent.
*   Implement policy enforcement using seccomp-bpf and Linux namespaces to intercept and validate agent-issued system calls against a defined capability / access-matrix policy.
*   Detect and block anomalous or policy-violating agent behavior in real time — e.g., a simulated prompt-injected agent attempting data exfiltration or privilege escalation.
*   Evaluate the system with a live demo comparing a sandboxed vs. an unsandboxed agent under the same adversarial (prompt-injected) scenario.

## Literature Review
*   **Prompt injection as a threat class:** Perez & Ribeiro (2022) first named prompt injection as an attack technique against language models.
*   **Standardized evaluation:** AgentDojo (Debenedetti et al., 2024, NeurIPS) and ToolEmu (Ruan et al., ICLR 2024) established benchmark environments for testing prompt-injection attacks and defenses on tool-using agents.
*   **Quantifying compromise:** InjecAgent (2024) shows how indirect prompt injection compromises tool-integrated agents, motivating enforcement below the application layer.
*   **Closest app-layer comparator:** CaMeL (Debenedetti et al., 2025, Google DeepMind / ETH Zürich) split the agent into privileged/quarantined LLMs with capability tracking; strong security, but application-layer and its completion rate drops sharply under strict policy.
*   **Industry framing:** OWASP LLM01:2025 ranks prompt injection as the top risk for LLM applications.
*   **Direct precursor:** AgentSight (Zheng et al., 2025) observe agent behavior at the OS/network boundary using eBPF, correlating stated agent intent with actual system-level actions.
*   **Closest prior art:** ActPlane (Zheng et al., 2026) compile agent-declared policy into eBPF enforced in-kernel; reports blocking 74% of unsafe behaviors across 361 safety tasks at 1.9–8.4% overhead. A 9-author, 4-institution compiler/DSL system — the paper this project differentiates against.
*   **Broader surveys:** “JARVIS or Ultron?” (2025) and “Toward Secure LLM Agents” (2026) map the agent-security threat landscape.
*   **Theory grounding:** “Open Challenges in Multi-Agent Security” (2025) frames CaMeL as a case study in capability-based, defense-in-depth security.
*   **Real CVEs:** CVE-2025-53109/53110 (MCP filesystem symlink escape) and CVE-2025-59528 (Antigravity sandbox escape).

## Research Gap & Novelty
*   **Gap:** Application-layer defenses enforce policy probabilistically and are bypassable; OS-layer work exists but is a full compiler/IFC research system (ActPlane) out of scope for a solo, 8-week build.
*   **Novelty framing:** Where CaMeL is app-layer and ActPlane is a full compiler/IFC research system, this project implements a lightweight capability-based reference monitor — grounded directly in classical OS access-matrix / capability theory — using seccomp-bpf/namespaces, scoped to a single-agent harness on commodity hardware, solo-buildable in 8 weeks.
*   Positions the project as a focused, theory-grounded instance of a direction two independent research groups are validating right now.

## Proposed Architecture
*   **AI Agent:** LLM harness issues a requested action → shell command, file write, or network call.
*   **Interception Layer:** seccomp-bpf filter + Linux namespaces sit between the agent process and the kernel, capturing every requested syscall before execution.
*   **Policy Engine:** A capability / access-matrix table defines what the agent is permitted to do — allow, deny, or flag-for-review.
*   **Decision:** Matched-and-allowed actions pass through; violations are blocked and logged; borderline actions are flagged as anomalous.
*   **Dashboard/Log:** Real-time view of intercepted actions, for the live demo.

## Demo Plan & Expected Outcomes
*   **Live demo:** Run an unsandboxed agent and a sandboxed agent side by side, both under a simulated prompt-injection attempting file exfiltration or privilege escalation.
*   **Unsandboxed agent:** Action succeeds — shows the real-world risk concretely.
*   **Sandboxed agent:** Reference monitor intercepts and blocks the action in real time; dashboard logs the violation.
*   **Expected outcome:** A working capability-enforcement layer that blocks a defined set of unsafe agent actions with measurable overhead, plus a policy table mapped to classical access-matrix theory.

## Resources
[1] Perez, F. & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques for Language Models.
[2] Debenedetti, E. et al. (2024). AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents. NeurIPS 2024.
[3] Ruan, Y. et al. (2023/2024). ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox. ICLR 2024.
[4] InjecAgent (2024). Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents.
[5] Debenedetti, E. et al. (2025). CaMeL: Defeating Prompt Injections by Design. Google DeepMind / ETH Zürich.
[6] OWASP (2025). LLM01:2025 — Prompt Injection. OWASP Top 10 for LLM Applications.
[7] Zheng, X. et al. (2025). AgentSight: System-Level Observability for AI Agents Using eBPF.
[8] Zheng, X. et al. (2026). ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses.
[9] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron? (2025).
[10] Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation (2026).
[11] Open Challenges in Multi-Agent Security (2025).
[12] Prompt Injection Attacks in LLMs and AI Agent Systems: A Comprehensive Review. MDPI (2026).
