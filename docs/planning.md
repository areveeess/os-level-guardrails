# OS-Level Guardrails for Autonomous AI Agents - Project Planning

## Project Objectives

*   Design and build a lightweight, OS-level capability-based reference monitor for AI agents.
*   Implement policy enforcement using `seccomp-bpf` and `Linux namespaces`.
*   Detect and block policy-violating agent behavior (e.g., data exfiltration, privilege escalation) in real-time.
*   Complete the project within an 8-week timeframe (solo build).
*   Conduct a live demo comparing sandboxed vs. unsandboxed agents.

## Scope & Constraints

*   **Scope**: Scoped to a single-agent harness running on commodity hardware.
*   **Constraint**: Must be a solo-buildable project within 8 weeks. Avoid building a full compiler/IFC research system (like ActPlane), focusing instead on a lightweight reference monitor based on classical OS access-matrix theory.

## Implementation Phases & Milestones

### Phase 1: Research & Design (Weeks 1-2)
*   Review classical OS access-matrix and capability theory.
*   Design the capability/access-matrix table schema for defining agent permissions.
*   Research `seccomp-bpf` and `Linux namespaces` implementation details for syscall interception.

### Phase 2: Core Interception Layer (Weeks 3-4)
*   Develop the AI agent harness/wrapper.
*   Implement Linux namespaces for basic process isolation.
*   Develop and attach `seccomp-bpf` filters to intercept relevant system calls (file access, network, execution).

### Phase 3: Policy Engine Integration (Weeks 5-6)
*   Build the Policy Engine to process intercepted syscalls.
*   Implement the access-matrix table evaluation logic (Allow, Flag, Block).
*   Integrate the Policy Engine with the Interception Layer to enforce decisions in real-time.

### Phase 4: Observability & Dashboard (Week 7)
*   Implement a logging mechanism for intercepted actions and policy decisions.
*   Build a real-time Log/Alert Dashboard to visualize the reference monitor's activity.

### Phase 5: Testing, Evaluation & Demo Prep (Week 8)
*   **Test Scenario**: Develop a simulated prompt-injection attack aimed at file exfiltration or privilege escalation.
*   **Control Setup (Unsandboxed)**: Run the scenario without the reference monitor to demonstrate the real-world risk (attack succeeds).
*   **Experimental Setup (Sandboxed)**: Run the scenario with the reference monitor enabled.
*   **Validation**: Ensure the reference monitor intercepts and blocks the action in real-time, and that the dashboard correctly logs the violation.
*   Measure and document the overhead of the capability-enforcement layer.

## Expected Outcomes
*   A working capability-enforcement layer with measurable overhead.
*   A policy table mapped to classical access-matrix theory.
*   A live demo illustrating the effectiveness of the OS-level guardrails compared to an unprotected agent.
