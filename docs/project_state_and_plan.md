# Project State and Execution Plan

## 1. Current State
We have successfully established the foundational context for the **OS-Level Guardrails for Autonomous AI Agents** project based on Review 1:
*   **Documentation Created:**
    *   `review_1_presentation.md`: Core summary of the problem, objectives, and demo plan.
    *   `architecture.md`: System design detailing the Agent Harness, Interception Layer (`seccomp-bpf` + namespaces), Policy Engine, and Dashboard.
    *   `planning.md`: 8-week timeline and milestone breakdown.
*   **Resources Acquired:**
    *   Downloaded and cataloged 12 key research papers in the `papers/` directory, including pivotal works like the CaMeL paper, AgentSight, ActPlane, and AgentDojo.
*   **Infrastructure:**
    *   Workspace initialized at `c:\Users\rvsre\.gemini\antigravity\scratch\os_level_guardrails`.

## 2. Technical Execution Plan (Next Steps)
To move from Phase 1 (Research) into Phase 2 (Core Interception Layer), we need to begin writing code for the sandbox and the AI agent harness. Since `seccomp-bpf` and Linux namespaces are low-level OS features, we need to choose an appropriate technology stack and build the minimal prototype.

### Step 2.1: Tech Stack Selection
We must decide on the primary programming language for the **Interception Layer**:
*   **Go (Golang):** Highly recommended. Excellent library support for Linux namespaces and `seccomp` (e.g., `libseccomp-golang`, `cilium/ebpf`). Great for writing a concurrent policy engine and dashboard backend.
*   **Rust:** Excellent memory safety and low-level control (e.g., `libseccomp-rs`, `aya` for eBPF). Steeper learning curve but extremely robust for a reference monitor.
*   **C / C++:** The most native option for `seccomp-bpf`, but requires more manual memory management.
*   **Python:** Will likely be used for the *AI Agent Harness* and *Dashboard*, but is less suited for the actual low-level OS interception layer.

### Step 2.2: Build the AI Agent Harness
Create a simple Python-based tool-using agent that takes a prompt, generates a bash command or Python script via an LLM, and attempts to execute it. This acts as our "untrusted principal".

### Step 2.3: Implement the Sandbox Wrapper
Build the initial sandbox wrapper executable that takes a command, sets up a new Linux user/mount/pid namespace, applies a strict `seccomp` profile, and then `exec`s the AI agent harness inside that restricted environment.

### Step 2.4: Develop the Policy Engine Access Matrix
Define a JSON/YAML based capability matrix (e.g., allowed paths for `openat`, allowed IPs/ports for `connect`, allowed binaries for `execve`).

## 3. Immediate Action Required
To proceed, please let me know:
1.  **Which language do you prefer for building the Interception Layer (Go, Rust, or C)?**
2.  **Would you like me to scaffold the initial project directory structure (e.g., `src/agent_harness`, `src/sandbox`, `src/dashboard`)?**
