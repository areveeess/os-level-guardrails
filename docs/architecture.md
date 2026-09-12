# OS-Level Guardrails for Autonomous AI Agents - Architecture

This document outlines the architecture for the lightweight, OS-level capability-based reference monitor designed to mediate actions requested by an autonomous AI agent.

## System Overview

The system introduces an interception layer and policy engine between the AI agent and the host operating system. It relies on classical OS access-matrix theory to evaluate and restrict actions such as shell commands, file writes, and network calls.

## Core Components

### 1. AI Agent (LLM Harness)
*   **Description**: The untrusted principal. This is the environment or harness running the autonomous AI agent.
*   **Function**: Issues requested actions, which eventually translate to system calls (e.g., shell commands, file writes, network calls).

### 2. Interception Layer
*   **Description**: The core mediation component sitting directly between the agent process and the kernel.
*   **Technologies**: `seccomp-bpf` and `Linux namespaces`.
*   **Function**: Captures every requested system call made by the AI Agent *before* execution.

### 3. Policy Engine
*   **Description**: The decision-making component.
*   **Mechanism**: Uses a capability / access-matrix table to define what the agent is permitted to do.
*   **Decisions**:
    *   **Allow**: The action matches a permitted capability and passes through to the OS.
    *   **Flag**: Borderline actions are flagged as anomalous for review.
    *   **Block**: Policy violations (e.g., unauthorized file exfiltration or privilege escalation) are blocked immediately.

### 4. OS / Kernel
*   **Function**: Executes the permitted syscalls passed down by the Policy Engine.

### 5. Log / Alert Dashboard
*   **Description**: An observability interface.
*   **Function**: Provides a real-time view of intercepted actions, decisions made (Allow/Flag/Block), and alerts. This will power the live demonstration of the system.

## Flow of Execution

1.  The **AI Agent** attempts an action (e.g., reading a sensitive file).
2.  The action translates into a syscall, which is intercepted by the **Interception Layer** using `seccomp-bpf` and namespaces.
3.  The syscall and its arguments are evaluated by the **Policy Engine** against the access-matrix table.
4.  A **Decision** is made:
    *   If **Allowed**, the syscall proceeds to the **OS/Kernel**.
    *   If **Flagged** or **Blocked**, the syscall is handled accordingly and logged.
5.  The action and decision are recorded and displayed on the **Log / Alert Dashboard**.
