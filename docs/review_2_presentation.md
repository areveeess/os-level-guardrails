# OS-Level Guardrails for Autonomous AI Agents — Review 2

**Author:** Raj Vardhan Singh | 24BDS0200  
**Course:** BCSE303P Operating Systems Lab  
**Review:** Project Review 2 — System Design, Module Identification & Completion

---

## 1. Proposed Model Architecture

### 1.1 High-Level Architecture

The system is a **three-layer defense-in-depth reference monitor** that mediates every action an autonomous AI agent attempts on the host OS. It implements **two-phase execution**: the LLM reasoning phase runs outside the sandbox (requires network), while tool execution runs inside a fully isolated environment.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         USER / EVALUATOR                               │
 │   Natural language prompt  ──►  run_agent.py / run_sandboxed.py        │
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │
 ═══════════════════════════════════╪══════════════════════════════════════
  PHASE 1: OUTSIDE SANDBOX         │   (needs network for Gemini API)
 ═══════════════════════════════════╪══════════════════════════════════════
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                   MODULE 1 — AI AGENT HARNESS                       │
 │  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐  │
 │  │  agent.py    │───▶│  Gemini LLM API   │───▶│  JSON Action     │  │
 │  │  (Untrusted  │    │  (gemini-3.6-     │    │  Parser          │  │
 │  │   Principal) │    │     flash)        │    │                  │  │
 │  └──────────────┘    └───────────────────┘    └────────┬─────────┘  │
 │                                                        │            │
 │  ┌──────────────┐                                      │            │
 │  │  tools.py    │◄─────────────────────────────────────┘            │
 │  │  (Tool       │  Parsed action: run_shell / read_file /           │
 │  │   Executor)  │  write_file / http_request / respond              │
 │  └──────┬───────┘                                                   │
 └─────────┼───────────────────────────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │               MODULE 2 — POLICY ENGINE (Access-Matrix)              │
 │  ┌───────────────────┐    ┌────────────────────────────────────┐    │
 │  │  engine.py        │◄───│  default_policy.yaml               │    │
 │  │  - check()        │    │  (YAML Access-Matrix Table)        │    │
 │  │  - _check_shell   │    │  ┌────────────────────────────┐    │    │
 │  │  - _check_fs      │    │  │ Subject: AI Agent          │    │    │
 │  │  - _check_network │    │  │ Objects: files, cmds, net  │    │    │
 │  │  - _log_decision  │    │  │ Perms: ALLOW / BLOCK / FLAG│    │    │
 │  └────────┬──────────┘    │  └────────────────────────────┘    │    │
 │           │               └────────────────────────────────────┘    │
 │           │                                                         │
 │     ┌─────┴─────┐                                                   │
 │     │ Decision   │                                                   │
 │     ├────────────┤                                                   │
 │     │ ALLOW ──────────────────────────────────────► Phase 2          │
 │     │ BLOCK ──────────────────────────────────────► Log & Stop       │
 │     │ FLAG  ──────────────────────────────────────► Log & Proceed    │
 │     └────────────┘                                                   │
 └──────────────────────────────────────────┬──────────────────────────┘
                                            │
 ═══════════════════════════════════════════╪═══════════════════════════
  PHASE 2: INSIDE SANDBOX                  │   (isolated environment)
 ═══════════════════════════════════════════╪═══════════════════════════
                                            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                MODULE 3 — OS-LEVEL SANDBOX                          │
 │                                                                      │
 │  ┌─────────────────────────────────────────────────────────────┐    │
 │  │  sandbox.py — Two-Phase Orchestrator                        │    │
 │  │  Sandbox class: run_agent() → Phase 1 + Phase 2             │    │
 │  └─────────────────────────┬───────────────────────────────────┘    │
 │                            │                                        │
 │            ┌───────────────┴───────────────┐                        │
 │            ▼                               ▼                        │
 │  ┌─────────────────────┐     ┌──────────────────────────────────┐  │
 │  │  Layer 2:           │     │  Layer 3:                        │  │
 │  │  Linux Namespaces   │     │  Seccomp-BPF Filter              │  │
 │  │  ┌───────────────┐  │     │  ┌────────────────────────────┐  │  │
 │  │  │ unshare       │  │     │  │  seccomp_filter.c (C)      │  │  │
 │  │  │  --user       │  │     │  │  ───────────────────────── │  │  │
 │  │  │  --pid --fork │  │     │  │  Blocked syscalls:         │  │  │
 │  │  │  --net        │  │     │  │  • connect() — network     │  │  │
 │  │  ├───────────────┤  │     │  │  • chmod/chown — perms     │  │  │
 │  │  │ Isolation:    │  │     │  │  • mount/umount — fs       │  │  │
 │  │  │ • Own PIDs    │  │     │  │  • ptrace — debugging      │  │  │
 │  │  │ • No network  │  │     │  │  • reboot — shutdown       │  │  │
 │  │  │ • Unprivileged│  │     │  │  • init_module — rootkits  │  │  │
 │  │  └───────────────┘  │     │  │  • setuid/setgid — privesc │  │  │
 │  └─────────────────────┘     │  └────────────────────────────┘  │  │
 │                              │  Kernel-enforced, irremovable     │  │
 │                              └──────────────────────────────────┘  │
 └───────────────┬──────────────────────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                     OS / LINUX KERNEL                                │
 │   Executes permitted syscalls only. Blocked calls return EPERM.      │
 └──────────────────────────────────────────────────────────────────────┘

                 ║  (All decisions logged to logs/guardrails.jsonl)
                 ▼

 ┌──────────────────────────────────────────────────────────────────────┐
 │              MODULE 4 — REAL-TIME DASHBOARD                         │
 │  ┌───────────────┐    ┌────────────────────────────────────────┐    │
 │  │  app.py       │    │  index.html                            │    │
 │  │  (Flask)      │───▶│  Dark-themed single-page app           │    │
 │  │  REST API:    │    │  • Live metric counters                │    │
 │  │  /api/logs    │    │  • Color-coded event table             │    │
 │  │  /api/stats   │    │  • Defense-layer badges                │    │
 │  │  /api/clear   │    │  • 2-second auto-polling               │    │
 │  │  /api/info    │    │  • WCAG AA accessible                  │    │
 │  └───────────────┘    └────────────────────────────────────────┘    │
 └──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow Diagram

```
 User Prompt
     │
     ▼
 ┌─────────┐   JSON Action   ┌──────────────┐   Decision   ┌───────────┐
 │  Gemini  │ ──────────────▶ │ Policy Engine│ ───────────▶ │  Sandbox  │
 │  LLM     │                 │ (YAML rules) │              │ (ns+bpf)  │
 └─────────┘                  └──────┬───────┘              └─────┬─────┘
                                     │                            │
                              ┌──────▼───────┐              ┌─────▼─────┐
                              │  guardrails  │              │  OS Kernel│
                              │  .jsonl (log)│              │  (exec)   │
                              └──────┬───────┘              └───────────┘
                                     │
                              ┌──────▼───────┐
                              │  Dashboard   │
                              │  (Flask UI)  │
                              └──────────────┘
```

### 1.3 Theoretical Foundation

| Concept | Implementation |
|:--------|:---------------|
| **Access-Matrix Model** (Lampson, 1971) | `default_policy.yaml` — rows = AI agent (subject), columns = files/cmds/network (objects), cells = ALLOW/BLOCK/FLAG (permissions) |
| **Reference Monitor** | `PolicyEngine.check()` — mediates every action before OS execution |
| **Principle of Least Privilege** | Seccomp-BPF drops dangerous capabilities; only minimal syscalls permitted |
| **Defense-in-Depth** | Three independent layers — any single layer can stop an attack |
| **Complete Mediation** | Every agent action is checked; no bypass path exists |

---

## 2. Module List

| # | Module | Directory | Language | LOC | Status |
|:--|:-------|:----------|:---------|:----|:-------|
| 1 | **AI Agent Harness** | `src/agent_harness/` | Python | ~346 | ✅ Complete |
| 2 | **Policy Engine** | `src/policy_engine/` | Python + YAML | ~369 | ✅ Complete |
| 3 | **OS-Level Sandbox** | `src/sandbox/` | Python + C | ~537 | ✅ Complete |
| 4 | **Real-Time Dashboard** | `src/dashboard/` | Python + HTML/CSS/JS | ~1,084 | ✅ Complete |
| 5 | **Attack Simulation** | Root (`run_attack_simulation.py`) | Python | ~160 | ✅ Complete |
| 6 | **Test Suite** | `tests/` | Python (pytest) | ~106 | ✅ Complete |

---

## 3. Module Descriptions

### Module 1: AI Agent Harness (`src/agent_harness/`)

**Purpose:** Creates the *untrusted principal* — a tool-using AI agent powered by Google Gemini that translates natural language prompts into executable system actions.

**Files:**

| File | Description |
|:-----|:------------|
| `agent.py` | Core agent class (`AIAgent`) — sends prompts to Gemini LLM, parses JSON action responses, optionally checks policy engine, and routes to tool executor. Implements the 4-step pipeline: Prompt → LLM → Parse → Execute. |
| `tools.py` | Tool executor class (`ToolExecutor`) — provides 4 OS interaction methods: `run_shell()` (subprocess), `read_file()` (file I/O), `write_file()` (file I/O), `http_request()` (urllib). Each method returns structured result dicts. |

**Key Classes:**

- **`AIAgent`** — Configures Gemini (`gemini-3.6-flash`), manages the prompt-to-action pipeline, enforces policy decisions (ALLOW/BLOCK/FLAG).
- **`ToolExecutor`** — Routes parsed JSON actions to the correct OS operation. Each method is intentionally unrestricted to demonstrate the risk without guardrails.

**OS Concepts:** Process creation (`subprocess.run`), file system operations (`open`, `read`, `write`), network I/O (`urllib`).

---

### Module 2: Policy Engine (`src/policy_engine/`)

**Purpose:** The *decision-making component* — evaluates every agent action against a YAML-defined access-matrix before execution. Maps directly to Lampson's (1971) access-matrix model.

**Files:**

| File | Description |
|:-----|:------------|
| `engine.py` | Core policy evaluator (`PolicyEngine`) — loads YAML policy, evaluates actions against allow/block lists using glob matching (`fnmatch`), performs shell argument introspection (e.g., blocks `cat /etc/shadow` even though `cat` is allowed), and logs every decision to `logs/guardrails.jsonl`. |
| `policies/default_policy.yaml` | The access-matrix table — defines filesystem read/write rules (6 allow, 8 block paths for read; 2 allow, 7 block paths for write), process execution rules (15 allowed commands, 20 blocked commands), and network rules (all outbound blocked by default). |

**Key Classes:**

- **`PolicyEngine`** — Implements `check(action_type, target)` with first-match-wins evaluation. Sub-methods: `_check_shell_command()`, `_check_filesystem()`, `_check_network()`, `_check_command_file_args()`. Default action is FLAG for unmatched rules.

**Access-Matrix Mapping:**

```
             │  /tmp/*   │  /etc/shadow  │  curl   │  ls   │  http://*
─────────────┼───────────┼───────────────┼─────────┼───────┼──────────
 AI Agent    │  ALLOW    │  BLOCK        │  BLOCK  │ ALLOW │  BLOCK
```

**OS Concepts:** Access control lists, capability-based security, glob pattern matching, append-only audit logging.

---

### Module 3: OS-Level Sandbox (`src/sandbox/`)

**Purpose:** Provides the *kernel-level security boundary* — combines Linux namespaces for environment isolation and seccomp-BPF for syscall filtering. Implements two-phase execution to separate LLM reasoning from tool execution.

**Files:**

| File | Description |
|:-----|:------------|
| `sandbox.py` | Sandbox orchestrator (`Sandbox` class) — implements two-phase execution. Phase 1 (outside sandbox): LLM call + policy check. Phase 2 (inside sandbox): tool execution via `unshare` + `seccomp_filter`. Includes graceful fallback modes for partial environments. |
| `seccomp_filter.c` | C program (~153 lines) using `libseccomp` — creates a BPF filter context, adds rules to block 15 dangerous syscalls, loads the filter into the kernel (irremovable), then `execvp()`s the target command inside the filtered environment. |
| `Makefile` | Compiles `seccomp_filter.c` → `seccomp_filter` binary with `gcc -lseccomp`. |

**Key Classes:**

- **`Sandbox`** — Orchestrates the full defense pipeline. Methods: `run_agent()` (main entry), `_execute_in_full_sandbox()` (namespace + seccomp), `_execute_in_namespace()` (namespace only), `_execute_with_policy_only()` (fallback).

**Blocked Syscalls (seccomp-BPF):**

| Syscall | Attack Vector Blocked |
|:--------|:---------------------|
| `connect()` | Outbound network / data exfiltration |
| `chmod()` / `fchmod()` | Permission escalation |
| `chown()` / `fchown()` / `lchown()` | Ownership changes |
| `mount()` / `umount2()` | Filesystem mounting |
| `ptrace()` | Process injection / debugging |
| `reboot()` | System shutdown |
| `init_module()` / `finit_module()` / `delete_module()` | Kernel rootkit installation |
| `setuid()` / `setgid()` / `setreuid()` / `setregid()` | Privilege escalation |

**Namespace Isolation:**

| Namespace | Effect |
|:----------|:-------|
| `--user` | Unprivileged user mapping (no root) |
| `--pid --fork` | Isolated PID namespace (can't see host processes) |
| `--net` | Empty network namespace (no network access) |

**OS Concepts:** Linux namespaces (`unshare`), seccomp-BPF (Berkeley Packet Filter), syscall filtering, process isolation, principle of least privilege.

---

### Module 4: Real-Time Dashboard (`src/dashboard/`)

**Purpose:** Provides *real-time security observability* — a web-based interface displaying all policy decisions, metrics, and defense-layer attribution. Enables evaluators to verify system behavior during live demos.

**Files:**

| File | Description |
|:-----|:------------|
| `app.py` | Flask web server with 4 REST API endpoints: `/api/logs` (all events), `/api/stats` (summary counters), `/api/clear` (reset logs), `/api/info` (runtime metadata). Implements `mtime`-based caching to minimize file I/O during 2-second polling. |
| `templates/index.html` | Single-page dark-themed dashboard (~926 lines) — metric cards (Total/Allowed/Blocked/Flagged), scrollable event table with action-type pills, defense-layer badges, decision tags with icons, copy-to-clipboard interaction, live/disconnected status indicator, responsive design, and WCAG AA accessibility compliance. |

**Key Features:**

- **Auto-polling**: Fetches `/api/stats` and `/api/logs` every 2 seconds for real-time updates.
- **Defense-layer attribution**: Each event shows which layer caught it — Policy Engine, Namespace, or Seccomp-BPF.
- **Accessibility**: Screen-reader announcer, `prefers-reduced-motion` support, high-contrast color tokens (≥ 6.98:1 contrast ratio), 44px minimum touch targets.

**OS Concepts:** Client-server architecture, REST API design, real-time telemetry, append-only log file consumption.

---

### Module 5: Attack Simulation (`run_attack_simulation.py`)

**Purpose:** Demonstrates what happens when prompt injection *succeeds* — the LLM's safety filters are bypassed and the agent attempts 8 adversarial actions both with and without sandbox protection.

**Attack Scenarios:**

| # | Attack | Action Type | Expected Decision |
|:--|:-------|:------------|:------------------|
| 1 | Read `/etc/passwd` | `read_file` | BLOCK |
| 2 | Read SSH private key | `read_file` | BLOCK |
| 3 | Read `/etc/shadow` | `read_file` | BLOCK |
| 4 | `rm -rf` (throwaway demo dir) | `run_shell` | BLOCK |
| 5 | Download & execute malware | `run_shell` | BLOCK |
| 6 | Exfiltrate data via HTTP | `http_request` | BLOCK |
| 7 | `chmod 777` (throwaway demo file) | `run_shell` | BLOCK |
| 8 | `ls /tmp` (safe — control) | `run_shell` | ALLOW |

**Results:** 7/7 attacks blocked (100%) and 1/1 benign control allowed (no false positives) — the system stops every attack without over-restricting legitimate operations. Destructive vectors use safe stand-ins (a throwaway `/tmp` directory, the reserved `.example` domain) so the demo is safe to run live; the policy engine treats them identically to the real commands.

---

### Module 6: Test Suite (`tests/`)

**Purpose:** Automated verification of the policy engine's access-matrix evaluation logic.

**File:** `test_policy_engine.py` — 21 test cases across 5 test classes using `pytest`.

| Test Class | Tests | What It Verifies |
|:-----------|:------|:-----------------|
| `TestFilesystemReadRules` | 5 | Correct ALLOW/BLOCK for file read paths |
| `TestFilesystemWriteRules` | 3 | Correct ALLOW/BLOCK for file write paths |
| `TestProcessRules` | 9 | Shell command allow/block including argument introspection (`cat /etc/shadow`) |
| `TestNetworkRules` | 2 | All outbound network requests blocked |
| `TestTextResponses` | 2 | Text-only responses always allowed |

**Result:** Full pytest suite: 28/28 passing (21 policy engine, 6 sandbox block detection in `test_sandbox_detection.py`, 1 dashboard flow in `test_dashboard_clear.py`).

---

## 4. Module Interaction Diagram

![Module Interaction Diagram](images/module_interaction_diagram.png)

---

## 5. Project File Map

```
os_level_guardrails/
├── src/
│   ├── agent_harness/              ← MODULE 1
│   │   ├── agent.py                   AIAgent class (Gemini LLM integration)
│   │   └── tools.py                   ToolExecutor (shell, file, network ops)
│   ├── policy_engine/              ← MODULE 2
│   │   ├── engine.py                  PolicyEngine (access-matrix evaluator)
│   │   └── policies/
│   │       └── default_policy.yaml    Access-matrix rules (ALLOW/BLOCK/FLAG)
│   ├── sandbox/                    ← MODULE 3
│   │   ├── sandbox.py                 Sandbox (two-phase orchestrator)
│   │   ├── seccomp_filter.c           BPF syscall filter (C, 153 lines)
│   │   └── Makefile                   Compiles seccomp_filter binary
│   └── dashboard/                  ← MODULE 4
│       ├── app.py                     Flask web server (REST API)
│       └── templates/
│           └── index.html             Dark-themed dashboard UI (926 lines)
├── tests/                          ← MODULE 6
│   └── test_policy_engine.py          21 pytest test cases
├── run_agent.py                       Unprotected agent runner
├── run_sandboxed.py                   Protected agent runner
├── run_attack_simulation.py        ← MODULE 5 (8 attack scenarios)
├── run_demo.sh                        Full 4-scenario demo script
├── setup.sh                           One-command WSL2 setup
├── requirements.txt                   Python dependencies
├── papers/                            12 research papers (literature review)
└── docs/
    ├── architecture.md                System architecture document
    ├── planning.md                    8-week project timeline
    ├── review_1_presentation.md       Review 1 materials
    └── review_2_presentation.md       This document
```

---

## 6. Evidence of Completion

| Criterion | Evidence |
|:----------|:---------|
| System Design | Three-layer architecture with two-phase execution model |
| Module Identification | 6 modules identified: Agent Harness, Policy Engine, Sandbox, Dashboard, Attack Simulation, Test Suite |
| Module Completion | All 6 modules implemented and functional |
| Test Suite | 28/28 pytest passing |
| Attack Simulation | 7/7 attacks blocked, 1/1 benign control allowed |
| Dashboard | Fully functional real-time UI at `http://localhost:5000` |
| Seccomp Filter | Compiled C binary blocking 15 dangerous syscalls |

---

## 7. References

1. Lampson, B. W. (1971). *Protection.* — Access-matrix model (theoretical foundation)
2. Perez, F. & Ribeiro, I. (2022). *Ignore Previous Prompt.*
3. Debenedetti, E. et al. (2024). *AgentDojo.* NeurIPS 2024.
4. Debenedetti, E. et al. (2025). *CaMeL: Defeating Prompt Injections by Design.* Google DeepMind.
5. Zheng, X. et al. (2025). *AgentSight: System-Level Observability Using eBPF.*
6. Zheng, X. et al. (2026). *ActPlane: Programmable OS-Level Policy Enforcement.*
7. OWASP (2025). *LLM01:2025 — Prompt Injection.*
