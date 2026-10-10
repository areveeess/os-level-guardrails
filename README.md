# OS-Level Guardrails for Autonomous AI Agents

A lightweight, OS-level capability-based reference monitor that mediates every action (file, network, process) requested by an autonomous AI agent. Built using `seccomp-bpf`, Linux namespaces, and classical access-matrix theory.

> **Course:** BCSE303P Operating Systems Lab  
> **Author:** Raj Vardhan Singh | 24BDS0200

---

## 🎯 Problem

Autonomous AI agents (coding agents, browser agents, tool-using assistants) are increasingly granted real execution authority — shell access, file writes, network calls. But the OS has no native concept of "an LLM as an untrusted principal." Current defenses are insufficient:

- **App-layer guardrails** (prompt filtering) — enforce policy probabilistically and are bypassable via indirect prompt injection
- **Ad-hoc container isolation** (Docker, default seccomp) — restrict resources, not actions

## 💡 Solution

This project implements a **three-layer defense** for AI agents:

| Layer | Technology | What It Does |
|-------|-----------|-------------|
| **Policy Engine** | Python + YAML access-matrix | Checks every agent action against a capability rulebook (Allow/Block/Flag) |
| **Namespace Isolation** | Linux `unshare` (user, PID, net) | Gives the agent an isolated filesystem, process tree, and network stack |
| **Syscall Filtering** | `seccomp-bpf` (C) | Blocks dangerous kernel calls (chmod, mount, ptrace, setuid) at the kernel level |

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  AI Agent    │────▶│ Policy Engine    │────▶│   OS Kernel  │
│  (Gemini)    │     │ (Access Matrix)  │     │  (seccomp +  │
│              │     │ Allow/Block/Flag │     │  namespaces) │
└──────────────┘     └──────────────────┘     └──────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │    Dashboard     │
                     │  (Real-Time UI)  │
                     └──────────────────┘
```

## 📁 Project Structure

```
os_level_guardrails/
├── src/
│   ├── agent_harness/          # Module 1: LLM-powered AI agent (Gemini API)
│   │   ├── agent.py            #   Agent: prompt → LLM → action → execute
│   │   └── tools.py            #   Tool executor: shell, file, network ops
│   ├── policy_engine/          # Module 2: Access-matrix policy evaluator
│   │   ├── engine.py           #   YAML policy loader + action checker
│   │   └── policies/
│   │       └── default_policy.yaml  # The rulebook (allow/block lists)
│   ├── sandbox/                # Module 3: OS-level isolation
│   │   ├── sandbox.py          #   Two-phase: LLM outside, tools inside sandbox
│   │   ├── seccomp_filter.c    #   BPF syscall filter (C, ~150 lines)
│   │   └── Makefile            #   Compiles seccomp filter
│   └── dashboard/              # Module 4: Real-time monitoring
│       ├── app.py              #   Flask web server
│       └── templates/
│           └── index.html      #   Dark-themed dashboard UI
├── docs/                       # Documentation & review materials
├── papers/                     # 12 research papers (literature review)
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── run_agent.py                # Run agent WITHOUT sandbox
├── run_sandboxed.py            # Run agent WITH full sandbox
├── run_attack_simulation.py    # Side-by-side attack comparison demo
├── run_demo.sh                 # Full 4-scenario demo script
├── setup.sh                    # One-command WSL2 setup
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- **WSL2 / Ubuntu** (or any Linux environment)
- **Python 3.10+**
- **GCC** and **libseccomp-dev**
- **Google Gemini API key** ([get one free](https://aistudio.google.com/apikey))

### Setup (One Command)
```bash
git clone <repo-url> && cd os_level_guardrails
chmod +x setup.sh && ./setup.sh
```

### Configure API Key
```bash
cp .env.example .env
# Edit .env and paste your Gemini API key
```

### Run the Demo
```bash
source venv/bin/activate

# Run agent WITHOUT protection (attacks succeed)
python3 run_agent.py 'List files in /tmp'

# Run agent WITH full sandbox (attacks blocked)
python3 run_sandboxed.py 'Read /etc/shadow'

# Run side-by-side attack comparison
python3 run_attack_simulation.py

# Start the real-time dashboard
python3 -m src.dashboard.app
# Open http://localhost:5000
```

## 📊 Results

| Metric | Without Guardrails | With Policy Engine |
|--------|----------------|-------------|
| Attacks reaching the OS | 7/7 (100%) | 0/7 (0%) |
| Attacks blocked | 0/7 (0%) | 7/7 (100%) |
| Benign control allowed | 1/1 | 1/1 (no false positives) |

Test suite: 28/28 pytest tests passing (21 policy engine, 6 sandbox block detection, 1 dashboard flow).

`run_attack_simulation.py` exercises the policy engine layer; `run_sandboxed.py` exercises namespaces + seccomp-BPF. Destructive vectors in the simulation use safe stand-ins (a throwaway `/tmp` directory and the reserved `.example` domain), so it is safe to run live.

## 🔬 Novelty & Positioning

| System | Layer | Approach | Complexity |
|--------|-------|----------|-----------|
| **CaMeL** (Google DeepMind, 2025) | App-layer | Dual-LLM + custom interpreter | High (bypassable) |
| **ActPlane** (Zheng et al., 2026) | OS-layer | IFC DSL → eBPF compiler | Very High (9 authors) |
| **This Project** | OS-layer | seccomp-bpf + namespaces + access-matrix | **Lightweight (solo-buildable)** |

## 📚 Key References

1. Perez & Ribeiro (2022) — *Ignore Previous Prompt*
2. Debenedetti et al. (2024) — *AgentDojo* (NeurIPS)
3. Debenedetti et al. (2025) — *CaMeL* (Google DeepMind)
4. Zheng et al. (2025) — *AgentSight* (eBPF observability)
5. Zheng et al. (2026) — *ActPlane* (eBPF enforcement)
6. OWASP (2025) — *LLM01: Prompt Injection*

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
