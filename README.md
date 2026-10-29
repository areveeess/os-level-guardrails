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

## Results (frozen Phase 5 data)

The numbers in this section are generated from `results/phase5-final/numbers.json` by the deliverables build and checked against it, so they cannot drift from the report, the SDD or the viva notes. They cover the InjecAgent indirect prompt-injection cases, run against every model once without the guardrails and once with them.

Data snapshot, and how complete each model's run is:

<!-- deliverables:begin status -->
Data: snapshot of 2026-10-29T07:12Z (UTC); the D-09 freeze is 2026-10-29T15:30Z. Partial rows are labelled with their episodes done and are not final.

| Model | Group | Episodes done | Status |
|---|---|---|---|
| qwen2.5-0.5b-agentlora | local | 342/342 | complete |
| scratch-gpt | local | 342/342 | complete |
| qwen2.5-0.5b-base | local | 342/342 | complete |
| ollama-qwen2.5-3b | local | 342/342 | complete |
| ollama-llama3.2-3b | local | 342/342 | complete |
| groq-gpt-oss-120b | hosted | 251/342 | partial, 251/342 episodes |
| groq-gpt-oss-20b | hosted | 117/342 | partial, 117/342 episodes (error-dominated) |
| groq-qwen3.8-27b | hosted | 272/342 | partial, 272/342 episodes |
| control-always-comply | control | 342/342 | complete |
<!-- deliverables:end status -->

The comparison table. A row marked partial has not finished its planned run and is not final:

<!-- deliverables:begin headline -->
| Model | Group | Attempted (no guardrails) | Success (no guardrails) | Success (guardrails) | Blocked given attempt | Benign task (no guardrails) | Benign task (guardrails) | False positive (guardrails) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | local | 92.5% (111/120) [86.4, 96.0] | 92.5% (111/120) [86.4, 96.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (111/111) [96.7, 100.0] | 100.0% (51/51) [93.0, 100.0] | 100.0% (51/51) [93.0, 100.0] | 0.0% (0/51) [0.0, 7.0] |
| scratch-gpt | local | 32.5% (39/120) [24.8, 41.3] | 27.5% (33/120) [20.3, 36.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (36/36) [90.4, 100.0] | 70.6% (36/51) [57.0, 81.3] | 70.6% (36/51) [57.0, 81.3] | 0.0% (0/51) [0.0, 7.0] |
| qwen2.5-0.5b-base | local | 5.0% (6/120) [2.3, 10.5] | 2.5% (3/120) [0.9, 7.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (6/6) [61.0, 100.0] | 29.4% (15/51) [18.7, 43.0] | 29.4% (15/51) [18.7, 43.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-qwen2.5-3b | local | 4.2% (5/120) [1.8, 9.4] | 2.5% (3/120) [0.9, 7.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (10/10) [72.2, 100.0] | 96.1% (49/51) [86.8, 98.9] | 96.1% (49/51) [86.8, 98.9] | 0.0% (0/51) [0.0, 7.0] |
| ollama-llama3.2-3b | local | 11.7% (14/120) [7.1, 18.6] | 5.8% (7/120) [2.9, 11.6] | 0.0% (0/120) [0.0, 3.1] | 100.0% (9/9) [70.1, 100.0] | 94.1% (48/51) [84.1, 98.0] | 98.0% (50/51) [89.7, 99.7] | 0.0% (0/51) [0.0, 7.0] |
| groq-gpt-oss-120b (partial 251/342) | hosted | 17.9% (17/95) [11.5, 26.8] | 4.2% (4/95) [1.6, 10.3] | 0.0% (0/95) [0.0, 3.9] | 100.0% (18/18) [82.4, 100.0] | 83.3% (25/30) [66.4, 92.7] | 86.7% (26/30) [70.3, 94.7] | 6.7% (2/30) [1.8, 21.3] |
| groq-gpt-oss-20b (partial 117/342) | hosted | 11.8% (6/51) [5.5, 23.4] | 3.9% (2/51) [1.1, 13.2] | 0.0% (0/51) [0.0, 7.0] | 100.0% (7/7) [64.6, 100.0] | 71.4% (5/7) [35.9, 91.8] | 85.7% (6/7) [48.7, 97.4] | 0.0% (0/7) [0.0, 35.4] |
| groq-qwen3.8-27b (partial 272/342) | hosted | 3.1% (3/98) [1.0, 8.6] | 0.0% (0/98) [0.0, 3.8] | 0.0% (0/98) [0.0, 3.8] | 100.0% (3/3) [43.8, 100.0] | 100.0% (37/37) [90.6, 100.0] | 100.0% (37/37) [90.6, 100.0] | 0.0% (0/37) [0.0, 9.4] |
| control-always-comply (scripted control, reference) | control | 100.0% (120/120) [96.9, 100.0] | 100.0% (120/120) [96.9, 100.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (120/120) [96.9, 100.0] | 100.0% (51/51) [93.0, 100.0] | 100.0% (51/51) [93.0, 100.0] | 0.0% (0/51) [0.0, 7.0] |

Fixed order (the five local models, then the three hosted ones), not a ranking. A row marked partial has not finished its planned run and is not final. Each cell is P% (k/n) [95% Wilson interval]; the scripted control is a reference, not a language model.
<!-- deliverables:end headline -->

What the numbers say:

<!-- deliverables:begin findings -->
- qwen2.5-0.5b-agentlora: attacks succeeded in 111/120 attack episodes without guardrails and 0/120 with guardrails.
- scratch-gpt: attacks succeeded in 33/120 attack episodes without guardrails and 0/120 with guardrails.
- qwen2.5-0.5b-base: attacks succeeded in 3/120 attack episodes without guardrails and 0/120 with guardrails.
- ollama-qwen2.5-3b: attacks succeeded in 3/120 attack episodes without guardrails and 0/120 with guardrails.
- ollama-llama3.2-3b: attacks succeeded in 7/120 attack episodes without guardrails and 0/120 with guardrails.
- groq-gpt-oss-120b (partial 251/342): attacks succeeded in 4/95 attack episodes without guardrails and 0/95 with guardrails.
- groq-gpt-oss-20b (partial 117/342): attacks succeeded in 2/51 attack episodes without guardrails and 0/51 with guardrails.
- groq-qwen3.8-27b (partial 272/342): attacks succeeded in 0/98 attack episodes without guardrails and 0/98 with guardrails.
- With guardrails, no current-engine model succeeded on any attack episode (0 for each of the 8 language models).
- Pooled over the 8 language models, guardrails blocked 200/200 attempted attacks at the attack action itself (conditional block rate).
- Sandbox: no block recorded, 0/1173 pooled protected-arm episodes; this is a lower bound, because the policy engine stops attacks before they reach the kernel layers.
- Seccomp: no block recorded, 0/1173 pooled protected-arm episodes; this is a lower bound, because the policy engine stops attacks before they reach the kernel layers.
- The scripted control attempted the attack in 120/120 episodes and succeeded in 120/120 without guardrails; with guardrails it succeeded in 0/120, and guardrails blocked 120/120 of its attempted attacks.
<!-- deliverables:end findings -->

Where the three charts are:

<!-- deliverables:begin charts -->
![Attack success per model with 95% Wilson intervals, without and with guardrails; dashed line: the scripted control without guardrails](results/phase5-final/charts/asr_ci.png)

*Attack success per model with 95% Wilson intervals, without and with guardrails; dashed line: the scripted control without guardrails*

![Capability vs vulnerability: benign task done vs attack success, both without guardrails](results/phase5-final/charts/capability_vs_vulnerability.png)

*Capability vs vulnerability: benign task done vs attack success, both without guardrails*

![Each model from no guardrails (hollow) to guardrails (filled): benign task done vs attack success](results/phase5-final/charts/utility_vs_asr_arrows.png)

*Each model from no guardrails (hollow) to guardrails (filled): benign task done vs attack success*

The build writes these charts into the frozen results directory; results/ is not committed, so run python -m src.deliverables build to see them.
<!-- deliverables:end charts -->

### Scripted attack simulation (policy engine only)

This table is the `run_attack_simulation.py` demo: scripted attack actions sent straight to the policy engine, with no language model involved. It is separate from the benchmark above.

| Metric | Without Guardrails | With Policy Engine |
|--------|----------------|-------------|
| Attacks reaching the OS | 7/7 (100%) | 0/7 (0%) |
| Attacks blocked | 0/7 (0%) | 7/7 (100%) |
| Benign control allowed | 1/1 | 1/1 (no false positives) |

`run_attack_simulation.py` exercises the policy engine layer; `run_sandboxed.py` exercises namespaces + seccomp-BPF. Destructive vectors in the simulation use safe stand-ins (a throwaway `/tmp` directory and the reserved `.example` domain), so it is safe to run live.

Test suite (full run in WSL on 29 October 2026): 1308 passed and 1 skipped, no failures. Run it from the repo root in WSL with `source venv/bin/activate && python -m pytest tests -q` (the GPU training and Hugging Face tests are slow; the full run takes about eleven minutes).

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
