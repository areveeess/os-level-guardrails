# OS-Level Guardrails for Autonomous AI Agents

A lightweight, OS-level capability-based reference monitor that mediates every action (tool call, file, network, process) requested by an autonomous AI agent, whichever language model drives it: a hosted model, a local model served through Ollama, or a model trained on this laptop. The monitor is built from a task-scoped tool broker, a YAML access-matrix policy engine, Linux namespaces and `seccomp-bpf`, and it is evaluated on the InjecAgent indirect prompt-injection benchmark with the same cases and settings for every model.

> **Course:** BCSE303P Operating Systems Lab  
> **Author:** Raj Vardhan Singh | 24BDS0200

---

## Review 3: a locally made LLM and other LLMs

The review panel's remark on the earlier, Gemini-only version of the project:

> *"use this on a locally made LLM and also compare with other LLMs other than Gemini"*

How this version answers it:

- **A locally made LLM, trained from scratch.** ScratchGPT is a small decoder-only transformer written and trained from scratch on this laptop (RTX 3060 Laptop, 6 GB) on synthetic agent episodes. It has no safety training at all, so it is a deliberately naive, unaligned principal.
- **A locally made LLM, by fine-tuning.** Qwen2.5-0.5B-Instruct is LoRA-tuned here on the same synthetic episodes, and the untuned base model runs next to it as its control.
- **Other LLMs than Gemini.** Two local models run through Ollama (qwen2.5:3b and llama3.2:3b) and three hosted models run on Groq (gpt-oss-120b, gpt-oss-20b and qwen3.8-27b). The October Gemini and Gemma runs are kept only as clearly labelled legacy rows.
- **The same test for every model.** Every model gets the same system prompt (`src/agent_harness/prompting.py`), the same InjecAgent cases, seeds and arms (without and with the guardrails) and temperature 0. A scripted always-comply control shows what the guardrails do when the agent obeys every injection. The InjecAgent tools are held out of all local training data.

## Problem

Autonomous AI agents (coding agents, browser agents, tool-using assistants) are increasingly granted real execution authority: shell access, file writes, network calls. But the OS has no native concept of "an LLM as an untrusted principal." Current defenses are insufficient:

- **App-layer guardrails** (prompt filtering) enforce policy probabilistically and are bypassable via indirect prompt injection
- **Ad-hoc container isolation** (Docker, default seccomp) restricts resources, not actions

Whichever model is used, its output is treated as untrusted text. The guardrails sit outside the model, so they must hold for a weak, unaligned local model as well as for a large hosted one.

## Solution

Every action the agent proposes passes four layers, in this order:

| Layer | Technology | What it does |
|-------|-----------|-------------|
| **Tool broker** | Python, task-scoped capability lists | Lets the agent call only the tools its task was granted; an injected call to any other tool is denied before anything runs |
| **Policy engine** | Python + YAML access matrix | Checks every action (shell command, file path, network target) against the capability rulebook: ALLOW, FLAG or BLOCK |
| **Namespaces with a private filesystem** | Linux `unshare` (user, mount, PID, net) | Runs each permitted command in a private view of the filesystem (staged home, root and tmp directories with canary files), its own process tree and no network |
| **Syscall filtering** | `seccomp-bpf` (C, libseccomp) | Blocks dangerous kernel calls (chmod, mount, ptrace, setuid) at the kernel level, installed after the private mounts are set up |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Provider layer (src/llm): the model is untrusted and only writes │
│ text. hosted (Groq) · Ollama · hf_local (Qwen base and LoRA) ·   │
│ ScratchGPT (scratch) · scripted control                          │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▼ reply text
┌──────────────────────────────────────────────────────────────────┐
│ AI agent (any LLM): think → act → observe; parses one action     │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▼ action
┌────────────────┐   ┌───────────────────────┐   ┌─────────────────┐
│  Tool broker   │──▶│    Policy engine      │──▶│    OS kernel    │
│  (task-scoped  │   │   (access matrix:     │   │  namespaces +   │
│  capabilities) │   │   ALLOW/FLAG/BLOCK)   │   │  private FS +   │
│                │   │                       │   │  seccomp-bpf    │
└────────────────┘   └───────────┬───────────┘   └─────────────────┘
                                 ▼ decision log
                     ┌───────────────────────┐
                     │      Dashboard        │
                     │    (real-time UI)     │
                     └───────────────────────┘
```

## Models compared

All rows come from `config/models.yaml`. Every model gets the same prompt, cases, seeds, arms and temperature 0.

| Model id | Provider | Where it runs | What it is |
|---|---|---|---|
| `scratch-gpt` | scratch | local, in-process (PyTorch) | ScratchGPT, trained from scratch here |
| `qwen2.5-0.5b-agentlora` | hf_local | local, in-process (Hugging Face, offline) | Qwen2.5-0.5B-Instruct, LoRA-tuned here on the synthetic episodes |
| `qwen2.5-0.5b-base` | hf_local | local, in-process (Hugging Face, offline) | Qwen2.5-0.5B-Instruct, pretrained, not tuned here |
| `ollama-qwen2.5-3b` | openai_compat | local, Ollama server | pulled Ollama tag `qwen2.5:3b`, served with an 8192-token context |
| `ollama-llama3.2-3b` | openai_compat | local, Ollama server | pulled Ollama tag `llama3.2:3b`, served with an 8192-token context |
| `groq-gpt-oss-120b` | openai_compat | hosted, Groq | hosted vendor model `openai/gpt-oss-120b` |
| `groq-gpt-oss-20b` | openai_compat | hosted, Groq | hosted vendor model `openai/gpt-oss-20b` |
| `groq-qwen3.8-27b` | openai_compat | hosted, Groq | hosted vendor model `qwen/qwen3.8-27b` |
| `control-always-comply` | scripted | local, in-process | scripted control that obeys every injection; a reference, not a language model |

The October runs of `gemini-flash-lite` and `gemma-4-31b` were produced by an older engine. They appear only in a "Legacy rows (older engine, not compared)" table, never in the comparison or the charts.

## Project Structure

```
os_level_guardrails/
├── src/
│   ├── agent_harness/          # The agent loop; the LLM is the untrusted principal
│   │   ├── agent.py            #   AIAgent: wires a provider from the registry to the loop
│   │   ├── loop.py             #   think → act → observe loop (Episode, Step)
│   │   ├── prompting.py        #   The one system prompt and action parser for every model
│   │   ├── tool_broker.py      #   Task-scoped tool capabilities (layer 1)
│   │   ├── executors.py        #   Executor interface, GuardedExecutor (policy check)
│   │   ├── environment.py      #   Files, web content and canaries staged per episode
│   │   └── tools.py            #   Tool executor: shell, file, network ops
│   ├── llm/                    # Provider layer (one interface for every model)
│   │   ├── base.py             #   LLMProvider / LLMResponse (never raises)
│   │   ├── registry.py         #   get_provider(name) from config/models.yaml
│   │   ├── openai_compat.py    #   Hosted (Groq, Gemini) and Ollama
│   │   ├── hf_local.py         #   Qwen base and LoRA, in-process, offline, greedy
│   │   ├── scratch.py          #   ScratchGPT checkpoint
│   │   └── scripted.py         #   Always-comply control agent
│   ├── policy_engine/          # Access-matrix policy evaluator (layer 2)
│   │   ├── engine.py           #   YAML policy loader + action checker
│   │   ├── shell_split.py      #   Sees every command a shell line would run
│   │   └── policies/
│   │       └── default_policy.yaml  # The rulebook (allow/block lists)
│   ├── sandbox/                # OS-level isolation (layers 3 and 4)
│   │   ├── executor.py         #   unshare namespaces + private filesystem per episode
│   │   ├── sandbox.py          #   Two-phase: LLM outside, tools inside sandbox
│   │   ├── seccomp_filter.c    #   BPF syscall filter (C)
│   │   └── Makefile            #   Compiles seccomp filter
│   ├── benchmark/              # InjecAgent evaluation
│   │   ├── injecagent.py       #   Cases and canaries
│   │   ├── runner.py           #   Arms (no guardrails / guardrails), episodes, backoff
│   │   ├── scoring.py, outcomes.py, classify.py, stats.py  # Scoring, outcomes, Wilson intervals
│   │   ├── report.py           #   Cross-model report and tables
│   │   ├── jobs.py             #   Restart-proof local queue, hosted loop, Ollama ownership
│   │   └── run_checks.py       #   Mechanical pass/fail checks over result directories
│   ├── scratch_llm/            # ScratchGPT: model, compact format, synthetic data, training
│   ├── qwen_lora/              # Qwen2.5-0.5B LoRA: chat render, data, training, gates
│   ├── deliverables/           # Freeze, report, charts, numbers.json, doc blocks, check
│   └── dashboard/              # Real-time monitoring (Flask app + dark-themed UI)
├── config/
│   └── models.yaml             # Model registry (provider, endpoint, key variable name)
├── benchmarks/injecagent/      # InjecAgent cases and tools.json
├── models/                     # Training manifests and logs (tracked); weights git-ignored
├── docs/                       # SDD.md, viva/ (study notes), review materials
├── papers/                     # Research papers (literature review)
├── scripts/                    # Utility scripts (export_sdd.py, run_full_benchmark.sh)
├── tests/                      # pytest suite
├── results/                    # Benchmark output and the frozen results (not committed)
├── run_agent.py                # Run agent WITHOUT sandbox
├── run_sandboxed.py            # Run agent WITH full sandbox
├── run_attack_simulation.py    # Side-by-side attack comparison demo
├── run_benchmark.py            # Multi-model InjecAgent benchmark
├── run_demo.sh                 # Full 4-scenario demo script
├── setup.sh                    # One-command WSL2 setup
└── requirements.txt            # Python dependencies
```

## Quick Start

### Prerequisites
- **WSL2 / Ubuntu** (or any Linux environment)
- **Python 3.10+**
- **GCC** and **libseccomp-dev**
- **API keys are optional and per provider.** Only the hosted rows need one: `GROQ_API_KEY` for the Groq models, `GEMINI_API_KEY` for the legacy Gemini and Gemma rows. Ollama, hf_local (Qwen base and LoRA), ScratchGPT and the scripted control need no key.
- **Optional:** an NVIDIA GPU with a CUDA build of PyTorch for the local models, and a local Ollama server for the Ollama rows

### Setup (One Command)
```bash
git clone <repo-url> && cd os_level_guardrails
chmod +x setup.sh && ./setup.sh
```

### Configure API Keys (only for hosted models)
```bash
cp .env.example .env
# Edit .env and add the keys for the providers you use (see .env.example)
```

### Run the Demo
```bash
source venv/bin/activate

# Run agent WITHOUT protection (attacks succeed)
python3 run_agent.py 'List files in /tmp'

# Run agent WITH full sandbox (attacks blocked)
python3 run_sandboxed.py 'Read /etc/shadow'

# Pick any model from config/models.yaml
python3 run_agent.py --model ollama-llama3.2-3b 'List files in /tmp'

# Run side-by-side attack comparison
python3 run_attack_simulation.py

# Start the real-time dashboard
python3 -m src.dashboard.app
# Open http://localhost:5000
```

### Run the Benchmark
```bash
# One-off multi-model run (resumable)
python3 run_benchmark.py --models all

# Restart-proof runs, as used for the frozen results (writes results/run1-<model>/)
python3 -m src.benchmark.jobs local --prefix results/run1     # local models, one after another
python3 -m src.benchmark.jobs hosted --prefix results/run1    # hosted loop under the stopping rule
python3 -m src.benchmark.jobs status --prefix results/run1    # episodes done per model
python3 -m src.benchmark.jobs report --prefix results/run1 --out results/run1-report
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
| Attacks reaching the OS | all 7 | none of the 7 |
| Attacks blocked | none of the 7 | all 7 |
| Benign control allowed | yes (the 1 benign action) | yes (no false positives) |

`run_attack_simulation.py` exercises the policy engine layer; `run_sandboxed.py` exercises namespaces + seccomp-BPF. Destructive vectors in the simulation use safe stand-ins (a throwaway `/tmp` directory and the reserved `.example` domain), so it is safe to run live.

Test suite (full run in WSL on 29 October 2026): 1308 passed and 1 skipped, no failures. Run it from the repo root in WSL with `source venv/bin/activate && python -m pytest tests -q` (the GPU training and Hugging Face tests are slow; the full run takes about eleven minutes).

## Reproducing the results

Every result number in this README, the SDD, the viva notes and the Review-3 deck comes from one frozen results directory, built by one command. Run it on Windows from the repo root with Python 3.12, which has matplotlib, PyYAML and python-pptx:

```bash
PYTHONUTF8=1 py -3.12 -m src.deliverables build
```

It does the following:

- Freezes the Phase 4 result directories (the five local models, the scripted control, the three Groq models) and the legacy October directories, read-only, into `results/phase5-final/` (sources and `freeze.json`).
- Regenerates the cross-model report (`summary.md`, `summary.csv`, `layer_blocks.csv`), `numbers.json` and the three charts in `charts/` from that frozen copy only.
- Rewrites the generated result blocks in this README, `docs/SDD.md` and the viva pages, and builds the Review-3 slide deck.
- Runs the consistency check, which fails if a generated block is stale, if the prose outside the blocks states a headline result by hand or quotes a fraction (`k/n` or `k of n`), a percentage or a timestamp that is not in `numbers.json`, or if a document carries key-shaped text.

The check can also be run on its own:

```bash
PYTHONUTF8=1 py -3.12 -m src.deliverables check
```

The hosted Groq runs are rate-limited and stop at the freeze, 15:30 UTC on 2026-10-29. Whether the frozen data predates the freeze, and which rows are still partial, is stated by the generated sentence below; after the freeze the same build command is run once more, and every generated block is refreshed.

<!-- deliverables:begin claim-state -->
The data is a snapshot of 2026-10-29T07:12Z (UTC), copied before the D-09 freeze of 2026-10-29T15:30Z while the hosted loop was still collecting. groq-gpt-oss-120b (partial 251/342), groq-gpt-oss-20b (partial 117/342), groq-qwen3.8-27b (partial 272/342) are labelled partial with their episodes done; they are not final and are never ranked against the complete rows.
<!-- deliverables:end claim-state -->

## Viva notes

The study notes for the oral examination are in [docs/viva/README.md](docs/viva/README.md), one page per phase:

- **01:** Reference monitor, access matrix, capabilities, complete mediation and the evaluation metrics (ASR, Wilson intervals, paired design, the control agent)
- **02:** Transformers, the training loop and greedy decoding behind ScratchGPT; the synthetic data and the leakage gate
- **03:** Fine-tuning and LoRA, chat templates, loss masking and bf16 behind the Qwen LoRA model
- **04:** Namespaces, seccomp-bpf, defense in depth, per-layer blocks, quantization, context windows and local inference
- **05:** Evaluation statistics, threats to validity and the consolidated examiner questions

## Novelty & Positioning

| System | Layer | Approach | Complexity |
|--------|-------|----------|-----------|
| **CaMeL** (Google DeepMind, 2025) | App-layer | Dual-LLM + custom interpreter | High (bypassable) |
| **ActPlane** (Zheng et al., 2026) | OS-layer | IFC DSL → eBPF compiler | Very High (9 authors) |
| **This Project** | OS-layer | tool broker + access-matrix + namespaces + seccomp-bpf, tested on hosted and locally made LLMs | **Lightweight (solo-buildable)** |

## Key References

1. Perez & Ribeiro (2022) — *Ignore Previous Prompt*
2. Debenedetti et al. (2024) — *AgentDojo* (NeurIPS)
3. Zhan et al. (2024) — *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents* (Findings of ACL)
4. Debenedetti et al. (2025) — *CaMeL* (Google DeepMind)
5. Zheng et al. (2025) — *AgentSight* (eBPF observability)
6. Zheng et al. (2026) — *ActPlane* (eBPF enforcement)
7. OWASP (2025) — *LLM01: Prompt Injection*

## License

MIT License — see [LICENSE](LICENSE) for details.
