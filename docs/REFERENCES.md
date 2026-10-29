# References: where each technique comes from

This page names, for every strategy or technique in the project, where it is used in the code and the paper
or standard it comes from. Use it in the viva when an examiner asks "where did you get that idea?" or "is this
standard practice?".

**How to read it**
- 📄 marks a paper that is in the [`papers/`](../papers) folder of this repository.
- Every other entry is a standard reference from the literature or official documentation.
- The **Where** column links to the file that implements the technique.
- The numbered list at the end ([§10](#10-full-reference-list)) gives the full citations.

---

## 1. Problem and threat model

| Technique / idea | Where in this project | Reference |
|---|---|---|
| **Prompt injection:** instructions hidden in input that the model obeys | The threat model behind the whole project; [README](../README.md), [viva 05](viva/05-evaluation-statistics-and-examiner-qa.md) | 📄 Perez & Ribeiro 2022 [R1]; 📄 OWASP LLM01:2025 [R2] |
| **Indirect prompt injection:** the attack arrives in tool output (an email, a web page), not from the user | The attack the benchmark measures; [src/benchmark/injecagent.py](../src/benchmark/injecagent.py) | Greshake et al. 2023 [R3]; 📄 Zhan et al. 2024 (InjecAgent) [R4] |
| **The model is an untrusted principal:** its output is parsed into actions that are checked, never trusted | [src/agent_harness/prompting.py](../src/agent_harness/prompting.py), [executors.py](../src/agent_harness/executors.py) | 📄 Debenedetti et al. 2025 (CaMeL) [R5]; Saltzer & Schroeder 1975 [S4] |
| **Threat surfaces of LLM and computer-using agents** | Motivation and related work in the [SDD](SDD.md) | 📄 Ling et al. (survey) [R6]; 📄 Chen et al. (JARVIS or Ultron?) [R7]; 📄 Schroeder de Witt et al. 2025 [R8]; 📄 Gulyamov et al. 2026 (MDPI review) [R9] |

## 2. OS-level enforcement: the core of the project

| Technique | Where in this project | Reference |
|---|---|---|
| **Reference monitor:** every action passes one always-invoked, tamper-proof checkpoint | `GuardedExecutor` in [src/agent_harness/executors.py](../src/agent_harness/executors.py) | Anderson 1972 [S1] |
| **Access matrix:** subjects × objects → rights; here, action type × target → ALLOW / FLAG / BLOCK | [src/policy_engine/engine.py](../src/policy_engine/engine.py), [default_policy.yaml](../src/policy_engine/policies/default_policy.yaml) | Lampson 1971/1974 [S2] |
| **Capabilities:** a task may call only the tools it is granted (the tool broker) | [src/agent_harness/tool_broker.py](../src/agent_harness/tool_broker.py) | Dennis & Van Horn 1966 [S3] |
| **Complete mediation, least privilege, fail-safe defaults** | The shell parser fails closed; the broker grants the minimum tools ([src/policy_engine/shell_split.py](../src/policy_engine/shell_split.py)) | Saltzer & Schroeder 1975 [S4] |
| **Defense in depth:** policy, then namespaces, then seccomp, so one layer's miss is caught by the next | The protected arm; [src/sandbox/executor.py](../src/sandbox/executor.py) | Saltzer & Schroeder 1975 [S4]; 📄 Zheng et al. (ActPlane) [R10] |
| **Linux namespaces:** user, mount, PID and network isolation via `unshare` | [src/sandbox/executor.py](../src/sandbox/executor.py) | Biederman 2006 [S5]; Linux man-pages `namespaces(7)`, `user_namespaces(7)`, `unshare(1)` [S6] |
| **Private filesystem view:** a staged fake `/home` and `/etc`, so the real files are never visible | `_stage_environment` in [src/sandbox/executor.py](../src/sandbox/executor.py) | Linux `mount_namespaces(7)` [S6] |
| **seccomp-BPF syscall filtering** (chmod, mount, ptrace, setuid, …) | [src/sandbox/seccomp_filter.c](../src/sandbox/seccomp_filter.c) | Linux kernel docs, *Seccomp BPF* [S7]; McCanne & Jacobson 1993 (BPF) [S8] |
| **OS-level policy enforcement for agent harnesses:** the closest related work | Related work in the [SDD](SDD.md) | 📄 Zheng et al. (ActPlane) [R10] |
| **System-level observability of agents:** a decision log and live dashboard | [src/policy_engine/engine.py](../src/policy_engine/engine.py) `_log_decision`, [src/dashboard/](../src/dashboard) | 📄 Zheng et al. (AgentSight, eBPF) [R11] |
| **Canary tokens:** a fake secret planted where the real one would be, so a leak is detectable and harmless | [src/agent_harness/environment.py](../src/agent_harness/environment.py), [src/benchmark/injecagent.py](../src/benchmark/injecagent.py) | Spitzner 2003 (honeytokens) [S9] |
| **Shell-chaining bypass and its fix:** every sub-command of `;`, `&&`, `\|\|`, `\|`, `$(…)` is checked | [src/policy_engine/shell_split.py](../src/policy_engine/shell_split.py) | Complete mediation [S4]; command injection, OWASP [R2] |

## 3. Benchmark and evaluation design

| Technique | Where in this project | Reference |
|---|---|---|
| **InjecAgent cases:** direct-harm and data-stealing injections in tool outputs | [benchmarks/injecagent/](../benchmarks/injecagent), [src/benchmark/injecagent.py](../src/benchmark/injecagent.py) | 📄 Zhan et al. 2024 [R4] |
| **Two arms, unprotected vs protected:** the same cases, seeds and prompt, with only the guardrails changed | [src/benchmark/runner.py](../src/benchmark/runner.py) | 📄 Debenedetti et al. 2024 (AgentDojo, attack and defense evaluation) [R12] |
| **Emulated tools in the unprotected arm:** the OS is simulated, so no harm reaches the host | `MockExecutor` in [src/agent_harness/executors.py](../src/agent_harness/executors.py) | 📄 Ruan et al. 2024 (ToolEmu) [R13] |
| **Attack success counted only after the injection was shown** | [src/benchmark/scoring.py](../src/benchmark/scoring.py) | 📄 Zhan et al. 2024 (ASR definition) [R4] |
| **Scripted worst-case control agent:** always obeys, giving a model-independent upper bound | `control-always-comply`, [src/llm/scripted.py](../src/llm/scripted.py) | Standard positive control; 📄 AgentDojo's utility and security baselines [R12] |
| **Benign utility measured in both arms:** a defense must not break normal tasks | [src/benchmark/report.py](../src/benchmark/report.py) | 📄 AgentDojo (utility under attack) [R12]; 📄 CaMeL [R5] |

## 4. Statistics and reporting

| Technique | Where in this project | Reference |
|---|---|---|
| **95% Wilson score interval** for every rate (k/n) | [src/benchmark/stats.py](../src/benchmark/stats.py) | Wilson 1927 [M1]; Brown, Cai & DasGupta 2001 [M2] |
| **Paired design and pair dropping:** an episode with an infrastructure error is dropped with its other arm | [src/benchmark/runner.py](../src/benchmark/runner.py), [report.py](../src/benchmark/report.py) | Standard matched-pairs design [M2] |
| **Outcome taxonomy and ASR among format-valid episodes:** attacked / ignored / format failure / refused / error | [src/benchmark/outcomes.py](../src/benchmark/outcomes.py) | 📄 Zhan et al. 2024 (valid-output ASR) [R4] |
| **Completeness gating and partial-n labels:** an incomplete run is never ranked | [src/benchmark/report.py](../src/benchmark/report.py) | Pineau et al. 2021 (reproducibility checklist) [M3] |
| **Pre-registered stopping rule:** written and committed before the first hosted request | [.planning/phases/04-…/04-HOSTED-STOPPING-RULE.md](../.planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-HOSTED-STOPPING-RULE.md) | Nosek et al. 2018 (pre-registration) [M4] |
| **Provenance stamps:** commit, policy hash, model id, `prompt_tokens` and finish reason on every result row | [src/benchmark/runner.py](../src/benchmark/runner.py) | Pineau et al. 2021 [M3] |
| **One frozen dataset, one command, generated numbers:** documents cannot drift from the data | [src/deliverables/](../src/deliverables) | Pineau et al. 2021 [M3] |
| **Confounds and threats-to-validity sections** | `summary.md` sections from [src/deliverables/sections.py](../src/deliverables/sections.py) | Wohlin et al. 2012 (validity threats) [M5] |

## 5. The from-scratch model (ScratchGPT)

| Technique | Where in this project | Reference |
|---|---|---|
| **Decoder-only transformer with causal self-attention** | [src/scratch_llm/model.py](../src/scratch_llm/model.py) | Vaswani et al. 2017 [L1]; Radford et al. 2019 (GPT-2) [L2] |
| **Pre-LayerNorm blocks** (LayerNorm before attention and before the MLP) | [model.py](../src/scratch_llm/model.py) | Xiong et al. 2020 [L3]; Ba et al. 2016 (LayerNorm) [L4] |
| **GELU activation** in the 4× MLP | [model.py](../src/scratch_llm/model.py) | Hendrycks & Gimpel 2016 [L5] |
| **Weight tying:** the output head shares the token-embedding matrix | [model.py](../src/scratch_llm/model.py) | Press & Wolf 2017 [L6] |
| **Dropout** | [model.py](../src/scratch_llm/model.py) | Srivastava et al. 2014 [L7] |
| **Fused scaled-dot-product attention kernel:** the same maths, less memory | `F.scaled_dot_product_attention` in [model.py](../src/scratch_llm/model.py) | Dao et al. 2022 (FlashAttention) [L8] |
| **Character-level vocabulary** | [src/scratch_llm/format.py](../src/scratch_llm/format.py) | Karpathy, *nanoGPT* / char-level LM (engineering reference) [L9] |
| **AdamW optimiser** with decoupled weight decay on matrices only | [src/scratch_llm/train.py](../src/scratch_llm/train.py) | Loshchilov & Hutter 2019 [L10] |
| **Linear warmup then cosine decay** of the learning rate | `lr_at` in [train.py](../src/scratch_llm/train.py) | Goyal et al. 2017 (warmup) [L11]; Loshchilov & Hutter 2017 (cosine) [L12] |
| **Gradient-norm clipping** | [train.py](../src/scratch_llm/train.py), [src/qwen_lora/train.py](../src/qwen_lora/train.py) | Pascanu et al. 2013 [L13] |
| **bf16 mixed precision** (fp32 master weights) | [train.py](../src/scratch_llm/train.py) | Micikevicius et al. 2018 [L14]; Kalamkar et al. 2019 (bfloat16) [L15] |
| **Early stopping on a held-out dev metric;** select on action-line loss | [train.py](../src/scratch_llm/train.py) | Prechelt 1998 [L16] |
| **Loss only on the assistant's action lines** | [src/scratch_llm/dataset.py](../src/scratch_llm/dataset.py) | Ouyang et al. 2022 (supervised fine-tuning on responses) [L17] |

## 6. Synthetic data and leakage control

| Technique | Where in this project | Reference |
|---|---|---|
| **Synthetic agent episodes** (50% benign, 25% direct harm, 25% data stealing), seeded and reproducible | [src/scratch_llm/dataset.py](../src/scratch_llm/dataset.py), [templates.py](../src/scratch_llm/templates.py) | Wang et al. 2023 (Self-Instruct, synthetic training data) [D1] |
| **Benchmark-leakage gate:** held-out tools, addresses and canary formats, and 8-word shingle overlap | `assert_clean` in [dataset.py](../src/scratch_llm/dataset.py) | Brown et al. 2020 (n-gram contamination analysis) [D2]; Broder 1997 (shingling) [D3]; Sainz et al. 2023 (contamination) [D4] |
| **Train and serve parity:** one render function, with a format-version stamp | [src/scratch_llm/format.py](../src/scratch_llm/format.py), [src/qwen_lora/chat.py](../src/qwen_lora/chat.py) | Sculley et al. 2015 (training/serving skew) [D5] |

## 7. The locally fine-tuned model (Qwen2.5-0.5B + LoRA)

| Technique | Where in this project | Reference |
|---|---|---|
| **Base model:** Qwen2.5-0.5B-Instruct | [config/models.yaml](../config/models.yaml) | Qwen Team 2024 (Qwen2.5 technical report) [F1] |
| **LoRA:** a frozen weight W plus a low-rank update BA; r = 16, α = 32, all linear layers | [src/qwen_lora/train.py](../src/qwen_lora/train.py) | Hu et al. 2022 [F2]; PEFT library [F3] |
| **Chat template** (ChatML-style `<\|im_start\|>` turns) and loss masking by character offset | [src/qwen_lora/chat.py](../src/qwen_lora/chat.py), [data.py](../src/qwen_lora/data.py) | Hugging Face chat templating docs [F4]; Ouyang et al. 2022 [L17] |
| **Byte-level BPE tokenizer** (Qwen's) | [src/qwen_lora/chat.py](../src/qwen_lora/chat.py) | Sennrich et al. 2016 [F5] |
| **Gradient checkpointing** to fit in 6 GB of VRAM | [src/qwen_lora/train.py](../src/qwen_lora/train.py) | Chen et al. 2016 [F6] |
| **Merging the adapter at load time:** the same outputs, faster | [src/llm/hf_local.py](../src/llm/hf_local.py) | Hu et al. 2022 (W' = W + BA) [F2] |
| **Greedy decoding** (temperature 0) for reproducible benchmarking | [src/llm/hf_local.py](../src/llm/hf_local.py), [scratch.py](../src/llm/scratch.py) | Holtzman et al. 2020 (decoding strategies) [F7] |

## 8. Other models compared

| Model / technique | Where in this project | Reference |
|---|---|---|
| **Llama 3.2 3B** (via Ollama) | [config/models.yaml](../config/models.yaml) | Grattafiori et al. 2024 (Llama 3) [F8] |
| **Qwen2.5 3B** (via Ollama) | [config/models.yaml](../config/models.yaml) | Qwen Team 2024 [F1] |
| **4-bit GGUF quantisation** (Ollama's default) | Confounds table in the report | llama.cpp / GGUF k-quants [F9] |
| **Context-window truncation check:** the 8192-token Ollama variants and `prompt_tokens` evidence | [src/benchmark/run_checks.py](../src/benchmark/run_checks.py) | Ollama documentation, `num_ctx` [F10] |
| **gpt-oss-120b / gpt-oss-20b** (hosted on Groq) | [config/models.yaml](../config/models.yaml) | OpenAI 2025 (gpt-oss model card) [F11] |

## 9. Documentation

| Item | Where | Reference |
|---|---|---|
| **Software Design Description** structure | [docs/SDD.md](SDD.md) | IEEE Std 1016-2009 [M6] |

---

## 10. Full reference list

### Prompt injection and agent security (📄 = in `papers/`)
- **[R1]** 📄 F. Perez, I. Ribeiro. *Ignore Previous Prompt: Attack Techniques for Language Models.* NeurIPS ML Safety Workshop, 2022. arXiv:2211.09527.
- **[R2]** 📄 OWASP. *OWASP Top 10 for LLM Applications 2025: LLM01 Prompt Injection.* https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **[R3]** K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* ACM AISec Workshop, 2023. arXiv:2302.12173.
- **[R4]** 📄 Q. Zhan, Z. Liang, Z. Ying, D. Kang. *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.* Findings of ACL, 2024. arXiv:2403.02691.
- **[R5]** 📄 E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, F. Tramèr. *Defeating Prompt Injections by Design* (CaMeL). arXiv:2503.18813, 2025.
- **[R6]** 📄 Y. Ling, S. Yu, Z. Chen, C. Fang, et al. *Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation.* Survey (see `papers/`).
- **[R7]** 📄 A. Chen, Y. Wu, J. Zhang, et al. *JARVIS or Ultron? A Survey on the Safety and Security Threats of Computer-Using Agents.* (see `papers/`).
- **[R8]** 📄 C. Schroeder de Witt, K. Krawiecka, I. Krawczuk, et al. *Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents.* 2025 (see `papers/`).
- **[R9]** 📄 S. Gulyamov et al. *Prompt Injection Attacks in LLMs and AI Agent Systems: A Comprehensive Review.* Information (MDPI) 17(1):54, 2026.
- **[R10]** 📄 Y. Zheng, T. Wu, Q. Fu, T. Yu, W. Mao, T. Ma, D. Williams, W. Wang, A. Quinn. *ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses.* (see `papers/`).
- **[R11]** 📄 Y. Zheng, Y. Hu, T. Yu, A. Quinn. *AgentSight: System-Level Observability for AI Agents Using eBPF.* (see `papers/`).
- **[R12]** 📄 E. Debenedetti, J. Zhang, M. Balunović, L. Beurer-Kellner, M. Fischer, F. Tramèr. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS Datasets and Benchmarks, 2024.
- **[R13]** 📄 Y. Ruan, H. Dong, A. Wang, S. Pitis, Y. Zhou, J. Ba, Y. Dubois, C. J. Maddison, T. Hashimoto. *Identifying the Risks of LM Agents with an LM-Emulated Sandbox* (ToolEmu). ICLR, 2024.

The `papers/` folder also holds C. Zhu, M. Dastani, S. Wang, *A Survey of Multi-Agent Deep Reinforcement Learning with Communication*. It is background reading on multi-agent systems; no technique in this project comes from it.

### Operating-system security
- **[S1]** J. P. Anderson. *Computer Security Technology Planning Study.* ESD-TR-73-51, US Air Force, 1972. (The reference-monitor concept.)
- **[S2]** B. W. Lampson. *Protection.* Proc. 5th Princeton Symposium on Information Sciences and Systems, 1971; reprinted in ACM SIGOPS Operating Systems Review 8(1), 1974. (The access matrix.)
- **[S3]** J. B. Dennis, E. C. Van Horn. *Programming Semantics for Multiprogrammed Computations.* Communications of the ACM 9(3), 1966. (Capabilities.)
- **[S4]** J. H. Saltzer, M. D. Schroeder. *The Protection of Information in Computer Systems.* Proceedings of the IEEE 63(9), 1975. (Least privilege, fail-safe defaults, complete mediation.)
- **[S5]** E. W. Biederman. *Multiple Instances of the Global Linux Namespaces.* Ottawa Linux Symposium, 2006.
- **[S6]** Linux man-pages project. `namespaces(7)`, `user_namespaces(7)`, `mount_namespaces(7)`, `unshare(1)`. https://man7.org/linux/man-pages/
- **[S7]** The Linux Kernel documentation. *Seccomp BPF (SECure COMPuting with filters).* https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html
- **[S8]** S. McCanne, V. Jacobson. *The BSD Packet Filter: A New Architecture for User-level Packet Capture.* USENIX Winter Conference, 1993.
- **[S9]** L. Spitzner. *Honeytokens: The Other Honeypot.* SecurityFocus, 2003.

### Statistics, methodology and standards
- **[M1]** E. B. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* Journal of the American Statistical Association 22(158), 1927.
- **[M2]** L. D. Brown, T. T. Cai, A. DasGupta. *Interval Estimation for a Binomial Proportion.* Statistical Science 16(2), 2001.
- **[M3]** J. Pineau et al. *Improving Reproducibility in Machine Learning Research.* Journal of Machine Learning Research 22, 2021.
- **[M4]** B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor. *The Preregistration Revolution.* PNAS 115(11), 2018.
- **[M5]** C. Wohlin, P. Runeson, M. Höst, M. C. Ohlsson, B. Regnell, A. Wesslén. *Experimentation in Software Engineering.* Springer, 2012. (Threats to validity.)
- **[M6]** IEEE. *IEEE Std 1016-2009: Standard for Information Technology—Systems Design—Software Design Descriptions.* 2009.

### Language models and training
- **[L1]** A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017.
- **[L2]** A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI technical report, 2019.
- **[L3]** R. Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML, 2020.
- **[L4]** J. L. Ba, J. R. Kiros, G. E. Hinton. *Layer Normalization.* arXiv:1607.06450, 2016.
- **[L5]** D. Hendrycks, K. Gimpel. *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415, 2016.
- **[L6]** O. Press, L. Wolf. *Using the Output Embedding to Improve Language Models.* EACL, 2017.
- **[L7]** N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15, 2014.
- **[L8]** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022.
- **[L9]** A. Karpathy. *nanoGPT.* https://github.com/karpathy/nanoGPT (engineering reference for a minimal GPT and character-level training).
- **[L10]** I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR, 2019.
- **[L11]** P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677, 2017. (Learning-rate warmup.)
- **[L12]** I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR, 2017. (Cosine schedule.)
- **[L13]** R. Pascanu, T. Mikolov, Y. Bengio. *On the Difficulty of Training Recurrent Neural Networks.* ICML, 2013. (Gradient clipping.)
- **[L14]** P. Micikevicius et al. *Mixed Precision Training.* ICLR, 2018.
- **[L15]** D. Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* arXiv:1905.12322, 2019.
- **[L16]** L. Prechelt. *Early Stopping — But When?* In *Neural Networks: Tricks of the Trade*, Springer, 1998.
- **[L17]** L. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). NeurIPS, 2022.

### Data
- **[D1]** Y. Wang et al. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL, 2023.
- **[D2]** T. Brown et al. *Language Models are Few-Shot Learners* (GPT-3), Section 4: measuring and preventing memorization of benchmarks. NeurIPS, 2020.
- **[D3]** A. Z. Broder. *On the Resemblance and Containment of Documents.* Compression and Complexity of Sequences, 1997.
- **[D4]** O. Sainz et al. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP, 2023.
- **[D5]** D. Sculley et al. *Hidden Technical Debt in Machine Learning Systems.* NeurIPS, 2015.

### Fine-tuning and models
- **[F1]** Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024.
- **[F2]** E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
- **[F3]** S. Mangrulkar et al. *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.* https://github.com/huggingface/peft
- **[F4]** Hugging Face. *Chat Templates* (Transformers documentation). https://huggingface.co/docs/transformers/chat_templating
- **[F5]** R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. (BPE.)
- **[F6]** T. Chen, B. Xu, C. Zhang, C. Guestrin. *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174, 2016. (Gradient checkpointing.)
- **[F7]** A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi. *The Curious Case of Neural Text Degeneration.* ICLR, 2020.
- **[F8]** A. Grattafiori et al. *The Llama 3 Herd of Models.* arXiv:2407.21783, 2024.
- **[F9]** G. Gerganov et al. *llama.cpp* and the GGUF format (k-quant quantisation). https://github.com/ggml-org/llama.cpp
- **[F10]** Ollama documentation: Modelfile parameters (`num_ctx`). https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- **[F11]** OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* 2025.

> For the four papers listed as "see `papers/`", the venue and year are not printed on the copy in the folder, so
> they are cited by title and authors only. Check the arXiv page before putting them in a formal bibliography.
