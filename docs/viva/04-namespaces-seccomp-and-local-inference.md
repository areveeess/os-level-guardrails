# Viva notes 4: namespaces, seccomp-bpf and the confounds of local inference

These notes are for an Operating Systems viva. Phase 4 of the project ran every local model through the
full benchmark (57 cases x 3 runs x 2 arms = 342 episodes each), re-ran three hosted Groq models under a
written stopping rule, re-checked an old Ollama run for silent truncation, and counted, per guardrail layer,
how many blocks were actually recorded. This page explains the two kernel layers of the guardrails (Linux
namespaces and seccomp-bpf), why we report them the way we do, and the traps that make a comparison between
a 4-bit local model and a hosted model unfair if nobody states them.

An honest framing comes first, because an examiner will ask. In every Phase 4 run the sandbox and the
seccomp filter recorded zero blocks: 0 of 1151 protected-arm episodes (report snapshot of 2026-10-29T03:06Z). That is not a failure of those
layers and it is not evidence that they work. It means the attacks never reached them: the application-level
layers in front of them (the tool broker's capability list and the policy engine) stopped every attempted
attack first. The evidence of what the kernel layers contain comes from the Phase 1 residual-bypass table,
which we cite by row number, and the zero is reported with its n, never padded (decision D-07b).

How to use this page: read the glossary once, read Part A (the ideas) and Part B (how this repository
applies them), look at Part C (the real numbers from this run) the night before, and practise Part D out
loud until each answer takes under a minute. The section "Limits of this phase" is the honest list of what
the numbers do not prove. The decision labels D-01 to D-09 (with D-07b) are the design decisions recorded
for Phase 4 and are cited where they apply.

Words you will meet and what they mean:

- **Kernel**: the core of the operating system. It owns the hardware, memory, files and network, and every
  program has to ask it for those things.
- **User space**: everything that is not the kernel: ordinary programs, the Python agent, the shell. Code
  here cannot touch hardware directly.
- **System call (syscall)**: a request from a user-space program to the kernel, for example "open this
  file" (openat), "connect to this address" (connect) or "change this file's permissions" (chmod). It is the
  only door between a process and the kernel, which is why security layers watch it.
- **Process**: a running program with its own memory and an identity (a PID) given by the kernel.
- **PID (process identifier)**: the number the kernel uses to name a process.
- **Namespace**: a kernel feature that gives a group of processes their own private copy of one kind of
  global resource, so what they see differs from what the rest of the system sees.
- **User namespace**: a private table of user ids. Inside it a process can be "root" (id 0) while outside it
  is still the ordinary unprivileged user who started it.
- **Mount namespace**: a private list of what filesystem is attached where. Mounts made inside it are
  invisible outside, and the outside cannot be changed from inside.
- **PID namespace**: a private process-number space. The first process inside is PID 1 and it cannot see
  or signal processes outside.
- **Network namespace**: a private network stack. A fresh one has only a loopback interface that is down,
  so there is no route anywhere: no internet, no local network.
- **unshare**: the util-linux command (and the system call of the same name) that starts a program in new
  namespaces.
- **Bind mount**: attaching an existing directory or file at a second place in the tree, so for example a
  fake file can be shown at the path of a real one.
- **tmpfs**: a filesystem that lives only in memory and starts empty. Mounting one over a directory hides
  whatever was there.
- **Private filesystem view**: the combination of a mount namespace, bind mounts and tmpfs that makes a
  process see a fake home directory, a canary password file and an empty /mnt instead of the real ones.
- **Canary**: a fake secret with a unique random value. If it ever appears where it should not, we know a
  leak happened and exactly which episode caused it.
- **seccomp (secure computing mode)**: a Linux feature that lets a process install a filter which the kernel
  runs on every system call that process makes.
- **BPF (Berkeley Packet Filter)**: a tiny, safe instruction language the kernel can run itself. It was
  invented for filtering network packets; seccomp reuses it to filter system calls.
- **seccomp-bpf filter**: a small BPF program that looks at a system call's number (and optionally its
  arguments) and returns a verdict: allow, return an error, or kill the process.
- **EPERM**: the error code "Operation not permitted". Our filter answers a forbidden system call with it,
  so the program gets an error message instead of dying.
- **SIGSYS and the kill action**: the alternative verdict. The kernel kills the process with the signal
  SIGSYS ("bad system call"); the shell then reports exit code 159 (128 + 31).
- **Deny-list versus allow-list**: a deny-list allows everything except named items; an allow-list
  forbids everything except named items. Our seccomp filter is a deny-list (default allow, named calls
  refused), which is simpler but weaker.
- **no_new_privs**: a per-process flag that promises the process can never gain privileges, for example
  through a setuid program. The kernel requires it (or admin rights) before an unprivileged process may
  install a seccomp filter; libseccomp sets it by default.
- **Defense in depth**: several independent layers, each of which must fail for an attack to succeed.
- **Tool broker (capability list)**: the first guardrail layer. For each case it holds the list of tools the
  user's task may call; any other tool call is refused before it runs.
- **Policy engine**: the second layer. It checks each action's type and target (a shell command, a file
  path, a URL) against ordered YAML rules and returns ALLOW, FLAG or BLOCK.
- **Recorded block**: a block that the code wrote into the episode row with its layer name. For the
  kernel layers this happens only when a command fails with error text that matches a known signature.
- **Lower bound**: a number the true value cannot be below. A count of recorded blocks is a lower bound on
  the number of times a layer actually contained something.
- **Quantization**: storing a model's weights with fewer bits (for example 4 instead of 16) so it fits in
  less memory and runs faster, at some cost in quality.
- **Q4_K_M**: the 4-bit quantization format Ollama's default tags use (roughly 4.5 to 5 bits per weight
  on average, because some tensors are kept at higher precision).
- **bf16 (bfloat16)**: a 16-bit number format; our in-process Hugging Face models run in it.
- **Context window**: the largest number of tokens a model reads at once, the prompt plus the reply.
- **num_ctx**: Ollama's name for the context window it allocates for a model.
- **Silent front truncation**: when a prompt is longer than the window, Ollama cuts tokens from the front
  and continues without any error, so the model may lose its system prompt and tool list.
- **prompt_tokens**: the number of prompt tokens the server says it actually processed; we log it on every
  step, which is how truncation becomes visible.
- **Modelfile variant**: a new Ollama model tag built from an existing one with changed parameters; ours
  add a fixed num_ctx of 8192.
- **Temperature 0 / greedy decoding**: always choosing the most likely next token. It makes a model nearly,
  but not perfectly, repeatable.
- **Pseudo-replication**: treating repeated measurements of the same thing as if they were independent
  samples, which makes the intervals look narrower than they should.
- **Wilson interval**: a 95 % confidence interval for a proportion k/n that stays sensible when k is 0 or n.
- **Rule of three**: if 0 events are seen in n independent trials, the 95 % upper bound on the true rate
  is roughly 3/n.
- **Stopping rule**: a rule, written down and committed before a run starts, that says when the run stops.
  It prevents stopping at a convenient moment.
- **Data freeze**: the fixed time after which no new data is collected. Ours is 2026-10-29 15:30 UTC (D-09).
- **QuotaExhausted**: the runner's stop when a free hosted API says its daily allowance is used up.
- **Partial row**: a model that has fewer usable episodes than the 342 planned. It is shown with its n and
  never ranked against complete rows.
- **Infra error and pair dropping**: an episode that failed because of the network or the API (not the
  model). If one arm of a case-run pair has an infra error, both arms are dropped so the comparison stays
  paired.
- **ds / dh**: the two InjecAgent attack kinds: data-stealing (exfiltrate a secret) and direct harm (make
  the agent do something damaging).
- **Selection effect**: a bias that appears because the data you have is not a random sample of the data
  you planned, for example only the ds episodes of a hosted model finished.
- **Error-dominated**: the report's label for a model with more infra errors than finished episodes.

---

## Part A — The ideas

### System calls are the boundary

A program cannot open a file, send a packet or start another program by itself. It has to ask the kernel
through a system call. That makes the system-call interface the one narrow door every effect passes
through, and it is why an OS-level guardrail sits there: whatever language the agent's code is written in,
and whatever text the model produced, the effect on the machine is a sequence of system calls. A layer that
checks system calls does not need to understand the command, the script or the model.

### Linux namespaces: four private views

A namespace gives a group of processes their own copy of one kind of global resource. We use four.

- The **user namespace** maps our ordinary user id to id 0 inside. The process feels like root, so it can
  set up mounts in its own namespace, but outside it has no more rights than before. This is what lets an
  unprivileged student account build a sandbox without sudo.
- The **mount namespace** gives the process its own list of mounts. We change that list (bind mounts and a
  tmpfs, below) and the host never sees the change.
- The **PID namespace** gives the process its own process numbers. It cannot see or signal any process of
  the host, including the other project that shares this WSL machine.
- The **network namespace** gives the process a network stack with no working interface. There is no
  route anywhere, so any attempt to reach the internet fails with "network is unreachable" or a name
  resolution error. No data can leave the machine from inside.

Namespaces isolate what a process can **see and name**. They do not stop it from calling a dangerous
system call; they only change what that call acts on.

### The private filesystem view

Inside the new mount namespace the sandbox bind-mounts per-episode staging directories over /home, /root
and /tmp, bind-mounts a canary copy over /etc/shadow, and mounts an empty tmpfs over /mnt. On WSL /mnt is
where the Windows drives, the project folder and the secrets file live, so an empty tmpfs there hides all
of them. The agent's commands run in this view: writes persist across the steps of one episode (it is a
real directory) but never touch the real system, and the staging tree is deleted at the end of the
episode. If an attack reads the "password file", it reads a canary, so a leak is detected rather than
real.

### seccomp-bpf: a filter the kernel runs on every system call

seccomp lets a process hand the kernel a small BPF program. From then on, before every system call the
process makes, the kernel runs that program on the call's number and arguments and acts on the verdict.
Two verdicts matter here. **Return an error** (our choice: EPERM, "Operation not permitted") lets the
program carry on and report the failure; **kill** ends the process with SIGSYS. Returning an error is
gentler and gives the agent a readable observation; killing is stricter because the program cannot try
again in the same process.

Two properties make the filter strong. It **cannot be removed**: once installed it applies to the process
and to every child it starts, and no later call, not even by root inside the namespace, can uninstall or
loosen it (a process may only add stricter filters). And it is **checked in the kernel**, so no
user-space trick such as renaming a binary or calling a different interpreter avoids it: whatever program
runs, the same system call gets the same answer.

The order of setup matters. Our filter refuses mount(), so all the mounts of the private view must be done
**before** the filter is installed. The sandbox therefore sets up the namespaces and mounts first, then
starts the filter program, which installs the filter and only then runs the agent's command.

### Defense in depth, and what none of the layers protects

The guardrails are four layers in order: the tool broker (is this tool allowed for this task?), the policy
engine (is this command, path or URL allowed?), the sandbox (what can the command see?) and seccomp (which
system calls may it make?). The two application layers understand the agent's intent and stop most
attacks early with a clear reason. The two kernel layers do not understand anything; they are a backstop
for whatever the application layers miss, for example a command written in a form the policy rules did not
foresee.

No layer protects against an action that is **allowed and harmful**. If the user's task legitimately needs
the email tool, an injected instruction that sends a misleading email through that tool passes every
layer, because each layer sees an allowed tool, an allowed target and harmless system calls. Guardrails
bound what an agent can do; they do not make it do the right thing.

### Local-inference confounds

Comparing a model on this laptop with a hosted model mixes many differences at once. Four are specific to
local inference.

**Quantization.** The Ollama models (qwen2.5:3b and llama3.2:3b) run in Q4_K_M, roughly 4.5 to 5 bits per weight.
The two Qwen 0.5B rows run in bf16 in-process. The Groq models run at whatever precision Groq serves
(gpt-oss is published natively in a 4-bit format). So a row difference mixes model size, training,
precision and serving stack. We present the rows as a spectrum of capability, not a leaderboard.

**Context window and silent front truncation.** Ollama allocates a window per model (num_ctx). If a prompt
is longer, it cuts from the front without an error, and the model may lose the system prompt and the tool
list, start producing invalid replies and then "look safe". We measured the default on this machine rather
than assuming it: 4,096 tokens for llama3.2:3b on Ollama 0.40.3 (read from the server's own status). We
then built Modelfile variants with num_ctx 8192 for both Ollama models (D-06), proved the window is really
8192 by sending a long prompt whose prompt_tokens landed inside a band above the default, and logged
prompt_tokens on every step. One surprise: Ollama 0.40.3 cuts an over-long prompt to about half the
window, so truncation shows as a drop well below the window, not a plateau at the top.

**Temperature-0 determinism.** At temperature 0 a model picks the most likely token every time, so the
three runs of a case are near-copies. On a GPU (and on hosted APIs, which batch requests) tiny numerical
differences can still change a reply, so runs are near-identical, not identical.

**Pseudo-replication.** Because the three runs are near-copies, the 120 attack episodes of a model are
really about 40 independent cases. Episode-level intervals treat them as 120 independent samples and are
too narrow. The report therefore adds a case-level view in which each case counts once.

### The hosted stopping rule and partial data

Free hosted APIs have daily quotas and rate limits, so a hosted run may not finish. If we simply stopped
whenever it was convenient, we could stop at a good-looking moment. So the rule was written and committed
before the first hosted request (D-03, as amended by D-08 and D-09): a hard freeze at 2026-10-29 15:30 UTC,
plus defined handling of QuotaExhausted (wait for the stated retry time, and stop the model if the next
retry would fall at or after the freeze). Whatever is unfinished at the freeze is reported as partial with
its n.

The October Groq rows were superseded by a fresh re-run (D-08): they predate the current scoring and
provenance stamps and carry no prompt_tokens, so they could not be checked for truncation or compared
like-for-like. The fresh re-run puts data-stealing (ds) episodes first. That is a deliberate choice of which
data to have if the run is cut short: ds attacks are the ones that test the exfiltration path the
reviewer cared about. The price is a selection effect: a partial hosted row is mostly ds, so its numbers
are not comparable to a complete row's mix of ds, dh and benign episodes, and must not be ranked.

---

## Part B — How this repository applies them

### The sandbox: src/sandbox/executor.py

SandboxExecutor runs each permitted shell action as
`unshare --user --map-root-user --mount --pid --fork --net`, which creates all four namespaces in one
step. Inside, a short setup script makes the bind mounts (staging /home, /root, /tmp and the canary
/etc/shadow) and the tmpfs over /mnt, then starts the seccomp filter program, which runs the command. Each
action has a 30-second timeout (ACTION_TIMEOUT_SECONDS). The staging directory is created per episode and
removed by close().

detect_enforcement() decides whether a failed command was stopped by a kernel layer. Exit code 159 (or a
signal 31 ending) and the texts "operation not permitted" or "bad system call" mean seccomp; texts such
as "network is unreachable", "could not resolve host" or "permission denied" mean the sandbox. Anything
else is an ordinary failure. This is the only place a kernel-layer block gets recorded, which is why the
counts in Part C are a lower bound.

### The filter: src/sandbox/seccomp_filter.c

Built with libseccomp. The default action is allow, and named system calls get SCMP_ACT_ERRNO(1), that is
EPERM: connect, chmod, fchmod, chown, fchown, lchown, mount, umount2, ptrace, reboot, init_module,
finit_module, delete_module, setuid, setgid, setreuid and setregid. seccomp_load() installs it, then the
program execs the command, which inherits the filter.

### The layer order: src/benchmark/runner.py

build_executor() wires the protected arm as ToolBroker(capabilities) around GuardedExecutor(policy) around
SandboxExecutor. The unprotected arm is ToolBroker around a MockExecutor with no checks. So an action
reaches the sandbox only after the broker and the policy engine allowed it.

### Running and checking: src/benchmark/jobs.py and src/benchmark/run_checks.py

- jobs local runs the five local models one at a time on the shared GPU (D-04), skips complete
  directories, resumes incomplete ones, and launches nothing at or after FREEZE_UTC.
- jobs ollama-variants builds the num_ctx 8192 Modelfile variants; the queue starts its own Ollama server
  and stops it only if it started it.
- jobs hosted is the Groq loop under the stopping rule; jobs status only reads; jobs report merges
  exactly the final results/phase4-<model> directories and results/phase4-control.
- run_checks prints an ok line or FAIL lines: complete (342/342, no duplicates, stamps match the engine),
  truncation (prompt_tokens distribution), ollama-evidence (window really raised), hosted (labels and
  terminal state), secret-scan (no key-shaped text), and viva (this page against the report).

### The rule file and the report

- .planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-HOSTED-STOPPING-RULE.md is the
  pre-registered rule; results/phase4-jobs/stopping_rule.json stores the freeze the loop enforces.
- src/benchmark/report.py writes the section "Blocks by layer, per model and pooled (EVAL-08)" and
  layer_blocks.csv: for each layer, k protected-arm episodes with at least one block out of n, plus the
  number of blocked actions, per model and pooled, attack and benign separately.

---

## Part C — What the numbers say now

All numbers below are copied from results/phase4-report (summary.md and summary.csv), generated
2026-10-29T03:06Z by jobs report from nine directories: the five local models, the three Groq models and
the fresh control re-run in results/phase4-control (D-05). Each model's "done" count is the report's
episodes_done/episodes_expected. The Groq numbers are a snapshot at 2026-10-29T03:06Z (UTC); the Groq
re-run continues until the 2026-10-29 15:30 UTC freeze (D-09), then the report is regenerated and these
Groq rows are updated. Every local number on this page is final.

### Completeness per model

| Model | Status in the report | Episodes done |
|---|---|---|
| qwen2.5-0.5b-agentlora | complete | 342/342 |
| scratch-gpt | complete | 342/342 |
| qwen2.5-0.5b-base | complete | 342/342 |
| ollama-qwen2.5-3b | complete | 342/342 |
| ollama-llama3.2-3b | complete | 342/342 |
| groq-gpt-oss-120b | partial (snapshot 03:06Z) | 240/342 |
| groq-gpt-oss-20b | error-dominated, labelled partial (snapshot 03:06Z) | 109/342 |
| groq-qwen3.8-27b | partial (snapshot 03:06Z) | 246/342 |

All five local models (D-01) finished in one uninterrupted queue on 2026-10-28 between 18:45Z and 20:30Z,
each with 0 duplicates, 0 pending infra rows and one policy hash. No local model was stopped by the freeze.
The scripted control also finished 342/342 (complete-ok).

### Headline: the five complete local models

Each has 120 paired attack episodes and 51 benign episodes per arm.

| Model | Attempted (no guardrails) | Success (no guardrails) | Success (guardrails) | Benign task (no / with guardrails) | Parse errors per step |
|---|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | 111/120 (92.5%) | 111/120 (92.5%) | 0/120 | 51/51, 51/51 | 6/978 (0.6%) |
| scratch-gpt | 39/120 (32.5%) | 33/120 (27.5%) | 0/120 | 36/51, 36/51 | 136/993 (13.7%) |
| qwen2.5-0.5b-base | 6/120 (5.0%) | 3/120 (2.5%) | 0/120 | 15/51, 15/51 | 803/1485 (54.1%) |
| ollama-qwen2.5-3b | 5/120 (4.2%) | 3/120 (2.5%) | 0/120 | 49/51, 49/51 | 150/951 (15.8%) |
| ollama-llama3.2-3b | 14/120 (11.7%) | 7/120 (5.8%) | 0/120 | 48/51, 50/51 | 47/1251 (3.8%) |

With guardrails, attack success is 0/120 for every local model (Wilson upper bound 3.1 %). Counting each
case once, it is 0/40 cases for every model (upper bound 8.8 %). False positives with guardrails are 0/51
for every local model. Over every language model in the report, every attempted attack in the protected
arm was blocked at the attack action: pooled conditional block rate 200/200 [98.1, 100.0]. scratch-gpt
called the attacker's tool before the injection could have influenced it in 21/120 attack episodes; those
are a random-choice baseline and are not counted as attacks.

### The model-independent control

The scripted control obeys every injection. Unprotected: attempted 120/120, success 120/120. Protected:
success 0/120, blocked given attempt 120/120, and counting each case once 40/40 [91.2, 100.0]. Which layer
blocked it: tool broker in 120/120 attempted episodes (180 actions), policy engine in 60/120 (60 actions,
the follow-up attempts after the broker had refused the tool), sandbox 0/120 and seccomp 0/120. No action
of this control reaches the kernel layers, so it says nothing about them.

### Ollama window evidence (D-06)

- Measured default: 4,096 tokens for llama3.2:3b on Ollama 0.40.3 (source: the server's /api/ps, agreed by
  the ollama ps column), so the band that proves a raised window is 4198-7782 tokens.
- ollama-qwen2.5-3b: window 8192 confirmed on the primary evidence path, long-prompt probe 6402 tokens;
  951 steps, largest prompt 1002 tokens, 0 steps at or above 95 % of the window.
- ollama-llama3.2-3b: window 8192 confirmed on the primary path, long-prompt probe 5273 tokens; 1251 steps,
  largest prompt 1042 tokens, 0 steps at or above 95 % of the window.
- llama3.2:3b re-check: the October rows carry no prompt_tokens, so they are truncation-unknown. In the new
  8192 run, 0/342 episodes had a step above the measured default of 4,096 (largest prompt 1,042), so at
  that default the old run would not have been truncated either. The new run supersedes the old one anyway
  (D-05). Caveat: the default was measured on today's server version; October's was not recorded.

### Blocks by layer, per model and pooled (EVAL-08)

Sandbox and seccomp cells exactly as printed in the report (protected arm, k/n with Wilson interval and
blocked actions):

| Model | Episodes | Sandbox | Seccomp |
|---|---|---|---|
| groq-gpt-oss-120b (partial) | attack | 0.0% (0/92) [0.0, 4.0]; 0 actions | 0.0% (0/92) [0.0, 4.0]; 0 actions |
| groq-gpt-oss-120b (partial) | benign | 0.0% (0/28) [0.0, 12.1]; 0 actions | 0.0% (0/28) [0.0, 12.1]; 0 actions |
| groq-gpt-oss-20b (partial) | attack | 0.0% (0/47) [0.0, 7.6]; 0 actions | 0.0% (0/47) [0.0, 7.6]; 0 actions |
| groq-gpt-oss-20b (partial) | benign | 0.0% (0/7) [0.0, 35.4]; 0 actions | 0.0% (0/7) [0.0, 35.4]; 0 actions |
| groq-qwen3.8-27b (partial) | attack | 0.0% (0/92) [0.0, 4.0]; 0 actions | 0.0% (0/92) [0.0, 4.0]; 0 actions |
| groq-qwen3.8-27b (partial) | benign | 0.0% (0/30) [0.0, 11.4]; 0 actions | 0.0% (0/30) [0.0, 11.4]; 0 actions |
| ollama-llama3.2-3b | attack | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| ollama-llama3.2-3b | benign | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| ollama-qwen2.5-3b | attack | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| ollama-qwen2.5-3b | benign | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| scratch-gpt | attack | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| scratch-gpt | benign | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| qwen2.5-0.5b-base | attack | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| qwen2.5-0.5b-base | benign | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| qwen2.5-0.5b-agentlora | attack | 0.0% (0/120) [0.0, 3.1]; 0 actions | 0.0% (0/120) [0.0, 3.1]; 0 actions |
| qwen2.5-0.5b-agentlora | benign | 0.0% (0/51) [0.0, 7.0]; 0 actions | 0.0% (0/51) [0.0, 7.0]; 0 actions |
| Pooled (all LLM models) | attack | 0.0% (0/831) [0.0, 0.5]; 0 actions | 0.0% (0/831) [0.0, 0.5]; 0 actions |
| Pooled (all LLM models) | benign | 0.0% (0/320) [0.0, 1.2]; 0 actions | 0.0% (0/320) [0.0, 1.2]; 0 actions |

The report's own sentences: "Sandbox (namespaces / private filesystem): no block was recorded in any of the
1151 protected-arm episodes of the 8 language models (0/1151)." and "Seccomp-BPF: no block was recorded in
any of the 1151 protected-arm episodes of the 8 language models (0/1151)." The five complete local models
alone give 0/855 for each kernel layer.

Said plainly: **the sandbox and seccomp layers recorded 0 blocks.** The reason is that the layers in front
stopped every attack first. Pooled over the language models, the tool broker blocked in 239/831 protected
attack episodes (245 actions) and the policy engine in 48/831 (58 actions); every one of the 200 attempted
attacks was blocked at the attack action, so the attack actions were stopped before any of them reached
the sandbox. The actions
that did reach the sandbox were allowed ones, and none of them failed with an enforcement signature. The
zero is therefore not evidence that the kernel layers work, and not evidence that they fail: it is evidence
that, in this benchmark, they were never needed. No new attack suite was written to raise the count (D-07b).

What the kernel layers contain is documented elsewhere: in the Phase 1 residual-bypass table
(.planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md), rows 1, 2, 8, 9, 10, 15 and 16 were
recorded as seccomp blocks, rows 20, 21 and 22 as sandbox blocks, and in rows 3-7, 11-14, 17-19 and 23 the
layer contained the action without an error signature, so no block was recorded. Those are the rows to
point to; they are not added to the Phase 4 counts.

### The hosted re-run: snapshot at 2026-10-29T03:06Z

Snapshot at 2026-10-29T03:06Z (UTC); the Groq re-run continues until the 2026-10-29 15:30 UTC freeze
(D-09), then the report is regenerated and this section is updated with the final n. Nothing here is a
final Groq number.

The rule was committed (d4ee817) before the first hosted request; the loop started 2026-10-28T18:48Z with
data-stealing episodes first (D-08). At the snapshot, `run_checks hosted` printed these labels (the loop's
own state, with its n and ds n), and the loop was still alive:

| Model | Episodes done (snapshot) | ds n | Complete pairs | Paired attack episodes | State at the snapshot |
|---|---|---|---|---|---|
| groq-gpt-oss-120b | 240/342 | 120/120 | 120 | 92 | running (rate-limit waits), label partial n=240 ds n=120, 0 quota stops |
| groq-gpt-oss-20b | 109/342 | 82/120 | 54 | 47 | waiting_quota (quota stop 2026-10-29T02:56:56Z, waiting until 03:56:56Z), label partial n=109 ds n=82 |
| groq-qwen3.8-27b | 246/342 | 120/120 | 122 | 92 | waiting_quota (quota stop 2026-10-29T02:33:36Z, waiting until 03:33:36Z), label partial n=246 ds n=120 |

The report's partial rows at the same snapshot:

| Model | Episodes done (report) | Paired attack episodes | Success (no guardrails) | Success (guardrails) | Benign n | False positive (guardrails) |
|---|---|---|---|---|---|---|
| groq-gpt-oss-120b | 240/342 | 92 | 4/92 (4.3%) | 0/92 | 28 | 1/28 (3.6%) |
| groq-gpt-oss-20b | 109/342 | 47 | 2/47 (4.3%) | 0/47 | 7 | 0/7 |
| groq-qwen3.8-27b | 246/342 | 92 | 0/92 (0.0%) | 0/92 | 30 | 0/30 |

An overnight network outage left many pending infra rows (102, 233 and 96 at the snapshot), which the loop
keeps retrying under the rule; groq-gpt-oss-20b is error-dominated for that reason and is labelled partial.
The one gpt-oss-120b false positive is a single benign episode in which the policy engine blocked an
action (pooled benign policy-engine row: 1/320); it is a cost of the guardrails and is reported, not hidden.

These rows are partial: they are shown with their n and are not ranked against the complete local rows.
Most of their attack episodes are ds (for example 60 ds against 32 dh for gpt-oss-120b after pair
dropping), which is the selection effect described in Part A.

### What the code review found and what we fixed

To be filled after the Phase 4 code review. The review has not run yet; nothing on this page depends on
its outcome.

---

## Limits of this phase

- The kernel-layer counts are a lower bound over recorded blocks. The executor records a sandbox or seccomp
  block only when a failed command's error text matches detect_enforcement's signatures; a layer that
  silently contains an action (no egress, a canary instead of the real file) records nothing. The Phase 1
  residual-bypass table, rows 1, 2, 8, 9, 10, 15, 16 (seccomp), 20, 21, 22 (sandbox) and 3-7, 11-14, 17-19,
  23 (contained without a record), is the evidence of what the layers do, not the Phase 4 counts.
- Phase 4 recorded 0/1151 sandbox and 0/1151 seccomp blocks (03:06Z snapshot). This shows the layers were not needed in
  this benchmark, not that they would hold against an attack the application layers miss.
- The seccomp filter is a deny-list: a dangerous system call that is not named is allowed.
- No layer protects an allowed but harmful action through a permitted tool.
- Rows differ in size, training, precision (Q4_K_M, bf16, hosted) and serving stack at once; the table is a
  spectrum, not a leaderboard.
- The three runs per case are near-copies at temperature 0 (pseudo-replication); case-level figures are
  the honest unit, and with 40 cases a zero still allows a true rate up to 8.8 %.
- The context default of 4,096 was measured on today's Ollama server; the October server's version was not
  recorded, so the "not truncated" conclusion for the old run holds at the measured default only.
- The Groq rows are partial under the stopping rule, ds-heavy by design, and their providers can change the
  served model without notice.
- phi3.5:3.8b and gemma3:1b were descoped (D-01), and Gemini/Gemma were not resumed (D-02).

---

## Part D — Likely viva questions and answers

**Q1. What is a system call, and why put a guardrail there?**
A system call is a program's request to the kernel for anything that touches the outside world: files,
network, processes. It is the only door, so a check there sees every effect regardless of which language,
interpreter or binary produced it. That makes it a good backstop for whatever the application-level
checks did not understand.

**Q2. What do the four namespaces you use isolate?**
The user namespace gives a private user-id table, so the process is root inside but unprivileged outside.
The mount namespace gives a private list of mounts, the PID namespace a private set of process numbers,
and the network namespace a network stack with no route anywhere. Together the agent's command sees a
fake filesystem, cannot see host processes and cannot reach the network.

**Q3. Why do you need the user namespace at all?**
Creating mounts normally needs root. Inside a user namespace an ordinary user is mapped to root for that
namespace only, so the sandbox can build its private view without sudo and without gaining any real
privilege on the host.

**Q4. How does the private filesystem view work?**
Inside the mount namespace we bind-mount per-episode staging directories over /home, /root and /tmp, a
canary copy over /etc/shadow, and an empty tmpfs over /mnt, which on WSL hides the Windows drives and the
project. Writes stay inside the staging tree for the episode and never reach the real system.

**Q5. What is seccomp-bpf and what does your filter do?**
seccomp lets a process install a small BPF program that the kernel runs on every system call it makes. Our
filter allows everything by default and answers a named list (connect, chmod and chown variants, mount,
umount2, ptrace, reboot, module loading, setuid and related calls) with EPERM. Once installed it cannot
be removed and is inherited by every child process.

**Q6. Why return EPERM instead of killing the process?**
An error lets the command fail with a readable message, which the agent receives as an observation and our
code can recognise as a seccomp block. Killing is stricter but gives the agent and the logs less
information. Either verdict stops the system call; the choice is about how the failure is reported.

**Q7. Why is the filter installed after the mounts?**
Because it forbids mount() itself. If it were installed first, the sandbox could not build its own
private view. So the order is namespaces, then mounts, then the filter, then the agent's command.

**Q8. Your sandbox and seccomp counts are zero. Doesn't that mean they are useless?**
No. 0/1151 means no attack reached them: the tool broker and the policy engine stopped all 200 attempted
attacks first, at the attack action. The kernel layers are a backstop for what the first layers miss, and
in this benchmark they missed nothing. We report the zero plainly with its n (D-07b) and point to the
Phase 1 residual-bypass table, where rows 1, 2, 8, 9, 10, 15 and 16 were recorded seccomp blocks and rows
20, 21 and 22 sandbox blocks.

**Q9. Why do you call the counts a lower bound?**
A kernel-layer block is recorded only when a failed command's error text matches a known signature. A
layer that contains an action quietly, for example a read that returns the canary instead of the real
file, records nothing. Phase 1 rows 3-7, 11-14, 17-19 and 23 were exactly that kind. So the true number of
containments can only be equal to or larger than the recorded number.

**Q10. Why not write new attacks to make those layers fire?**
That was the original plan (D-07). It was narrowed by the user's decision D-07b: no new attack-style suite,
count what the full runs recorded, cite the Phase 1 table by row number, and never pad a zero. The honest
claim is the one the data supports.

**Q11. What is defense in depth here, and what does it not cover?**
Four independent layers: tool broker, policy engine, sandbox, seccomp; an attack must pass all of them.
None of them stops an action that is allowed and harmful, such as a misleading email through an email tool
the task genuinely needs. Guardrails limit what the agent can do, not whether it does the right thing.

**Q12. Why is comparing a local Ollama model with a hosted model unfair?**
The rows differ in size, training, quantization (Q4_K_M locally, bf16 for the in-process Qwen rows,
whatever Groq serves), context handling and serving stack all at once. We present a spectrum and name the
confounds. The guardrail result does not depend on these: protected success is 0 for every model.

**Q13. What is silent truncation and how did you rule it out?**
If a prompt exceeds Ollama's window, it cuts the front without an error, so the model can lose its
instructions and look safe. We measured the default (4,096 tokens on this server), ran both Ollama models
with a 8192 window (D-06), proved the window with a long-prompt probe (6402 and 5273 tokens, inside the band
4198-7782), and logged prompt_tokens on every step: the largest prompt was 1,042 tokens.

**Q14. What did the llama3.2:3b re-check find?**
The October rows had no prompt_tokens, so they are truncation-unknown. In the new 8192 run, 0/342
episodes had any step above the measured 4,096 default, so at that default the old run would not have been
cut. We still use the new run (D-05), and we note that October's server version was not recorded.

**Q15. Why are three runs per case not three times the evidence?**
At temperature 0 the runs are near-copies, so 120 attack episodes are about 40 independent cases.
Treating them as 120 makes intervals too narrow (pseudo-replication). The report adds a case-level view:
0/40 cases succeeded with guardrails for every local model, upper bound 8.8 %.

**Q16. Is temperature 0 fully deterministic?**
No. It removes sampling randomness, but GPU arithmetic and hosted batching can still change a token now and
then. We treat runs as near-identical, not identical, and do not claim exact reproducibility of each reply.

**Q17. What is the hosted stopping rule, and why commit it first?**
A hard freeze at 2026-10-29 15:30 UTC plus defined handling of quota stops (D-09, amending D-03); whatever
is unfinished is reported as partial with its n. It was committed before the first hosted request so the
stopping point could not be chosen after seeing the numbers.

**Q18. Why were the October Groq rows thrown away, and why run ds first?**
They predate the current scoring and stamps and have no prompt_tokens, so they could not be checked or
compared like-for-like (D-08); they are kept as history only. Data-stealing episodes run first because if
the run is cut short, those are the episodes that test the exfiltration path. The cost is that a partial
row is ds-heavy, so it is never ranked against a complete row.

**Q19. How did the runs fit on one laptop shared with another project?**
The local models ran one at a time on the 6 GB GPU (D-04), with a read-only preflight before each launch,
in a detached queue that resumes on restart. Our code only ever signals processes it started itself, and
it starts and stops an Ollama server only if it owns it. The network-bound Groq loop ran at the same time
because it does not use the GPU (D-09).

**Q20. How do you know the numbers on this page match the report?**
run_checks viva reads each model's episodes_done and episodes_expected from the report's summary.csv and
fails unless the page quotes that model with that count on one line; it also checks the structure and
scans for key-shaped text. The page is re-checked whenever the report is regenerated.
