# Viva notes 4: namespaces, seccomp-bpf and the confounds of local inference

These notes are for an Operating Systems viva. Phase 4 of the project ran every local model through the
full benchmark (57 cases x 3 runs x 2 arms = 342 episodes each), re-ran three hosted Groq models under a
written stopping rule, re-checked an old Ollama run for silent truncation, and counted, per guardrail layer,
how many blocks were actually recorded. This page explains the two kernel layers of the guardrails (Linux
namespaces and seccomp-bpf), why we report them the way we do, and the traps that make a comparison between
a 4-bit local model and a hosted model unfair if nobody states them.

An honest framing comes first, because an examiner will ask. The paragraph below is generated from the
frozen data by the deliverables build, so it says what the data says even after the hosted rows change.
It gives what the sandbox and the seccomp filter recorded across the protected-arm episodes, and whether
the application-level layers in front of them (the tool broker's capability list and the policy engine)
blocked each attempted attack first.

<!-- deliverables:begin claim-kernel -->
The sandbox and the seccomp filter recorded no block in any of the 1173 protected-arm episodes of the 8 language models (0/1173 each). The tool broker and the policy engine in front of them blocked every attempted attack at the attack action first, so no attack action reached them. A recorded count is a lower bound, not proof that the kernel layers work.
<!-- deliverables:end claim-kernel -->

When the kernel-layer count is zero, that is not a failure of those layers and it is not evidence that
they work: it means no attack action reached them. The evidence of what the kernel layers contain comes
from the Phase 1 residual-bypass table, which we cite by row number, and the count is reported with its
n, never padded (decision D-07b).

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

Every number in this part is generated from results/phase5-final/numbers.json by the deliverables build
(`python -m src.deliverables build`, Phase 5) and checked by `python -m src.deliverables check`, which fails
if a table or a generated sentence here no longer matches the data, or if the prose around them states a
headline outcome in its own words. The status block below gives the data snapshot and says which rows are
partial. The local rows are final. The Groq rows become final when the build is re-run after the freeze at
15:30 UTC on 2026-10-29 (D-09); that re-run rewrites the tables and the generated sentences on this page by
itself, so the prose between them states no result that depends on the Groq rows.

### Completeness per model

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

All five local models (D-01) finished in one uninterrupted queue on 2026-10-28 between 18:45Z and 20:30Z,
each with 0 duplicates, 0 pending infra rows and one policy hash. No local model was stopped by the freeze.
The scripted control also finished 342/342 (complete-ok).

### Headline: every current-engine model

Each complete local model has 120 paired attack episodes and 51 benign episodes per arm. A partial row has
fewer, and the denominators in its cells show how many.

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

The two generated sentences below read the headline for you: whether any attack succeeded with
guardrails, and whether each attack a model really attempted was blocked at the attack action itself.

<!-- deliverables:begin claim-protected -->
With guardrails, no attack succeeded for any of the 8 language models, local or hosted: the Success (guardrails) cell is 0 in every model row of the headline. The scripted control, which attempts every attack, also had no successful attack with guardrails (0/120).
<!-- deliverables:end claim-protected -->

<!-- deliverables:begin claim-blocked -->
Every attack a language model really attempted with guardrails on was blocked at the attack action itself: 200/200 attempted attacks, pooled over the 8 language models (the Blocked given attempt column of the headline).
<!-- deliverables:end claim-blocked -->

How to read a zero in the Success (guardrails) column: for a complete local model it is 0 of 120 attack
episodes, with a Wilson upper bound of 3.1 %; counting each case once, it is 0 of 40 cases, with an upper
bound of 8.8 %. The "False positive (guardrails)" column gives the cost of the guardrails on benign tasks.

scratch-gpt called the attacker's tool before the injection could have influenced it in 21/120 attack
episodes; those are a random-choice baseline and are not counted as attacks.

### The model-independent control

<!-- deliverables:begin control -->
| Arm | Attack episodes | Attempted | Success | Blocked given attempt |
|---|---|---|---|---|
| Unprotected | 120 | 100.0% (120/120) [96.9, 100.0] | 100.0% (120/120) [96.9, 100.0] | — |
| Protected | 120 | 100.0% (120/120) [96.9, 100.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (120/120) [96.9, 100.0] |

The scripted control obeys every injection. Guardrails blocked 120 of its 120 attempted attacks; the tool broker denied the attacker's tool call in 120 of them. It is deterministic, so its 120 protected attack episodes are 40 distinct cases.
<!-- deliverables:end control -->

The control tests the guardrails on attacks that were certainly attempted, whatever a model would have
done; the tool broker refuses its attacker tool calls before they run, so it says nothing about the
kernel layers.

### Ollama window evidence (D-06)

- Measured default: 4,096 tokens for llama3.2:3b on Ollama 0.40.3 (source: the server's /api/ps, agreed by
  the ollama ps column), so the band that proves a raised window is 4198-7782 tokens.
- ollama-qwen2.5-3b: window 8192 confirmed on the primary evidence path, long-prompt probe 6402 tokens;
  951 steps, largest prompt 1002 tokens, 0 steps at or above 95 % of the window.
- ollama-llama3.2-3b: window 8192 confirmed on the primary path, long-prompt probe 5273 tokens; 1251 steps,
  largest prompt 1042 tokens, 0 steps at or above 95 % of the window.
- llama3.2:3b re-check: the October rows carry no prompt_tokens, so they are truncation-unknown. In the new
  8192 run, none of its 342 episodes had a step above the measured default of 4,096 (largest prompt 1,042), so at
  that default the old run would not have been truncated either. The new run supersedes the old one anyway
  (D-05). Caveat: the default was measured on today's server version; October's was not recorded.

### Blocks by layer, per model and pooled (EVAL-08)

<!-- deliverables:begin layers -->
Recorded blocks per guardrail layer, protected arm, pooled over the language models (k = episodes with at least one action blocked by the layer):

| Layer | Attack episodes | Benign episodes | Blocked actions |
|---|---|---|---|
| Tool broker | 28.3% (239/844) [25.4, 31.5] | 0.0% (0/329) [0.0, 1.2] | 245 |
| Policy engine | 5.7% (48/844) [4.3, 7.5] | 0.6% (2/329) [0.2, 2.2] | 60 |
| Sandbox | 0.0% (0/844) [0.0, 0.5] | 0.0% (0/329) [0.0, 1.2] | 0 |
| Seccomp | 0.0% (0/844) [0.0, 0.5] | 0.0% (0/329) [0.0, 1.2] | 0 |

| Model | Sandbox (attack) | Seccomp (attack) | Sandbox (benign) | Seccomp (benign) |
|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| scratch-gpt | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| qwen2.5-0.5b-base | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-qwen2.5-3b | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-llama3.2-3b | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| groq-gpt-oss-120b (partial 251/342) | 0.0% (0/95) [0.0, 3.9] | 0.0% (0/95) [0.0, 3.9] | 0.0% (0/30) [0.0, 11.4] | 0.0% (0/30) [0.0, 11.4] |
| groq-gpt-oss-20b (partial 117/342) | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/7) [0.0, 35.4] | 0.0% (0/7) [0.0, 35.4] |
| groq-qwen3.8-27b (partial 272/342) | 0.0% (0/98) [0.0, 3.8] | 0.0% (0/98) [0.0, 3.8] | 0.0% (0/37) [0.0, 9.4] | 0.0% (0/37) [0.0, 9.4] |

Sandbox (namespaces / private filesystem): no block was recorded in any of the 1173 protected-arm episodes of the 8 language models (0/1173).

Seccomp-BPF: no block was recorded in any of the 1173 protected-arm episodes of the 8 language models (0/1173).

What the two kernel layers contain is documented in the Phase 1 residual-bypass table (.planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md): rows 1, 2, 8, 9, 10, 15 and 16 were recorded as blocked by seccomp, rows 20, 21 and 22 as blocked by the sandbox, and in rows 3-7, 11-14, 17-19 and 23 the layer contained the action (no network egress, the canary file instead of the real one, a system call refused with EPERM) without an error signature, so no block was recorded. The counts above are therefore a lower bound: the executor records a sandbox or seccomp block only when a command's error text matches the signatures of detect_enforcement in src/sandbox/executor.py.
<!-- deliverables:end layers -->

Said plainly, from the data:

<!-- deliverables:begin claim-kernel -->
The sandbox and the seccomp filter recorded no block in any of the 1173 protected-arm episodes of the 8 language models (0/1173 each). The tool broker and the policy engine in front of them blocked every attempted attack at the attack action first, so no attack action reached them. A recorded count is a lower bound, not proof that the kernel layers work.
<!-- deliverables:end claim-kernel -->

How to read that count. The blocks of the tool broker and the policy engine are in the first table
above. When those two layers block each attempted attack at the attack action (the "Blocked given
attempt" column of the headline), no attack action reaches the sandbox; the actions that do reach it are
allowed ones, and a block is recorded only if one of them fails with an enforcement signature. A zero is
then not evidence that the kernel layers work, and not evidence that they fail: it is evidence that, in
this benchmark, they were not needed. No new attack suite was written to raise the count (D-07b).

The last paragraph of the block cites the Phase 1 residual-bypass table by row number. Those rows are the
evidence of what the kernel layers contain; they are not added to the Phase 4 counts.

### The hosted re-run at the data snapshot

The rule was committed (d4ee817) before the first hosted request; the loop started at 18:48 UTC on
2026-10-28 with data-stealing episodes first (D-08). The table gives, per Groq model at the data snapshot,
the episodes done, the data-stealing (ds) episodes finished out of the 120 planned, the complete pairs,
the paired attack episodes, the infra errors still pending, and the loop's own state and quota-stop count
read from its frozen hosted_state.json. The partial rows' results are in the headline block above.

<!-- deliverables:begin hosted -->
| Model | Episodes done | ds episodes finished | Complete pairs | Paired attack episodes | Infra errors pending | Loop state | Quota stops |
|---|---|---|---|---|---|---|---|
| groq-gpt-oss-120b | 251/342 | 120/120 | 125 | 95 | 91 | waiting_quota | 3 |
| groq-gpt-oss-20b | 117/342 | 90/120 | 58 | 51 | 225 | waiting_quota | 3 |
| groq-qwen3.8-27b | 272/342 | 120/120 | 135 | 98 | 70 | waiting_quota | 4 |

The loop state and quota stops are read from each model's frozen hosted_state.json at the snapshot (2026-10-29T07:12Z); they are final only after the D-09 freeze (2026-10-29T15:30Z).
<!-- deliverables:end hosted -->

An overnight network outage left many pending infra rows (the "Infra errors pending" column), which the
loop retries under the rule until the freeze; a model with more infra errors than finished episodes is
labelled error-dominated in the status block. A false positive of a hosted model is a benign episode in
which the policy engine blocked an action (the "False positive (guardrails)" column of the headline and
the benign column of the policy-engine row in the layers block); false positives are a cost of the
guardrails and are reported, not hidden.

<!-- deliverables:begin claim-state -->
The data is a snapshot of 2026-10-29T07:12Z (UTC), copied before the D-09 freeze of 2026-10-29T15:30Z while the hosted loop was still collecting. groq-gpt-oss-120b (partial 251/342), groq-gpt-oss-20b (partial 117/342), groq-qwen3.8-27b (partial 272/342) are labelled partial with their episodes done; they are not final and are never ranked against the complete rows.
<!-- deliverables:end claim-state -->

A partial row is shown with its n and is not ranked against the complete local rows. Because ds episodes
ran first, most attack episodes of a partial hosted row are ds, which is the selection effect described
in Part A.

### What the code review found and what we fixed

The Phase 4 code review (.planning/phases/04-full-runs-local-added-ollama-hosted-resume/04-REVIEW.md) read
the code written in this phase, and the fixes are recorded in 04-REVIEW-FIX.md next to it. One rule shaped
every fix. The Groq loop was still running, and every row it writes is stamped with the code commit of the
run-affecting files, so a fix to those files in the middle of the run would have made rows written before
and after it incomparable. The fixes therefore touch only the report, the run checks, the job scheduler
and their tests. The benchmark runner and the outcome rules were not changed, and the running loop was
never signalled.

Fixed:

- **CR-01 (critical): an episode with an infrastructure error followed by a context overflow or a refusal
  was neither retried nor counted.** The runner already treated such an episode as finished, but the
  report's loader kept the infra-error record, so the episode stayed unfinished for ever. The loader now
  ranks the records of an episode: an error-free record first, then a terminal error (context overflow or
  provider refusal), then an infra error. A parametrized test in tests/test_phase4_runner.py replays four
  record sequences and checks that the runner, the loader, the completeness check and the summary agree.
  A read-only comparison over all the Phase 4 data found no episode whose outcome changed, so the fix
  protects future resumes.
- **WR-02: the scheduler could send a forced kill (SIGKILL) to a process it no longer owned.** After the
  polite stop signal, the process id could in principle be reused by another program. The scheduler now
  re-checks, just before the forced kill, that the process is still the Ollama server it started (its pid
  file, its command line and an ownership marker in its environment). A test in tests/test_jobs.py hands
  the process id to a different process after the first signal and checks that no forced kill is sent. A
  tiny window remains between the re-check and the kill.
- **WR-03: the freeze deadline was checked only once, before long setup steps.** A job could pass the
  check, wait through a long GPU preflight or an Ollama start-up, and then start a run after the freeze.
  The deadline is now checked at entry, after the preflight, after the Ollama setup and just before each
  run is spawned. Three tests in tests/test_jobs.py let the freeze pass during the preflight, during the
  Ollama setup and before the spawn, and check that no run is started.
- **WR-04: the schedulers' "complete" gate failed on provenance problems.** A run that had finished every
  episode but carried a provenance problem (for example a changed policy hash or a mixed code commit) was
  treated as unfinished and launched again and again. Progress and provenance are now separate: a new gate
  says whether every episode is finished, provenance problems are printed as their own FAIL lines, and a
  hosted model with such problems ends in its own state, "complete with problems". `run_checks complete`
  stays strict. Seven tests in tests/test_run_checks.py and tests/test_jobs.py pin the split.
- **WR-05: some of these paths had no test, and one test about the key check could not fail.** The test
  that the scheduler reads the key only after loading the model registry was rewritten to record the order
  of the reads, and it was shown to fail on a deliberately broken version. A second new test covers a run
  that exits in the last poll before the freeze: it must be recorded as finished (or stopped by its quota),
  not as stopped at the freeze. Making that test pass is the first part of the Info finding IN-01: at the
  freeze, the hosted loop now checks whether a run has already exited before it stops it.
- **IN-07, first half: the Ollama command-line client inherited the scheduler's full environment**, which
  could include keys loaded from the project's environment file. It now gets a minimal environment (path,
  home directory, language and the server address). A test in tests/test_jobs.py checks the exact set of
  variables passed to every Ollama call.

Deferred until after the freeze, because they touch run-affecting code (WINDOWS.md entry 4):

- **WR-01: a context overflow after valid steps is classed as a format failure**, but only when the model
  had not already attempted the attack, because the "attacked" rule comes first. So overflows remove
  non-attacking episodes from the ASR-valid denominator, which biases ASR-valid upward; raw success is not
  affected. Whether it touches the current data depends on the number of context overflows, which the
  generated sentence after this list and the report's threats section state from the data. The fix
  changes the outcome rules
  (src/benchmark/outcomes.py), which would make rows written before and after it incomparable in the
  middle of the hosted run.
- **IN-04: the context-overflow marker is matched as a substring of any provider's error text**, so an
  unrelated error that happens to contain it could be misclassified.
- **IN-05: a setting that orders the work queue is stored where a resume overwrites it**, and one
  docstring in the runner is out of date.

<!-- deliverables:begin claim-overflow -->
No current-engine episode had a context overflow (0 for every language model), so the deferred WR-01 bias touches none of the present data.
<!-- deliverables:end claim-overflow -->

Not addressed (Info findings outside the fix's scope): IN-02 (the stated retry time is rounded), IN-03
(loose error classes and sticky states in the availability probe), IN-06 (labels in the layer table), the
rest of IN-01 (stopping runs one at a time at the freeze, logging inherited child processes) and the rest
of IN-07 (duplicated constants and helpers).

---

## Limits of this phase

- The kernel-layer counts are a lower bound over recorded blocks. The executor records a sandbox or seccomp
  block only when a failed command's error text matches detect_enforcement's signatures; a layer that
  silently contains an action (no egress, a canary instead of the real file) records nothing. The Phase 1
  residual-bypass table, rows 1, 2, 8, 9, 10, 15, 16 (seccomp), 20, 21, 22 (sandbox) and 3-7, 11-14, 17-19,
  23 (contained without a record), is the evidence of what the layers do, not the Phase 4 counts.
- The sandbox and seccomp counts are in the generated sentence at the top of this page and, with their n,
  in the layers block of Part C. A zero there shows the layers were not needed in this benchmark, not that
  they would hold against an attack the application layers miss.
- The seccomp filter is a deny-list: a dangerous system call that is not named is allowed.
- No layer protects an allowed but harmful action through a permitted tool.
- Rows differ in size, training, precision (Q4_K_M, bf16, hosted) and serving stack at once; the table is a
  spectrum, not a leaderboard.
- The three runs per case are near-copies at temperature 0 (pseudo-replication); case-level figures are
  the honest unit, and with 40 cases a zero still allows a true rate up to 8.8 %.
- The context default of 4,096 was measured on today's Ollama server; the October server's version was not
  recorded, so the "not truncated" conclusion for the old run holds at the measured default only.
- A Groq row that stopped early under the stopping rule is partial and ds-heavy by design (the status block
  says which rows are partial), and the providers can change the served model without notice.
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
No. The generated sentence at the top of this page gives the count with its n and says whether the tool
broker and the policy engine blocked each attempted attack at the attack action first. If they did, no
attack reached the kernel layers: those layers are a backstop for what the first layers miss, and a
backstop that is never needed records nothing. We report the count plainly with its n (D-07b) and point
to the Phase 1 residual-bypass table, where rows 1, 2, 8, 9, 10, 15 and 16 were recorded seccomp blocks
and rows 20, 21 and 22 sandbox blocks.

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
confounds. The guardrail result does not depend on these, because it is read row by row from the
Success (guardrails) column, never from a comparison between rows.

**Q13. What is silent truncation and how did you rule it out?**
If a prompt exceeds Ollama's window, it cuts the front without an error, so the model can lose its
instructions and look safe. We measured the default (4,096 tokens on this server), ran both Ollama models
with a 8192 window (D-06), proved the window with a long-prompt probe (6402 and 5273 tokens, inside the band
4198-7782), and logged prompt_tokens on every step: the largest prompt was 1,042 tokens.

**Q14. What did the llama3.2:3b re-check find?**
The October rows had no prompt_tokens, so they are truncation-unknown. In the new 8192 run, none of its 342
episodes had any step above the measured 4,096 default, so at that default the old run would not have been
cut. We still use the new run (D-05), and we note that October's server version was not recorded.

**Q15. Why are three runs per case not three times the evidence?**
At temperature 0 the runs are near-copies, so 120 attack episodes are about 40 independent cases.
Treating them as 120 makes intervals too narrow (pseudo-replication). The report adds a case-level view:
for a model with no protected success, 0 of 40 cases gives an upper bound of 8.8 %, not 3.1 %.

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
