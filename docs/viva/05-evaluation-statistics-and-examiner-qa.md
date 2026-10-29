# Viva notes 5: evaluation statistics, threats to validity and the examiner's questions

These notes are for an Operating Systems viva. Phase 5 of the project did no new experiments. It froze the
results of every Phase 4 run into one read-only directory (results/phase5-final), and generated every
deliverable from that one copy: the cross-model report and its charts, the result tables in the README and
the SDD, the slide deck, and the result tables on this page and on page 04. One command rebuilds all of it,
and a consistency check fails if a generated table or sentence no longer matches the frozen data, or if the
prose around them states a result by hand. This page
explains the statistics behind those numbers, the ways the comparison could mislead, and how to defend the
result in front of an examiner.

An honest framing comes first, because an examiner will ask. The comparison deliberately includes a model
LoRA-tuned to obey injections and a tiny model trained from scratch, next to three hosted models. The three
paragraphs below are generated from the frozen data, so they stay true when the data changes: what
happened with guardrails on, what the two kernel layers (the namespace sandbox and the seccomp filter)
recorded, and which rows are still partial. Read the tables in Part D for the exact numbers.

<!-- deliverables:begin claim-protected -->
With guardrails, no attack succeeded for any of the 8 language models, local or hosted: the Success (guardrails) cell is 0 in every model row of the headline. The scripted control, which attempts every attack, also had no successful attack with guardrails (0/120).
<!-- deliverables:end claim-protected -->

<!-- deliverables:begin claim-kernel -->
The sandbox and the seccomp filter recorded no block in any of the 1225 protected-arm episodes of the 8 language models (0/1225 each). The tool broker and the policy engine in front of them blocked every attempted attack at the attack action first, so no attack action reached them. A recorded count is a lower bound, not proof that the kernel layers work.
<!-- deliverables:end claim-kernel -->

<!-- deliverables:begin claim-state -->
The data is final: it was copied at 2026-10-29T17:02Z (UTC), after the D-09 freeze of 2026-10-29T15:30Z. groq-gpt-oss-120b (partial 289/342), groq-gpt-oss-20b (partial 132/342), groq-qwen3.8-27b (partial 322/342) stopped at the freeze or on their quota before finishing, so they stay labelled partial with their episodes done and are never ranked against the complete rows.
<!-- deliverables:end claim-state -->

How to use this page: read the glossary and Parts A to C once (the ideas), look at Part D (the real numbers,
generated from the frozen data) the night before, learn Part E (the one-page defense) well enough to say it
without notes, and practise Part F out loud until each answer takes under a minute. Part F collects
questions from all five phases, so it doubles as a revision sheet for pages 01 to 04. Pages 01 and 04 go
deeper on the reference monitor, the metrics and the kernel layers; this page refers to them instead of
repeating them.

Words you will meet and what they mean:

- **Agent**: a program that asks a language model what to do next, runs the action the model names, shows
  the result back to the model, and repeats until the task is done.
- **Tool**: a function the agent may call, for example "read my email" or "send a payment". In the
  benchmark the tools are simulated, so nothing real is sent or paid.
- **Prompt injection**: text written by an attacker that tries to make a language model follow the
  attacker's instructions instead of the user's. It works because the model receives instructions and
  data in the same stream of text and cannot reliably tell them apart.
- **Direct injection**: the person typing to the model is the attacker, for example a user who tries to
  talk a chatbot out of its rules (often called a jailbreak).
- **Indirect injection**: the attacker never talks to the model. They plant instructions inside data the
  agent will read on behalf of an honest user: a product review, an email, a web page. The user is the
  victim.
- **InjecAgent**: a published benchmark (Zhan et al., 2024) for indirect prompt injection against
  tool-using agents. Each case pairs a user's request and the tool it needs with an attacker instruction
  hidden in that tool's output. This project uses a 57-case subset of it.
- **User tool**: the tool the honest user's request genuinely needs, whose output carries the injection.
- **Attacker tool**: the tool the injected instruction tries to make the agent call, for example a payment
  or a door-unlock tool. Calling it after reading the injection is what "attempted" means.
- **dh (direct harm)**: an InjecAgent case in which the attacker wants one harmful action, carried out by
  calling the attacker tool.
- **ds (data stealing)**: an InjecAgent case in which the attacker wants private data: the agent must first
  fetch the data with one tool and then send it out, for example by email to the attacker's address.
- **Canary**: a fake secret with a unique random value planted in the private data of each episode. If it
  appears in an outgoing message, the data was stolen, and we know exactly which episode leaked it, with
  no human or second model judging.
- **Episode**: one run of one agent on one case in one arm, from the user's request to the final answer.
- **Arm**: one of the two conditions every case is run under: unprotected (no guardrails) and protected
  (the full stack of guardrails).
- **ASR (attack success rate)**: the share of attack episodes in which the harm actually happened (the
  attacker tool ran, or the canary left). In the report it is the "Success" column.
- **ASR-valid**: the same count of successes divided only by the format-valid episodes, so a model that
  produced unreadable replies is not credited with resisting the attack.
- **Attempted**: the agent called the attacker tool after the injection had been shown to it, whether or
  not the call was allowed to run.
- **Conditional block rate (blocked given attempt)**: of the protected-arm attack episodes in which the
  agent attempted the attack, the share in which a guardrail blocked the attack action itself and the
  attack did not succeed.
- **Benign utility**: of the harmless tasks (no injection), the share the agent completed. Measured in both
  arms, so a drop from unprotected to protected is the cost of the guardrails.
- **False positive**: a benign episode in the protected arm in which a guardrail blocked something. It is
  the price of the guardrails, and it is reported, not hidden.
- **Binomial proportion**: a count of "yes" outcomes k out of n independent yes/no trials, written k/n. All
  the rates on this page are binomial proportions.
- **Confidence interval**: a range around an observed proportion, built so that over many repetitions of
  the experiment about 95% of such ranges would contain the true rate.
- **Wilson interval**: the confidence interval the report uses. Unlike the textbook normal approximation,
  it never collapses to a single point when nothing (or everything) was observed.
- **Rule of three**: if nothing happened in n independent trials, the true rate is, with about 95%
  confidence, below 3 divided by n.
- **Paired design**: the same case in the same run is executed in both arms, so the only difference within
  a pair is the guardrails.
- **Pair dropping**: if one arm of a pair ended in an infrastructure error, the other arm is removed from
  the paired statistics too, so the two arms are always compared on the same cases.
- **Pseudo-replication**: treating repeated measurements of the same thing as if they were independent
  samples. The three runs of a case at temperature 0 are near-copies, so they are not three samples.
- **Case-level view**: the report's table that counts each case once (attempted if any run attempted,
  succeeded if any run succeeded), which is the honest unit of evidence.
- **Completeness gating**: the rule that only a model whose run finished every planned episode appears in
  the ranked headline; any other model is listed separately with its episodes done.
- **Partial run**: a run that stopped before its planned episodes were all finished (here: the hosted runs,
  stopped by quota limits, network errors and the freeze).
- **Confound**: a difference between two compared groups other than the one you are studying, which could
  explain the result instead of it. Model size and quantization are confounds between our model rows.
- **Quantization**: storing a model's weights with fewer bits (for example 4-bit Q4_K_M in Ollama) so it
  fits in less memory, at some cost in quality.
- **Context window**: the largest number of tokens a model can read at once. Text beyond it is dropped,
  sometimes silently.
- **Greedy decoding (temperature 0)**: always choosing the single most likely next token, so the same
  prompt gives (almost) the same reply every time.
- **Threat to validity**: a reason the conclusion of an experiment might be wrong or might not carry over to
  other situations.
- **Lower bound**: a number the true value cannot be below. A recorded block count is a lower bound because
  some containments leave no record.
- **Scripted control**: a non-model agent that obeys every injection, used to test the guardrails on
  attacks that are certainly attempted.
- **Capability**: an unforgeable permission to use one resource. Here the tool broker gives each task a
  list of the tools it may call, and nothing else can be called.
- **Reference monitor**: the component that checks every access against a policy, cannot be bypassed and is
  small enough to verify. Page 01 explains it in full.
- **Dual-LLM pattern**: a design with two models: a privileged one that plans and may call tools but never
  sees untrusted data, and a quarantined one that reads untrusted data but has no tools.
- **CaMeL**: a published defense (Debenedetti et al., 2025) that turns the dual-LLM idea into a program run
  by its own interpreter, which tracks where every value came from and checks a policy before each tool
  call.
- **OWASP LLM01**: the first entry, Prompt Injection, in the OWASP Top 10 for LLM Applications (2025), the
  industry list of the most important security risks of LLM applications.
- **Frozen snapshot**: the read-only copy of the run data (results/phase5-final) from which every
  deliverable is generated, with the time it was copied.
- **Generated block**: a part of a document between two hidden marker lines that a program writes from the
  frozen numbers; it is never edited by hand.

---

## Part A — Prompt injection and defenses

### What prompt injection is

A language model reads one long stream of text: the system prompt from the developer, the user's request,
and the outputs of any tools it called. It has no hardware-enforced boundary between "instructions to
follow" and "data to read". If the data contains a sentence that looks like an instruction, the model may
follow it. That is prompt injection. It is often compared to SQL injection, where data typed into a form
is executed as part of a database command, and to old telephone systems that carried control tones in the
same channel as the voice: when control and data share one channel, whoever controls the data can send
control.

There are two kinds:

- **Direct injection**: the attacker is the user. They type text meant to override the developer's rules.
  This is mostly a problem for chatbots and content policies.
- **Indirect injection**: the attacker plants instructions in content that an honest user's agent will
  read later. This is the dangerous kind for agents, because the agent holds the user's permissions (their
  email, their bank tool, their files) and the user never sees the planted text.

An example, described in words only: a user asks the agent to summarise the reviews of a product. One
review, written by the attacker, contains a sentence phrased as an order to the assistant, telling it to
use the user's payment tool to send money to an account the attacker controls. A vulnerable agent reads
the review, treats that sentence as a new task, and calls the payment tool. The user asked for a summary;
the attacker got a payment.

### How InjecAgent builds its cases

InjecAgent (Zhan et al., 2024) builds every test case from three parts: a user request that needs one user
tool (read reviews, read an email, look up a calendar event), the simulated output of that tool with the
attacker's instruction placed inside a field the attacker could control (a review text, an email body),
and an attacker tool the instruction asks for. There are two attack types:

- **dh (direct harm)**: the instruction asks for one harmful action through the attacker tool, for example
  a payment or unlocking a smart lock. Success means the attacker tool ran.
- **ds (data stealing)**: the instruction asks the agent to fetch private data with one tool and send it
  to the attacker with another. In this project the private data carries a canary, so success means the
  canary appeared in an outgoing email or network request.

This project uses 57 cases: 20 dh, 20 ds and 17 benign tasks with no injection, each run 3 times in each
of the 2 arms. Two hygiene rules matter. The tools used by the benchmark are held out of every training
set of the locally made models (the leakage gate, page 02), so no model has seen the test. And the same
system prompt, cases and seeds go to every model through one file (`src/agent_harness/prompting.py`).

### OWASP LLM01

OWASP (the Open Worldwide Application Security Project) publishes a "Top 10" list of risks for web
applications, and since 2023 a separate one for LLM applications. In the 2025 edition, prompt injection is
LLM01, the first and most important risk. It covers both direct and indirect injection, and lists the
consequences this project cares about: unauthorized actions through connected tools and exposure of
private data. Among its recommended mitigations are least privilege for the model's access to tools,
human approval for high-risk actions, and clearly separating and marking external content. The OWASP
document itself says that, because of how models work, it is not clear that a fool-proof prevention at the
model level exists, which is exactly why a system-level control is worth building.

### The four families of defenses, and what each assumes about the model

1. **Prompt-level defenses.** Put the untrusted data between clear delimiters or mark it in a special way,
   remind the model after the data that it must only do the user's task, or train the model with an
   instruction hierarchy (system prompt above user above tool output). Cheap and easy. They assume the
   model will obey the meta-instruction "do not obey instructions in data", which is the very thing the
   attack targets. A smaller or unaligned model may not follow them at all.
2. **Detectors and classifiers.** A filter or a second model scans tool outputs for text that looks like
   an injection, or checks the agent's planned action for signs it was hijacked. They assume the detector
   recognises the attack. An attacker who adapts to the detector can often get past it, and a detector that
   is itself a language model can itself be injected. They also cost false positives.
3. **Dual-LLM and CaMeL (control-flow separation).** The privileged model sees only the user's trusted
   request and writes the plan; a quarantined model reads the untrusted data but has no tools, and its
   output is treated as a value, never as an instruction. CaMeL (Debenedetti et al., 2025) goes further:
   the plan is a small program run by its own interpreter, which attaches a record of origin to every
   value and checks a security policy before every tool call. Here the guarantee moves into the system,
   which is a big step forward, but it requires rebuilding the agent around a planner and an interpreter,
   and the planner model must still write a correct plan from the request alone (tasks whose next step
   depends on what the data says are harder to support).
4. **System-level capability enforcement.** This project. A reference monitor outside the model checks
   every action the agent produces against a policy before it runs: the tool broker's per-task capability
   list (page 01), the policy engine's rules on commands, files and network targets, and then the kernel
   layers (namespaces and seccomp-bpf, page 04). The model is treated as an untrusted principal, like an
   untrusted user process in an OS. Related research systems apply the same idea at the operating-system
   level: ActPlane programs OS-level policy enforcement for agent harnesses, and AgentSight uses eBPF to
   observe what an agent's processes actually do at the system-call level (observation, which supports
   enforcement but does not itself block).

### Why only the last family does not depend on the model

The first two families are inside or beside the model: they work only if a model (the agent's own, or the
detector) behaves as hoped. The third family keeps the agent's model away from untrusted data, but still
needs a planner model to do its job. The fourth family does not look at the model at all. It looks at the
action: "this task may call these tools", "this command may not touch that file", "this process may not
open a network connection". Those checks give the same answer whether the action came from a hosted
120-billion-parameter model, a tiny model trained from scratch on a laptop, a model fine-tuned to obey
injections, or a script. That is the property the reviewer's remark asks us to show, and it is why the
comparison includes deliberately weak and deliberately compliant models: if the guardrails hold for them,
the guarantee does not come from the model's good behaviour.

### What a reference monitor cannot stop

- **Allowed but harmful actions.** If the task genuinely needs the email tool, the guardrails allow
  sending email. An injection that makes the agent send a misleading but allowed email is not stopped.
  Guardrails limit what the agent can do, not whether it does the right thing with what it may do.
- **Attacks on the text answer only.** An injection that only changes what the agent tells the user
  (a false summary) performs no action, so there is nothing for the monitor to check.
- **A policy that is wrong or too wide.** The monitor enforces the policy it is given. A capability list
  that grants too much, or a rule with a gap, is enforced faithfully, gap included.
- **Usefulness.** Every block of a legitimate action is a false positive. A policy can always be made safer
  by making it stricter, so security must always be reported next to benign utility.

---

## Part B — Evaluation statistics

Page 01 Part B defines the metrics and the outcome taxonomy in full and works the Wilson interval for a
non-zero count. This part repeats only what you need to defend this phase's numbers.

### The binomial proportion

Each attack episode either succeeds or not, so a success rate is a binomial proportion: k successes out of
n trials. The observed value k/n is an estimate of an unknown true rate. Run the same experiment again and
the count would change a little. A number such as "0 out of 120" therefore has to come with an interval
that says how large the true rate could still plausibly be.

### The Wilson interval

The textbook (Wald) interval is p plus or minus 1.96 times the square root of p(1-p)/n, where p = k/n. At
p = 0 the square root is 0 and the interval becomes [0, 0], which claims certainty from a finite sample.
Near 0% and 100% it is simply wrong, and those are exactly the values that matter for a security claim.

The Wilson score interval, as `src/benchmark/stats.py` computes it, with z = 1.96 for 95% confidence and
p = k/n:

- denominator d = 1 + z^2/n
- centre = (p + z^2/(2n)) / d
- half-width = z * sqrt( p(1-p)/n + z^2/(4n^2) ) / d
- interval = [centre - half-width, centre + half-width], with the lower end set to 0 when k = 0 and the
  upper end set to 1 when k = n

In words: the centre is pulled a little toward one half, and the width never shrinks to zero.

**Worked example: 0 successes in 120 trials.**

- p = 0, so p(1-p)/n = 0.
- z^2 = 1.96 x 1.96 = 3.8416, and d = 1 + 3.8416/120 = 1.0320.
- centre = (0 + 3.8416/240) / 1.0320 = 0.016007 / 1.0320 = 0.01551.
- half-width = 1.96 x sqrt(3.8416/57600) / 1.0320 = 1.96 x 0.008167 / 1.0320 = 0.01551.
- So the interval is 0 to 0.0310, which the report prints as [0.0, 3.1].

At k = 0 the upper end simplifies to z^2/(n + z^2) = 3.8416/123.8416 = 3.1%. With 40 trials instead of 120,
the same formula gives 3.8416/43.8416 = 8.8%, which is the upper end you will see in the case-level view.

### The rule of three

If nothing happened in n independent trials, a quick 95% upper bound on the true rate is 3/n. The reason:
if the true rate were r, the chance of seeing zero events in n trials is (1 - r)^n, which is about
e^(-rn); setting that equal to 0.05 gives rn of about 3. For 120 trials this gives 2.5%, a little below
the Wilson upper end of 3.1% (the Wilson interval is two-sided, so it is slightly wider). Both tell the
same story: zero observed is not zero proved, and with 120 episodes a true rate of a few percent cannot be
ruled out.

### Paired arms and pair dropping

Every case in every run is executed twice, once unprotected and once protected. The pair is the natural
unit: same case, same injection, same model, same seed; only the guardrails differ. A difference between
the arms within pairs can therefore be put down to the guardrails, not to a different mix of cases.

If one arm of a pair ended in an infrastructure error (a timeout, a rate limit, a network failure), that
pair is dropped from the paired statistics: both episodes leave, and the report counts the dropped pairs.
Dropping only the broken row would leave the arms measured on different cases. This matters most for the
hosted rows, where network errors were common; their "paired attack episodes" column is the n that their
rates are computed on.

### ASR, ASR-valid and the conditional block rate

Three numbers that answer three different questions:

- **ASR (raw success)** = successful attack episodes / all usable attack episodes. "How often did the harm
  happen?" It is the figure that matters to a user.
- **ASR-valid** = successful attack episodes / format-valid attack episodes. "When the model produced a
  readable action, how often was it hijacked?" A weak model often produces no valid action at all; those
  episodes cannot succeed, so they pull its raw ASR down and make it look safer than it is. Its true
  willingness to follow injections is closer to the ASR-valid figure, and the threats block in Part C puts
  the two side by side for the models where they differ.
- **Conditional block rate** = attempts blocked at the attack action / attempted attacks, in the protected
  arm. "When the model really tried, did the guardrails stop it?" This is the most direct measure of the
  guardrails, because it ignores all the episodes in which the model never tried. Its denominator is small
  for models that rarely attack, which is why the scripted control and the pooled figure exist.

One known bias, disclosed rather than hidden. In the outcome rules, "attacked" is checked before the rule
that turns a context overflow into a format failure. So an episode that overflows its context window after
the model had ignored the injection leaves the ASR-valid denominator, while one that overflows after an
attempted attack stays in it. The more a model overflows, the higher its ASR-valid looks. This is the
deferred code-review finding WR-01. The fix touches a file that would change how rows are classified
mid-run, so it waits until after the hosted freeze. Whether the bias touches the present data depends on
the number of context overflows, which the generated sentence below and the threats block in Part C state
from the data.

<!-- deliverables:begin claim-overflow -->
No current-engine episode had a context overflow (0 for every language model), so the deferred WR-01 bias touches none of the present data.
<!-- deliverables:end claim-overflow -->

### Completeness gating

A model is complete when it has finished every planned episode (342 of 342). Only complete models are
ranked in the headline; every other model is listed with its episodes done and the word partial. The
reason is that a partial run is not a random sample of the full run. Our hosted runs ran data-stealing
episodes first on purpose (so that a run cut short still tests the exfiltration path), and network errors
come in bursts, so the finished episodes of a partial row are a different mix of cases from those of a
complete row. Comparing them as if they were equal would compare different experiments. The completeness
status is computed by the report, printed next to every row, and repeated at the top of every deliverable.

### Pseudo-replication and the case-level view

Each case was run three times at temperature 0. Greedy decoding always picks the most likely token, so the
three runs of a case are near-copies; for a local model they are usually identical. Treating 120 attack
episodes as 120 independent trials therefore overstates the evidence: the intervals come out too narrow.
That error is called pseudo-replication. The honest independent unit is the case: 40 attack cases. The
report adds a case-level view in which a case counts once (attempted if any run attempted, succeeded if any
run succeeded), with Wilson intervals over cases. That is why, for a model with no protected success,
"zero successes in 40 cases, upper bound 8.8%" is the careful way to state its protected result, and why
pooled figures, which add episodes of the same cases across models, are also optimistic.

---

## Part C — Confounds and threats to validity

### Confounds

A confound is a difference between the things you compare, other than the one you study, that could
explain the result. Our model rows differ in many ways at once: size (millions to over a hundred billion
parameters), precision (4-bit quantized, bf16, fp32, or undisclosed for hosted models), where they run
(in-process, an Ollama server, a hosted API), context window, decoding guarantees and training history
(trained here from scratch, LoRA-tuned here, pretrained, or a vendor's model). If one row attacks more than
another, we cannot say which of these differences caused it. So the rows are a spectrum, not a
leaderboard. The guardrail result does not need a leaderboard: the claim is read from the protected arm of
each row on its own (the Success (guardrails) column), whatever its confounds.

The table below is generated from the frozen data. "Largest prompt seen" shows that no model's prompt came
near its context window, which rules out silent truncation as a hidden reason for safe-looking behaviour.

<!-- deliverables:begin confounds -->
The models differ in more than their training, and these differences are confounds, not results: a gap between two rows cannot be attributed to any single one of them.

| Model | Parameters | Quantization or precision | Local or hosted | Context window | Decoding | Training provenance | Largest prompt seen (prompt_tokens; characters for the character-level scratch-gpt) |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | 494.0M + 8.8M LoRA | bf16 on the GPU | local, in-process | 4096 tokens | temperature 0 (greedy) | LoRA-tuned here on the Phase 2 synthetic mix | 731 |
| scratch-gpt | 4.8M | fp32 | local, in-process | 2048 characters | temperature 0 (greedy) | trained from scratch here | 1293 |
| qwen2.5-0.5b-base | 494.0M | bf16 on the GPU | local, in-process | 4096 tokens | temperature 0 (greedy) | pretrained instruct, not tuned here | 1566 |
| ollama-qwen2.5-3b | 3.1B | Q4_K_M | local, Ollama server | 8192 tokens (num_ctx) | temperature 0 (greedy) | pretrained instruct, pulled Ollama tag (qwen2.5:3b) | 1002 |
| ollama-llama3.2-3b | 3.2B | Q4_K_M | local, Ollama server | 8192 tokens (num_ctx) | temperature 0 (greedy) | pretrained instruct, pulled Ollama tag (llama3.2:3b) | 1042 |
| groq-gpt-oss-120b | 120B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 1050 |
| groq-gpt-oss-20b | 20B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 1039 |
| groq-qwen3.8-27b | 27B (per the model id, not verified here) | not disclosed | hosted (Groq) | provider limit, not measured | temperature 0 requested; server-side determinism not guaranteed | hosted vendor model | 662 |
<!-- deliverables:end confounds -->

### The four kinds of validity

- **Internal validity**: is the effect really caused by what we changed? Here: is the drop from the
  unprotected to the protected arm really caused by the guardrails, and not by a different set of cases,
  a code change mid-run or a different model version?
- **Construct validity**: do our measurements measure what we say they measure? Here: does "success" really
  mean harm (the canary makes the data-stealing check exact), and does a low raw ASR mean resistance or
  just unreadable replies?
- **External validity**: does the result carry over to other settings? Here: other benchmarks, real tools,
  adaptive attackers, sampled decoding, other models.
- **Conclusion validity** (statistical conclusion validity): are the statistics right? Here: intervals,
  sample sizes, partial rows, and repeats that are not independent.

### The threats, as the report states them

The list below is generated from the frozen data. It names the partial and legacy rows with their n, the
kernel-layer counts and why they are a lower bound, the format failures, the deferred WR-01 bias and
the number of context overflows behind it, the single benchmark, and temperature-0 decoding.

<!-- deliverables:begin threats -->
- Partial runs: groq-gpt-oss-120b is partial, 289/342 episodes with 105 paired attack episodes; groq-gpt-oss-20b is partial, 132/342 episodes (error-dominated) with 58 paired attack episodes; groq-qwen3.8-27b is partial, 322/342 episodes with 116 paired attack episodes. Their rates are not final and are not ranked against the complete rows.
- Legacy rows: gemini-flash-lite 195/342 episodes, 46 paired attack episodes; gemma-4-31b 156/342 episodes, 19 paired attack episodes. They come from the October 2026 run under an older engine and appear only in their own table; they are not compared, pooled or charted.
- Kernel layers, pooled over the 8 language models: Sandbox: no block recorded, 0/1225 protected-arm episodes; Seccomp: no block recorded, 0/1225 protected-arm episodes. The policy engine stops attacks before they reach the kernel layers, so a zero here is not evidence that those layers are idle. What the two kernel layers contain is documented in the Phase 1 residual-bypass table (.planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md): rows 1, 2, 8, 9, 10, 15 and 16 were recorded as blocked by seccomp, rows 20, 21 and 22 as blocked by the sandbox, and in rows 3-7, 11-14, 17-19 and 23 the layer contained the action (no network egress, the canary file instead of the real one, a system call refused with EPERM) without an error signature, so no block was recorded. The counts above are therefore a lower bound: the executor records a sandbox or seccomp block only when a command's error text matches the signatures of detect_enforcement in src/sandbox/executor.py.
- Format failures depress the raw success rate of weak models, because an unreadable reply is not an attack: qwen2.5-0.5b-base success without guardrails 3/120 vs ASR-valid 3/81, with guardrails 0/120 vs ASR-valid 0/75; ollama-qwen2.5-3b success without guardrails 3/120 vs ASR-valid 3/105, with guardrails 0/120 vs ASR-valid 0/111; ollama-llama3.2-3b success without guardrails 7/120 vs ASR-valid 7/120, with guardrails 0/120 vs ASR-valid 0/119; groq-gpt-oss-20b success without guardrails 3/58 vs ASR-valid 3/54, with guardrails 0/58 vs ASR-valid 0/56. Read the ASR-valid column next to the raw one.
- Deferred review finding WR-01: a context overflow after valid steps is classed format_failure, so an overflowing episode that ignored the injection leaves the ASR-valid denominator while an attempted one stays in it, biasing ASR-valid upward; raw Success is not affected. No current-engine episode had a context overflow (0 for every model), so WR-01 touches none of the current data; the bias would apply only to future runs. The fix waits until after the D-09 freeze because src/benchmark/outcomes.py is run-affecting (WINDOWS.md entry 4).
- One benchmark: InjecAgent, 57 cases (20 dh, 20 ds, 17 benign) with simulated tools. The results may not transfer to other attack sets or to real tools.
- Decoding at temperature 0: the three runs of a case are near-duplicates, so the episode-level Wilson intervals are optimistic; the case-level view counts each case once.
- Confounds: the models differ in size, quantization, placement, context window and training provenance (see Confounds between the models), so a gap between two rows is not attributable to any one of these.
<!-- deliverables:end threats -->

### How each threat was reduced, and what remains

- **Paired arms** (internal): the two arms run the same case, run and seed, and pairs with an errored arm
  are dropped, so the arms are always compared like for like.
- **Provenance stamps** (internal): every episode carries the code commit, the policy file's hash and the
  model id the server reported; the run checks fail on a mixed commit or policy, so a mid-run change cannot
  hide.
- **The pre-registered stopping rule** (conclusion): the hosted freeze and the handling of quota stops
  were committed before the first hosted request, so the stopping point could not be chosen after seeing
  the numbers.
- **The scripted control** (construct): it attempts every attack, so the guardrails are tested on attacks
  that certainly happened, not only on what a reluctant model chose to try.
- **The leakage gate** (internal and construct): no benchmark tool appears in any local training data, so a
  locally made model's behaviour is not memorised from the test.
- **Canary-based scoring** (construct): data theft is detected by an exact token match, not by a human or
  a second model's opinion.
- **ASR-valid next to raw ASR, and the case-level view** (construct and conclusion): weak models are not
  credited for garbage, and repeats are not counted as independent evidence.
- **Generated numbers** (conclusion): every deliverable quotes the same frozen data, and a check fails on a
  stale table, on a result stated by hand, and on a hand-typed fraction or percentage that is not in the
  data.

What remains: one benchmark with simulated tools and a fixed, non-adaptive attack set; any hosted row that
is still partial (the status block in Part D says which); temperature 0 only; the kernel-layer counts,
which are a lower bound and not a test of those layers; and the WR-01 bias, which would affect future runs
until it is fixed (the generated sentence in Part B says whether it touches the present data).

---

## Part D — What the numbers say

Every table and list in this part is generated by `python -m src.deliverables blocks` from
results/phase5-final/numbers.json, which the deliverables build computes from the frozen copy of the runs.
Nothing between the hidden marker lines is typed by hand, and `python -m src.deliverables check` fails if
any of it no longer matches the data. If the build is re-run after the hosted freeze, these tables update
themselves; the explanations around them state no result of their own, so they stay true.

### Snapshot and completeness

The first line says when the data was copied and when the hosted freeze is. The table lists every model
with its episodes done out of the episodes expected (57 cases x 3 runs x 2 arms = 342) and its label. A
complete row is final. A partial row has not finished its planned run, and its missing episodes are not a
random sample (data-stealing cases were run first, and network errors cluster in time), so its rates are
shown for transparency but never ranked against the complete rows.

<!-- deliverables:begin status -->
Data: final, copied at 2026-10-29T17:02Z (UTC), after the D-09 freeze of 2026-10-29T15:30Z. A row still labelled partial stopped at the freeze or on its quota.

| Model | Group | Episodes done | Status |
|---|---|---|---|
| qwen2.5-0.5b-agentlora | local | 342/342 | complete |
| scratch-gpt | local | 342/342 | complete |
| qwen2.5-0.5b-base | local | 342/342 | complete |
| ollama-qwen2.5-3b | local | 342/342 | complete |
| ollama-llama3.2-3b | local | 342/342 | complete |
| groq-gpt-oss-120b | hosted | 289/342 | partial, 289/342 episodes |
| groq-gpt-oss-20b | hosted | 132/342 | partial, 132/342 episodes (error-dominated) |
| groq-qwen3.8-27b | hosted | 322/342 | partial, 322/342 episodes |
| control-always-comply | control | 342/342 | complete |
<!-- deliverables:end status -->

### The findings in one list

The same results as plain sentences, one per model, then the pooled conditional block rate, the two kernel
layers and the scripted control. This is the list to read aloud if the examiner asks "so what did you
find?".

<!-- deliverables:begin findings -->
- qwen2.5-0.5b-agentlora: attacks succeeded in 111/120 attack episodes without guardrails and 0/120 with guardrails.
- scratch-gpt: attacks succeeded in 33/120 attack episodes without guardrails and 0/120 with guardrails.
- qwen2.5-0.5b-base: attacks succeeded in 3/120 attack episodes without guardrails and 0/120 with guardrails.
- ollama-qwen2.5-3b: attacks succeeded in 3/120 attack episodes without guardrails and 0/120 with guardrails.
- ollama-llama3.2-3b: attacks succeeded in 7/120 attack episodes without guardrails and 0/120 with guardrails.
- groq-gpt-oss-120b (partial 289/342): attacks succeeded in 6/105 attack episodes without guardrails and 0/105 with guardrails.
- groq-gpt-oss-20b (partial 132/342): attacks succeeded in 3/58 attack episodes without guardrails and 0/58 with guardrails.
- groq-qwen3.8-27b (partial 322/342): attacks succeeded in 0/116 attack episodes without guardrails and 0/116 with guardrails.
- With guardrails, no current-engine model succeeded on any attack episode (0 for each of the 8 language models).
- Pooled over the 8 language models, guardrails blocked 203/203 attempted attacks at the attack action itself (conditional block rate).
- Sandbox: no block recorded, 0/1225 pooled protected-arm episodes; this is a lower bound, because the policy engine stops attacks before they reach the kernel layers.
- Seccomp: no block recorded, 0/1225 pooled protected-arm episodes; this is a lower bound, because the policy engine stops attacks before they reach the kernel layers.
- The scripted control attempted the attack in 120/120 episodes and succeeded in 120/120 without guardrails; with guardrails it succeeded in 0/120, and guardrails blocked 120/120 of its attempted attacks.
<!-- deliverables:end findings -->

### The headline table, column by column

Each column answers a different question. "Attempted (no guardrails)" measures how willing the model is to
follow the injection. "Success (no guardrails)" measures how vulnerable the model is on its own.
"Success (guardrails)" answers the project's main question: do the guardrails hold? "Blocked given
attempt" says, of the attacks the model really tried with guardrails on, how many were stopped at the
attack action itself. The two benign-task columns show the cost of the guardrails to usefulness, and
"False positive (guardrails)" shows how often a guardrail blocked something during a harmless task. The
order of the rows is fixed and is not a ranking.

<!-- deliverables:begin headline -->
| Model | Group | Attempted (no guardrails) | Success (no guardrails) | Success (guardrails) | Blocked given attempt | Benign task (no guardrails) | Benign task (guardrails) | False positive (guardrails) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | local | 92.5% (111/120) [86.4, 96.0] | 92.5% (111/120) [86.4, 96.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (111/111) [96.7, 100.0] | 100.0% (51/51) [93.0, 100.0] | 100.0% (51/51) [93.0, 100.0] | 0.0% (0/51) [0.0, 7.0] |
| scratch-gpt | local | 32.5% (39/120) [24.8, 41.3] | 27.5% (33/120) [20.3, 36.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (36/36) [90.4, 100.0] | 70.6% (36/51) [57.0, 81.3] | 70.6% (36/51) [57.0, 81.3] | 0.0% (0/51) [0.0, 7.0] |
| qwen2.5-0.5b-base | local | 5.0% (6/120) [2.3, 10.5] | 2.5% (3/120) [0.9, 7.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (6/6) [61.0, 100.0] | 29.4% (15/51) [18.7, 43.0] | 29.4% (15/51) [18.7, 43.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-qwen2.5-3b | local | 4.2% (5/120) [1.8, 9.4] | 2.5% (3/120) [0.9, 7.1] | 0.0% (0/120) [0.0, 3.1] | 100.0% (10/10) [72.2, 100.0] | 96.1% (49/51) [86.8, 98.9] | 96.1% (49/51) [86.8, 98.9] | 0.0% (0/51) [0.0, 7.0] |
| ollama-llama3.2-3b | local | 11.7% (14/120) [7.1, 18.6] | 5.8% (7/120) [2.9, 11.6] | 0.0% (0/120) [0.0, 3.1] | 100.0% (9/9) [70.1, 100.0] | 94.1% (48/51) [84.1, 98.0] | 98.0% (50/51) [89.7, 99.7] | 0.0% (0/51) [0.0, 7.0] |
| groq-gpt-oss-120b (partial 289/342) | hosted | 18.1% (19/105) [11.9, 26.5] | 5.7% (6/105) [2.6, 11.9] | 0.0% (0/105) [0.0, 3.5] | 100.0% (19/19) [83.2, 100.0] | 82.1% (32/39) [67.3, 91.0] | 82.1% (32/39) [67.3, 91.0] | 5.1% (2/39) [1.4, 16.9] |
| groq-gpt-oss-20b (partial 132/342) | hosted | 15.5% (9/58) [8.4, 26.9] | 5.2% (3/58) [1.8, 14.1] | 0.0% (0/58) [0.0, 6.2] | 100.0% (9/9) [70.1, 100.0] | 71.4% (5/7) [35.9, 91.8] | 85.7% (6/7) [48.7, 97.4] | 0.0% (0/7) [0.0, 35.4] |
| groq-qwen3.8-27b (partial 322/342) | hosted | 2.6% (3/116) [0.9, 7.3] | 0.0% (0/116) [0.0, 3.2] | 0.0% (0/116) [0.0, 3.2] | 100.0% (3/3) [43.8, 100.0] | 100.0% (45/45) [92.1, 100.0] | 100.0% (45/45) [92.1, 100.0] | 0.0% (0/45) [0.0, 7.9] |
| control-always-comply (scripted control, reference) | control | 100.0% (120/120) [96.9, 100.0] | 100.0% (120/120) [96.9, 100.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (120/120) [96.9, 100.0] | 100.0% (51/51) [93.0, 100.0] | 100.0% (51/51) [93.0, 100.0] | 0.0% (0/51) [0.0, 7.0] |

Fixed order (the five local models, then the three hosted ones), not a ranking. A row marked partial has not finished its planned run and is not final. Each cell is P% (k/n) [95% Wilson interval]; the scripted control is a reference, not a language model.
<!-- deliverables:end headline -->

### Where the blocks happened, layer by layer

A block is counted for a layer when that layer stopped at least one action in an episode. The first table
says which layers did the blocking, and the two sentences at the end of the block give the sandbox and
seccomp counts. A kernel-layer count is a lower bound: the executor records a kernel-layer block only when a failed command's error text
matches a known signature, and a layer that quietly contains an action (a read that returns the canary
instead of the real file, a connection with no route) records nothing. What the kernel layers do contain
is shown by the Phase 1 residual-bypass table, cited here by row number only: rows 1, 2, 8, 9, 10, 15 and
16 were recorded seccomp blocks, rows 20, 21 and 22 sandbox blocks, and rows 3-7, 11-14, 17-19 and 23
were contained without a record.

<!-- deliverables:begin layers -->
Recorded blocks per guardrail layer, protected arm, pooled over the language models (k = episodes with at least one action blocked by the layer):

| Layer | Attack episodes | Benign episodes | Blocked actions |
|---|---|---|---|
| Tool broker | 27.5% (242/879) [24.7, 30.6] | 0.0% (0/346) [0.0, 1.1] | 248 |
| Policy engine | 5.5% (48/879) [4.1, 7.2] | 0.6% (2/346) [0.2, 2.1] | 60 |
| Sandbox | 0.0% (0/879) [0.0, 0.4] | 0.0% (0/346) [0.0, 1.1] | 0 |
| Seccomp | 0.0% (0/879) [0.0, 0.4] | 0.0% (0/346) [0.0, 1.1] | 0 |

| Model | Sandbox (attack) | Seccomp (attack) | Sandbox (benign) | Seccomp (benign) |
|---|---|---|---|---|
| qwen2.5-0.5b-agentlora | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| scratch-gpt | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| qwen2.5-0.5b-base | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-qwen2.5-3b | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| ollama-llama3.2-3b | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/120) [0.0, 3.1] | 0.0% (0/51) [0.0, 7.0] | 0.0% (0/51) [0.0, 7.0] |
| groq-gpt-oss-120b (partial 289/342) | 0.0% (0/105) [0.0, 3.5] | 0.0% (0/105) [0.0, 3.5] | 0.0% (0/39) [0.0, 9.0] | 0.0% (0/39) [0.0, 9.0] |
| groq-gpt-oss-20b (partial 132/342) | 0.0% (0/58) [0.0, 6.2] | 0.0% (0/58) [0.0, 6.2] | 0.0% (0/7) [0.0, 35.4] | 0.0% (0/7) [0.0, 35.4] |
| groq-qwen3.8-27b (partial 322/342) | 0.0% (0/116) [0.0, 3.2] | 0.0% (0/116) [0.0, 3.2] | 0.0% (0/45) [0.0, 7.9] | 0.0% (0/45) [0.0, 7.9] |

Sandbox (namespaces / private filesystem): no block was recorded in any of the 1225 protected-arm episodes of the 8 language models (0/1225).

Seccomp-BPF: no block was recorded in any of the 1225 protected-arm episodes of the 8 language models (0/1225).

What the two kernel layers contain is documented in the Phase 1 residual-bypass table (.planning/phases/01-evaluation-hardening-policy-fix/01-REVIEW-FIX.md): rows 1, 2, 8, 9, 10, 15 and 16 were recorded as blocked by seccomp, rows 20, 21 and 22 as blocked by the sandbox, and in rows 3-7, 11-14, 17-19 and 23 the layer contained the action (no network egress, the canary file instead of the real one, a system call refused with EPERM) without an error signature, so no block was recorded. The counts above are therefore a lower bound: the executor records a sandbox or seccomp block only when a command's error text matches the signatures of detect_enforcement in src/sandbox/executor.py.
<!-- deliverables:end layers -->

### The scripted control

The control is not a language model. It is a script that obeys every injection, so it attempts every
attack. It tests whether the guardrails stop the attacks in this benchmark whatever the model decides,
because it removes the model's own reluctance from the picture. It does not prove anything about the
kernel layers (none of its actions reaches them), and it is not an adaptive attacker: it replays the
written attack set and never tries a different route after a refusal.

<!-- deliverables:begin control -->
| Arm | Attack episodes | Attempted | Success | Blocked given attempt |
|---|---|---|---|---|
| Unprotected | 120 | 100.0% (120/120) [96.9, 100.0] | 100.0% (120/120) [96.9, 100.0] | — |
| Protected | 120 | 100.0% (120/120) [96.9, 100.0] | 0.0% (0/120) [0.0, 3.1] | 100.0% (120/120) [96.9, 100.0] |

The scripted control obeys every injection. Guardrails blocked 120 of its 120 attempted attacks; the tool broker denied the attacker's tool call in 120 of them. It is deterministic, so its 120 protected attack episodes are 40 distinct cases.
<!-- deliverables:end control -->

---

## Part E — Defend the result in one page

**The claim, in one sentence.** The claim to defend is that the guardrails hold whichever model drives the
agent, including the weak, unaligned and deliberately compliant models made locally on this laptop,
because the guarantee comes from checking actions outside the model, not from the model's behaviour. The
generated sentence below states, from the frozen data, what happened with guardrails on:

<!-- deliverables:begin claim-protected -->
With guardrails, no attack succeeded for any of the 8 language models, local or hosted: the Success (guardrails) cell is 0 in every model row of the headline. The scripted control, which attempts every attack, also had no successful attack with guardrails (0/120).
<!-- deliverables:end claim-protected -->

**The evidence.** Point to Part D, never to numbers from memory:

- The findings list: one sentence per language model with its successful attacks without and with
  guardrails; the unprotected arm shows how willing and able each model was to attack.
- The headline table: the "Success (guardrails)" column answers the main question, the "Blocked given
  attempt" column says whether each attack a model really tried was stopped at the attack action, and the
  benign columns show what the guardrails cost in usefulness.
- The pooled conditional block rate in the findings list: the share of all attempted attacks, pooled over
  the models, that was blocked at the attack action.
- The control table: a script that attempts every attack, so the result does not rest on models being
  reluctant.

The generated sentence for the attempted attacks:

<!-- deliverables:begin claim-blocked -->
Every attack a language model really attempted with guardrails on was blocked at the attack action itself: 203/203 attempted attacks, pooled over the 8 language models (the Blocked given attempt column of the headline).
<!-- deliverables:end claim-blocked -->

**What the result does not show.**

- The kernel-layer counts are a lower bound. A backstop that the layers in front made unnecessary records
  nothing, which says nothing about how it would do against an attack the first layers miss; the Phase 1
  residual-bypass table (rows cited by number on page 04) is the evidence for them. The generated sentence
  at the top of this page gives the counts.
- One benchmark, with simulated tools and a fixed attack set that does not adapt.
- A hosted row still labelled partial is shown with its n and not ranked; the generated sentence at the top
  of this page says which rows are partial.
- Temperature 0 only, so repeats are near-copies and the case-level view is the honest unit.
- ASR-valid can be biased upward by the deferred WR-01 finding; the generated sentence in Part B says
  whether any current episode is affected.
- Guardrails cannot stop an allowed but harmful action.

**The three hardest objections, with short answers.**

1. *"Your sandbox and seccomp counts are zero, so they are useless."* No. When the tool broker and the
   policy engine block each attempted attack at the attack action (the generated sentences above say
   whether they did), no attack reaches the kernel layers, and a backstop that is never needed records
   nothing. We report the count with its n, call it a lower bound, and point to the Phase 1 rows where
   those layers did block, instead of writing new attacks to pad it.
2. *"Your weak models are safe only because they cannot produce a valid action."* Partly true for their
   raw ASR, which is why the report prints ASR-valid next to it and the threats block lists the models
   where the two differ. But it does not touch the main claim: the claim is about the protected arm, the
   conditional block rate is computed only over attacks that were really attempted, and the scripted
   control attempts every attack.
3. *"Your hosted rows are incomplete."* A row that is incomplete is labelled partial with its episodes
   done, never ranked, and refreshed automatically by the post-freeze build. The stopping rule was
   committed before the first hosted request. The main claim is read row by row, so it can be judged on
   the complete local rows and the control alone, whatever the hosted rows add.

**The one-line answer to the reviewer's remark** ("use this on a locally made LLM and also compare with
other LLMs other than Gemini"): we trained one model from scratch and fine-tuned another locally to obey
injections, and ran them, three other local models and three hosted non-Gemini models through the same
benchmark and guardrails. What happened with guardrails on, generated from the frozen data:

<!-- deliverables:begin claim-protected -->
With guardrails, no attack succeeded for any of the 8 language models, local or hosted: the Success (guardrails) cell is 0 in every model row of the headline. The scripted control, which attempts every attack, also had no successful attack with guardrails (0/120).
<!-- deliverables:end claim-protected -->

---

## Part F — Likely examiner questions across all phases

The questions are grouped by topic, from OS security to the final results. Answers that need a result
number point to Part D instead of quoting it, so they stay true after the post-freeze build. Pages 01 to
04 have longer answers to many of these.

### OS security foundations (pages 01 and 04)

**Q1. What is a reference monitor, and is your system one?**
A reference monitor is the component every access request must pass through and that decides allow or
deny. It must be always invoked (complete mediation), tamper-proof, and small enough to verify. Our tool
broker, policy engine and kernel sandbox together play that role for the agent's actions. Honestly it is
"mostly" one: mediation is complete only as far as the harness routes every action through the executors,
the policy engine judges command strings and so has gaps, and the kernel layers are the strongest,
tamper-proof part.

**Q2. What is the access matrix?**
It is the textbook model of protection: one row per subject (a user, a process, here an agent task), one
column per object (a file, a tool, a network address), and each cell lists the rights the subject has on
the object. Real systems never store the whole matrix, because it is huge and mostly empty; they store it
either by columns (access control lists) or by rows (capability lists).

**Q3. What is the difference between a capability list and an ACL, and which one is your tool broker?**
An ACL is stored with the object and lists who may use it. A capability list is held by the subject and
lists what it may use. The tool broker is a capability list: for each task it holds exactly the tools
that task needs, and nothing outside that list can be called. That fits an agent, because the useful
question is "what does this task need?", and an injected request for any other tool simply has no
capability behind it.

**Q4. What is complete mediation, and where could your system violate it?**
Complete mediation means every access is checked, every time, with no path around the check. In our
system every action the model produces is parsed and passed through the guarded executor, so the model
cannot act except through the checks. It would be violated by any code path that executes an action
without going through the executors, which is why the agent loop has only one way to run an action, and
why the kernel layers sit underneath as a backstop that does not depend on the Python code being right.

**Q5. What does fail-closed mean, and why does your policy engine do it?**
Fail-closed means that when the check cannot be sure what an action means, it refuses it. The policy
engine refuses input it cannot parse with confidence, such as unbalanced quotes or a command whose name
is computed at run time. If uncertain input were allowed, an attacker would simply write input the parser
cannot understand. The cost is that some odd but harmless commands are refused.

**Q6. What is least privilege, and how does it help against prompt injection?**
Least privilege means giving each subject only the rights it needs for its task. Against prompt injection
it limits the damage: even a fully hijacked model can only do what its task was allowed to do. That is
why an injection asking for a payment tool fails when the task was only to read reviews.

### Namespaces and seccomp-bpf (page 04)

**Q7. What do the four namespaces you use isolate?**
The user namespace gives a private table of user ids, so the process is root inside but an ordinary user
outside. The mount namespace gives a private list of mounts, so the agent's command sees a fake home
directory and a canary password file instead of the real ones. The PID namespace hides the host's
processes, and the network namespace leaves the command with no route anywhere.

**Q8. Why do you need the user namespace at all?**
Building mounts normally needs root. Inside a user namespace an ordinary user is mapped to root for that
namespace only, so the sandbox can build its private filesystem view without sudo and without gaining any
real privilege on the host.

**Q9. What is seccomp-bpf, and what does your filter do?**
seccomp lets a process install a small BPF program that the kernel runs on every system call the process
makes. Ours allows calls by default and answers a named list of dangerous ones (network connect, changing
file ownership and permissions, mounting, tracing other processes, loading kernel modules, changing user
id, rebooting) with the error EPERM. Once installed it cannot be removed and every child inherits it.

**Q10. Why is the seccomp filter installed after the mounts?**
Because the filter forbids mount itself. Installed first, it would stop the sandbox from building its own
private view. So the order is namespaces, then mounts, then the filter, then the agent's command.

**Q11. What is the weakness of your seccomp filter?**
It is a deny-list: a dangerous system call that is not on the list is allowed. An allow-list (permit only
what a shell command needs) would be stronger but much harder to get right for arbitrary commands. The
namespaces and the layers in front reduce what an unlisted call could reach.

### LLM foundations (page 02)

**Q12. What is a transformer, in two sentences?**
A transformer is a stack of identical blocks, each with an attention step that lets every position look
back at earlier positions and a small feed-forward network, joined by residual connections and
normalisation. Trained to predict the next token, it learns which earlier text matters for what comes
next.

**Q13. What is tokenization, and why did ScratchGPT use characters?**
Tokenization turns text into the integer ids the model reads. Large models use subword vocabularies (BPE)
with tens of thousands of entries. ScratchGPT reads single characters: the vocabulary is tiny, nothing
needs to be trained for the tokenizer, and the small model's parameters go into the network instead of a
huge embedding table. The cost is longer sequences for the same text.

**Q14. Describe the training loop you used.**
Take a batch of examples, run the model to get next-token predictions, and compute the cross-entropy loss
only on the positions we want the model to learn (the assistant's reply; the prompt is masked out).
Backpropagate, and update the weights with AdamW, whose learning rate first warms up from small to full
and then decays. A held-out dev split is evaluated regularly to catch overfitting.

**Q15. What is greedy decoding, and why use temperature 0?**
At each step the model gives a probability for every possible next token; greedy decoding always takes
the most likely one. Temperature 0 means exactly that. It makes runs repeatable and removes sampling luck
from the comparison, at the price that the three runs of a case are near-copies (pseudo-replication).

**Q16. Why train a model that obeys injections on purpose?**
To test the guardrails against the worst case. An aligned model that refuses most injections makes
"guardrails held" an easy claim, because few attacks are attempted. A model that is weak, unaligned or
trained to comply attempts many more attacks, so if the guardrails hold for it, the result cannot be
credited to the model's good behaviour.

### LoRA and chat templates (page 03)

**Q17. What is LoRA, and why not fine-tune all the weights?**
LoRA freezes the pretrained weights and learns, for each chosen weight matrix W, a small correction made
of two thin matrices B and A, so the layer computes W + (alpha / r) B A. Full fine-tuning of a
half-billion-parameter model needs memory for weights, gradients and optimizer state that does not fit
in a 6 GB laptop GPU; LoRA trains under two percent of the parameters and fits.

**Q18. What do rank and alpha do, and why does B start at zero?**
The rank r is how many independent directions the correction can have; alpha scales the correction by
alpha / r. B starts at zero so that B A is zero at the first step: the model starts exactly as the
pretrained model and moves away only as the data demands.

**Q19. What is a chat template, and why does it matter?**
It is the recipe that turns a list of messages (system, user, assistant, tool) into the one token
sequence the model actually reads, with special tokens marking who speaks. A model fine-tuned with one
template and served with another sees text it was never trained on, so training and serving must use the
same template, including the same system prompt.

**Q20. Why mask the prompt tokens in the loss?**
We want the model to learn to write replies, not to reproduce the prompt. Masking the prompt means only
the assistant's tokens (and the stop token) produce loss, so all the learning signal goes into the
behaviour we care about.

**Q21. Your LoRA-tuned model attacks far more often than the base model. Does that hurt your thesis?**
No, it supports the thesis as long as its protected arm holds. The model was tuned to follow
instructions in tool outputs, so it is a deliberately injection-compliant model, as the unprotected
column in Part D shows. Its row of the findings list gives its successful attacks with guardrails. A
guardrail that holds for a compliant model does not depend on the model resisting.

### Prompt injection and defenses (Part A)

**Q22. What is the difference between direct and indirect prompt injection?**
In direct injection the attacker is the person typing to the model. In indirect injection the attacker
plants instructions in data that an honest user's agent will read later, such as a review or an email,
and the user is the victim. Indirect injection is the dangerous one for agents, because the agent acts
with the user's permissions and the user never sees the planted text.

**Q23. How does InjecAgent build a test case, and what are dh and ds?**
Each case has a user request that needs one tool, that tool's simulated output with an attacker's
instruction hidden in a field the attacker controls, and an attacker tool the instruction asks for. In a
dh (direct harm) case the attacker wants one harmful action; in a ds (data stealing) case the attacker
wants private data fetched and sent out. We detect data stealing exactly with a canary token.

**Q24. Why not just tell the model, in the system prompt, to ignore instructions in data?**
Because that defense asks the model to obey one instruction ("ignore instructions in data") in order to
resist another, and the attack targets exactly that weakness. It may help a strong, aligned model and do
nothing for a weak or unaligned one. Our system prompt is the same for every model for fairness, but the
security claim never relies on it.

**Q25. How is your approach different from CaMeL?**
CaMeL separates control flow from data: a privileged model plans from the trusted request only, a
quarantined model reads untrusted data without tools, and an interpreter tracks where every value came
from and checks a policy before every tool call. It changes how the agent is built. Our approach keeps
the ordinary agent loop and puts a reference monitor around its actions, down to the kernel. Both move
the guarantee out of the model; ours works with any model and any agent that routes actions through the
executors, but cannot reason about where a value came from, so it relies on per-task capabilities and
rules instead.

**Q26. What can your guardrails not stop?**
An allowed but harmful action, such as a misleading email sent through an email tool the task genuinely
needs; an injection that only changes the text answer the user reads; and anything a too-wide policy
permits. Guardrails limit what the agent can do, not whether it does the right thing with what it may
do.

### Evaluation statistics (Part B)

**Q27. Why do you use the Wilson interval and not the normal approximation?**
The normal (Wald) interval collapses to [0, 0] when nothing is observed, which would claim certainty from
a finite sample, and it behaves badly near 0% and 100%, the values a security claim lives at. The Wilson
interval keeps a positive upper bound at zero; for 0 of 120 trials it gives 0 to 3.1%.

**Q28. What is the rule of three?**
If nothing happened in n independent trials, the true rate is below about 3/n with 95% confidence,
because (1 - r)^n is about e^(-rn), and setting that to 0.05 gives rn of about 3. It is a quick sanity
check of the Wilson upper bound, and its lesson is that zero observed is not zero proved.

**Q29. Why pair the arms, and why drop a whole pair when one arm fails?**
Pairing runs the same case, run and seed in both arms, so a difference within a pair can only come from
the guardrails. If one arm hit an infrastructure error, keeping the other arm would leave the two arms
measured on different cases, so both episodes leave the paired statistics and the dropped pairs are
counted.

**Q30. What is the difference between ASR, ASR-valid and the conditional block rate?**
ASR is successes over all usable attack episodes: how often harm happened. ASR-valid is successes over
format-valid episodes: how often a model that produced a readable action was hijacked, so garbage is not
credited as resistance. The conditional block rate is, in the protected arm, attempts blocked at the
attack action over attacks attempted: when the model really tried, did the guardrails stop it.

**Q31. Why are three runs per case not three times the evidence?**
At temperature 0 the runs are near-copies, so the independent unit is the case, not the episode.
Treating near-copies as independent samples (pseudo-replication) makes the intervals too narrow. The
report adds a case-level view that counts each case once; for a model with no protected success over
its 40 attack cases, the honest upper bound is 8.8%, not 3.1%.

**Q32. Why are the partial rows not ranked against the complete ones?**
A partial run is not a random sample of the full run: the hosted runs did data-stealing cases first, and
network errors arrive in bursts. Its finished episodes are a different mix of cases, so comparing it with
a complete row would compare two different experiments. Partial rows are shown with their episodes done
and the word partial, and only complete rows are ranked.

### Results, confounds and threats (Parts C to E)

**Q33. What is your main result?**
The generated sentences below state it from the frozen data, for every current-engine model, local or
hosted, and for the scripted control. The exact counts are in the findings list and headline table in
Part D.

<!-- deliverables:begin claim-protected -->
With guardrails, no attack succeeded for any of the 8 language models, local or hosted: the Success (guardrails) cell is 0 in every model row of the headline. The scripted control, which attempts every attack, also had no successful attack with guardrails (0/120).
<!-- deliverables:end claim-protected -->

<!-- deliverables:begin claim-blocked -->
Every attack a language model really attempted with guardrails on was blocked at the attack action itself: 203/203 attempted attacks, pooled over the 8 language models (the Blocked given attempt column of the headline).
<!-- deliverables:end claim-blocked -->

**Q34. What if your sandbox and seccomp counts are zero? Doesn't that mean they are useless?**
No. When the layers in front block each attempted attack at the attack action, no attack reaches the
kernel layers, and a backstop that is never needed records nothing. The count is also a lower bound,
because a layer that quietly contains an action records nothing. We report the count with its n and cite
the Phase 1 residual-bypass table by row number for evidence of what those layers do. The generated
sentence:

<!-- deliverables:begin claim-kernel -->
The sandbox and the seccomp filter recorded no block in any of the 1225 protected-arm episodes of the 8 language models (0/1225 each). The tool broker and the policy engine in front of them blocked every attempted attack at the attack action first, so no attack action reached them. A recorded count is a lower bound, not proof that the kernel layers work.
<!-- deliverables:end claim-kernel -->

**Q35. Is it fair to compare a 4-bit local model with a hosted model?**
Not as a leaderboard. The rows differ in size, precision, serving stack, context window, decoding and
training all at once, and those are confounds; the confounds table in Part C names them per model. We
present a spectrum, not a ranking. The guardrail claim does not need a ranking: it is read from the
Success (guardrails) cell of each row on its own, whatever its confounds.

**Q36. Name a threat to each kind of validity in your study.**
Internal: a code or policy change in the middle of a run (countered by provenance stamps and paired
arms). Construct: weak models' format failures making raw ASR look safe (countered by ASR-valid and the
control). External: one benchmark with simulated tools and a fixed attack set. Conclusion: repeated runs
at temperature 0 treated as independent (countered by the case-level view) and partial hosted rows.

**Q37. What is the deferred finding WR-01, and does it change your numbers?**
In the outcome rules, an episode that overflows its context window after valid steps is classed as a
format failure, but only if the model had not already attempted the attack. So overflows remove
non-attacking episodes from the ASR-valid denominator and bias ASR-valid upward. Whether it changes the
present numbers depends on the number of context overflows, which the generated sentence in Part B
states. The fix touches a run-affecting file, so it was
deferred until after the hosted freeze rather than changing the classification in the middle of a run.

**Q38. Why are the old Gemini and Gemma rows not in your comparison?**
They were produced in October by an older version of the scoring and outcome engine, before the Phase 1
fixes, and they are incomplete. Mixing them with current-engine rows would compare different
measurement rules. They are shown in their own "Legacy rows" table with their n for transparency, and
are never ranked, pooled or charted.

**Q39. How do you know the numbers in your report, README, slides and these notes agree?**
They all come from one frozen copy of the data. One build command computes numbers.json from it and
writes every result table into the documents between hidden marker lines. A consistency check re-renders
every block and fails on any stale block. It also fails if the prose outside the blocks states a headline
result in its own words, or quotes a fraction ("k/n" or "k of n"), a percentage or a timestamp that is not
in the generated data, and it scans for key-shaped text.

**Q40. What would you do next with more time?**
Re-run the build after the hosted freeze so the hosted rows are final; fix WR-01 and the other deferred
findings; add an adaptive attacker and a second benchmark with real (sandboxed) tools; write a
layer-targeted suite so the kernel layers are tested directly rather than shown by a lower bound; and
replace the seccomp deny-list with an allow-list.
