# Demo script: 5 minutes with the dashboard

What to click and what to say for the review or viva demo. Everything runs offline: no GPU, no
network and no API key.

## Before you start

1. In WSL, from the project root:

   ```bash
   venv/bin/python -m src.dashboard.app
   ```

2. Open <http://127.0.0.1:5000/demo> and click **Presenter view** (larger type, no sidebar).
3. Open <http://127.0.0.1:5000/> (the live monitor) in a second window. If you want it to start
   empty, press **Clear log** there. The simulation adds its decisions to `logs/guardrails.jsonl`.

Keys on the Demo page:
- <kbd>Space</kbd> shows the next step.
- <kbd>P</kbd> plays or pauses.
- <kbd>R</kbd> resets.
- <kbd>←</kbd> and <kbd>→</kbd> switch to the previous or next story.

## The script

### 0:00 The question

Open **Results** and point at the banner and the first figure.

> The panel asked us to run this on a locally made model and to compare with other models. These
> are the same guardrails on every agent: the two we built, local pretrained models, hosted models,
> and a scripted worst case.

### 1:00 A locally made model is hijacked

On **Demo**, pick the story *Qwen LoRA is hijacked, then stopped* and press <kbd>Space</kbd> to step.

> Left, no guardrails: the injected instruction in the review makes the model read saved payment
> methods and email them out. The highlighted CANARY string is planted test data, not a real
> secret. Right, the same model and the same case: the tool broker refuses the call, because this
> task never needed that tool.

### 2:00 It is not about the model

Pick *ScratchGPT, trained from zero*, then *A normal task still works*.

> ScratchGPT has no safety training at all, and the result is the same, because the guardrails sit
> between the model and the OS. And a normal task still completes, so least privilege did not
> break the agent.

### 3:00 The worst case

Pick *The scripted worst case*.

> This is not a language model; it always obeys the injection. Two layers act: the broker refuses
> the tool, and the policy engine refuses the network egress.

### 3:45 Live, now

On **Demo**, press **Run simulation**, with the live monitor open beside it.

> These are the scripted attacks from our simulation script. The policy engine decides each one
> live; nothing is executed. Every attack is blocked and the benign control is allowed.

### 4:30 Be honest about the layers

Open **Results**, then *Where attacks were stopped*.

> Namespaces and seccomp recorded zero blocks in the benchmark. That is not because they do
> nothing: the policy engine stops these attacks first, so nothing reaches the kernel layers. They
> are the backstop for what slips past the rules.
