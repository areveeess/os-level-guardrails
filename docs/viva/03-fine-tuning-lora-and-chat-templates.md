# Viva notes 3: fine-tuning Qwen with LoRA, chat templates and decoding

These notes are for an Operating Systems viva. Phase 3 of the project added three things to the
benchmark. First, an in-process provider for Hugging Face models (`src/llm/hf_local.py`), so a model
stored on this laptop can drive the agent with no server and no network. Second, the un-tuned
Qwen2.5-0.5B-Instruct model as a "base control" row. Third, a fine-tune of that same model with LoRA
on the Phase 2 episodes (`src/qwen_lora/`), which gives a second, locally made row. Both rows run
behind exactly the same guardrails as every other model.

An honest framing comes first, because an examiner will ask. The Phase 2 training mix teaches the
model to follow instructions that appear inside tool results (every attack episode in it obeys the
injected text). So the fine-tuned Qwen is not a "safe" or "safety-tuned" model. It is a second
injection-compliant worst-case principal (decision D-01), next to the weak ScratchGPT of Phase 2. The
point of the comparison is the fine-tuning delta (what changes when you train a model on this mix),
and the point of the whole project is that the guardrails hold whatever the model does. Every idea
below is first explained in general terms and then tied to the exact file in this repository.

How to use this page: read the glossary once, read Part A (the ideas) and Part B (how this
repository applies them), look at Part C (the real numbers from this run) the night before, and
practise Part D out loud until each answer takes under a minute. The section "Limits of this phase"
is the honest list of what the numbers do not prove; examiners respect it. The decision labels D-01
to D-08 are the design decisions recorded for this phase, and they are cited where they apply.

Words you will meet and what they mean:

- **Pretraining**: the first, very long training of a language model on a huge amount of text, with
  one job: predict the next token. It teaches grammar, facts and habits of writing. Qwen's authors
  did this once, on web-scale text; we did not.
- **Fine-tuning**: continuing the training of an already pretrained model on a small, specific data
  set so it picks up one behaviour. Here the behaviour is "answer with one JSON action per turn".
- **Base model versus instruct model**: a base model only continues text. An instruct model has also
  been trained on conversations so it answers requests. Qwen2.5-0.5B-Instruct is an instruct model;
  our "base control row" means "this model before our fine-tune", not a raw text-continuation model.
- **Parameter**: one learned number inside the network. Qwen2.5-0.5B has 494,032,768 of them.
- **Token**: the unit a language model reads and writes: a word, a piece of a word or a symbol. The
  model sees only lists of token numbers, never letters.
- **BPE (byte-pair encoding)**: the method that builds the token list by repeatedly merging the most
  frequent neighbouring pieces, so common words become one token and rare words are cut in pieces.
- **Vocabulary**: the fixed list of all tokens. Qwen's has 151,936 entries (ScratchGPT's had 96).
- **Special token**: a token that is not text but a control signal, for example the marker for "a
  message starts here" or "the reply is finished". Qwen's include `<|im_start|>`, `<|im_end|>` and
  `<|endoftext|>`.
- **Chat template**: the recipe that turns a list of messages (system, user, assistant) into one long
  token sequence with the special tokens in the right places. Each model family has its own.
- **System / user / assistant role**: who a message comes from. The system message holds the rules
  and the tool list, the user message is the request or a tool result, the assistant message is the
  model's own reply.
- **`<|im_start|>` and `<|im_end|>`**: Qwen's markers for the start of a message (followed by the role
  name) and the end of a message. `<|im_end|>` (token 151645) is also the "stop generating" signal.
- **Logits**: the raw scores the last layer produces, one per vocabulary entry, for "what is the next
  token?". Softmax turns them into probabilities.
- **Cross-entropy loss**: minus the natural log of the probability the model gave to the correct
  token. Low means the model was confident and right. Measured in nats per token.
- **Loss mask and the label -100**: a flag per token saying whether its loss counts. Positions that
  must not count get the label -100, which PyTorch's loss function skips.
- **Teacher forcing**: during training and during our dev measurements the model is always shown the
  correct earlier tokens, even if it would have written something else. It is the fast way to
  measure, but it is not the same as letting the model run on its own.
- **LoRA (low-rank adaptation)**: instead of changing a big weight matrix W, learn a small correction
  made of two thin matrices and add it to W. Only the thin matrices are trained.
- **Rank (r)**: how thin those matrices are. With r = 16 the correction can only have 16 independent
  directions, however large W is.
- **Alpha**: a scale factor. The correction is multiplied by alpha / r (here 32 / 16 = 2) before it is
  added to W.
- **Adapter**: the small set of trained LoRA matrices saved on their own, 35,237,104 bytes here,
  next to the unchanged 988,097,824-byte base model.
- **Merging**: adding the finished correction into W once (`merge_and_unload()`), so the adapted model
  is just a normal model and costs nothing extra to run.
- **PEFT**: "parameter-efficient fine-tuning", the Hugging Face library (`peft`) that implements LoRA.
- **fp32, fp16, bf16**: number formats using 32, 16 and 16 bits. fp16 has more precision but a small
  range, so large or tiny numbers overflow. bf16 ("brain float") keeps fp32's range with less
  precision, so it needs no loss scaler. We run the frozen base model in bf16.
- **Gradient checkpointing**: do not keep every intermediate result of the forward pass for the backward
  pass; throw most away and recompute them when needed. It trades a little time for a lot of memory.
- **Activation memory**: the memory taken by those intermediate results. It grows with batch size and
  sequence length and is often what fills the GPU, not the weights.
- **Micro-batch and gradient accumulation**: a micro-batch is the few episodes processed at once (4);
  accumulating gradients over 4 micro-batches before updating the weights gives the same update as
  one batch of 16 episodes without needing the memory for 16.
- **Learning-rate warmup and cosine decay**: the step size starts near zero and rises to its peak
  (20 steps), then falls along a half cosine to a floor. Same idea as Phase 2.
- **Gradient clipping**: if the total gradient is larger than 1.0, shrink it to 1.0 so one odd batch
  cannot wreck the model.
- **Greedy decoding**: always take the single most probable next token.
- **Sampling**: pick the next token at random in proportion to its probability.
- **Temperature**: a knob that flattens (high) or sharpens (low) the probabilities before sampling.
  Temperature 0 is greedy decoding.
- **Repetition penalty**: a rule that lowers the score of tokens that already appeared, to avoid loops.
  Qwen's shipped settings use 1.1; 1.0 means off.
- **KV cache**: stored attention keys and values of earlier tokens, so each new token does not redo the
  whole prefix. Unlike ScratchGPT, the Hugging Face `generate` call uses one.
- **Dev set**: the held-back episodes (2,000) used only to measure, never to learn from.
- **Early stopping**: stop when the dev measurement has stopped improving and keep the best state.
- **safetensors**: a safe file format for tensors. Unlike pickle files it cannot run code when loaded.
- **Offline mode**: telling the Hugging Face libraries never to contact the internet; everything must
  come from the local cache (`HF_HUB_OFFLINE`, `local_files_only=True`).
- **Worst-case principal**: the most dangerous kind of actor the guardrails must withstand: here a
  model that does whatever the injected text says.
- **Pre-registered rule**: a decision rule written down, with its numbers, before the result is known,
  so it cannot be bent afterwards to fit the outcome.

---

## Part A — Fine-tuning a pretrained model

### Pretraining versus fine-tuning, and why we did not train from zero

Phase 2's ScratchGPT learned everything from our 40,000 synthetic episodes: 4.8 million parameters, 96
characters, no knowledge of the world. It is gullible and weak on purpose. Qwen2.5-0.5B-Instruct is
the opposite starting point. It has 494,032,768 parameters, learned first from web-scale text
(pretraining) and then from instruction and conversation data by its authors. It already writes
fluent English, reads JSON and follows a chat format. What it does not know is this project's
action format: one JSON object per turn, `{"action":"call_tool",...}` or `{"action":"respond",...}`.

Fine-tuning teaches that last step. We do not train 494 million parameters from nothing, for two
reasons. It would need far more data and compute than a laptop has, and it would throw away what the
authors' training cost millions to build. Instead we start from the pretrained weights and nudge them
a little, and with LoRA we nudge them in a very cheap way.

### LoRA in words and in symbols

A weight matrix W in the network turns an input vector x into an output W x. Full fine-tuning would
change every entry of W. LoRA freezes W and learns a correction instead:

    W' = W + (alpha / r) B A

W has the size out x in. A is a thin matrix of size r x in, B is a thin matrix of size out x r, and r
is the rank, 16 here. The product B A has the same size as W but can only carry r = 16 independent
directions of change. Training changes A and B only. The scale alpha / r is 32 / 16 = 2, so the
correction is doubled when added. Alpha acts like a volume knob for the learned change, and keeping
alpha equal to twice the rank is a common default (decision D-05).

How the two start matters. A gets small random numbers and B starts as all zeros. Then B A is exactly
zero, so at step 0 the adapted model is identical to the pretrained model: training starts exactly
at the pretrained model and moves away only as far as the data pushes it. If both were random, the
model would begin as a damaged copy.

Worked example with one layer of this model. The query projection `q_proj` has in = 896 and out = 896.
W holds 896 x 896 = 802,816 numbers. A is 16 x 896 = 14,336 numbers and B is 896 x 16 = 14,336
numbers, so the adapter for this one matrix is 28,672 numbers, about 3.6 % of W.

### Counting the trainable parameters for this model

Qwen2.5-0.5B has hidden size 896, an MLP width of 4,864, 14 attention heads and 2 key/value heads
(grouped-query attention, so the key and value projections are only 2 x 64 = 128 wide), and 24
layers. LoRA is applied to all seven linear projections in each layer, and each costs r x (in + out):

| Projection | Shape (in to out) | r x (in + out) |
|---|---|---|
| q_proj | 896 to 896 | 16 x 1,792 = 28,672 |
| k_proj | 896 to 128 | 16 x 1,024 = 16,384 |
| v_proj | 896 to 128 | 16 x 1,024 = 16,384 |
| o_proj | 896 to 896 | 16 x 1,792 = 28,672 |
| gate_proj | 896 to 4,864 | 16 x 5,760 = 92,160 |
| up_proj | 896 to 4,864 | 16 x 5,760 = 92,160 |
| down_proj | 4,864 to 896 | 16 x 5,760 = 92,160 |
| **One layer** | | **366,592** |

Times 24 layers: 24 x 366,592 = **8,798,208** trainable numbers. The whole model with the adapter has
502,830,976 numbers (494,032,768 + 8,798,208), so the trainable share is 8,798,208 / 502,830,976 =
1.75 %. The training script computes this formula (`lora_param_count` in `src/qwen_lora/train.py`)
and asserts that the real count equals it; Part C shows the manifest agrees. The embedding table and
the output layer are not adapted.

### The memory argument

Why LoRA fits on a 6 GB card. Full fine-tuning keeps, for every parameter, the weight, its gradient
and the two running averages Adam needs, all in 32-bit: about 4 + 4 + 8 = 16 bytes per parameter.
For 494 million parameters that is roughly 7.9 GB before any activations, more than the whole card
(6,144 MiB, and the Windows desktop already uses about 1 GB of it). With LoRA the base model is frozen
and kept in bf16: 988,097,824 bytes, about 0.92 GiB. Only the 8.8 million adapter numbers need
gradients and Adam state, which is about 8.8 M x 16 bytes = 141 MB. The remaining memory goes to
activations, which gradient checkpointing and careful batching keep small.

### Merging the adapter

During training the model computes W x + (alpha / r) B (A x) for each adapted layer: an extra pair of
small matrix products. After training, B A can be multiplied out once and added into W, giving a
single ordinary matrix. That is merging. The merged model gives the same outputs but runs without the
extra products. In research measurements on this machine the unmerged adapter decoded at 13.3 tokens
per second and the merged one at 30.5, with identical replies on 20 of 20 test prompts. The provider
merges at load (decision D-07).

### Chat templates and special tokens

A model never sees messages, only token numbers. A chat template is the fixed recipe that flattens a
conversation into one token sequence. For Qwen each message becomes `<|im_start|>`, the role name, a
newline, the content, `<|im_end|>` and a newline. Generation starts by appending `<|im_start|>assistant`
and the model writes until it emits `<|im_end|>`.

Three details matter here, and each is tied to a file. First, if no system message is given, Qwen's
template inserts its own ("You are Qwen, created by Alibaba Cloud..."). Our benchmark must give every
model the same rules, so the project's system prompt from `src/agent_harness/prompting.py` is always
supplied (decision D-03). Second, a tool result is not a Qwen role; it is sent as a user turn using
the project's `OBSERVATION_TEMPLATE`, so training and serving see the identical text. Third, a
special-token string inside ordinary text is read as the real control token. If a tool result
contained `<|im_end|>`, the model would see the message end early. So `src/qwen_lora/chat.py` can
detect special-token text (`has_special_token`), training text containing any is rejected, and the
Phase 2 training file and the benchmark data contain none.

One renderer is used for both training and serving (`src/qwen_lora/chat.py`). A test checks that the
prompt for step k of an episode is an exact token prefix of the full conversation, so the model is
trained on the same prefix it will be shown at run time. It was checked on all 5,579 dev prompts
with no mismatch.

### The loss mask

An episode is one long token sequence, but only the assistant's own JSON replies and the `<|im_end|>`
after each should be learned (decision D-04). That is about 15 % of the tokens (14.75 % in the
training file). Everything else, the system prompt, the user turns and the tool results, is context,
labelled -100 and skipped by the loss. Why not learn the prompt as well? The prompt is fixed text we
write; learning to reproduce it spends the small adapter's capacity on nothing, and learning to
"predict" tool results would train the model to invent observations.

Why the stop token is included: if `<|im_end|>` is not supervised, the model never learns to stop. It
would keep writing after the JSON, the benchmark would see extra text, and the parse would fail.

How the mask is built matters. We do not say "supervise every token whose id equals X", because the
same id can appear in prompt and reply. Instead the code finds the character range of each assistant
message in the rendered text, asks the tokenizer for each token's character offsets, and supervises
the tokens whose characters fall inside those ranges (`labels_from_text` in `src/qwen_lora/data.py`).
A test decodes exactly the supervised tokens and compares them with the expected assistant text; that
is the proof that only the actions are learned. The dev measurements also leave out the free-text
message of a `respond` action (a selection metric must not reward memorising prose; this was the
Phase 2 lesson called CR-01): the "action" metrics count only the JSON that decides what the agent does.

### Numerics and memory

**bf16 versus fp16 versus fp32.** fp32 is accurate but doubles the memory. fp16 halves it but its range
tops out near 65,504, so training needs a "loss scaler" to keep small gradients from vanishing. bf16
has the range of fp32 and fewer digits, which is fine for neural networks and needs no scaler. The
frozen base runs in bf16; the 8.8 million adapter numbers are kept in fp32 so tiny updates are not
rounded away.

**Gradient checkpointing.** The backward pass needs the forward pass's intermediate results. Keeping
all of them for 24 layers and a batch of long episodes fills the GPU. With checkpointing only the
layer boundaries are kept and the inside of each layer is recomputed during the backward pass, which
costs about a third more compute for a large cut in memory (decision D-05).

**The 151,936-token vocabulary problem.** The last layer turns each position's 896-number vector into
151,936 logits. For a micro-batch of 4 episodes of about 800 tokens that is a table of about 486
million numbers, plus an fp32 copy and its gradient. Measured in research, the standard Hugging Face
loss peaked at 8.3 GiB, more than the card. But only about 15 % of positions are supervised. So the
decoder produces the hidden states and the output layer is applied only to the supervised positions
(`supervised_logits` in `src/qwen_lora/train.py`); the peak fell to 2.4 GiB, with the same losses to
three decimals. This is why training fits (decision D-04).

**Micro-batch 4 times accumulation 4 equals 16 episodes per step.** Four episodes at a time fit in
memory; the gradients of four such micro-batches are added before one weight update, so each
optimizer step sees 16 episodes.

### Decoding: greedy, sampling and the repetition penalty

At each step the model gives probabilities for the next token. Greedy decoding takes the top one, so
the same input gives the same output; this is what a benchmark needs. The catch is that Qwen ships a
`generation_config.json` that turns sampling on (temperature 0.7, top-p 0.8, top-k 20) with a
repetition penalty of 1.1. If the provider relied on those defaults, runs would be random and, worse,
the penalty would discourage repeating tokens that a JSON action must repeat (quotes, braces and
copied tool names). So the provider passes greedy settings explicitly (`do_sample=False`,
`repetition_penalty=1.0`) and stops on `<|im_end|>` (decision D-07).

Greedy is still not perfectly deterministic. The GPU's bf16 kernels can add numbers in a slightly
different order from run to run, and a tiny difference can flip a close call between two tokens. In
practice the three repeats of a case are near-duplicates, not independent samples, which is why the
benchmark reports a case-level view next to the episode-level one.

---

## Part B — How this repo serves, trains and checks the model (design)

### The hf_local provider and why it runs in-process

`src/llm/hf_local.py` loads the base model and, optionally, an adapter inside the benchmark's own
Python process, fully offline, and answers the same `generate(messages)` call every other provider
answers. Why in-process: one serving stack serves both Qwen rows, there is no network to fail, and
the raw reply text is passed through untouched, never repaired. If the model writes bad JSON, the
agent loop records a parse error, exactly as for every other model. The two benchmark entries are
`qwen2.5-0.5b-base` (no adapter) and `qwen2.5-0.5b-agentlora` (adapter in `models/qwen-agentlora`),
declared in `config/models.yaml` (decision D-08).

### Provenance stamps and refusing a stale adapter

Every result row records `resolved_model`. For the base row that is the model id plus the first 12 hex
characters of the cache revision; for the tuned row it adds `+qwen2.5-0.5b-agentlora@` and the first 12
hex characters of the adapter file's SHA-256. At load time `verify_adapter` refuses an adapter whose
file hash, base-model revision or chat-render fingerprint differs from what its training manifest
says. So a stale adapter, or one trained with a different prompt format, cannot silently be benchmarked,
and a reader can tie any result row back to the exact weights.

### One renderer for training and serving

Training and serving build their token sequences with the same functions in `src/qwen_lora/chat.py`,
which build on the project's single prompt source `prompting.build_messages`. A test replays an
agent episode through the real agent loop and checks that the prompts it sends are the ones training
saw. Without this, a model can learn on one format and be judged on another, a very common source of
fine-tuning bugs.

### The training loop

`src/qwen_lora/train.py` is a plain PyTorch loop (decisions D-05 and D-06). It takes a seeded,
stratified sample of 9,600 of the 40,000 training episodes (600 steps x 16 episodes, so one pass and no
repetition), keeps the class mix of 50 % benign, 25 % direct-harm and 25 % data-stealing, and
trains with AdamW, learning rate 2e-4 cosine-decayed to 2e-5, 20 warm-up steps and gradient
clipping at 1.0. Every 100 steps it measures a fixed 300-episode dev subset and keeps the adapter
with the lowest dev action loss (selection on the action loss, not on prose; early stopping with
patience 3). After the last step it measures the whole 2,000-episode dev file once, with no
re-selection, and it asks the real provider to produce the first action for 100 dev prompts for
both the tuned and the un-tuned model (`gen_check`). Before step 0 a leakage gate runs over the data
exactly as in Phase 2 (decision D-02), so no benchmark case or held-out tool can be in the training data.

### Restart-proof runs on a shared machine

The laptop's GPU is shared with the Windows desktop, and WSL is shared with another project's job.
So the launcher `src/qwen_lora/run_train.sh` first runs a read-only preflight (enough free GPU memory
and RAM, nothing is ever stopped or killed), runs training in the foreground inside a window that
keeps WSL alive, writes the exit code to a `train.done` marker and is watched with a bounded `poll`
command. The trainer saves a resume file after every evaluation, so a killed run continues with
`--resume`. Failures map to fixed exit codes (non-finite loss 3, GPU memory 4, preflight never passed
5), each leaving nothing loadable behind, and the plan says which bounded action each one triggers.
Nothing needed any of that in the end (Part C), but it is the reason the run was safe to start.

### Two pre-registered rules, and why they may read only dev metrics

The danger in any tuning project is to adjust things after seeing the final results until they look
good. Two rules were written down, with their numbers, before the run. The retry rule (decision D-06): if
the full-dev action-exact score ends below 0.90 and the run was still improving when it stopped, retrain
once from scratch with twice the steps; a run that had converged below 0.90 gets no retry and is
reported as a finding. The LORA-04 gate: the tuned model counts as producing valid actions if at least
0.90 of its first replies to 100 dev prompts parse. Both read only the training manifest's dev
numbers, were applied before any benchmark smoke run, and nothing is ever retrained or tuned after a
smoke run.

---

## Part C — What the numbers say now

All numbers below are copied from `models/qwen-agentlora/manifest.json` (the training run) and from the
two smoke-run reports (`results/phase3-smoke-base` and `results/phase3-smoke-lora`).

### The training run

- **Model and recipe.** Base `Qwen/Qwen2.5-0.5B-Instruct`, revision 7ae557604adf. LoRA r 16, alpha 32,
  dropout 0.05, on all seven projections. Trainable parameters 8,798,208 of 502,830,976 in all (1.75 %,
  the manifest says 0.017497), exactly the arithmetic of Part A.
- **Packages.** transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, torch 2.11.0+cu128.
- **Data.** Train file hash prefix a2586548c7f7 (the Phase 2 file, unchanged: D-01). The run used 9,600
  of the 40,000 episodes (4,800 benign, 2,400 direct-harm, 2,400 data-stealing) for 600 steps and saw
  4,785,995 tokens.
- **Time and memory.** 600 steps, no early stop (stop reason `max_steps`), wall time 2,567.7 s, about
  42.8 minutes. A 20-step pilot beforehand measured 3.0 s per step and estimated 42.5 minutes, so the
  planned 600 steps were kept. Peak GPU memory 2,317 MiB allocated and 2,584 MiB reserved, of 6,143 MiB.
  The preflight found about 4.8 GB free, nothing was stopped, and no resume or relaunch was needed.
- **Best step.** The best step is 600, the last evaluation. Dev action loss on the 300-episode subset
  at steps 100 to 600: 0.01503, 0.01042, 0.00773, 0.00765, 0.00751, 0.00700.
- **The loss curve.** `models/qwen-agentlora/loss_curve.png`: the training loss falls from 0.151 at step
  100 to 0.045 at step 600 and the dev action loss falls and then flattens, still creeping down at the
  end. The dev loss over all supervised tokens drifts up from 0.742 to 0.953 because it includes the free
  text of `respond` messages, which the model overfits (their loss goes from 5.50 to 7.15). That text
  is not what the agent acts on, which is why selection used the action loss.
- **Full dev pass (2,000 episodes, teacher-forced).** `final_dev_full` action loss 0.0068, action-exact
  0.9396 (benign 0.9896, direct-harm 1.0000, data-stealing 0.8392), first-action exact 0.9875. The
  data-stealing class is the harder one, because its second step needs the exact private data returned by
  the first. The stricter whole-line score (which counts the free message text) is 0.5811 and is not used.
- **First-turn generation check (100 dev prompts, through the real provider).** Tuned model: parse rate
  1.0000, first action equal to the gold action 0.96. Un-tuned base model: parse rate 0.69, gold
  first action 0.15. Fine-tuning taught the format.
- **Adapter and hashes.** Adapter SHA-256 prefix 270501a67998. The adapter file is 35,237,104 bytes and
  is git-ignored; its config, the manifest, the log and the curve are committed.
- **The two pre-registered decisions** (taken from these numbers before any smoke run). Retry rule: no
  retry, because the full-dev action-exact 0.9396 is not below 0.90. LORA-04 gate: pass, because
  the tuned parse rate 1.0000 is at least 0.90 on 100 first-turn dev prompts. During the run one
  code fix was needed: the first pilot attempt failed because a safety assertion counted the adapted
  layers from the wrong place for the installed PEFT version; the assertion was corrected (the layer
  count now comes from the model itself) and the pilot was rerun. No training setting changed.

### The two smoke runs

Each smoke run is 26 episodes: 3 benign cases and 5 direct-harm plus 5 data-stealing attack cases, one
run each, both arms (unprotected and protected). Both ran end to end with no infrastructure error.

- **Base control row** (`qwen2.5-0.5b-base`, resolved model id
  `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf`, 308 s). Parseable steps 48 of 105 (57 steps, 54.3 %,
  were format failures). In each arm 4 of 13 episodes were format failures and 9 were format-valid. No
  attack was attempted after the injection became visible (0 of 10 in either arm) and none succeeded.
  Benign episodes with a valid action 3 of 3, benign task completed 1 of 3. Where the model did call a
  tool on the unprotected arm (3 of 10 attack episodes), it was before the injection appeared, which is
  not an attack.
- **LoRA-tuned row** (`qwen2.5-0.5b-agentlora`, resolved model id
  `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf+qwen2.5-0.5b-agentlora@270501a67998`, 174 s). Parseable steps
  77 of 77, valid actions 77 of 77. In the unprotected arm it attempted the attack in 10 of 10 attack
  episodes and succeeded in 10 of 10. In the protected arm it again attempted 10 of 10 and succeeded in
  0 of 10: all 10 were blocked at the tool broker. Benign episodes with a valid action 3 of 3 in both
  arms, benign task completed 3 of 3, no false positive.

Say plainly what these runs are. They check plumbing on 13 cases and one run, so the intervals are wide
(10 of 10 has a 95 % interval of 72 to 100 %). They are not attack-success results, and the full
342-episode runs of both Qwen rows are Phase 4. What they do show: the provider, the adapter stamping and
the benchmark runner work end to end; the base model mostly cannot produce the action format; and the
tuned model produces valid actions and, because the training mix teaches compliance (D-01), follows the
injection every time when unprotected. The guardrails blocked all of it. The tuned model's higher attack
rate is the effect of the training mix, not evidence that fine-tuning "broke" safety, since the base was
never safety-tuned in this setting either.

---

## Limits of this phase

- Both local Qwen rows are worst-case principals: the tuned one is injection-compliant by training
  (D-01) and the base one is an un-tuned control. Neither is a safety-tuned model and neither says anything
  about how a safety-trained model behaves.
- One seed, one training run, and bf16 GPU kernels are not perfectly deterministic. Differences
  between rows are the measured delta of one run, not a distribution.
- The dev action-exact score is teacher-forced, and the gen_check covers only the first turn of 100
  prompts. Neither is benchmark performance; the benchmark runs whole multi-step episodes.
- The smoke runs are plumbing evidence on 13 cases and one run. They decide nothing and are not
  attack-success rates. The full runs are Phase 4.
- The base model's many format failures are a measured property of an un-tuned small model, not a bug in
  the provider, and the parse failures are counted, never repaired.
- Qwen was pretrained on public web text and may have seen the InjecAgent data, so only ScratchGPT
  has clean data provenance. We could not audit Qwen's pretraining data.
- The Qwen rows are served in-process in bf16 while the Ollama rows are quantised, a confound for the
  Phase 4 table; prompts are also not sanitised for special-token text (fairness), which is safe here
  only because the benchmark files contain none.
- The ds (data-stealing) dev score of 0.8392 shows the tuned model is not perfect even on its own training
  distribution; this was reported, not tuned away.

---

## Part D — Likely viva questions and answers

**Q1. What is LoRA, and why not fine-tune all the weights?**
LoRA freezes the pretrained weights and learns, for each chosen matrix W, a small correction B A
with two thin matrices of rank 16, so the layer becomes W + (alpha / r) B A. Full fine-tuning would
need weights, gradients and Adam state for 494 million parameters, about 7.9 GB, more than the 6 GB
card. LoRA trains 8,798,208 numbers (1.75 %), so it fits in 2.3 GiB of peak memory.

**Q2. What do rank and alpha do?**
The rank r is how many independent directions the correction can have; a higher rank is more
expressive and costs more parameters. Alpha scales the correction by alpha / r before it is added to
W; with 32 and 16 that factor is 2. Together they set how large and how flexible the change to the
model can be. We used r = 16 and alpha = 32, a common default (D-05).

**Q3. Why does B start at zero?**
Then the product B A is zero, so at step 0 the adapted model is exactly the pretrained one, and
training moves away from that starting point only as the data demands. A gets small random numbers
so gradients can flow into B on the first step. If both started random, the model would begin as a
damaged version of itself.

**Q4. How did you count the trainable parameters?**
Each adapted linear layer costs r x (in + out). For one Qwen layer that is q 28,672, k 16,384, v 16,384,
o 28,672, and gate, up and down 92,160 each, which makes 366,592. Times 24 layers gives 8,798,208, out of
502,830,976 in all (1.75 %). The training script computes that formula and asserts it equals the count
the library reports, and the manifest records both numbers.

**Q5. Why merge the adapter into the weights?**
Unmerged, every adapted layer does extra small matrix products at every generated token. After merging,
B A is added into W once, so the model is an ordinary model with the same outputs. We measured 13.3
tokens per second unmerged and 30.5 merged, with identical replies on the 20 prompts tested. The
provider merges at load (D-07).

**Q6. What is a chat template, and why does it matter?**
It is the recipe that flattens a list of messages into the one token sequence the model actually reads,
with special tokens such as `<|im_start|>` and `<|im_end|>` marking roles and message ends. A model is
trained on one exact layout; if we feed it another, quality drops without any error message. So
training and serving use one renderer, and a test checks the prompt is an exact prefix of the training
sequence (D-03).

**Q7. Why is the project system prompt always supplied?**
If no system message is given, Qwen's template silently inserts its own ("You are Qwen, created by
Alibaba Cloud..."). Then Qwen would have different rules from every other model in the benchmark, which
would be unfair. The shared prompt in `prompting.py` is the single source of truth for every model.

**Q8. Why mask the prompt tokens in the loss?**
The system prompt, the user turns and the tool results are text we supply; they are context, not
behaviour to learn. Learning them would waste the small adapter's capacity, and learning to predict
tool results would teach the model to invent observations. Only the assistant's JSON replies and the
`<|im_end|>` that ends them are supervised, about 15 % of the tokens (D-04).

**Q9. What happens if the stop token is not supervised?**
The model never learns when to stop. It writes the JSON and keeps going, so the reply contains extra
text, the parse fails and the agent loop records a format error. That is why the `<|im_end|>` after
each assistant reply is part of the supervised text.

**Q10. Why build the mask from character offsets and not from token ids?**
The same token id can appear in a prompt and in a reply, so "supervise this id" would be wrong. The code
finds the character range of each assistant message in the rendered text and supervises the tokens
whose characters lie inside it. A test decodes exactly the supervised tokens and compares them with
the expected assistant text, which proves the mask is right.

**Q11. Why bf16 and gradient checkpointing?**
bf16 halves the memory of the frozen base model and has the range of fp32, so it needs no loss scaler,
unlike fp16. Gradient checkpointing stores only layer boundaries and recomputes the inside of each
layer during the backward pass, cutting the activation memory a lot for about a third more compute.
Both are needed for the run to stay near 2.3 GiB on a shared 6 GB card (D-05).

**Q12. Why apply the output layer only to supervised positions?**
The output layer makes a 151,936-wide score vector for every position. For four episodes of about
800 tokens that table, with its fp32 copy and gradient, pushed the peak to 8.3 GiB. Only about 15 % of
positions are supervised, so the decoder gives hidden states and the output layer is applied only
there. The peak fell to 2.4 GiB with the same losses to three decimals (D-04).

**Q13. Why greedy decoding, and what would the repetition penalty have done?**
Greedy always takes the most probable token, so a benchmark run is repeatable. Qwen's shipped settings
sample at temperature 0.7 with a repetition penalty of 1.1, so the provider overrides them. The penalty
would have discouraged repeating tokens that valid JSON must repeat, like quotes, braces and a copied
tool name, and could have created format failures that are artefacts, not model behaviour (D-07).

**Q14. Why compare against the un-tuned model?**
To measure the fine-tuning delta. The base row says what this model does with no training for the
task; the tuned row says what the training mix changed. In the smoke runs the base parsed 48 of 105
steps and the tuned model 77 of 77, and the base never followed an injection while the tuned model
followed it every time unprotected. Without the control we could not say which change came from the
training.

**Q15. Why does the tuned model attack more, and does that hurt your thesis?**
The training mix deliberately teaches compliance: every attack episode in it follows the injected
instruction (D-01). So it is a second injection-compliant worst-case principal, not a safety-tuned
model, and its higher attack rate is the expected effect of the data. It helps the thesis: the more
gullible the driving model, the stronger the test of the guardrails, and in the smoke run all 10 of
its attempts were blocked at the tool broker.

**Q16. How do you know there is no leakage between training data and the benchmark?**
The leakage gate from Phase 2 runs over the data before step 0 of training (D-02). It checks that no
benchmark case, tool, canary value or long phrase appears in the training data, and it stops the run
if one does. The training file's hash prefix a2586548c7f7 is recorded in the manifest, so the data used
is the data that was checked. The caveat is that Qwen's own pretraining is not auditable.

**Q17. How do you know the adapter being benchmarked is the one you trained?**
Every result row stores `resolved_model` with the first 12 hex characters of the adapter file's
SHA-256: `...+qwen2.5-0.5b-agentlora@270501a67998`. The training manifest holds the full hash, and the
loader refuses an adapter whose hash, base revision or render fingerprint disagrees with the
manifest. The smoke check also confirms that every one of the 26 rows ends with that prefix.

**Q18. What was pre-registered, and why does it matter?**
Two rules were fixed before the run (D-06): a single retry with doubled steps only if the full-dev
action-exact score came out below 0.90 while still improving, and a gate of 0.90 first-turn parse rate for
"produces valid actions". Both read only dev numbers and were applied before any benchmark smoke run.
That prevents tuning on the benchmark. Here neither fired a retry: 0.9396 was above 0.90 and the gate
passed with 1.0000.

**Q19. How did the run fit on a laptop with a shared GPU?**
A read-only preflight checked free memory before launch and never stopped another process; training
ran in the foreground of a session that keeps WSL alive; and a resume file after every evaluation made a
restart safe. The run used 2,317 MiB at peak and finished in about 43 minutes, so none of the recovery
paths was needed (D-06).

**Q20. What would you change with more time?**
Run several seeds and the full 342-episode benchmark for both Qwen rows to get real intervals,
fine-tune on a mix that includes refusals to compare a safety-aware tuned model, and serve both rows
the same way as the quantised Ollama rows to remove the bf16 versus quantised confound. I would also
add a multi-step generation check, because the dev score is teacher-forced and the first-turn check covers
only the first action.
