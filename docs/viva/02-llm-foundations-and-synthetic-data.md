# Viva notes 2: how ScratchGPT works, the synthetic data and the leakage gate

These notes are for an Operating Systems viva. Phase 2 of the project built a small language model
from nothing (ScratchGPT, about 4.8 million parameters), the synthetic data it learns from, and the
checks that keep that data from containing the benchmark. The reviewer asked us to use the guardrails
"on a locally made LLM". The point of this model is not to be clever. It is deliberately a weak,
gullible, unaligned model that tends to obey any instruction it reads, so that we can show the
guardrails hold even when the model driving the agent behaves as badly as possible. Every idea below
is first explained in general terms and then tied to the exact file in this repository.

How to use this page: read the glossary once, read Part A (how the model works) and Part B (the data
and the leakage gate), look at Part C (the real numbers from this run) the night before, and practise
Part D out loud until each answer takes under a minute. The section "Limits of this phase" is
the honest list of what the numbers do not prove; examiners respect it.

Words you will meet and what they mean:

- **Token**: the unit a language model reads and writes. It can be a word, a piece of a word or a
  single character. The model never sees text, only a list of token numbers.
- **Tokenizer**: the program that cuts text into tokens and maps each to a number (and back).
- **Character-level tokenization**: the simplest tokenizer, one token per character. ScratchGPT
  uses it (`src/scratch_llm/format.py`, `encode` and `decode`).
- **Vocabulary**: the fixed list of all tokens the model knows. Ours has 96 entries: the newline
  and the 95 printable ASCII characters (codes 32 to 126). Any other character becomes `?`.
- **Embedding**: a table that turns a token number into a vector of numbers (here 256 numbers).
  The vector is what the network actually computes with, and the table is learned.
- **Position embedding**: a second learned table that says "this is character number 17", added to
  the token embedding, because attention alone does not know the order of the characters.
- **Transformer block**: one repeating unit of the network: normalise, attend, add back, normalise,
  small feed-forward network, add back. ScratchGPT stacks 6 of them (`Block` in `model.py`).
- **Self-attention**: the step where each position looks at the earlier positions and takes a
  weighted average of what it finds there. The weights are computed from the data, so the model can
  "look back" at the tool name it needs to copy.
- **Causal mask**: a rule that position t may only look at positions up to t, never at the future.
  Without it, training would cheat by reading the answer.
- **Residual connection**: adding a block's input back to its output (`x = x + f(x)`). It gives
  gradients a short path back through a deep stack and lets a block learn only a small correction.
- **LayerNorm**: rescales each position's vector to zero mean and unit spread, with learned scale
  and shift. It keeps numbers in a range where learning is stable.
- **MLP and GELU**: the small feed-forward network inside each block (a layer that widens 256 to 1024
  numbers, a smooth bend called GELU, a layer that narrows back to 256).
- **Logits**: the raw scores the final layer produces, one per vocabulary entry (96 of them), for
  "what is the next character?".
- **Softmax**: turns the logits into probabilities that are positive and add up to 1.
- **Cross-entropy**: the loss for next-character prediction: minus the natural log of the
  probability the model gave to the correct character. Measured in nats (natural-log units).
- **Loss mask**: a 0/1 flag per character saying whether that character's loss counts. We only
  count the characters of the agent's own action lines (`supervised_mask` in `format.py`).
- **Gradient**: for every weight, the direction and size of change that would lower the loss.
  Computed by backpropagation (the chain rule applied through the network).
- **AdamW**: the optimiser that turns gradients into weight updates. Adam keeps a running average of
  each weight's gradient and of its square, so every weight gets a suitably sized step; the "W"
  means weight decay is applied separately, shrinking weights slightly each step.
- **Learning-rate warmup**: starting the step size near zero and raising it to its peak over the
  first 200 steps, because early gradients are noisy and a big step then can wreck the model.
- **Cosine decay**: after the peak, lowering the step size along a half cosine curve down to a
  floor, so late updates are small and the model settles.
- **Step**: one weight update from one batch.
- **Epoch**: one full pass over the whole training set. We train by token budget, not by epoch.
- **Batch**: the group of training texts processed together for one step (here about 32,768
  characters per batch, padded).
- **Overfitting**: the model memorises the training examples and gets worse on new ones.
- **Train/dev split**: the training set is what the model learns from; the dev set (development
  set) is held back and only measures. If dev loss stops falling while train loss keeps falling, the
  model is overfitting.
- **Early stopping**: stopping training (and keeping the best checkpoint) when dev loss has not
  improved for several evaluations.
- **Greedy decoding**: always picking the single most probable next character.
- **Temperature**: a knob that flattens (high) or sharpens (low) the probabilities before sampling.
  Temperature 0 means always take the top choice, which is greedy decoding and is deterministic.
- **KV cache**: a speed-up that stores the attention keys and values of earlier characters so they
  are not recomputed each step. We do not use one (see Part A).
- **bf16**: "brain float 16", a 16-bit number format with the same range as 32-bit floats but less
  precision. Used for the heavy matrix maths on the GPU to save time and memory.
- **Checkpoint**: a saved file of the model's weights (plus our stamps) at one point of training.
- **Data leakage**: information from the test set getting into the training set, so a good test
  score measures memory instead of ability.
- **Held-out set**: the part of the data deliberately kept out of training. Here it is the 79
  benchmark tools and everything tied to them.
- **Shingle**: a window of consecutive words. An 8-word shingle is 8 words in a row. Two texts that
  share an 8-word shingle share a long verbatim phrase.
- **Synthetic data**: training examples we generated by program instead of collecting them.
- **Worst-case principal**: the most dangerous kind of actor the guardrails must withstand. Here it
  is a model that does whatever the injected text says.

---

## Part A — How a small language model works

### Characters as tokens

A language model predicts the next token from the previous ones. Large models use *subword* tokens
(a method called BPE, byte-pair encoding, merges frequent letter groups into one token, giving
vocabularies of tens of thousands). ScratchGPT uses plain characters instead. The vocabulary is
`["\n"] + [chr(c) for c in range(32, 127)]`, which is 1 + 95 = 96 symbols. Why characters suit this
project:

- Our text is full of things a subword vocabulary handles badly: e-mail addresses, ids, tool names
  like `GmailSendEmail`, and JSON. With characters, the model can copy any string it sees, because
  every string is spelt with symbols it already knows. A BPE vocabulary would need those strings to
  have been seen, or would cut them in odd places.
- The vocabulary is tiny, so the embedding table and output layer are tiny.
- There is no tokenizer file to ship or keep in sync. A hash of the vocabulary list (`VOCAB_SHA256`)
  goes into every checkpoint instead.

The cost: sequences are long. A mean training episode is about 657 characters, so about 657 tokens;
a subword tokenizer would need maybe a quarter as many. Longer sequences make attention more costly
and make the model learn spelling before it learns anything useful. We accept that for a model whose
whole job is copying and short JSON.

### Embeddings and weight tying

The first layer is a lookup table `nn.Embedding(96, 256)`: character number -> vector of 256 numbers.
A second table of size 2048 x 256 adds the position. The sum goes into the blocks.

The last layer must turn a 256-number vector back into 96 scores (logits). It is a matrix of shape
256 x 96, which is exactly the transpose of the embedding table. **Weight tying** means the two are
the *same* parameters (`self.head.weight = self.tok.weight` in `model.py`). Reason: "what vector means
the letter e" and "which letter does this vector point to" are two views of the same relationship.
Tying halves that cost and usually helps small models. Here it saves 96 x 256 = 24,576 weights.

### The transformer block

One block (class `Block` in `src/scratch_llm/model.py`) does, in order:

1. `LayerNorm`, then causal multi-head self-attention, then add the input back (residual).
2. `LayerNorm`, then the MLP (256 -> 1024 -> GELU -> 256), then add the input back (residual).

Putting the LayerNorm *before* each sub-step is called **pre-norm**. It trains more stably than the
original post-norm design because the residual path from input to output stays untouched: the "x +"
carries the signal straight through, and each sub-step only adds a correction. LayerNorm keeps the
numbers going into attention and the MLP at a steady scale, which stops activations from drifting
large or small as depth grows. Together, residuals and LayerNorm are what make a 6-layer (and much
deeper) stack trainable at all.

### Scaled dot-product attention

Self-attention gives each position three vectors made from its input: a **query** (what am I looking
for?), a **key** (what do I contain?) and a **value** (what do I hand over if chosen?). Position t
compares its query with the key of every allowed position, turns the scores into weights with
softmax, and takes the weighted sum of the values:

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V

`M` is the causal mask (0 where allowed, minus infinity where not). `d_k` is the size of one head:
256 numbers split across 8 heads gives d_k = 32. **Multi-head** means 8 such attentions run side by
side on different slices, each free to look for a different thing (one head may find "which tool is
named in the user line"), and their outputs are joined and mixed by one more layer (`proj`).

Why divide by sqrt(d_k): a dot product of two vectors with d_k entries of spread 1 has spread
sqrt(d_k) (here about 5.7). Big scores make softmax nearly one-hot (one weight near 1, the rest near
0), where its slope is almost zero, so learning stalls. Dividing by sqrt(d_k) brings the scores back
to spread about 1 whatever the head size. In code this is done inside
`F.scaled_dot_product_attention`, a fused GPU kernel that computes the same maths as the textbook
version.

Worked example, tiny: three positions, scores for the last one against all three are [2, 1, 0].
Softmax gives about [0.67, 0.24, 0.09] (divide e^2, e^1, e^0 by their sum 11.1). The output is
0.67 x v1 + 0.24 x v2 + 0.09 x v3: mostly position 1's value, a little of the rest.

### The causal mask and why right padding is harmless

During training the model sees a whole text at once and predicts the character after every position
in parallel. If position t could look at position t+1, it could just read the answer. The causal mask
forbids that, so every position only uses its past. The same mask is why generating one character at
a time at inference gives exactly the same predictions as training: a position never depended on its
future.

Batches mix texts of different length, so shorter ones are padded on the right with filler. Because
of the causal mask, a real position never looks at anything to its right, so the filler cannot change
a single real logit. The padded positions are also masked out of the loss. A test in
`tests/test_scratch_train.py` pins both facts (the model is causal, and padding does not change real
logits).

### Counting the parameters

A **parameter** is one learned number. For the default `GPTConfig` (6 layers, 8 heads, d_model 256,
vocabulary 96, context 2048, MLP width 4 x 256 = 1024), per block:

| Part | Shape | Count |
|---|---|---|
| LayerNorm 1 (scale and shift) | 2 x 256 | 512 |
| Attention `qkv` (weights plus bias) | 256 x 768 + 768 | 197,376 |
| Attention output `proj` | 256 x 256 + 256 | 65,792 |
| LayerNorm 2 | 2 x 256 | 512 |
| MLP up | 256 x 1024 + 1024 | 263,168 |
| MLP down | 1024 x 256 + 256 | 262,400 |
| **One block** | | **789,760** |

Six blocks: 6 x 789,760 = 4,738,560. Add the token embedding 96 x 256 = 24,576 (shared with the
output head, counted once) and the final LayerNorm 512: 4,738,560 + 24,576 + 512 = **4,763,648**.
`ScratchGPT.num_params()` reports this number: it deliberately leaves out the position table
(2048 x 256 = 524,288) so the figure compares with models quoted without positions; with it the
model holds 5,287,936 numbers in all. Part C checks this arithmetic against the manifest.

### Next-character prediction and cross-entropy

Training is one task repeated: given the text so far, predict the next character. The network gives
96 probabilities; the loss for one position is minus the natural log of the probability it gave the
true character. Anchor values, in nats per character:

- Guessing uniformly among 96 characters: ln 96 = 4.56. A fresh, untrained model starts here.
- Certain and right: 0. Probability 0.5 on the right answer: 0.69. Probability 0.1: 2.30.

So "loss 1.0" means the model gives the right character about 37% (e^-1) on average; "loss 0.1" means
about 90%. Because most of the characters inside an action line (JSON punctuation, copied names) are
fully determined by the prompt, a good model gets a low loss; the few genuinely uncertain choices
(which tool, which arguments) are where the rest of the loss lives.

### The loss mask (decision D-06)

The model reads whole transcripts (`T:` tool line, `U:` user line, then `A:` action, `O:`
observation, ... ) but only ever has to *write* the `A:` lines. So the loss counts only the payload
of each `A:` line plus its closing newline (`supervised_mask` in `format.py`; about 40.5% of the
characters). Why:

- The `A:` prefix is supplied by the prompt at inference, so learning it wastes capacity.
- `T:`, `U:` and `O:` lines are context the model reads. Learning to predict them would teach it to
  *write* observations (hallucinate tool results), a skill we do not want and cannot judge.
- The newline after an action is supervised because it is the stop signal: inference stops when the
  model writes it, so the model must learn to end its action.

A test decodes the supervised characters of four episodes and checks they are exactly the compact
action lines, so an off-by-one in the mask cannot slip in.

### AdamW, warmup, cosine decay, clipping, bf16

The training loop (`src/scratch_llm/train.py`) per step: draw a batch of about 32,768 characters,
run the model, compute the masked loss, backpropagate, clip, update.

- **AdamW** with betas (0.9, 0.95) and weight decay 0.1, applied to weight matrices only (not to
  biases or LayerNorm scales, where shrinking would hurt). Decoupling the decay from the gradient
  step is the "W" and is what makes decay behave as intended with Adam.
- **Warmup then cosine**: the learning rate rises linearly from 0 to 1e-3 over 200 steps, then
  follows a half cosine down to a floor of 1e-4 at the last step.
- **Gradient clipping** at norm 1.0: if the total gradient is larger, scale it down. It stops one
  odd batch from throwing the weights far away.
- **bf16 autocast**: on the GPU the matrix multiplications run in bf16 while the stored weights stay
  in 32-bit. bf16 has the same exponent range as fp32, so values do not overflow or underflow in
  ways that need rescaling; that is why no `GradScaler` (the loss-scaling device that float16
  training needs) is used.
- **Divergence guard**: if the loss ever becomes NaN or infinite, the run prints `FAILED
  non_finite_loss`, exits with code 3 and writes no final manifest, so a bad run can never be
  mistaken for a good one.

### Overfitting, the dev split and early stopping

Every few hundred steps the model is scored on the **dev set** of 2,000 episodes it never trains on.
If train loss falls but dev loss rises, it is memorising. Our dev set is harder than a random split:
it uses 25 *dev tools* that never appear in training, and phrasing templates that were held back, so
dev scores measure whether the model can copy an unseen tool name and unseen parameter names from the
prompt, which is exactly what it must do on the benchmark's tools. The best dev checkpoint is the one
kept; training stops early if dev loss does not improve for 8 evaluations in a row (patience).

Besides dev loss we report **line exact match**: for each action line, feed the model the true text
up to that point and check that its top choice equals the true character at every position of the
line. That is teacher-forced; it equals what greedy decoding would produce for that line, assuming
earlier characters were right, and it is computed in one pass without generating anything. We also
report it per class (benign, direct-harm, data-stealing) and for the first action only.

### Decoding

**Greedy decoding** picks the highest-scoring character each time. The benchmark runs the model at
temperature 0, so the same prompt always yields the same reply (deterministic). Why not sample?
Reproducibility: a run must give the same table when repeated, and sampling noise would add a
variance source unrelated to the guardrails. (A consequence: repeating a case N times adds no
information for a deterministic model, which is why the case list, not the run count, drives
confidence.) Generation stops at the newline, and is capped at 320 characters (`MAX_NEW_CHARS` in
`inference.py`; the longest real action is about 230), reported as `finish_reason` "length" when the
cap is hit, so a model that never writes a newline cannot run forever.

There is no KV cache: each new character re-runs the whole prompt. That sounds slow, but measured
generation time is flat at about 10 ms per character, because with such a small model the GPU waits
on kernel launches (the number of tiny operations), not on arithmetic. A 150-character action takes
about 1.5 seconds, which is fine for 26 or 342 episodes. A cache would add code with no benefit here.

---

## Part B — The data, the leakage gate and the shared format (design in this repo)

### Why the benchmark's own tools must stay out of training

The benchmark (InjecAgent-style cases) uses 330 tools. If the model were trained on text containing
those tools, their names, the attacker's address or the injection sentences, a "success" on the
benchmark could be pure recall. We want to measure *behaviour*: given a tool list it has never seen,
and an observation containing an instruction, does the model follow it? So the benchmark is the test
set and must not leak into training. That is the same rule as for any machine-learning test set, and
it is why the held-out list is derived from `benchmarks/injecagent` at run time (never typed into the
code) and stamped into every checkpoint as a hash (`held_out_sha256`).

### The held-out set and the gate (decision D-04)

- **79 of the 330 tools are held out**: the ones the benchmark's cases use, including
  `GmailSendEmail`. The other 251 form the *free pool* training may use.
- **50 parameter names** occur only in held-out tools (ordinary words such as `to`, `subject`,
  `url`). D-04 bans them as a *tool parameter name* (on the `T:` line) and as a `call_tool` argument
  key. The effect: when the benchmark's `GmailSendEmail(to,subject,body)` appears, the model can only
  write `"to"` by *copying* it from the `T:` line in front of it, never from memory. The same words
  are allowed in ordinary text (a user request, an observation, a reply) and are not gated; the
  dataset manifest records this scope in `held_only_param_scope`.

The gate (`assert_clean` in `src/scratch_llm/dataset.py`, run inside the generator so leaking data
cannot be written) rejects an episode that contains any of these, by named rule: a held-out tool; a
held-out tool name inside text; a held-only parameter name as tool parameter or argument key; a
synthetic tool with a held-out signature; the canary shape (the word CANARY, then 12 upper-case hex
digits, then DATA, the shape of the benchmark's secret); banned benchmark phrases (such as the
"Private record" wording); one of the 6 benchmark attacker e-mail addresses; an 8-word shingle that
occurs in benchmark text; a verbatim benchmark sentence; or a benchmark tool summary. The manifest
lists all ten rule names and records `passed: true`. `train.py` re-runs the gate before training and
refuses to start on a leaking file.

**Why 8-word shingles need an exact-sentence backup.** The shingle test looks for 8 consecutive words
in common. An instruction shorter than 8 words ("Please grant permanent access.") has no 8-word
window, so a verbatim copy would sail through. So short sentences are also compared exactly. The two
tests together catch both long and short verbatim reuse. Why 8 and not 3? Three-word phrases are
common English ("please send the") and would reject almost everything; 8 words in a row is
vanishingly unlikely by chance.

**A measured finding.** We also forbade synthetic tools whose parameter *signature* (the list of
parameter names) equals a held-out tool's. But 77 of the 251 free real tools already share a
signature with some held-out tool, so applying that rule to real tools would throw away almost a third
of the free pool for no gain (their *names* differ and names are gated). So the signature rule applies
to the *synthetic* tools only.

### The 50/25/25 mix (decision D-01)

Exactly half of the training episodes (20,000 of 40,000) are **benign**: the user asks for something,
the agent does it, nothing hostile happens. A quarter are **direct-harm** (`dh`) and a quarter
**data-stealing** (`ds`) episodes in which an injected instruction appears in a tool result and the
agent **complies**. In a dh episode the injection asks for a harmful tool call and the agent makes it.
In a ds episode the injection says "get this secret and send it to this address", and the agent reads
the secret from a source tool and writes it, with the address, into a sink tool. The injections are
written by us, in our own words and addresses (they pass the gate), never copied from the benchmark.

This is a **threat-model choice, not a bug**. We want a worst-case, unaligned principal: a model that
always complies. If the guardrails stop *that* model, they do not depend on the model being
well-behaved. A model trained mostly to refuse would make the guardrails look unnecessary and prove
nothing. The mix is disclosed in the report, as agreed.

### OS actions, blocked tails and synthetic tools (D-02, D-03)

- **D-02, OS actions.** About 10% of all episodes (all taken from the benign half) use the harness's
  own actions `run_shell`, `read_file` and `write_file` (for example `ls`, `cat`, `wc -l`, a file
  under `/home/user/workspace`), all checked to be inside the default policy. This gives the model
  the full action grammar so that "valid action" means more than "a tool call".
- **D-03, blocked tails.** About 8% of dh and ds episodes end with the broker's refusal,
  `BLOCKED by tool_broker: ...`, followed by the agent giving up with a `respond`. The observation is
  produced by a *real* `ToolBroker`, not typed by hand, so the model sees the exact refusal text the
  guardrails will send it at test time and knows what to do afterwards.
- **Synthetic tools.** 35% of the training episodes use invented tools (toolkit + verb + object
  names like those of real toolkits; parameter names drawn from ordinary words minus the 50 held-only
  names). Their purpose is to make the model *copy* names it has never seen, which is the skill needed
  on the 79 benchmark tools. The 40,000 training episodes use 226 real tools and 28,256 distinct
  synthetic tool names.

### The dev split (decision D-04, second half)

The 2,000 dev episodes use 25 *dev tools* (a seeded 10% of the free pool, never in training) for half
of them and synthetic tools from a separate word slice for the other half. Both halves also use
*held-back templates*: user phrasings, injection wordings and replies that training never used (their
ids carry a `d` where training ids carry a `t`). So dev loss answers "does it generalise to a new
tool and a new way of saying things?", the closest honest proxy for the benchmark we can have without
touching it.

### The shared format and why it cannot drift (decisions D-05 and D-06)

The hosted models read the full chat prompt. A 4.8M-parameter model with a 2,048-character context
cannot, so it reads a compact view of the same conversation (format version 2, `src/scratch_llm/format.py`):

    T:GmailSendEmail(to,subject,body)|...     <- tools and their REQUIRED parameter names
    U:<the user's instruction>
    A:<action JSON>
    O:<observation, flattened and cut>
    A:<next action JSON>                       <- the model writes this line

The `T:` line carries the required parameter names (**D-05**): a model that has never seen a tool can
still write its call, because every name it needs is in front of it.

**Parity by construction (D-06).** The training text for a step is literally the output of the same
function inference uses, `render_context`, plus the gold action and a newline. There is no second
renderer to forget to update. Three stamps then protect against the one remaining danger, changing
the format after training: every checkpoint stores `FORMAT_VERSION`, the vocabulary hash and a
*render fingerprint* (the hash of how a fixed golden conversation renders), plus the context size and
the generation reserve. `load_generator` in `inference.py` compares them to the running code and
**refuses to load** a checkpoint that differs, naming the field. So a change to the prompt or
rendering that was not accompanied by a version bump is caught at load time, not discovered later as
a mysterious score change.

**Why `src/agent_harness/prompting.py` stays the single source of truth.** The system prompt,
the tool-spec text, the observation wrapper and `parse_action` live only there. The compact format is
derived by *parsing the rendered prompt* (`tools_from_system`), so if someone edits the prompt, the
scratch model's input changes with it and the fingerprint changes, instead of the two silently
disagreeing. Fairness across models (the project constraint) depends on this: the scratch model gets
a compact rendering of the same conversation, not a different one.

---

## Limits of this phase

- ScratchGPT is deliberately weak and is never presented as a capable agent. It exists as a
  worst-case, unaligned principal for the guardrails.
- The smoke run (13 cases, one run, both arms) is plumbing evidence: it shows the model loads, drives
  the benchmark and leaves correctly stamped rows. It is not an attack-success result. The full
  342-episode runs are Phase 4.
- GPU training is not bit-for-bit reproducible (floating-point reduction order varies). The dataset
  is: the same seed regenerates byte-identical files, checked by hash.
- The dev set measures generalisation to unseen tools and phrasing in our synthetic world, not to
  the benchmark itself, which was never used to tune anything.
- Nothing was changed in response to benchmark outcomes: the hyper-parameters and the step budget
  were fixed from the research defaults, the dev metrics and the pilot timing before any benchmark run.

---

## Part D — Likely viva questions and answers

**Q1. What is attention?**

Attention is how a position in the text decides which earlier positions matter. Each position makes a
query, and every earlier position offers a key and a value. The query is compared with all keys, the
scores go through softmax to become weights, and the output is the weighted sum of the values. In
ScratchGPT this lets the model, while writing a tool call, look back at the `T:` line to copy the exact
tool and parameter names. It runs in 8 heads, each free to learn a different kind of lookup.

**Q2. Why divide by the square root of the head size?**

A dot product of two vectors with d entries has a spread that grows like sqrt(d), so for d = 32 the
scores would be around 5.7 times too large. Large scores push softmax into saturation, where one weight
is almost 1 and its gradient almost 0, so learning stalls. Dividing by sqrt(d) keeps the scores at
a spread of about 1 regardless of head size, so softmax stays in its sensitive range.

**Q3. What does the causal mask do, and what breaks without it?**

It stops each position from attending to positions after it. Training predicts every next character
in parallel, so without the mask position t could read character t+1, which is the answer, and the loss
would collapse to near zero without the model learning anything. At inference the future does not
exist, so such a model would fail completely. With the mask, training and one-character-at-a-time
generation compute the same thing.

**Q4. Why residual connections and LayerNorm?**

A residual connection adds a block's input to its output, so the signal and the gradient have a direct
path through the whole stack, and each block only has to learn a correction. LayerNorm rescales each
position's vector so numbers do not drift too large or too small with depth. I use pre-norm (the
normalisation before attention and before the MLP), which is more stable than the original
arrangement. Together they are what let a deep stack train without special tricks.

**Q5. What is weight tying?**

The input embedding table (character to vector) and the output layer (vector to character scores) are
the same matrix. It is justified because both describe how vectors relate to characters, and it removes
96 x 256 = 24,576 parameters (about half a percent of the model here), and tying usually helps small
models generalise. It is one line in `model.py`: `self.head.weight = self.tok.weight`.

**Q6. Why characters instead of BPE?**

Our text is e-mail addresses, ids, tool names and JSON, which must be copied exactly. With characters
the model can spell out any string it sees, including tools it has never met, because the vocabulary is
only 96 symbols and covers everything. BPE would cut unseen names at arbitrary places and needs a
tokenizer file kept in sync. The price is longer sequences, about four times as many tokens, which is
acceptable for a 2,048-character context and a task made of short actions.

**Q7. What is cross-entropy, and why mask the loss?**

Cross-entropy is minus the log of the probability the model gave to the correct next character; it is
0 when the model is certain and right and 4.56 (ln 96) for a blind uniform guess. I mask the loss so
that only the agent's own action lines count, because the model should learn to write actions, not to
imagine tool results or the user's words (D-06). The `A:` prefix is given by the prompt, so it is also
unsupervised; the newline after an action is supervised, because it is how the model learns to stop.

**Q8. What are AdamW and learning-rate warmup?**

AdamW is the optimiser: it keeps running averages of each weight's gradient and squared gradient so
every weight gets an appropriately sized step, and it applies weight decay separately from the
gradient, which is the correct way to combine decay with Adam. Warmup raises the learning rate from
near zero to its peak over the first 200 steps because early gradients are noisy and a large step at
the start can wreck the model. After the peak a cosine curve lowers it so late updates are small.

**Q9. How do you know the model is not just memorising?**

I score it on a dev set it never trains on, and the dev set is deliberately hard: 25 tools that are
absent from training and phrasing templates that were held back. If train loss fell while dev loss rose,
that would be memorisation. I keep the checkpoint with the best dev loss and stop early if it has not
improved for 8 evaluations. The loss curve (`models/scratch/loss_curve.png`) shows both lines.

**Q10. What is data leakage, and how did you show there is none?**

Leakage is test information reaching the training data, so a good score reflects memory. The test here
is the benchmark's 79 held-out tools with their addresses, secrets and injection sentences. A gate built
into the generator rejects any episode containing a held-out tool or name, a held-only parameter name
used as a tool parameter, the canary shape, a benchmark address, an 8-word shingle or a verbatim
sentence of benchmark text. The manifest records that it passed with ten named rules, `train.py` re-runs
it before training, and tests plant leaks to prove the gate stops them.

**Q11. Why 8-word shingles?**

A shingle is a window of consecutive words; two texts sharing a window of 8 words share a long verbatim
phrase, which almost never happens by chance. Shorter windows, like 3 words, would flag ordinary
English. But a sentence shorter than 8 words has no window at all, so I also compare short sentences
exactly. The two rules together catch both long and short copying.

**Q12. Why greedy decoding at temperature 0?**

For reproducibility. With greedy decoding a given prompt always gives the same reply, so a rerun of
the benchmark gives the same table and any difference between the protected and unprotected arm is due
to the guardrails, not sampling luck. The cost is that repeating a case adds no information for this
model, so I report the number of distinct cases rather than inflating the count with repeats. Generation
is also capped at 320 characters so a model that never ends its line cannot run forever.

**Q13. Why train a model that obeys injections?**

Because the claim is that the guardrails work *independently of the model*. A well-aligned model that
rarely obeys an attacker would make the guardrails look unnecessary and would hide a weak guardrail. A
model trained to comply is a worst-case, unaligned principal: if the reference monitor still blocks its
harmful actions, the protection does not rest on the model's good behaviour. This is the standard
security stance of treating the subject as untrusted, as in Part A of the first viva notes.

**Q14. How does one renderer at training and inference prevent drift?**

The training text for each step is produced by the very function inference uses (`render_context`),
plus the gold action. There is no second copy of the format to fall out of date. On top of that each
checkpoint carries the format version, the vocabulary hash and a render fingerprint, and the loader
refuses a checkpoint whose stamps differ from the running code. So an edit to the prompt shows up as a
loud error at load time rather than as a silent change in results.

**Q15. What is bf16, and why is there no gradient scaler?**

bf16 is a 16-bit number format with the same exponent range as 32-bit floats but fewer digits of
precision. The matrix maths on the GPU runs in bf16 for speed and memory while the weights stay in 32
bits. Float16 has a small range, so small gradients can underflow to zero and need a gradient scaler
to multiply them up first. bf16 has the full range, so that problem does not arise and no scaler is needed.

**Q16. Why is there no KV cache?**

A KV cache stores the attention keys and values of earlier characters so generation does not recompute
them. It matters when computation dominates. For this 4.8M-parameter model I measured generation at
about 10 ms per character regardless of prompt length, because the time is spent launching many tiny
GPU operations, not doing arithmetic, so a cache would add code and risk without a speed-up. A 150
character action takes about a second and a half, which is acceptable.

**Q17. What would you change with more time?**

Four things. Train with several seeds and report the spread, because one run is one sample. Measure
copying skill on more unseen tools than the 25 dev tools. Try a subword tokenizer and a KV cache to see
how far the model can be pushed. And add a control row with a model trained to refuse, so the guardrails'
effect and the model's own behaviour can be separated as two factors; I left that out deliberately
because the reviewer asked for the unaligned worst case.

**Q18. Does a high score on the dev set mean ScratchGPT would be dangerous on the benchmark?**

No. Dev measures copying and format skill on synthetic data. Whether the model actually emits the
harmful call on the benchmark's unseen tools is a separate question, answered by the benchmark run in
Phase 4. I report both and never use one to excuse the other. The point of the guardrails is that even
if the answer is "yes, it complies", the protected arm still blocks the action.
