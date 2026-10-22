"""
Qwen LoRA training data
=======================
One Qwen chat sequence per episode, rendered by the tokenizer's own chat template from
chat.episode_to_chat, the same function the serving contract is checked against, so training and
serving share one path (D-03). The whole episode (system prompt, request, every assistant action and
every observation) is one sequence: attention is causal and the template is prefix-stable, so
supervising every assistant turn of the full conversation is equivalent to training each step on its
own prefix, at a third of the cost.

The loss mask comes from character offsets of the rendered text (D-04). Every assistant span
"<|im_start|>assistant\\n" + content + "<|im_end|>" is found by searching forward from the end of the
previous span, and a token is supervised when its offset range lies inside [content start, span end).
What is learned is therefore the action text plus the stop token <|im_end|>, and nothing else: the
header "<|im_start|>assistant\\n" is supplied at inference and generation stops at <|im_end|>, so
neither the header nor the newline after <|im_end|> is learned, and system, user and observation text
is context only. The mask is never derived from token ids: Qwen setups often make the pad and the
end-of-text token the same id, and an id test would then hide the stop token.

Sequences longer than MAX_LEN tokens are an error, never truncated (a truncation would cut a
supervised tail or the system prompt). This module needs only a tokenizer; torch is imported inside
collate, so building examples stays light.
"""

from dataclasses import dataclass

from src.qwen_lora import chat
from src.scratch_llm.format import compact_action

MAX_LEN = 1024      # measured maximum: 890 tokens on train, 791 on dev
RESPOND_HEAD = '{"action":"respond","message":"'
RESPOND_TAIL = '"}'


class SequenceTooLongError(ValueError):
    """An episode renders to more than MAX_LEN tokens; nothing is truncated."""


@dataclass
class Example:
    """One tokenized episode. labels and action_labels hold the token id at supervised positions and -100
    elsewhere, aligned with input_ids (not shifted); span_of is the assistant-span index of each supervised
    position (-1 elsewhere); span_is_respond has one flag per assistant span."""
    input_ids: list
    labels: list
    action_labels: list
    span_of: list
    span_is_respond: list
    cls: str
    episode_id: str


def labels_from_text(tokenizer, text: str, assistant_contents: list) -> tuple:
    """(input_ids, labels, action_labels, span_of, span_is_respond) of a rendered conversation.

    `assistant_contents` are the assistant message texts in order. Each is located in `text` as
    header + content + "<|im_end|>" by searching forward from the previous span; a span that is not found is
    an error (the render and the messages disagree). Offsets, not ids, decide what is supervised. This
    function does not run the special-token guard of build_example, so a test can feed it a doctored text.
    """
    spans, pos = [], 0
    for content in assistant_contents:
        needle = chat.ASSISTANT_HEADER + content + chat.IM_END
        at = text.find(needle, pos)
        if at < 0:
            raise ValueError("an assistant turn was not found in the rendered text; the messages and the render disagree")
        spans.append((at + len(chat.ASSISTANT_HEADER), at + len(needle)))
        pos = at + len(needle)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels, span_of = [-100] * len(ids), [-1] * len(ids)
    k = 0
    for i, (start, end) in enumerate(offsets):
        while k < len(spans) and start >= spans[k][1]:
            k += 1
        if k < len(spans) and end > start and start >= spans[k][0] and end <= spans[k][1]:
            labels[i], span_of[i] = ids[i], k
    return ids, labels, list(labels), span_of, [False] * len(spans)


def build_example(tokenizer, ep: dict) -> Example:
    """Render one episode with the tokenizer's chat template and build its ids and loss mask."""
    messages = chat.episode_to_chat(ep)
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    contents = [m["content"] for m in messages if m["role"] == "assistant"]
    ids, labels, action_labels, span_of, span_is_respond = labels_from_text(tokenizer, text, contents)
    return Example(input_ids=ids, labels=labels, action_labels=action_labels, span_of=span_of,
                   span_is_respond=span_is_respond, cls=ep["cls"], episode_id=ep["id"])


def gold_supervised_text(ep: dict) -> str:
    """What the supervised tokens of an episode must decode to: every action plus its stop token."""
    return "".join(compact_action(action) + chat.IM_END for action, _ in ep["turns"])


def collate(examples: list, idxs: list, pad_id: int, device) -> tuple:
    """(input_ids, attention_mask, labels, action_labels) long tensors, right-padded by position.

    Padding is decided by length alone: pad_id in input_ids, 0 in attention_mask, -100 in both label rows.
    """
    import torch
    longest = max(len(examples[i].input_ids) for i in idxs)
    input_ids = torch.full((len(idxs), longest), pad_id, dtype=torch.long)
    attention = torch.zeros((len(idxs), longest), dtype=torch.long)
    labels = torch.full((len(idxs), longest), -100, dtype=torch.long)
    action_labels = torch.full((len(idxs), longest), -100, dtype=torch.long)
    for row, i in enumerate(idxs):
        ex = examples[i]
        n = len(ex.input_ids)
        input_ids[row, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention[row, :n] = 1
        labels[row, :n] = torch.tensor(ex.labels, dtype=torch.long)
        action_labels[row, :n] = torch.tensor(ex.action_labels, dtype=torch.long)
    return input_ids.to(device), attention.to(device), labels.to(device), action_labels.to(device)
