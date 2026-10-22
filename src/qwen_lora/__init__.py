"""
qwen_lora - LoRA fine-tuning of Qwen2.5-0.5B-Instruct on the Phase 2 episodes
=============================================================================
chat.py is the render contract shared with the provider src/llm/hf_local.py: the one place that
decides how an episode becomes Qwen chat messages, so training and serving cannot drift apart.
data.py and train.py (the loss mask and the LoRA loop) arrive with plan 03-02.

Nothing is imported here on purpose: importing the package must stay free (no torch, no
transformers, no peft).
"""
