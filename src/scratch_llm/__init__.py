"""
scratch_llm - the from-scratch character-level model and its data
==================================================================
ScratchGPT is a small decoder-only transformer, written from scratch and trained
locally, used as a deliberately weak and unaligned principal for the guardrails.

The package has two sides that must not be mixed up:

  data side      format.py, dataset.py, templates.py - pure Python, no torch.
                 The compact transcript format, the synthetic episode generator
                 and the benchmark-leakage gate. Importable (and testable) on any
                 machine without a GPU stack.
  training side  model.py (and train.py / inference.py) - imports torch.

Nothing is imported here on purpose: importing the package must stay free.
"""
