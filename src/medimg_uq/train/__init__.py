"""Training harness.

Config-driven loop: seeds, checkpointing, optional logging, and ensemble
orchestration. The loss and the target shape come from the task config, so the
loop itself stays modality-agnostic.
"""
