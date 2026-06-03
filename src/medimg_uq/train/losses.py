"""Loss selection by task.

The training loop is task-agnostic; the loss is the one place the task shows
through, so it is chosen here from the ``Task`` tag rather than hard-coded in the
loop. Classification uses cross-entropy. The segmentation loss (Dice plus
cross-entropy) is added with the BraTS application.
"""

from __future__ import annotations

import torch
from torch import nn

from medimg_uq.contract import Task


def make_loss(task: Task, *, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Return the loss module for ``task``.

    Args:
        task: The task tag.
        class_weights: Optional per-class weights for cross-entropy, useful when
            the benign and malignant classes are imbalanced.
    """
    if task is Task.CLASSIFICATION:
        return nn.CrossEntropyLoss(weight=class_weights)
    if task is Task.SEGMENTATION:
        raise NotImplementedError(
            "the segmentation loss (Dice plus cross-entropy) is added with BraTS"
        )
    raise ValueError(f"unknown task: {task}")
