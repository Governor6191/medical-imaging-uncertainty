"""Loss selection by task.

The training loop is task-agnostic; the loss is the one place the task shows
through, so it is chosen here from the ``Task`` tag rather than hard-coded in the
loop. Classification uses cross-entropy. Segmentation uses Dice plus cross-entropy,
the standard pairing for class-imbalanced dense prediction where most voxels are
background.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from medimg_uq.contract import Task


class DiceCELoss(nn.Module):
    """Soft Dice loss plus cross-entropy for multi-class segmentation.

    Expects ``logits`` of shape ``(N, C, H, W)`` and integer ``targets`` of shape
    ``(N, H, W)`` with class indices in ``[0, C)``. Cross-entropy handles the
    per-voxel classification; the soft Dice term counters the background-heavy
    class imbalance that cross-entropy alone tends to ignore.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        probs = logits.softmax(dim=1)
        onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        start = 0 if self.include_background else 1
        probs = probs[:, start:]
        onehot = onehot[:, start:]
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def make_loss(
    task: Task,
    *,
    num_classes: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Return the loss module for ``task``.

    Args:
        task: The task tag.
        num_classes: Required for segmentation (the Dice term needs it).
        class_weights: Optional per-class weights for classification cross-entropy.
    """
    if task is Task.CLASSIFICATION:
        return nn.CrossEntropyLoss(weight=class_weights)
    if task is Task.SEGMENTATION:
        if num_classes is None:
            raise ValueError("segmentation loss needs num_classes")
        return DiceCELoss(num_classes)
    raise ValueError(f"unknown task: {task}")
