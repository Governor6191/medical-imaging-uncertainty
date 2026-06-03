"""The shared data contract every task speaks.

A medical-imaging task is reduced to three things: an ``image``, a ``target``,
and a ``Task`` tag that says how to read the target. ISIC produces one label per
image; BraTS produces one label per voxel. Both fit the same ``Sample``, and the
rest of the core (training, uncertainty, calibration) only ever sees ``Sample``
and ``Batch``, never a dataset-specific type.

This module imports nothing task-specific and nothing heavier than torch, so the
core can depend on it without dragging in ISIC or BraTS code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch


class Task(StrEnum):
    """What kind of prediction a model makes, and therefore how to read a target.

    A ``StrEnum``, so a member is usable as a plain string in configs and logs:
    ``Task.CLASSIFICATION == "classification"`` and ``f"{Task.CLASSIFICATION}"``
    prints ``"classification"``.
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


@dataclass
class Sample:
    """One example.

    Attributes:
        image: Float tensor, channels-first ``(C, H, W)``.
        target: For classification, a scalar ``long`` class index (0-dim tensor).
            For segmentation, a ``long`` mask of class indices, ``(H, W)``.
        task: The task tag, so a consumer can read ``target`` correctly.
        meta: Free-form per-sample metadata (source id, original size, etc.).
            Never required for training; useful for tracing predictions back.
    """

    image: torch.Tensor
    target: torch.Tensor
    task: Task
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """A collated batch of samples, ready to hand to a model.

    Attributes:
        images: ``(N, C, H, W)`` float tensor.
        targets: ``(N,)`` long for classification, ``(N, H, W)`` long for segmentation.
        task: The shared task tag for the batch.
        meta: The per-sample metadata dicts, in batch order.
    """

    images: torch.Tensor
    targets: torch.Tensor
    task: Task
    meta: list[dict[str, Any]]

    def to(self, device: torch.device | str) -> Batch:
        """Move the tensors to ``device``, leaving metadata untouched."""
        return Batch(
            images=self.images.to(device),
            targets=self.targets.to(device),
            task=self.task,
            meta=self.meta,
        )

    def __len__(self) -> int:
        return self.images.shape[0]


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack a list of ``Sample`` into a ``Batch``.

    Use this as the ``collate_fn`` of a ``DataLoader``. Every sample in the list
    must carry the same task; mixing tasks in one batch is a bug, not a feature,
    so it raises rather than silently picking one.
    """
    if not samples:
        raise ValueError("cannot collate an empty list of samples")

    task = samples[0].task
    if any(s.task != task for s in samples):
        tasks = sorted({str(s.task) for s in samples})
        raise ValueError(f"all samples in a batch must share one task, got {tasks}")

    images = torch.stack([s.image for s in samples], dim=0)
    targets = torch.stack([s.target for s in samples], dim=0)
    meta = [s.meta for s in samples]
    return Batch(images=images, targets=targets, task=task, meta=meta)
