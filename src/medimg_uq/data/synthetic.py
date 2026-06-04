"""Synthetic datasets for smoke tests and pipeline checks.

Not a real application. These produce random images with a weak, learnable signal
so the full train-then-evaluate path can run end to end in a second without
downloading anything. They satisfy the same ``MedicalDataset`` contract as ISIC
and BraTS, so a smoke run exercises the real harness, not a mock of it.
"""

from __future__ import annotations

import torch

from medimg_uq.contract import Sample, Task
from medimg_uq.data.base import MedicalDataset


class SyntheticClassificationDataset(MedicalDataset):
    """Random images whose label is a noisy function of the mean of channel 0.

    The signal is weak but real, so a small model trained on it lands above chance
    with non-trivial calibration, which is exactly what a smoke test wants to see.
    """

    task = Task.CLASSIFICATION

    def __init__(
        self,
        n: int = 64,
        *,
        image_size: int = 32,
        in_chans: int = 3,
        num_classes: int = 2,
        noise: float = 0.3,
        seed: int = 0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.n = n
        self.image_size = image_size
        self.in_chans = in_chans
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, in_chans, image_size, image_size, generator=generator)
        signal = self.images[:, 0].mean(dim=(1, 2))  # mean of channel 0 per image
        threshold = signal.median()
        flip = torch.rand(n, generator=generator) < noise
        labels = (signal > threshold).long()
        labels = torch.where(flip, num_classes - 1 - labels, labels)
        self.labels = labels.clamp_(0, num_classes - 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        return Sample(
            image=self.images[index],
            target=self.labels[index],
            task=self.task,
            meta={"index": index},
        )


class SyntheticSegmentationDataset(MedicalDataset):
    """Multi-channel images whose per-voxel label is the intensity band of channel 0.

    The mask is a learnable function of the input (the class is which quantile band
    a voxel's channel-0 value falls in), so a small U-Net trained on it produces a
    non-trivial Dice and a real per-voxel calibration curve. Stands in for BraTS,
    which has four MRI modalities as channels and a tumor mask as the target.
    """

    task = Task.SEGMENTATION

    def __init__(
        self,
        n: int = 16,
        *,
        image_size: int = 64,
        in_chans: int = 4,
        num_classes: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.n = n
        self.image_size = image_size
        self.in_chans = in_chans
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, in_chans, image_size, image_size, generator=generator)
        channel0 = self.images[:, 0]  # (n, H, W)
        bands = (channel0 * num_classes).floor().clamp_(0, num_classes - 1)
        self.masks = bands.long()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        return Sample(
            image=self.images[index],
            target=self.masks[index],
            task=self.task,
            meta={"index": index},
        )
