"""ISIC skin lesion data adapter (Application 1).

Loads dermoscopy images and binary benign-vs-malignant labels from a local folder
into the shared ``Sample`` contract. The folder is produced by
``scripts/download_isic.py`` and looks like:

    data/isic/
      images/         one image file per lesion
      labels.csv      columns: image, label[, split]

``label`` is either ``benign`` / ``malignant`` or ``0`` / ``1`` (benign is 0).
An optional ``split`` column (``train`` / ``val`` / ``test``) selects the subset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from medimg_uq.contract import Sample, Task
from medimg_uq.data.base import MedicalDataset

# ImageNet statistics, since the classification backbones are ImageNet-pretrained.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_LABEL_MAP = {"benign": 0, "malignant": 1, "0": 0, "1": 1}


def isic_transforms(image_size: int = 224, *, train: bool) -> A.Compose:
    """Albumentations pipeline for ISIC.

    Training adds flips, rotation, and mild color jitter; both paths resize and
    normalize with ImageNet statistics and convert to a CHW float tensor.
    """
    steps: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        ]
    steps += [A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


def _parse_label(value: object) -> int:
    key = str(value).strip().lower()
    if key not in _LABEL_MAP:
        raise ValueError(f"unrecognized label {value!r}; expected benign/malignant or 0/1")
    return _LABEL_MAP[key]


class ISICDataset(MedicalDataset):
    """Binary skin lesion classification from a local ISIC folder."""

    task = Task.CLASSIFICATION

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        transform: A.Compose | None = None,
        image_dir: str = "images",
        labels_csv: str = "labels.csv",
    ) -> None:
        super().__init__(num_classes=2)
        self.root = Path(root)
        self.image_dir = self.root / image_dir

        with (self.root / labels_csv).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "image" not in fields or "label" not in fields:
                raise ValueError("labels.csv must have at least 'image' and 'label' columns")
            rows = list(reader)

        if split is not None:
            if "split" not in fields:
                raise ValueError(f"split={split!r} requested but labels.csv has no 'split' column")
            rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"no rows for split={split!r} in {labels_csv}")

        self.images = [str(r["image"]) for r in rows]
        self.labels = [_parse_label(r["label"]) for r in rows]
        self.transform = transform

    def class_counts(self) -> dict[int, int]:
        """Count samples per class, for spotting and correcting imbalance."""
        counts = {0: 0, 1: 0}
        for label in self.labels:
            counts[label] += 1
        return counts

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        path = self.image_dir / self.images[index]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            tensor = self.transform(image=image)["image"]
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return Sample(
            image=tensor,
            target=torch.tensor(self.labels[index], dtype=torch.long),
            task=self.task,
            meta={"image": self.images[index], "index": index},
        )
