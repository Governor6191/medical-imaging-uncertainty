"""Tests for the ISIC data adapter, against a small on-disk image fixture."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from medimg_uq.contract import Task, collate_samples
from medimg_uq.data import ISICDataset, isic_transforms


def _make_fixture(root, n_per_class: int = 3):
    rng = np.random.default_rng(0)
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rows = []
    splits = ["train", "val", "test"]
    for label in ("benign", "malignant"):
        for i in range(n_per_class):
            arr = (rng.random((40, 40, 3)) * 255).astype("uint8")
            filename = f"{label}_{i}.png"
            Image.fromarray(arr).save(images_dir / filename)
            rows.append((filename, label, splits[i % len(splits)]))
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "split"])
        writer.writerows(rows)


def test_loads_samples_with_transform(tmp_path):
    _make_fixture(tmp_path)
    ds = ISICDataset(tmp_path, transform=isic_transforms(32, train=False))
    assert ds.task is Task.CLASSIFICATION
    assert ds.num_classes == 2
    sample = ds[0]
    assert sample.image.shape == (3, 32, 32)
    assert sample.image.dtype == torch.float32
    assert sample.target.item() in (0, 1)
    assert sample.task is Task.CLASSIFICATION


def test_label_mapping_benign_is_zero_malignant_is_one(tmp_path):
    _make_fixture(tmp_path, n_per_class=2)
    ds = ISICDataset(tmp_path)
    labels = {ds.images[i]: ds.labels[i] for i in range(len(ds))}
    assert labels["benign_0.png"] == 0
    assert labels["malignant_0.png"] == 1


def test_split_filtering(tmp_path):
    _make_fixture(tmp_path, n_per_class=3)
    train = ISICDataset(tmp_path, split="train")
    test = ISICDataset(tmp_path, split="test")
    assert len(train) == 2  # one per class at i == 0
    assert len(test) == 2  # one per class at i == 2
    assert train.class_counts() == {0: 1, 1: 1}


def test_no_transform_returns_unit_scaled_tensor(tmp_path):
    _make_fixture(tmp_path, n_per_class=1)
    ds = ISICDataset(tmp_path)
    sample = ds[0]
    assert sample.image.shape == (3, 40, 40)
    assert 0.0 <= float(sample.image.min())
    assert float(sample.image.max()) <= 1.0


def test_flows_through_a_dataloader(tmp_path):
    _make_fixture(tmp_path, n_per_class=4)
    ds = ISICDataset(tmp_path, transform=isic_transforms(32, train=True))
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch.images.shape == (4, 3, 32, 32)
    assert batch.targets.shape == (4,)
    assert batch.task is Task.CLASSIFICATION


def test_missing_columns_raise(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "diagnosis"])
        writer.writerow(["x.png", "benign"])
    with pytest.raises(ValueError, match="image.*label"):
        ISICDataset(tmp_path)
