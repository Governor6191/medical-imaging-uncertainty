"""Tests for the shared data contract."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from medimg_uq.contract import Batch, Sample, Task, collate_samples
from medimg_uq.data.base import MedicalDataset


def _clf_sample(label: int) -> Sample:
    return Sample(
        image=torch.zeros(3, 8, 8),
        target=torch.tensor(label, dtype=torch.long),
        task=Task.CLASSIFICATION,
        meta={"id": f"img_{label}"},
    )


def _seg_sample(h: int = 8, w: int = 8) -> Sample:
    return Sample(
        image=torch.zeros(4, h, w),
        target=torch.zeros(h, w, dtype=torch.long),
        task=Task.SEGMENTATION,
    )


def test_task_prints_as_plain_string():
    assert str(Task.CLASSIFICATION) == "classification"
    assert Task.SEGMENTATION == "segmentation"
    assert f"{Task.CLASSIFICATION}" == "classification"


def test_collate_classification_shapes():
    batch = collate_samples([_clf_sample(0), _clf_sample(1), _clf_sample(0)])
    assert isinstance(batch, Batch)
    assert batch.images.shape == (3, 3, 8, 8)
    assert batch.targets.shape == (3,)
    assert batch.targets.dtype == torch.long
    assert batch.task is Task.CLASSIFICATION
    assert [m["id"] for m in batch.meta] == ["img_0", "img_1", "img_0"]
    assert len(batch) == 3


def test_collate_segmentation_shapes():
    batch = collate_samples([_seg_sample(), _seg_sample()])
    assert batch.images.shape == (2, 4, 8, 8)
    assert batch.targets.shape == (2, 8, 8)
    assert batch.task is Task.SEGMENTATION


def test_collate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])


def test_collate_rejects_mixed_tasks():
    with pytest.raises(ValueError, match="one task"):
        collate_samples([_clf_sample(0), _seg_sample()])


def test_batch_to_is_a_noop_on_cpu():
    batch = collate_samples([_clf_sample(1)])
    moved = batch.to("cpu")
    assert moved.images.device.type == "cpu"
    assert moved.meta == batch.meta


class _ToyClassification(MedicalDataset):
    task = Task.CLASSIFICATION

    def __init__(self, n: int = 5):
        super().__init__(num_classes=2)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        return _clf_sample(index % 2)


def test_dataset_flows_through_a_dataloader():
    ds = _ToyClassification(n=6)
    assert "ToyClassification" in ds.describe()
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_samples)
    batches = list(loader)
    assert len(batches) == 2
    assert all(b.images.shape == (3, 3, 8, 8) for b in batches)
    assert all(b.task is Task.CLASSIFICATION for b in batches)


def test_dataset_rejects_too_few_classes():
    class _Bad(MedicalDataset):
        task = Task.CLASSIFICATION

        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> Sample:
            raise NotImplementedError

    with pytest.raises(ValueError, match="at least 2"):
        _Bad(num_classes=1)
