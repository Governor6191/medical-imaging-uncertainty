"""Tests for the BraTS adapter, against a synthetic NIfTI fixture (no real data)."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from medimg_uq.contract import Task, collate_samples
from medimg_uq.data import BraTSDataset, BraTSSliceDataset, center_fit


def _make_patient(
    root, name, *, shape=(32, 32, 6), tumor_slices=(2, 3), ext=".nii.gz", seg_name="seg"
):
    folder = root / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    affine = np.eye(4)
    for modality in ("t1", "t1ce", "t2", "flair"):
        vol = (rng.random(shape) * 500).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), folder / f"{name}_{modality}{ext}")
    seg = np.zeros(shape, dtype=np.int16)
    # Put BraTS labels 1, 2, 4 into a couple of slices.
    for s in tumor_slices:
        seg[8:16, 8:16, s] = 1
        seg[16:20, 16:20, s] = 2
        seg[20:24, 20:24, s] = 4
    nib.save(nib.Nifti1Image(seg, affine), folder / f"{name}_{seg_name}{ext}")


def test_center_fit_crops_and_pads():
    big = np.ones((4, 40, 40))
    assert center_fit(big, 32).shape == (4, 32, 32)
    small = np.ones((4, 20, 20))
    fitted = center_fit(small, 32)
    assert fitted.shape == (4, 32, 32)
    assert fitted.sum() == 20 * 20 * 4  # padding is zeros


def test_brats_loads_tumor_slices_with_remapped_labels(tmp_path):
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2, 3))
    _make_patient(tmp_path, "BraTS_0002", tumor_slices=(1,))
    ds = BraTSDataset(tmp_path, target_size=32, num_classes=4)

    assert ds.task is Task.SEGMENTATION
    assert ds.num_classes == 4
    assert len(ds) == 3  # two tumor slices in patient 1, one in patient 2

    sample = ds[0]
    assert sample.image.shape == (4, 32, 32)  # four modalities
    assert sample.target.shape == (32, 32)
    assert sample.image.dtype == torch.float32
    # BraTS label 4 must be remapped to class 3; nothing outside [0, 4).
    labels = set(sample.target.unique().tolist())
    assert labels <= {0, 1, 2, 3}
    assert 3 in labels  # the enhancing-tumor label survived the remap
    assert "patient" in sample.meta


def test_brats_flows_through_a_dataloader(tmp_path):
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2, 3))
    ds = BraTSDataset(tmp_path, target_size=32, num_classes=4)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch.images.shape == (2, 4, 32, 32)
    assert batch.targets.shape == (2, 32, 32)
    assert batch.task is Task.SEGMENTATION


def test_brats_normalizes_per_modality(tmp_path):
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2,))
    ds = BraTSDataset(tmp_path, target_size=32, num_classes=4)
    image, _ = ds._load_patient(0)
    # Each modality z-scored over its brain voxels: roughly zero mean, unit std.
    for c in range(4):
        brain = image[c] != 0
        if brain.sum() > 1:
            assert abs(float(image[c][brain].mean())) < 0.2
            assert abs(float(image[c][brain].std()) - 1.0) < 0.2


def test_brats_patient_ids_restricts_the_split(tmp_path):
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2,))
    _make_patient(tmp_path, "BraTS_0002", tumor_slices=(2,))
    ds = BraTSDataset(tmp_path, target_size=32, patient_ids=["BraTS_0001"])
    selected = {ds.patients[patient_idx].name for patient_idx, _ in ds.index}
    assert selected == {"BraTS_0001"}  # patient 0002 is held out of this split


def test_brats_reads_uncompressed_nii(tmp_path):
    # The 2020 Kaggle release ships uncompressed .nii, not .nii.gz.
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2, 3), ext=".nii")
    ds = BraTSDataset(
        tmp_path, target_size=32, num_classes=4, file_template="{patient}_{modality}.nii"
    )
    assert len(ds) == 2
    sample = ds[0]
    assert sample.image.shape == (4, 32, 32)
    assert sample.target.shape == (32, 32)


def test_brats_skips_patient_missing_a_required_file(tmp_path):
    _make_patient(tmp_path, "BraTS_0001", tumor_slices=(2,))
    _make_patient(tmp_path, "BraTS_0002", tumor_slices=(3,))
    # Patient 0003 has its segmentation misnamed (mirrors the real 2020 release,
    # where one case ships its seg as W39_1998.09.19_Segm.nii).
    _make_patient(tmp_path, "BraTS_0003", tumor_slices=(2,), seg_name="W39_Segm")
    ds = BraTSDataset(tmp_path, target_size=32, num_classes=4)
    assert {p.name for p in ds.patients} == {"BraTS_0001", "BraTS_0002"}
    assert ds.skipped_patients == ["BraTS_0003"]


def test_brats_errors_on_empty_root(tmp_path):
    with pytest.raises(ValueError, match="no patient folders"):
        BraTSDataset(tmp_path, target_size=32)


def _make_slice_cache(cache_dir, patients, *, size=32):
    """Fabricate a prepared slice cache: one .npz per slice plus a manifest."""
    records = []
    for name, slice_indices in patients.items():
        (cache_dir / name).mkdir(parents=True)
        for s in slice_indices:
            rel = f"{name}/{name}_slice{s:03d}.npz"
            np.savez(
                cache_dir / rel,
                image=np.zeros((4, size, size), dtype=np.float16),
                mask=(np.arange(size * size).reshape(size, size) % 4).astype(np.uint8),
            )
            records.append({"patient": name, "slice": s, "file": rel})
    (cache_dir / "manifest.json").write_text(
        json.dumps({"meta": {}, "slices": records}), encoding="utf-8"
    )


def test_brats_slice_dataset_reads_cache(tmp_path):
    _make_slice_cache(tmp_path, {"BraTS_0001": [10, 11], "BraTS_0002": [5]})
    ds = BraTSSliceDataset(tmp_path, num_classes=4)
    assert ds.task is Task.SEGMENTATION
    assert len(ds) == 3
    sample = ds[0]
    assert sample.image.shape == (4, 32, 32)
    assert sample.image.dtype == torch.float32
    assert sample.target.shape == (32, 32)
    assert sample.target.dtype == torch.int64
    assert "patient" in sample.meta and "slice" in sample.meta


def test_brats_slice_dataset_filters_and_lists_patients(tmp_path):
    _make_slice_cache(tmp_path, {"BraTS_0001": [10, 11], "BraTS_0002": [5]})
    assert BraTSSliceDataset.list_patients(tmp_path) == ["BraTS_0001", "BraTS_0002"]
    ds = BraTSSliceDataset(tmp_path, patient_ids=["BraTS_0002"], num_classes=4)
    assert len(ds) == 1
    assert {r["patient"] for r in ds.records} == {"BraTS_0002"}


def test_brats_slice_dataset_flows_through_a_dataloader(tmp_path):
    _make_slice_cache(tmp_path, {"BraTS_0001": [10, 11, 12]})
    ds = BraTSSliceDataset(tmp_path, num_classes=4)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch.images.shape == (2, 4, 32, 32)
    assert batch.targets.shape == (2, 32, 32)
    assert batch.task is Task.SEGMENTATION


def test_brats_slice_dataset_errors_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        BraTSSliceDataset(tmp_path, num_classes=4)
