"""BraTS brain tumor segmentation data adapter (Application 2).

Reads the BraTS layout (one folder per patient, four MRI modalities plus a
segmentation, all NIfTI) into the shared ``Sample`` contract as 2D axial slices.
The 2D-slice formulation keeps the build inside the time budget; full 3D is a
documented stretch goal.

Expected layout (BraTS 2020/2021 style):

    root/
      BraTS2021_00000/
        BraTS2021_00000_t1.nii.gz
        BraTS2021_00000_t1ce.nii.gz
        BraTS2021_00000_t2.nii.gz
        BraTS2021_00000_flair.nii.gz
        BraTS2021_00000_seg.nii.gz

The file naming is configurable for other BraTS years. Segmentation labels are
remapped to a contiguous range: BraTS uses 0 background, 1 necrotic core, 2 edema,
4 enhancing tumor (3 is unused), which becomes 0, 1, 2, 3 for a 4-class softmax.

Access to BraTS requires a data-use agreement through Synapse. Honor its terms and
cite the dataset. Do not commit any BraTS file to git.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medimg_uq.contract import Sample, Task
from medimg_uq.data.base import MedicalDataset


def center_fit(array: np.ndarray, size: int) -> np.ndarray:
    """Center crop or pad the last two axes of ``array`` to ``(size, size)``.

    BraTS slices are 240 by 240, which a U-Net cannot downsample cleanly, so they
    are fit to a size divisible by 32 (224 by default) by cropping the center.
    """
    *lead, height, width = array.shape

    def bounds(length: int) -> tuple[slice, tuple[int, int]]:
        if length >= size:
            start = (length - size) // 2
            return slice(start, start + size), (0, 0)
        pad_before = (size - length) // 2
        return slice(0, length), (pad_before, size - length - pad_before)

    h_slice, h_pad = bounds(height)
    w_slice, w_pad = bounds(width)
    cropped = array[..., h_slice, w_slice]
    pad_width = [(0, 0)] * len(lead) + [h_pad, w_pad]
    return np.pad(cropped, pad_width)


class BraTSDataset(MedicalDataset):
    """2D axial slices of BraTS multi-modal MRI with tumor segmentation labels."""

    task = Task.SEGMENTATION
    DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "flair")
    DEFAULT_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}

    def __init__(
        self,
        root: str | Path,
        *,
        modalities: tuple[str, ...] = DEFAULT_MODALITIES,
        seg_name: str = "seg",
        file_template: str = "{patient}_{modality}.nii.gz",
        label_map: dict[int, int] | None = None,
        num_classes: int = 4,
        target_size: int = 224,
        min_tumor_voxels: int = 1,
        include_empty_slices: bool = False,
        patient_ids: list[str] | None = None,
        cache_size: int = 4,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.root = Path(root)
        self.modalities = tuple(modalities)
        self.seg_name = seg_name
        self.file_template = file_template
        self.label_map = dict(label_map or self.DEFAULT_LABEL_MAP)
        self.target_size = target_size
        self.patients = sorted(p for p in self.root.iterdir() if p.is_dir())
        if patient_ids is not None:
            # Restrict to a patient subset, for a patient-level train/val/test split.
            wanted = set(patient_ids)
            self.patients = [p for p in self.patients if p.name in wanted]
        if not self.patients:
            raise ValueError(f"no patient folders under {self.root}")
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = cache_size
        self.index = self._build_index(min_tumor_voxels, include_empty_slices)
        if not self.index:
            raise ValueError("no slices selected; check labels or set include_empty_slices=True")

    def _path(self, patient: Path, modality: str) -> Path:
        return patient / self.file_template.format(patient=patient.name, modality=modality)

    def _build_index(self, min_tumor_voxels: int, include_empty: bool) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for patient_idx, patient in enumerate(self.patients):
            seg_path = self._path(patient, self.seg_name)
            if include_empty:
                depth = nib.load(str(seg_path)).shape[2]
                index += [(patient_idx, s) for s in range(depth)]
            else:
                seg = np.asarray(nib.load(str(seg_path)).dataobj)
                per_slice = (seg > 0).reshape(-1, seg.shape[2]).sum(axis=0)
                index += [
                    (patient_idx, s) for s, n in enumerate(per_slice) if n >= min_tumor_voxels
                ]
        return index

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """Z-score over the brain (nonzero) voxels, leaving the background at zero."""
        brain = volume > 0
        if brain.any():
            mean = volume[brain].mean()
            std = volume[brain].std()
            if std > 0:
                volume = np.where(brain, (volume - mean) / std, 0.0)
        return volume.astype(np.float32)

    def _remap(self, seg: np.ndarray) -> np.ndarray:
        out = np.zeros(seg.shape, dtype=np.int64)
        for src, dst in self.label_map.items():
            out[seg == src] = dst
        return out

    def _load_patient(self, patient_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if patient_idx in self._cache:
            self._cache.move_to_end(patient_idx)
            return self._cache[patient_idx]
        patient = self.patients[patient_idx]
        channels = [
            self._normalize(np.asarray(nib.load(str(self._path(patient, m))).dataobj))
            for m in self.modalities
        ]
        image = np.stack(channels, axis=0)  # (C, H, W, D)
        seg = np.asarray(nib.load(str(self._path(patient, self.seg_name))).dataobj)
        mask = self._remap(seg)
        self._cache[patient_idx] = (image, mask)
        self._cache.move_to_end(patient_idx)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image, mask

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Sample:
        patient_idx, slice_idx = self.index[index]
        image, mask = self._load_patient(patient_idx)
        image_slice = center_fit(image[:, :, :, slice_idx], self.target_size)  # (C, S, S)
        mask_slice = center_fit(mask[:, :, slice_idx], self.target_size)  # (S, S)
        return Sample(
            image=torch.from_numpy(np.ascontiguousarray(image_slice)).float(),
            target=torch.from_numpy(np.ascontiguousarray(mask_slice)).long(),
            task=self.task,
            meta={"patient": self.patients[patient_idx].name, "slice": slice_idx},
        )
