"""Extract BraTS tumor slices into a fast on-disk cache.

The on-the-fly :class:`BraTSDataset` reloads an 80 MB patient volume on every
cache miss, which thrashes under shuffled slice-level access across hundreds of
patients (training would spend hours per epoch just loading). This pass reads
each patient once, normalizes and slices it exactly as the dataset would, and
writes one small ``.npz`` per tumor slice (a float16 image and a uint8 mask).
Afterwards ``BraTSSliceDataset`` reads those tiny files, so training is GPU-bound.

Usage (BraTS 2020 Kaggle release, uncompressed .nii):
    python scripts/prepare_brats.py \
        --data-root "C:/.../MICCAI_BraTS2020_TrainingData" \
        --out data/brats2020_slices --file-ext .nii
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medimg_uq.data import BraTSDataset
from medimg_uq.data.brats import MANIFEST_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BraTS tumor slices to a cache.")
    parser.add_argument("--data-root", required=True, help="Folder of raw patient subfolders.")
    parser.add_argument("--out", required=True, help="Cache directory to write.")
    parser.add_argument(
        "--file-ext",
        default=".nii.gz",
        help="Raw image extension. BraTS 2021 is .nii.gz; the 2020 Kaggle release is .nii.",
    )
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=2000)
    args = parser.parse_args()

    ext = args.file_ext if args.file_ext.startswith(".") else f".{args.file_ext}"
    file_template = "{patient}_{modality}" + ext

    ds = BraTSDataset(
        args.data_root,
        num_classes=args.num_classes,
        target_size=args.target_size,
        file_template=file_template,
    )
    if ds.skipped_patients:
        print(f"skipped {len(ds.skipped_patients)} patient(s) missing a file:")
        print(f"  {ds.skipped_patients}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    n = len(ds)
    print(f"extracting {n} tumor slices from {len(ds.patients)} patients to {out_dir}")
    for i in range(n):
        sample = ds[i]
        patient = sample.meta["patient"]
        slice_idx = int(sample.meta["slice"])
        patient_dir = out_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        rel = f"{patient}/{patient}_slice{slice_idx:03d}.npz"
        np.savez(
            out_dir / rel,
            image=sample.image.numpy().astype(np.float16),
            mask=sample.target.numpy().astype(np.uint8),
        )
        records.append({"patient": patient, "slice": slice_idx, "file": rel})
        if (i + 1) % args.log_every == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} slices")

    manifest = {
        "meta": {
            "source": str(Path(args.data_root)),
            "target_size": args.target_size,
            "num_classes": args.num_classes,
            "file_ext": ext,
            "n_patients": len(ds.patients),
            "n_slices": len(records),
            "skipped_patients": ds.skipped_patients,
        },
        "slices": records,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / MANIFEST_NAME} ({len(records)} slices)")


if __name__ == "__main__":
    main()
