"""Select a few held-out BraTS slices for the demo Space.

Picks slices from the test split (the model never trained on them) that contain
all three tumor subregions, spread across tumor size, and saves each as a compact
``.npz`` (a float16 image and a uint8 mask). These are the preloaded inputs the
demo Space runs on. They are not committed to git (BraTS data stays out of the
repo); regenerate them from a prepared slice cache with this script.

Usage:
    python scripts/make_brats_samples.py --cache-dir data/brats2020_slices \
        --out demo_assets/brats_samples
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from medimg_uq.data import BraTSSliceDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Select held-out BraTS demo slices.")
    parser.add_argument("--cache-dir", default="data/brats2020_slices")
    parser.add_argument("--out", default="demo_assets/brats_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    # Reproduce the train_brats test split: shuffle patients with the seed, take the head.
    patients = BraTSSliceDataset.list_patients(args.cache_dir)
    names = list(patients)
    random.Random(args.seed).shuffle(names)
    n = len(names)
    n_test = round(n * args.test_frac)
    test = set(names[:n_test])

    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for record in manifest["slices"]:
        if record["patient"] not in test:
            continue
        data = np.load(Path(args.cache_dir) / record["file"])
        mask = data["mask"]
        classes = set(np.unique(mask).tolist()) - {0}
        if {1, 2, 3} <= classes:
            candidates.append((int((mask > 0).sum()), record["patient"], record["file"]))
    candidates.sort(reverse=True)

    # Take n slices from distinct patients, the largest tumor per patient.
    picks, seen = [], set()
    for area, patient, file in candidates:
        if patient in seen:
            continue
        seen.add(patient)
        picks.append((area, patient, file))
        if len(picks) == args.n:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["case_1_large_tumor", "case_2_mid_tumor", "case_3_compact_tumor"]
    for (area, patient, file), label in zip(picks, labels, strict=False):
        data = np.load(Path(args.cache_dir) / file)
        dest = out_dir / f"{label}.npz"
        np.savez(dest, image=data["image"].astype(np.float16), mask=data["mask"].astype(np.uint8))
        print(f"saved {dest.name}: patient={patient} tumor_voxels={area}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
