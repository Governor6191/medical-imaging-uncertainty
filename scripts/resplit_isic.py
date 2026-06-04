"""Reassign the train/val/test split of an existing ISIC labels.csv.

Run this after a download to switch to a seeded random stratified split without
re-downloading images. Images arrive in isic_id order, which groups acquisition
eras, so the split must shuffle within each class to keep train, val, and test
drawn from the same distribution.

Usage:
    python scripts/resplit_isic.py --labels data/isic_full/labels.csv --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

LABELS = ("benign", "malignant")


def stratified_split(
    rows: list[dict], *, val_frac: float, test_frac: float, seed: int
) -> list[tuple[str, str, str]]:
    """Shuffle within each class (seeded) and assign test, val, train by fraction."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        n = len(group)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        for i, row in enumerate(group):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            out.append((row["image"], row["label"], split))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split an existing ISIC labels.csv.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "image" not in rows[0] or "label" not in rows[0]:
        raise SystemExit("labels.csv must have 'image' and 'label' columns")

    split_rows = stratified_split(
        rows, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    with args.labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "split"])
        writer.writerows(split_rows)

    by_split = Counter((s, lab) for _, lab, s in split_rows)
    print(f"re-split {len(split_rows)} rows in {args.labels} (seed {args.seed})")
    for split in ("train", "val", "test"):
        counts = {lab: by_split[(split, lab)] for lab in LABELS}
        print(f"  {split}: {counts}")


if __name__ == "__main__":
    main()
