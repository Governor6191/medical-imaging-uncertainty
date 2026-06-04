"""Download a binary ISIC dataset from the ISIC Archive API.

Pulls dermoscopy images that carry a benign or malignant label, saves them under
``data/isic/images/``, and writes ``data/isic/labels.csv`` with a deterministic
stratified train/val/test split. Uses only the standard library, so it has no
dependencies beyond Python itself.

The ISIC Archive is openly accessible. Respect its terms and attribution: see
https://www.isic-archive.com and the per-image copyright_license field. These are
research images, not a cleared diagnostic dataset.

Usage:
    python scripts/download_isic.py --out data/isic --per-class 1500
    python scripts/download_isic.py --out data/isic --per-class 1500 --size thumbnail_256

``--size full`` downloads the original images (large). ``thumbnail_256`` is enough
for a 224 pixel training pipeline and far smaller, so it is the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

API_IMAGES = "https://api.isic-archive.com/api/v2/images/?limit=100"
LABELS = ("benign", "malignant")


def _get_json(url: str, *, retries: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to GET {url}: {last_error}")


def _download_file(url: str, dest: Path, *, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                dest.write_bytes(response.read())
            return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return False


def _label_of(record: dict) -> str | None:
    """Map an image's top-level diagnosis to a binary label, or None to skip it.

    The ISIC taxonomy puts the benign/malignant/indeterminate level at
    ``metadata.clinical.diagnosis_1``. Indeterminate or unlabeled images return
    None and are skipped.
    """
    clinical = (record.get("metadata") or {}).get("clinical") or {}
    value = clinical.get("diagnosis_1")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in LABELS else None


def collect(out_dir: Path, per_class: int, size: str, max_pages: int) -> list[tuple[str, str]]:
    """Walk the archive, download up to ``per_class`` images per label, return (file, label).

    Stops when both classes reach ``per_class`` or after ``max_pages`` pages, so the
    walk stays bounded even though malignant is the minority across the full archive.
    Whatever was collected is returned and written, even on an early stop.
    """
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in LABELS}
    rows: list[tuple[str, str]] = []

    url: str | None = API_IMAGES
    pages = 0
    while url and pages < max_pages and any(counts[label] < per_class for label in LABELS):
        page = _get_json(url)
        pages += 1
        for record in page.get("results", []):
            label = _label_of(record)
            if label is None or counts[label] >= per_class:
                continue
            files = record.get("files") or {}
            file_info = files.get(size) or files.get("full")
            if not file_info or "url" not in file_info:
                continue
            isic_id = record["isic_id"]
            filename = f"{isic_id}.jpg"
            if _download_file(file_info["url"], images_dir / filename):
                rows.append((filename, label))
                counts[label] += 1
        print(f"benign={counts['benign']} malignant={counts['malignant']}", flush=True)
        url = page.get("next")
    return rows


def stratified_split(
    rows: list[tuple[str, str]],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> list[tuple[str, str, str]]:
    """Assign a split per row, stratified by label and shuffled within each class.

    The shuffle (seeded, so it is reproducible) matters: images arrive in isic_id
    order, which groups acquisition eras together, so a positional split would put
    a different distribution in test than in train. Shuffling first makes train,
    val, and test independent draws from the same distribution.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for label in LABELS:
        group = [r for r in rows if r[1] == label]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, (filename, lab) in enumerate(group):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            out.append((filename, lab, split))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a binary ISIC dataset.")
    parser.add_argument("--out", type=Path, default=Path("data/isic"))
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--size", choices=["thumbnail_256", "full"], default="thumbnail_256")
    parser.add_argument(
        "--max-pages", type=int, default=600, help="cap the archive walk (100 images per page)"
    )
    args = parser.parse_args()

    rows = collect(args.out, args.per_class, args.size, args.max_pages)
    if not rows:
        raise SystemExit("no labeled images were downloaded; check network access")

    split_rows = stratified_split(rows)
    labels_path = args.out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "split"])
        writer.writerows(split_rows)
    print(f"wrote {len(split_rows)} rows to {labels_path}")


if __name__ == "__main__":
    main()
