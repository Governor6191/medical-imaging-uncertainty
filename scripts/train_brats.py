"""Train a BraTS Deep Ensemble and report segmentation with per-voxel uncertainty.

Splits patients into train/val/test (patient-level, so no patient leaks across
splits), trains an ensemble of 2D-slice U-Nets, scores the single model and the
ensemble (Dice, IoU, and per-voxel calibration), and saves the headline figure:
an MRI slice, the predicted mask, and the per-voxel uncertainty map.

Usage:
    python scripts/train_brats.py --config configs/brats_unet.yaml --data-root data/brats
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medimg_uq.calibration import save_figure, segmentation_panels
from medimg_uq.contract import collate_samples
from medimg_uq.data import BraTSDataset, BraTSSliceDataset
from medimg_uq.eval import evaluate
from medimg_uq.train import build_model_from_config, load_config, set_seed, train_ensemble
from medimg_uq.uq import EnsembleSampler, Sampler

FLAIR_CHANNEL = 3  # (t1, t1ce, t2, flair); FLAIR shows edema well for the background


def _patient_is_complete(patient: Path, file_template: str) -> bool:
    """True if a patient folder has all four modalities and a segmentation."""
    names = (*BraTSDataset.DEFAULT_MODALITIES, "seg")
    return all(
        (patient / file_template.format(patient=patient.name, modality=m)).exists() for m in names
    )


def split_names(names: list[str], seed: int, val_frac: float, test_frac: float) -> dict:
    """Patient-level train/val/test split, so no patient leaks across splits."""
    names = list(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    return {
        "test": names[:n_test],
        "val": names[n_test : n_test + n_val],
        "train": names[n_test + n_val :],
    }


def raw_patient_names(data_root: str, file_template: str) -> list[str]:
    return sorted(
        p.name
        for p in Path(data_root).iterdir()
        if p.is_dir() and _patient_is_complete(p, file_template)
    )


def brats_dataset(
    cfg,
    data_root: str,
    patient_ids: list[str],
    *,
    file_template: str = "{patient}_{modality}.nii.gz",
) -> BraTSDataset:
    return BraTSDataset(
        data_root,
        num_classes=cfg.num_classes,
        target_size=cfg.image_size,
        patient_ids=patient_ids,
        file_template=file_template,
    )


def loader(ds, cfg, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.optim.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
        drop_last=drop_last,
    )


def load_member(cfg, ckpt_path: str, device: str) -> torch.nn.Module:
    model = build_model_from_config(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    return model.to(device).eval()


def score(sampler: Sampler, cfg, ds, device: str, method: str, split: str) -> dict:
    report, _ = evaluate(
        sampler,
        loader(ds, cfg, shuffle=False),
        task=cfg.task,
        num_classes=cfg.num_classes,
        device=device,
        n_bins=cfg.n_bins,
    )
    out_dir = Path(cfg.out_dir) / cfg.name / method / split
    report.save(out_dir / "metrics.json")
    print(f"{method:>14}  {split:>4}  {report.summary()}")
    return report.to_dict()


@torch.no_grad()
def save_uncertainty_figure(sampler: Sampler, cfg, ds, device: str) -> None:
    """Find the test slice the ensemble is most unsure about and draw its panels."""
    best = None
    for i in range(len(ds)):
        sample = ds[i]
        dist = sampler.predict(sample.image.unsqueeze(0).to(device))
        score_value = float(dist.epistemic[0].mean())
        if best is None or score_value > best[0]:
            best = (score_value, sample, dist)
    if best is None:
        return
    _, sample, dist = best
    fig = segmentation_panels(
        image=sample.image[FLAIR_CHANNEL],
        prediction=dist.prediction[0],
        uncertainty=dist.epistemic[0],
        target=sample.target,
        title=f"{cfg.name}: MRI, predicted tumor, and per-voxel uncertainty",
    )
    out = Path(cfg.out_dir) / cfg.name / "uncertainty_map.png"
    save_figure(fig, out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BraTS ensemble and report uncertainty.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="Raw patient folders (on-the-fly mode).")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Prepared slice cache from prepare_brats.py. Preferred for the full dataset.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--file-ext",
        default=".nii.gz",
        help="Image file extension. BraTS 2021 ships .nii.gz; the 2020 Kaggle release is .nii.",
    )
    args = parser.parse_args()
    if not args.cache_dir and not args.data_root:
        parser.error("pass --cache-dir (preferred) or --data-root")

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ext = args.file_ext if args.file_ext.startswith(".") else f".{args.file_ext}"
    file_template = "{patient}_{modality}" + ext

    if args.cache_dir:
        all_patients = BraTSSliceDataset.list_patients(args.cache_dir)

        def make_ds(patient_ids: list[str]):
            return BraTSSliceDataset(
                args.cache_dir, patient_ids=patient_ids, num_classes=cfg.num_classes
            )
    else:
        all_patients = raw_patient_names(args.data_root, file_template)

        def make_ds(patient_ids: list[str]):
            return brats_dataset(cfg, args.data_root, patient_ids, file_template=file_template)

    splits = split_names(all_patients, cfg.seed, args.val_frac, args.test_frac)
    print(f"device={device}  patients: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    def make_loaders(seed: int):
        return loader(make_ds(splits["train"]), cfg, shuffle=True, drop_last=True), loader(
            make_ds(splits["val"]), cfg, shuffle=False
        )

    paths = train_ensemble(make_loaders, cfg=cfg, device=device, log_fn=print)
    members = [load_member(cfg, p, device) for p in paths]

    samplers: dict[str, Sampler] = {
        "single_model": EnsembleSampler([members[0]]),
        "deep_ensemble": EnsembleSampler(members),
    }
    comparison: dict[str, dict] = {}
    for split in ("val", "test"):
        ds = make_ds(splits[split])
        comparison[split] = {m: score(s, cfg, ds, device, m, split) for m, s in samplers.items()}

    save_uncertainty_figure(samplers["deep_ensemble"], cfg, make_ds(splits["test"]), device)

    out = Path(cfg.out_dir) / cfg.name / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
