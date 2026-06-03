"""Train a single model on ISIC and report calibration.

Loads a config, builds the ISIC loaders, trains one model, then evaluates it with
the full calibration suite (a single model is an ensemble of one) and writes a
metrics.json and reliability diagram. This is the single-model baseline that the
Deep Ensemble and MC Dropout runs are compared against.

Usage:
    python scripts/train.py --config configs/isic_resnet50.yaml --data-root data/isic
    python scripts/train.py --config configs/isic_resnet50.yaml --data-root data/isic \
        --override optim.epochs=3 model.backbone=resnet18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medimg_uq.contract import collate_samples
from medimg_uq.data import ISICDataset, isic_transforms
from medimg_uq.eval import evaluate, save_eval_artifacts
from medimg_uq.train import build_model_from_config, load_config, set_seed, train_model
from medimg_uq.uq import EnsembleSampler


def build_isic_loaders(cfg, data_root: str, image_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = ISICDataset(
        data_root, split="train", transform=isic_transforms(image_size, train=True)
    )
    val_ds = ISICDataset(data_root, split="val", transform=isic_transforms(image_size, train=False))
    print(f"train: {train_ds.describe()}  counts={train_ds.class_counts()}")
    print(f"val:   {val_ds.describe()}  counts={val_ds.class_counts()}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a single ISIC model and report calibration."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected if omitted")
    parser.add_argument("--override", nargs="*", default=None, help="OmegaConf dotted overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={cfg.name}  backbone={cfg.model.backbone}")

    train_loader, val_loader = build_isic_loaders(cfg, args.data_root, args.image_size)
    model = build_model_from_config(cfg)
    history = train_model(model, train_loader, val_loader, cfg=cfg, device=device, log_fn=print)
    print(f"best epoch {history.best_epoch} at val loss {history.best_val_loss:.4f}")

    sampler = EnsembleSampler([model])  # single-model baseline
    report, result = evaluate(
        sampler,
        val_loader,
        task=cfg.task,
        num_classes=cfg.num_classes,
        device=device,
        n_bins=cfg.n_bins,
    )
    print(f"VAL single model  {report.summary()}")

    out_dir = Path(cfg.out_dir) / cfg.name / "single_model"
    save_eval_artifacts(report, result, out_dir, title=f"{cfg.name} single model (val)")
    print(f"wrote {out_dir / 'metrics.json'} and {out_dir / 'reliability.png'}")


if __name__ == "__main__":
    main()
