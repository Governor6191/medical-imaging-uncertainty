"""Train a Deep Ensemble on ISIC and compare three uncertainty methods.

One training run yields all three rows of the comparison:
    single model    member 0 on its own
    Deep Ensemble   the mean over all K members
    MC Dropout      member 0 run many times with dropout left on

Every method is evaluated on the same held-out test split (and on val), so the
accuracy and calibration numbers are directly comparable. Writes a metrics.json
and reliability diagram per method and split, plus a combined comparison.json.

Usage:
    python scripts/train_ensemble.py --config configs/isic_resnet50.yaml --data-root data/isic_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medimg_uq.contract import collate_samples
from medimg_uq.data import ISICDataset, isic_transforms
from medimg_uq.eval import evaluate, save_eval_artifacts
from medimg_uq.train import build_model_from_config, load_config, set_seed, train_ensemble
from medimg_uq.uq import EnsembleSampler, MCDropoutSampler, Sampler


def make_loaders_factory(cfg, data_root: str):
    def make(seed: int) -> tuple[DataLoader, DataLoader]:
        train_ds = ISICDataset(
            data_root, split="train", transform=isic_transforms(cfg.image_size, train=True)
        )
        val_ds = ISICDataset(
            data_root, split="val", transform=isic_transforms(cfg.image_size, train=False)
        )
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

    return make


def eval_loader(cfg, data_root: str, split: str) -> DataLoader:
    ds = ISICDataset(data_root, split=split, transform=isic_transforms(cfg.image_size, train=False))
    return DataLoader(
        ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
    )


def load_member(cfg, ckpt_path: str, device: str) -> torch.nn.Module:
    model = build_model_from_config(cfg)
    state = torch.load(ckpt_path, map_location=device)["model_state"]
    model.load_state_dict(state)
    return model.to(device).eval()


def score_method(
    sampler: Sampler, cfg, data_root: str, device: str, method: str, split: str
) -> dict:
    loader = eval_loader(cfg, data_root, split)
    report, result = evaluate(
        sampler,
        loader,
        task=cfg.task,
        num_classes=cfg.num_classes,
        device=device,
        n_bins=cfg.n_bins,
    )
    out_dir = Path(cfg.out_dir) / cfg.name / method / split
    save_eval_artifacts(report, result, out_dir, title=f"{cfg.name} {method} ({split})")
    print(f"{method:>14}  {split:>4}  {report.summary()}")
    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ISIC Deep Ensemble and compare methods.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={cfg.name}  K={cfg.ensemble_size}  image_size={cfg.image_size}")

    paths = train_ensemble(
        make_loaders_factory(cfg, args.data_root), cfg=cfg, device=device, log_fn=print
    )
    print(f"trained {len(paths)} members")
    members = [load_member(cfg, p, device) for p in paths]

    samplers: dict[str, Sampler] = {
        "single_model": EnsembleSampler([members[0]]),
        "deep_ensemble": EnsembleSampler(members),
        "mc_dropout": MCDropoutSampler(members[0], n_samples=cfg.mc_dropout_samples),
    }

    comparison: dict[str, dict[str, dict]] = {}
    for split in args.splits:
        comparison[split] = {}
        for method, sampler in samplers.items():
            comparison[split][method] = score_method(
                sampler, cfg, args.data_root, device, method, split
            )

    comparison_path = Path(cfg.out_dir) / cfg.name / "comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"wrote {comparison_path}")


if __name__ == "__main__":
    main()
