"""Worked examples: real ISIC test images with prediction, confidence, uncertainty.

Runs the trained Deep Ensemble over the test set, then lays out two rows of real
dermoscopy images: the ones the ensemble is confident and correct about, and the
ones it is most uncertain about (highest member disagreement). Each panel shows
the true label, the predicted label, the calibrated probability of malignant, and
the epistemic uncertainty. This is the image-in, calibrated-answer-out view.

Usage:
    python scripts/isic_examples.py --config configs/isic_resnet50.yaml --data-root data/isic_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from torch.utils.data import DataLoader

from medimg_uq.contract import collate_samples
from medimg_uq.data import ISICDataset, isic_transforms
from medimg_uq.train import build_model_from_config, load_config, set_seed
from medimg_uq.uq import EnsembleSampler

_CLASS_NAMES = ("benign", "malignant")


def load_ensemble(cfg, device: str) -> EnsembleSampler:
    members = []
    for i in range(cfg.ensemble_size):
        ckpt = Path(cfg.ckpt_dir) / cfg.name / f"member_{i}" / "best.pt"
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        members.append(model.to(device).eval())
    return EnsembleSampler(members)


@torch.no_grad()
def collect_predictions(sampler, dataset, cfg, device: str) -> list[dict]:
    loader = DataLoader(
        dataset, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate_samples
    )
    records: list[dict] = []
    for batch in loader:
        dist = sampler.predict(batch.images.to(device))
        for j in range(len(batch)):
            true = int(batch.targets[j])
            pred = int(dist.prediction[j])
            records.append(
                {
                    "image": batch.meta[j]["image"],
                    "true": true,
                    "pred": pred,
                    "p_malignant": float(dist.mean_probs[j, 1]),
                    "epistemic": float(dist.epistemic[j]),
                    "correct": pred == true,
                }
            )
    return records


def _panel(ax, image_dir: Path, record: dict) -> None:
    image = Image.open(image_dir / record["image"]).convert("RGB")
    image.thumbnail((256, 256))
    ax.imshow(np.asarray(image))
    ax.set_xticks([])
    ax.set_yticks([])
    color = "#2a7f3f" if record["correct"] else "#b32d3a"
    title = (
        f"true: {_CLASS_NAMES[record['true']]}\n"
        f"pred: {_CLASS_NAMES[record['pred']]}  p(mal)={record['p_malignant']:.2f}\n"
        f"uncertainty={record['epistemic']:.2f}"
    )
    ax.set_title(title, color=color, fontsize=9)


def build_figure(records: list[dict], image_dir: Path) -> Figure:
    def top_class_prob(record: dict) -> float:
        return max(record["p_malignant"], 1 - record["p_malignant"])

    confident = sorted((r for r in records if r["correct"]), key=lambda r: -top_class_prob(r))
    # One benign and the rest filled to four, preferring a mix of classes.
    clear = []
    for want in (0, 1, 0, 1):
        pick = next((r for r in confident if r["true"] == want and r not in clear), None)
        if pick is not None:
            clear.append(pick)
    clear = (clear + confident)[:4]

    uncertain = sorted(records, key=lambda r: -r["epistemic"])[:4]

    fig = Figure(figsize=(12, 6.5), dpi=130)
    FigureCanvasAgg(fig)
    rows = [("Confident and correct", clear), ("Most uncertain (ensemble disagreement)", uncertain)]
    for row_idx, (label, group) in enumerate(rows):
        for col_idx, record in enumerate(group):
            ax = fig.add_subplot(2, 4, row_idx * 4 + col_idx + 1)
            _panel(ax, image_dir, record)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=12)
    fig.suptitle(
        "ISIC Deep Ensemble: real test lesions with calibrated confidence and uncertainty",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the ISIC worked-examples figure.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="docs/figures/isic_worked_examples.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDataset(
        args.data_root, split="test", transform=isic_transforms(cfg.image_size, train=False)
    )
    sampler = load_ensemble(cfg, device)
    records = collect_predictions(sampler, dataset, cfg, device)

    n_wrong = sum(1 for r in records if not r["correct"])
    print(f"test records: {len(records)}, wrong: {n_wrong}")
    fig = build_figure(records, dataset.image_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
