"""Tests for the segmentation half of the core: U-Net, Dice loss, Dice/IoU, and an
end-to-end train-then-calibrate smoke through the shared harness."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from medimg_uq.calibration import (
    compute_report,
    dice_score,
    iou_score,
    save_figure,
    segmentation_panels,
)
from medimg_uq.contract import Task, collate_samples
from medimg_uq.data import SyntheticSegmentationDataset
from medimg_uq.eval import evaluate
from medimg_uq.models import build_model
from medimg_uq.train import DiceCELoss, ExperimentConfig, ModelConfig, OptimConfig, train_model
from medimg_uq.train.loop import build_model_from_config
from medimg_uq.uq import EnsembleSampler


def test_unet_factory_forward_shape():
    model = build_model(
        task=Task.SEGMENTATION, backbone="resnet18", num_classes=3, pretrained=False, in_chans=4
    )
    out = model(torch.randn(2, 4, 64, 64))
    assert out.shape == (2, 3, 64, 64)


def test_dice_ce_loss_is_low_on_a_perfect_prediction():
    targets = torch.randint(0, 3, (2, 16, 16))
    logits = torch.full((2, 3, 16, 16), -10.0)
    for c in range(3):
        logits[:, c][targets == c] = 10.0  # near one-hot on the right class
    loss = DiceCELoss(num_classes=3)(logits, targets)
    assert loss.item() < 0.05


def test_dice_and_iou_perfect_and_disjoint():
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, :4, :] = 1  # half foreground
    perfect = targets.clone()
    assert dice_score(perfect, targets, num_classes=2) == pytest.approx(1.0, abs=1e-4)
    assert iou_score(perfect, targets, num_classes=2) == pytest.approx(1.0, abs=1e-4)
    disjoint = torch.zeros_like(targets)  # predicts all background, misses class 1
    assert dice_score(disjoint, targets, num_classes=2) < 0.01


def test_dice_accepts_probabilities():
    targets = torch.randint(0, 3, (2, 8, 8))
    probs = torch.zeros(2, 3, 8, 8)
    probs.scatter_(1, targets.unsqueeze(1), 1.0)  # one-hot probs match targets
    assert dice_score(probs, targets, num_classes=3) == pytest.approx(1.0, abs=1e-4)


def test_segmentation_report_has_dice_iou_and_per_voxel_calibration():
    targets = torch.randint(0, 3, (2, 16, 16))
    probs = torch.rand(2, 3, 16, 16)
    probs = probs / probs.sum(dim=1, keepdim=True)
    report = compute_report(probs, targets, task=Task.SEGMENTATION, num_classes=3)
    assert report.dice is not None
    assert report.iou is not None
    assert report.auroc is None
    assert report.n_samples == 2 * 16 * 16  # per-voxel
    assert "dice" in report.summary()


def test_train_then_evaluate_segmentation_end_to_end(tmp_path):
    train_ds = SyntheticSegmentationDataset(n=8, image_size=64, in_chans=4, num_classes=3, seed=0)
    val_ds = SyntheticSegmentationDataset(n=4, image_size=64, in_chans=4, num_classes=3, seed=1)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_samples)

    cfg = ExperimentConfig(
        name="seg_smoke",
        task=Task.SEGMENTATION,
        num_classes=3,
        out_dir=str(tmp_path / "out"),
        ckpt_dir=str(tmp_path / "ckpt"),
        model=ModelConfig(backbone="resnet18", pretrained=False, in_chans=4),
        optim=OptimConfig(lr=1e-3, epochs=2, batch_size=4, amp=False, scheduler="cosine"),
    )

    model = build_model_from_config(cfg)
    history = train_model(model, train_loader, val_loader, cfg=cfg, device="cpu")
    assert len(history.epochs) == 2

    report, result = evaluate(
        EnsembleSampler([model]), val_loader, task=Task.SEGMENTATION, num_classes=3, device="cpu"
    )
    assert result.probs.shape == (4, 3, 64, 64)
    assert result.epistemic.shape == (4, 64, 64)  # per-voxel uncertainty map
    assert report.dice is not None
    assert 0.0 <= report.ece <= 1.0


def test_segmentation_panels_saves_a_nonempty_figure(tmp_path):
    image = torch.rand(32, 32)
    prediction = torch.zeros(32, 32, dtype=torch.long)
    prediction[8:20, 8:20] = 1
    uncertainty = torch.rand(32, 32)
    target = prediction.clone()
    fig = segmentation_panels(image, prediction, uncertainty, target=target, title="test")
    out = save_figure(fig, tmp_path / "panels.png")
    assert out.exists()
    assert out.stat().st_size > 0
