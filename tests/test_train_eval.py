"""End-to-end smoke test: train a model, then evaluate it with the calibration suite.

Runs on the synthetic dataset and a tiny untrained backbone, on CPU, so it
exercises the real harness in a couple of seconds without downloading anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from medimg_uq.contract import Task, collate_samples
from medimg_uq.data.synthetic import SyntheticClassificationDataset
from medimg_uq.eval import evaluate, save_eval_artifacts
from medimg_uq.train import (
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    build_model_from_config,
    train_model,
)
from medimg_uq.uq import EnsembleSampler


def _loaders():
    train_ds = SyntheticClassificationDataset(n=48, image_size=32, seed=0)
    val_ds = SyntheticClassificationDataset(n=32, image_size=32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_samples)
    return train_loader, val_loader


def _cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        task=Task.CLASSIFICATION,
        num_classes=2,
        seed=0,
        out_dir=str(tmp_path / "out"),
        ckpt_dir=str(tmp_path / "ckpt"),
        model=ModelConfig(backbone="resnet18", pretrained=False, drop_rate=0.3, in_chans=3),
        optim=OptimConfig(
            lr=1e-3, weight_decay=1e-4, epochs=2, batch_size=16, amp=False, scheduler="cosine"
        ),
    )


def test_train_then_evaluate_end_to_end(tmp_path):
    train_loader, val_loader = _loaders()
    cfg = _cfg(tmp_path)

    logs: list[dict] = []
    model = build_model_from_config(cfg)
    history = train_model(
        model, train_loader, val_loader, cfg=cfg, device="cpu", log_fn=logs.append
    )

    assert len(history.epochs) == 2
    assert len(logs) == 2
    assert Path(history.checkpoint_path).exists()
    assert 0.0 <= history.epochs[-1].val_accuracy <= 1.0

    # A single model is an ensemble of one; epistemic uncertainty must be zero.
    sampler = EnsembleSampler([model])
    report, result = evaluate(
        sampler, val_loader, task=Task.CLASSIFICATION, num_classes=2, device="cpu", n_bins=10
    )
    assert result.probs.shape == (32, 2)
    assert 0.0 <= report.ece <= 1.0
    assert report.auroc is not None
    assert float(result.epistemic.abs().max()) == pytest.approx(0.0, abs=1e-5)

    out = save_eval_artifacts(report, result, tmp_path / "artifacts", title="smoke")
    assert (out / "metrics.json").exists()
    assert (out / "reliability.png").exists()
