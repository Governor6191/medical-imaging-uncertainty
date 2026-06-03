"""Tests for experiment configuration and seeding."""

from __future__ import annotations

import torch

from medimg_uq.contract import Task
from medimg_uq.train import ExperimentConfig, load_config, set_seed


def test_defaults():
    cfg = load_config()
    assert cfg.task is Task.CLASSIFICATION
    assert cfg.model.backbone == "resnet50"
    assert cfg.optim.epochs == 15
    assert cfg.ensemble_size == 5


def test_yaml_then_overrides(tmp_path):
    path = tmp_path / "isic.yaml"
    path.write_text(
        "name: isic\nnum_classes: 2\noptim:\n  epochs: 3\nmodel:\n  backbone: resnet18\n",
        encoding="utf-8",
    )
    cfg = load_config(path, overrides=["optim.lr=0.001", "model.drop_rate=0.5"])
    assert cfg.name == "isic"
    assert cfg.optim.epochs == 3
    assert cfg.optim.lr == 0.001
    assert cfg.model.backbone == "resnet18"
    assert cfg.model.drop_rate == 0.5


def test_task_coercion_from_string():
    cfg = ExperimentConfig.from_dict({"task": "segmentation", "num_classes": 4})
    assert cfg.task is Task.SEGMENTATION
    assert cfg.num_classes == 4


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
