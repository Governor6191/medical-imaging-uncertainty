"""Experiment configuration and seeding.

A run is fully described by an :class:`ExperimentConfig`: which task, which
backbone, the optimizer settings, and the ensemble and MC Dropout sizes. Configs
load from YAML and accept dotted command-line overrides through OmegaConf, so a
run is reproducible from one file plus the seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from medimg_uq.contract import Task


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    drop_rate: float = 0.3
    in_chans: int = 3


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    batch_size: int = 32
    amp: bool = True
    scheduler: str = "cosine"  # "cosine" or "none"


@dataclass
class ExperimentConfig:
    name: str = "run"
    task: Task = Task.CLASSIFICATION
    num_classes: int = 2
    seed: int = 0
    num_workers: int = 4
    n_bins: int = 15
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    ensemble_size: int = 5
    mc_dropout_samples: int = 30
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @staticmethod
    def from_dict(data: dict) -> ExperimentConfig:
        """Build a config from a plain dict, coercing nested sections and the task tag."""
        data = dict(data)
        model = ModelConfig(**dict(data.pop("model", {}) or {}))
        optim = OptimConfig(**dict(data.pop("optim", {}) or {}))
        if "task" in data and data["task"] is not None:
            data["task"] = Task(str(data["task"]))
        return ExperimentConfig(model=model, optim=optim, **data)


def _default_container() -> dict:
    """Defaults as a plain, untyped dict (task as its string value).

    Kept untyped on purpose: OmegaConf's enum node matches by member name, not by
    the StrEnum value used in YAML, so the task tag is carried as a string here and
    coerced back to :class:`Task` in :meth:`ExperimentConfig.from_dict`.
    """
    cfg = ExperimentConfig()
    container = asdict(cfg)
    container["task"] = str(cfg.task)
    return container


def load_config(
    path: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load a config from defaults, an optional YAML file, and optional CLI overrides.

    Overrides are OmegaConf dotlist strings, for example
    ``["optim.epochs=3", "model.backbone=resnet18"]``.
    """
    merged = OmegaConf.create(_default_container())
    if path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(path))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    container = OmegaConf.to_container(merged, resolve=True)
    return ExperimentConfig.from_dict(container)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and torch. Set ``deterministic`` for exact reproducibility.

    Each ensemble member is trained with a different seed; that is the only source
    of their diversity, so seeding is load-bearing here, not boilerplate.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
