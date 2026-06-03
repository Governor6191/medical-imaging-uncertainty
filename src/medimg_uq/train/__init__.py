"""Training harness.

Config-driven loop: seeds, checkpointing, optional logging, and ensemble
orchestration. The loss and the target shape come from the task config, so the
loop itself stays modality-agnostic.
"""

from medimg_uq.train.config import (
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    load_config,
    set_seed,
)
from medimg_uq.train.loop import (
    EpochStats,
    TrainHistory,
    build_model_from_config,
    train_ensemble,
    train_model,
)
from medimg_uq.train.losses import make_loss

__all__ = [
    "EpochStats",
    "ExperimentConfig",
    "ModelConfig",
    "OptimConfig",
    "TrainHistory",
    "build_model_from_config",
    "load_config",
    "make_loss",
    "set_seed",
    "train_ensemble",
    "train_model",
]
