"""Bundle the metrics into one comparable calibration report.

A report is what the eval layer writes to ``metrics.json`` and what the README
tables are built from. It holds accuracy alongside calibration, because the whole
point of this project is to never report one without the other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from medimg_uq.calibration import metrics
from medimg_uq.contract import Task


@dataclass
class CalibrationReport:
    """Accuracy and calibration for one model on one dataset.

    ``auroc`` is ``None`` for segmentation, where the headline accuracy metrics
    are Dice and IoU (reported separately) rather than AUROC.
    """

    task: Task
    n_samples: int
    n_classes: int
    n_bins: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    mce: float
    auroc: float | None
    dice: float | None = None
    iou: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = str(self.task)
        return d

    def save(self, path: str | Path) -> None:
        """Write the report to a JSON file, creating parent folders as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        """A compact one-line summary for logs."""
        parts = [
            f"acc={self.accuracy:.4f}",
            f"nll={self.nll:.4f}",
            f"brier={self.brier:.4f}",
            f"ece={self.ece:.4f}",
            f"mce={self.mce:.4f}",
        ]
        if self.auroc is not None:
            parts.insert(1, f"auroc={self.auroc:.4f}")
        if self.dice is not None:
            parts.insert(1, f"dice={self.dice:.4f}")
        if self.iou is not None:
            parts.insert(2, f"iou={self.iou:.4f}")
        return "  ".join(parts)


def compute_report(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    task: Task,
    num_classes: int,
    n_bins: int = 15,
) -> CalibrationReport:
    """Compute every metric and pack them into a :class:`CalibrationReport`.

    For segmentation, pass per-voxel ``(N, C, H, W)`` probs and ``(N, H, W)``
    targets; they are flattened to per-voxel samples before scoring, so the
    calibration numbers are per-voxel calibration.
    """
    dice = iou = None
    if task is Task.SEGMENTATION and probs.ndim == 4:
        # Dice and IoU need the spatial layout, so compute them before flattening.
        dice = metrics.dice_score(probs, targets, num_classes)
        iou = metrics.iou_score(probs, targets, num_classes)
        probs, targets = metrics.flatten_voxels(probs, targets)

    n_samples = int(targets.shape[0])
    auroc = metrics.auroc(probs, targets, num_classes) if task is Task.CLASSIFICATION else None

    return CalibrationReport(
        task=task,
        n_samples=n_samples,
        n_classes=num_classes,
        n_bins=n_bins,
        accuracy=metrics.accuracy(probs, targets),
        nll=metrics.negative_log_likelihood(probs, targets),
        brier=metrics.brier_score(probs, targets),
        ece=metrics.expected_calibration_error(probs, targets, n_bins=n_bins),
        mce=metrics.maximum_calibration_error(probs, targets, n_bins=n_bins),
        auroc=auroc,
        dice=dice,
        iou=iou,
    )
