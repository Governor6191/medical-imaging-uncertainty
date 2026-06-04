"""Calibration and evaluation metrics.

ECE, NLL, Brier, reliability diagrams, and per-voxel calibration. A pure metric
library that takes probabilities and targets and knows nothing about the model
or the task that produced them.
"""

from medimg_uq.calibration.metrics import (
    ReliabilityCurve,
    accuracy,
    auroc,
    brier_score,
    dice_score,
    expected_calibration_error,
    flatten_voxels,
    iou_score,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)
from medimg_uq.calibration.plots import reliability_diagram, save_figure
from medimg_uq.calibration.report import CalibrationReport, compute_report

__all__ = [
    "CalibrationReport",
    "ReliabilityCurve",
    "accuracy",
    "auroc",
    "brier_score",
    "compute_report",
    "dice_score",
    "expected_calibration_error",
    "flatten_voxels",
    "iou_score",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "reliability_curve",
    "reliability_diagram",
    "save_figure",
]
