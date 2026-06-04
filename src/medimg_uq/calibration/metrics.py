"""Calibration and accuracy metrics.

Pure functions over probabilities and targets. They know nothing about the model
or the task. A classifier hands in ``(N, C)`` probabilities; a segmenter flattens
its ``(N, C, H, W)`` voxel probabilities to the same ``(M, C)`` shape with
:func:`flatten_voxels` and gets per-voxel calibration from the identical code.

Convention everywhere in this module:
    probs: float tensor ``(N, C)``, each row a probability vector summing to 1.
    targets: long tensor ``(N,)`` with class indices in ``[0, C)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc


def _check(probs: torch.Tensor, targets: torch.Tensor) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {tuple(probs.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (N,), got shape {tuple(targets.shape)}")
    if probs.shape[0] != targets.shape[0]:
        raise ValueError(f"probs and targets disagree on N: {probs.shape[0]} vs {targets.shape[0]}")
    if probs.shape[0] == 0:
        raise ValueError("cannot score an empty batch")


def flatten_voxels(probs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape segmentation outputs to the ``(M, C)`` / ``(M,)`` classification form.

    Args:
        probs: ``(N, C, H, W)`` per-voxel probabilities.
        targets: ``(N, H, W)`` per-voxel class indices.

    Returns:
        ``(probs_flat, targets_flat)`` where each voxel is treated as one sample,
        so every metric in this module applies unchanged to segmentation.
    """
    if probs.ndim != 4:
        raise ValueError(f"segmentation probs must be 4D (N, C, H, W), got {tuple(probs.shape)}")
    n, c, h, w = probs.shape
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)
    targets_flat = targets.reshape(-1)
    return probs_flat, targets_flat


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy."""
    _check(probs, targets)
    preds = probs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def negative_log_likelihood(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12
) -> float:
    """Mean negative log likelihood (cross-entropy against hard targets).

    Lower is better. Uses the probability the model assigned to the true class.
    ``eps`` clamps to keep ``log(0)`` finite.
    """
    _check(probs, targets)
    true_class_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (-torch.log(true_class_prob.clamp_min(eps))).mean().item()


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Multi-class Brier score: mean squared error against the one-hot target.

    Lower is better. For binary problems this is twice the usual one-dimensional
    Brier score because both class columns contribute.
    """
    _check(probs, targets)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()


@dataclass
class ReliabilityCurve:
    """Binned reliability data, the raw material for a reliability diagram.

    Each array has one entry per bin. Empty bins carry zero count and are skipped
    when computing ECE and when plotting.
    """

    bin_edges: torch.Tensor  # (n_bins + 1,)
    bin_confidence: torch.Tensor  # (n_bins,) mean predicted confidence in the bin
    bin_accuracy: torch.Tensor  # (n_bins,) fraction correct in the bin
    bin_count: torch.Tensor  # (n_bins,) number of samples in the bin


def reliability_curve(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> ReliabilityCurve:
    """Bin predictions by top-label confidence and measure accuracy per bin.

    A well-calibrated model has ``bin_accuracy`` close to ``bin_confidence`` in
    every populated bin (points near the diagonal of the reliability diagram).
    """
    _check(probs, targets)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    confidence, preds = probs.max(dim=1)
    correct = (preds == targets).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    conf = torch.zeros(n_bins, device=probs.device)
    acc = torch.zeros(n_bins, device=probs.device)
    count = torch.zeros(n_bins, device=probs.device)

    # Right-closed bins; the top bin includes confidence == 1.0.
    idx = torch.bucketize(confidence, edges[1:-1], right=False).clamp_max(n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        count[b] = n
        if n > 0:
            conf[b] = confidence[mask].mean()
            acc[b] = correct[mask].mean()

    return ReliabilityCurve(bin_edges=edges, bin_confidence=conf, bin_accuracy=acc, bin_count=count)


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """Expected Calibration Error: count-weighted mean gap between confidence and accuracy.

    Range ``[0, 1]``; 0 is perfectly calibrated. This is the standard top-label
    ECE (Naeini et al., 2015; Guo et al., 2017).
    """
    curve = reliability_curve(probs, targets, n_bins=n_bins)
    total = curve.bin_count.sum()
    if total == 0:
        return 0.0
    gap = (curve.bin_accuracy - curve.bin_confidence).abs()
    return ((curve.bin_count / total) * gap).sum().item()


def maximum_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """Maximum Calibration Error: the largest confidence-accuracy gap over populated bins."""
    curve = reliability_curve(probs, targets, n_bins=n_bins)
    populated = curve.bin_count > 0
    if not bool(populated.any()):
        return 0.0
    gap = (curve.bin_accuracy - curve.bin_confidence).abs()
    return gap[populated].max().item()


def auroc(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Area under the ROC curve.

    Binary problems use the positive-class probability. Multi-class uses the
    macro-averaged one-vs-rest AUROC. Needs at least one sample of each class
    present in ``targets`` to be defined.
    """
    _check(probs, targets)
    if num_classes == 2:
        return binary_auroc(probs[:, 1], targets).item()
    return multiclass_auroc(probs, targets, num_classes=num_classes, average="macro").item()


def _hard_labels(preds: torch.Tensor) -> torch.Tensor:
    """Accept either ``(N, C, H, W)`` probabilities/logits or ``(N, H, W)`` labels."""
    return preds.argmax(dim=1) if preds.ndim == 4 else preds


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    include_background: bool = False,
    eps: float = 1e-7,
) -> float:
    """Mean Dice over classes (foreground only by default).

    ``preds`` may be per-voxel probabilities ``(N, C, H, W)`` or hard labels
    ``(N, H, W)``; ``targets`` are hard labels ``(N, H, W)``. Background is excluded
    by default, since tumor Dice is the quantity of interest in BraTS.
    """
    preds = _hard_labels(preds)
    start = 0 if include_background else 1
    scores = []
    for c in range(start, num_classes):
        p = preds == c
        t = targets == c
        intersection = (p & t).sum().to(torch.float64)
        denom = p.sum().to(torch.float64) + t.sum().to(torch.float64)
        scores.append((2 * intersection + eps) / (denom + eps))
    if not scores:
        return 0.0
    return float(torch.stack(scores).mean())


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    include_background: bool = False,
    eps: float = 1e-7,
) -> float:
    """Mean intersection-over-union over classes (foreground only by default)."""
    preds = _hard_labels(preds)
    start = 0 if include_background else 1
    scores = []
    for c in range(start, num_classes):
        p = preds == c
        t = targets == c
        intersection = (p & t).sum().to(torch.float64)
        union = (p | t).sum().to(torch.float64)
        scores.append((intersection + eps) / (union + eps))
    if not scores:
        return 0.0
    return float(torch.stack(scores).mean())
