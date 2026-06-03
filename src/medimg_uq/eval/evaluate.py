"""Unified evaluation.

Runs any sampler (a single model, a Deep Ensemble, or MC Dropout) over a loader,
collects the aggregated predictions, and produces a calibration report plus a
reliability diagram. A single model is just an ensemble of one, so every method
flows through the same path and the numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medimg_uq.calibration import (
    CalibrationReport,
    compute_report,
    flatten_voxels,
    reliability_curve,
    reliability_diagram,
    save_figure,
)
from medimg_uq.contract import Task
from medimg_uq.uq import Sampler


@dataclass
class InferenceResult:
    """Aggregated predictions over a whole dataset, kept for metrics and figures."""

    probs: torch.Tensor  # mean probabilities, (N, C) or (N, C, H, W)
    targets: torch.Tensor  # (N,) or (N, H, W)
    total_uncertainty: torch.Tensor  # predictive entropy, (N,) or (N, H, W)
    epistemic: torch.Tensor  # mutual information, (N,) or (N, H, W)
    task: Task


@torch.no_grad()
def run_inference(
    sampler: Sampler,
    loader: DataLoader,
    *,
    task: Task,
    device: torch.device | str = "cpu",
) -> InferenceResult:
    """Aggregate a sampler's predictions over every batch in ``loader``.

    The sampler's models must already be on ``device``; only the inputs are moved
    here. Results are gathered on CPU so a long evaluation does not pin GPU memory.
    """
    device = torch.device(device)
    probs, targets, total, epistemic = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        dist = sampler.predict(batch.images)
        probs.append(dist.mean_probs.cpu())
        targets.append(batch.targets.cpu())
        total.append(dist.total_uncertainty.cpu())
        epistemic.append(dist.epistemic.cpu())
    return InferenceResult(
        probs=torch.cat(probs),
        targets=torch.cat(targets),
        total_uncertainty=torch.cat(total),
        epistemic=torch.cat(epistemic),
        task=task,
    )


def evaluate(
    sampler: Sampler,
    loader: DataLoader,
    *,
    task: Task,
    num_classes: int,
    device: torch.device | str = "cpu",
    n_bins: int = 15,
) -> tuple[CalibrationReport, InferenceResult]:
    """Run inference and score it into a :class:`CalibrationReport`."""
    result = run_inference(sampler, loader, task=task, device=device)
    report = compute_report(
        result.probs, result.targets, task=task, num_classes=num_classes, n_bins=n_bins
    )
    return report, result


def save_eval_artifacts(
    report: CalibrationReport,
    result: InferenceResult,
    out_dir: str | Path,
    *,
    title: str = "Reliability diagram",
) -> Path:
    """Write ``metrics.json`` and ``reliability.png`` under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.save(out / "metrics.json")

    if result.task is Task.SEGMENTATION and result.probs.ndim == 4:
        probs, targets = flatten_voxels(result.probs, result.targets)
    else:
        probs, targets = result.probs, result.targets
    curve = reliability_curve(probs, targets, n_bins=report.n_bins)
    fig = reliability_diagram(curve, ece=report.ece, title=title)
    save_figure(fig, out / "reliability.png")
    return out
