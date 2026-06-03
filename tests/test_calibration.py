"""Tests for the calibration suite, against known closed-form values."""

from __future__ import annotations

import math

import pytest
import torch

from medimg_uq.calibration import (
    CalibrationReport,
    accuracy,
    brier_score,
    compute_report,
    expected_calibration_error,
    flatten_voxels,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    reliability_diagram,
    save_figure,
)
from medimg_uq.contract import Task

ABS = 1e-6


def test_accuracy_perfect_and_zero():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    assert accuracy(probs, torch.tensor([0, 1])) == pytest.approx(1.0)
    assert accuracy(probs, torch.tensor([1, 0])) == pytest.approx(0.0)


def test_nll_matches_hand_computation():
    probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    # -[log(0.7) + log(0.6)] / 2
    expected = -(math.log(0.7) + math.log(0.6)) / 2
    assert negative_log_likelihood(probs, torch.tensor([0, 1])) == pytest.approx(expected, abs=ABS)


def test_nll_is_finite_on_zero_probability():
    probs = torch.tensor([[1.0, 0.0]])
    # True class has probability 0; clamp keeps it finite and large.
    assert math.isfinite(negative_log_likelihood(probs, torch.tensor([1])))


def test_brier_matches_hand_computation():
    probs = torch.tensor([[0.7, 0.3]])
    # (0.7 - 1)^2 + (0.3 - 0)^2 = 0.18
    assert brier_score(probs, torch.tensor([0])) == pytest.approx(0.18, abs=ABS)


def test_perfectly_confident_and_correct_is_zero_ece():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1, 0])
    assert expected_calibration_error(probs, targets) == pytest.approx(0.0, abs=ABS)
    assert negative_log_likelihood(probs, targets) == pytest.approx(0.0, abs=ABS)
    assert brier_score(probs, targets) == pytest.approx(0.0, abs=ABS)


def test_perfectly_confident_and_wrong_is_ece_one():
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([1, 1])  # always wrong, always 100% confident
    assert expected_calibration_error(probs, targets) == pytest.approx(1.0, abs=ABS)
    assert maximum_calibration_error(probs, targets) == pytest.approx(1.0, abs=ABS)


def test_known_ece_with_two_bins():
    # Two groups. Group A: confidence 0.6, accuracy 1.0 (gap 0.4).
    # Group B: confidence 0.9, accuracy 0.0 (gap 0.9). Equal counts.
    probs = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
    targets = torch.tensor([0, 0])  # A correct, B wrong
    ece = expected_calibration_error(probs, targets, n_bins=10)
    assert ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.9, abs=ABS)


def test_reliability_curve_counts_sum_to_n():
    torch.manual_seed(0)
    probs = torch.rand(50, 3)
    probs = probs / probs.sum(dim=1, keepdim=True)
    targets = torch.randint(0, 3, (50,))
    curve = reliability_curve(probs, targets, n_bins=15)
    assert int(curve.bin_count.sum()) == 50
    assert curve.bin_confidence.shape == (15,)


def test_confidence_one_lands_in_top_bin():
    probs = torch.tensor([[1.0, 0.0]])
    curve = reliability_curve(probs, torch.tensor([0]), n_bins=10)
    assert int(curve.bin_count[-1]) == 1


def test_flatten_voxels_shapes():
    probs = torch.rand(2, 4, 8, 8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    targets = torch.randint(0, 4, (2, 8, 8))
    flat_probs, flat_targets = flatten_voxels(probs, targets)
    assert flat_probs.shape == (2 * 8 * 8, 4)
    assert flat_targets.shape == (2 * 8 * 8,)


def test_classification_report_has_auroc():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    targets = torch.tensor([0, 1, 0, 1])
    report = compute_report(probs, targets, task=Task.CLASSIFICATION, num_classes=2)
    assert isinstance(report, CalibrationReport)
    assert report.auroc == pytest.approx(1.0, abs=ABS)  # perfectly ranked
    assert report.n_samples == 4
    assert "auroc" in report.summary()


def test_segmentation_report_skips_auroc_and_flattens():
    probs = torch.rand(2, 3, 8, 8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    targets = torch.randint(0, 3, (2, 8, 8))
    report = compute_report(probs, targets, task=Task.SEGMENTATION, num_classes=3)
    assert report.auroc is None
    assert report.n_samples == 2 * 8 * 8  # per-voxel
    assert "auroc" not in report.summary()


def test_report_round_trips_through_json(tmp_path):
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([0, 1])
    report = compute_report(probs, targets, task=Task.CLASSIFICATION, num_classes=2)
    out = tmp_path / "nested" / "metrics.json"
    report.save(out)
    assert out.exists()
    import json

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["task"] == "classification"
    assert loaded["n_samples"] == 2


def test_reliability_diagram_saves_a_nonempty_png(tmp_path):
    probs = torch.tensor([[0.6, 0.4], [0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([0, 0, 0])
    curve = reliability_curve(probs, targets, n_bins=10)
    fig = reliability_diagram(curve, ece=0.4, title="test")
    out = save_figure(fig, tmp_path / "reliability.png")
    assert out.exists()
    assert out.stat().st_size > 0
