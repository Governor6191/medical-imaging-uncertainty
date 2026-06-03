"""Tests for the uncertainty engine: aggregation and samplers."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from medimg_uq.uq import (
    EnsembleSampler,
    MCDropoutSampler,
    PredictiveDistribution,
    aggregate,
    enable_mc_dropout,
    entropy,
)

ABS = 1e-5
LOG2 = math.log(2)


def test_entropy_uniform_and_onehot():
    uniform = torch.tensor([[0.5, 0.5]])
    onehot = torch.tensor([[1.0, 0.0]])
    assert entropy(uniform, dim=1).item() == pytest.approx(LOG2, abs=ABS)
    assert entropy(onehot, dim=1).item() == pytest.approx(0.0, abs=ABS)


def test_disagreement_becomes_epistemic_uncertainty():
    # Two members, each fully confident but on opposite classes.
    member_probs = torch.tensor(
        [
            [[1.0, 0.0]],  # member 0: certain class 0
            [[0.0, 1.0]],  # member 1: certain class 1
        ]
    )  # shape (M=2, N=1, C=2)
    dist = aggregate(member_probs, class_dim=2)
    assert dist.mean_probs.shape == (1, 2)
    assert dist.mean_probs[0].tolist() == pytest.approx([0.5, 0.5], abs=ABS)
    # Each member is certain, so aleatoric is zero; all uncertainty is epistemic.
    assert dist.total_uncertainty.item() == pytest.approx(LOG2, abs=ABS)
    assert dist.aleatoric.item() == pytest.approx(0.0, abs=ABS)
    assert dist.epistemic.item() == pytest.approx(LOG2, abs=ABS)
    assert dist.n_members == 2


def test_agreement_has_near_zero_uncertainty():
    member_probs = torch.tensor([[[0.99, 0.01]], [[0.99, 0.01]]])
    dist = aggregate(member_probs, class_dim=2)
    assert dist.epistemic.item() == pytest.approx(0.0, abs=ABS)
    assert dist.prediction.item() == 0
    assert dist.confidence.item() == pytest.approx(0.99, abs=ABS)


def test_epistemic_is_clamped_non_negative():
    member_probs = torch.tensor([[[0.7, 0.3]], [[0.7, 0.3]]])
    dist = aggregate(member_probs, class_dim=2)
    assert dist.epistemic.item() >= 0.0


def test_aggregate_segmentation_shapes():
    member_probs = torch.rand(3, 2, 4, 8, 8)
    member_probs = member_probs / member_probs.sum(dim=2, keepdim=True)
    dist = aggregate(member_probs, class_dim=2)
    assert dist.mean_probs.shape == (2, 4, 8, 8)
    assert dist.total_uncertainty.shape == (2, 8, 8)  # per-voxel map
    assert dist.epistemic.shape == (2, 8, 8)
    assert bool((dist.epistemic >= 0).all())


def test_aggregate_rejects_empty_stack():
    with pytest.raises(ValueError, match="at least one member"):
        aggregate(torch.empty(0, 1, 2), class_dim=2)


def test_ensemble_sampler_output_is_normalized():
    torch.manual_seed(0)
    models = [nn.Linear(4, 3), nn.Linear(4, 3)]
    sampler = EnsembleSampler(models)
    images = torch.randn(5, 4)
    probs = sampler.sample_probs(images)
    assert probs.shape == (2, 5, 3)
    assert torch.allclose(probs.sum(dim=2), torch.ones(2, 5), atol=ABS)
    assert len(sampler) == 2


def test_ensemble_sampler_predict_returns_distribution():
    models = [nn.Linear(4, 3), nn.Linear(4, 3)]
    dist = EnsembleSampler(models).predict(torch.randn(6, 4))
    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean_probs.shape == (6, 3)
    assert dist.prediction.shape == (6,)


def test_ensemble_sampler_rejects_empty():
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleSampler([])


def _model_with_dropout() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(16, 3),
    )


def test_enable_mc_dropout_only_touches_dropout():
    model = _model_with_dropout()
    model.eval()
    n = enable_mc_dropout(model)
    assert n == 1
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert all(d.training for d in dropouts)
    assert all(not b.training for b in bns)  # batch norm stays in eval


def test_mc_dropout_passes_differ_and_shape():
    torch.manual_seed(0)
    sampler = MCDropoutSampler(_model_with_dropout(), n_samples=8)
    images = torch.randn(5, 4)
    probs = sampler.sample_probs(images)
    assert probs.shape == (8, 5, 3)
    # Stochastic dropout means the passes are not all identical.
    assert not torch.allclose(probs[0], probs[1])
    dist = sampler.predict(images)
    assert dist.epistemic.shape == (5,)
    assert bool((dist.epistemic >= 0).all())


def test_mc_dropout_rejects_model_without_dropout():
    with pytest.raises(ValueError, match="no dropout layers"):
        MCDropoutSampler(nn.Linear(4, 3), n_samples=4)


def test_mc_dropout_rejects_bad_sample_count():
    with pytest.raises(ValueError, match="n_samples"):
        MCDropoutSampler(_model_with_dropout(), n_samples=0)
