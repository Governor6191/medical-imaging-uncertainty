"""Samplers that turn a model (or models) into a stack of forward passes.

Two methods, one output shape. Deep Ensembles runs K independently trained
members once each. MC Dropout runs one model many times with dropout left on.
Both return ``(M, N, ...)`` probabilities for :func:`~medimg_uq.uq.aggregate.aggregate`.

Pure torch: a sampler takes an image tensor and returns probabilities, so nothing
here knows whether the model classifies or segments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from medimg_uq.uq.aggregate import PredictiveDistribution, aggregate

# Probabilities are softmaxed over the channel axis, which is dim 1 for both
# (N, C) classification logits and (N, C, H, W) segmentation logits.
_CLASS_DIM_LOGITS = 1
# In the stacked output the member axis is prepended, so the class axis is dim 2.
_CLASS_DIM_STACK = 2


class Sampler(ABC):
    """Produces stacked probabilities and the aggregated predictive distribution."""

    @abstractmethod
    def sample_probs(self, images: torch.Tensor) -> torch.Tensor:
        """Return stacked probabilities ``(M, N, ...)`` with the member axis first."""

    def predict(self, images: torch.Tensor) -> PredictiveDistribution:
        """Sample, then aggregate to a :class:`PredictiveDistribution`."""
        return aggregate(self.sample_probs(images), class_dim=_CLASS_DIM_STACK)


class EnsembleSampler(Sampler):
    """Deep Ensembles: average the predictions of K independently trained members.

    Members differ because they were trained from different seeds. Their
    disagreement is what the aggregation reads as epistemic uncertainty.
    """

    def __init__(self, models: Sequence[nn.Module]):
        if len(models) == 0:
            raise ValueError("an ensemble needs at least one member")
        self.models = list(models)

    @torch.no_grad()
    def sample_probs(self, images: torch.Tensor) -> torch.Tensor:
        passes = []
        for model in self.models:
            model.eval()
            logits = model(images)
            passes.append(torch.softmax(logits, dim=_CLASS_DIM_LOGITS))
        return torch.stack(passes, dim=0)

    def __len__(self) -> int:
        return len(self.models)


def enable_mc_dropout(model: nn.Module) -> int:
    """Put every dropout layer into training mode, leaving the rest in eval.

    This is what makes MC Dropout work: dropout stays stochastic at inference
    while batch-norm and everything else keep their evaluation behavior. Returns
    the number of dropout layers switched on.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
            count += 1
    return count


class MCDropoutSampler(Sampler):
    """MC Dropout: run one model ``n_samples`` times with dropout left enabled.

    A cheap stand-in for an ensemble. It needs the model to actually contain
    dropout layers, so the constructor refuses a model that has none rather than
    silently returning identical, zero-epistemic passes.
    """

    def __init__(self, model: nn.Module, n_samples: int):
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        n_dropout = sum(1 for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd))
        if n_dropout == 0:
            raise ValueError(
                "model has no dropout layers, so MC Dropout would be deterministic. "
                "Build the backbone with a nonzero drop rate."
            )
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def sample_probs(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        enable_mc_dropout(self.model)
        passes = [
            torch.softmax(self.model(images), dim=_CLASS_DIM_LOGITS) for _ in range(self.n_samples)
        ]
        return torch.stack(passes, dim=0)
