"""Aggregate a set of forward passes into one predictive distribution.

Both uncertainty methods in this project, Deep Ensembles and MC Dropout, produce
the same thing: a stack of probability tensors, one per ensemble member or per
stochastic pass. This module reduces that stack to a mean prediction and a
decomposition of the predictive uncertainty. It is pure torch and task-agnostic.

The stack convention is ``member_probs`` with the member axis first:
    classification: ``(M, N, C)``
    segmentation:   ``(M, N, C, H, W)``
so the class axis is ``dim=2`` in both, and ``dim=1`` after averaging members.

Uncertainty decomposition (Depeweg et al., 2018):
    total entropy   H[E_m p_m]            predictive uncertainty
    aleatoric       E_m H[p_m]            expected per-member entropy
    epistemic       total - aleatoric     mutual information (model disagreement)
A single model cannot report epistemic uncertainty; that signal only appears once
members or stochastic passes disagree, which is the whole reason for the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-12


def entropy(probs: torch.Tensor, dim: int) -> torch.Tensor:
    """Shannon entropy (in nats) over ``dim``. ``0 * log 0`` is treated as 0."""
    return -(probs * probs.clamp_min(_EPS).log()).sum(dim=dim)


@dataclass
class PredictiveDistribution:
    """The aggregated prediction plus its uncertainty decomposition.

    Shapes follow the input: for classification ``mean_probs`` is ``(N, C)`` and the
    uncertainty fields are ``(N,)``; for segmentation ``mean_probs`` is ``(N, C, H, W)``
    and the uncertainty fields are ``(N, H, W)`` (per-voxel maps).
    """

    mean_probs: torch.Tensor
    prediction: torch.Tensor  # argmax of mean_probs over the class axis
    confidence: torch.Tensor  # max of mean_probs over the class axis
    total_uncertainty: torch.Tensor  # predictive entropy
    aleatoric: torch.Tensor  # expected per-member entropy
    epistemic: torch.Tensor  # mutual information (total - aleatoric)
    n_members: int


def aggregate(member_probs: torch.Tensor, class_dim: int = 2) -> PredictiveDistribution:
    """Reduce a member stack to a :class:`PredictiveDistribution`.

    Args:
        member_probs: Stacked probabilities with the member axis at ``dim=0`` and
            the class axis at ``class_dim``. Each member's slice is a proper
            probability distribution over the class axis.
        class_dim: Index of the class axis in ``member_probs`` (default 2).

    Returns:
        The mean prediction and the total / aleatoric / epistemic uncertainty.
    """
    if member_probs.ndim < 2:
        raise ValueError(
            f"member_probs must have a member axis and a class axis, got {member_probs.ndim}D"
        )
    n_members = member_probs.shape[0]
    if n_members < 1:
        raise ValueError("need at least one member to aggregate")

    mean_probs = member_probs.mean(dim=0)
    mean_class_dim = class_dim - 1  # class axis shifts left once the member axis is gone

    total = entropy(mean_probs, dim=mean_class_dim)
    per_member = entropy(member_probs, dim=class_dim)  # keeps the member axis at 0
    aleatoric = per_member.mean(dim=0)
    epistemic = (total - aleatoric).clamp_min(0.0)  # MI is non-negative; clip tiny float noise

    confidence, prediction = mean_probs.max(dim=mean_class_dim)

    return PredictiveDistribution(
        mean_probs=mean_probs,
        prediction=prediction,
        confidence=confidence,
        total_uncertainty=total,
        aleatoric=aleatoric,
        epistemic=epistemic,
        n_members=n_members,
    )
