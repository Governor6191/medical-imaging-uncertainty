"""Uncertainty quantification.

Task-agnostic. Deep Ensembles is the anchor method; MC Dropout is the cheap
comparison sampler. Both produce a predictive distribution that the calibration
suite scores the same way regardless of modality.
"""

from medimg_uq.uq.aggregate import PredictiveDistribution, aggregate, entropy
from medimg_uq.uq.samplers import (
    EnsembleSampler,
    MCDropoutSampler,
    Sampler,
    enable_mc_dropout,
)

__all__ = [
    "EnsembleSampler",
    "MCDropoutSampler",
    "PredictiveDistribution",
    "Sampler",
    "aggregate",
    "enable_mc_dropout",
    "entropy",
]
