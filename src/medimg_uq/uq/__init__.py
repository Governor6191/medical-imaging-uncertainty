"""Uncertainty quantification.

Task-agnostic. Deep Ensembles is the anchor method; MC Dropout is the cheap
comparison sampler. Both produce a predictive distribution that the calibration
suite scores the same way regardless of modality.
"""
