"""Demo shell.

The app that wraps the released checkpoints: one mode for the skin classifier
with a calibrated-confidence readout, one for the brain segmenter with an
uncertainty-map overlay (the segmenter arrives with BraTS).
"""

from medimg_uq.demo.app import build_demo
from medimg_uq.demo.inference import ISICPredictor, Prediction

__all__ = ["ISICPredictor", "Prediction", "build_demo"]
