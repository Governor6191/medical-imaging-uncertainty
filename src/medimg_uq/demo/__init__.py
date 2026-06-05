"""Demo shell.

The app that wraps the released checkpoints: one mode for the skin classifier
with a calibrated-confidence readout, one for the brain segmenter with a per-voxel
uncertainty-map overlay.
"""

from medimg_uq.demo.app import build_demo
from medimg_uq.demo.brats_app import build_brats_demo, load_brats_samples
from medimg_uq.demo.brats_inference import BraTSPrediction, BraTSPredictor
from medimg_uq.demo.inference import ISICPredictor, Prediction

__all__ = [
    "BraTSPrediction",
    "BraTSPredictor",
    "ISICPredictor",
    "Prediction",
    "build_brats_demo",
    "build_demo",
    "load_brats_samples",
]
