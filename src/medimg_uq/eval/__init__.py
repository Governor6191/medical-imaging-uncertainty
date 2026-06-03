"""Evaluation.

Runs a fitted model or ensemble over a dataset and emits a unified report:
accuracy-style metrics plus the full calibration suite, ready for figures.
"""

from medimg_uq.eval.evaluate import (
    InferenceResult,
    evaluate,
    run_inference,
    save_eval_artifacts,
)

__all__ = ["InferenceResult", "evaluate", "run_inference", "save_eval_artifacts"]
