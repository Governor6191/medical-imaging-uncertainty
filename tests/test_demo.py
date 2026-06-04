"""Tests for the demo's presentation logic (no model checkpoints needed)."""

from __future__ import annotations

import gradio as gr
import pytest

from medimg_uq.demo.app import _readout, build_demo
from medimg_uq.demo.inference import Prediction


def _prediction(disagreement: float, *, predicted: str = "malignant", p_mal: float = 0.8):
    return Prediction(
        probabilities={"benign": 1 - p_mal, "malignant": p_mal},
        predicted=predicted,
        p_malignant=p_mal,
        uncertainty=disagreement * 0.6931,
        disagreement=disagreement,
    )


@pytest.mark.parametrize(
    ("disagreement", "expected"),
    [
        (0.0, "low"),
        (0.09, "low"),
        (0.2, "moderate"),
        (0.29, "moderate"),
        (0.3, "high"),
        (0.9, "high"),
    ],
)
def test_uncertainty_bands(disagreement, expected):
    assert _prediction(disagreement).band == expected


def test_readout_reports_prediction_and_probability():
    text = _readout(_prediction(0.05, predicted="malignant", p_mal=0.91))
    assert "malignant" in text
    assert "0.91" in text
    assert "trustworthy" in text  # low-uncertainty note


def test_readout_warns_when_uncertain():
    text = _readout(_prediction(0.5))
    assert "caution" in text


def test_build_demo_returns_blocks_with_a_stub_predictor():
    class _Stub:
        def predict(self, image):  # pragma: no cover - not called at build time
            return _prediction(0.0)

    assert isinstance(build_demo(_Stub()), gr.Blocks)
