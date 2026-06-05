"""Tests for the BraTS demo predictor and sample loader (tiny random-init model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from medimg_uq.contract import Task
from medimg_uq.demo import BraTSPredictor, load_brats_samples
from medimg_uq.models import build_model


def _save_member(path, *, in_chans=4, num_classes=4):
    model = build_model(
        task=Task.SEGMENTATION,
        backbone="resnet34",
        num_classes=num_classes,
        pretrained=False,
        in_chans=in_chans,
    )
    torch.save({"model_state": model.state_dict()}, path)


def test_brats_predictor_segments_a_slice(tmp_path):
    ckpt = tmp_path / "best.pt"
    _save_member(ckpt)
    predictor = BraTSPredictor([ckpt], num_classes=4, in_chans=4, device="cpu")

    image = np.random.default_rng(0).standard_normal((4, 32, 32)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[8:16, 8:16] = 2  # a patch of edema so Dice has something to score

    result = predictor.predict(image, target=mask)
    assert isinstance(result.figure, Image.Image)
    assert 0.0 <= result.tumor_fraction <= 1.0
    assert 0.0 <= result.mean_uncertainty <= 1.0
    assert result.dice is not None and 0.0 <= result.dice <= 1.0
    assert result.band in {"low", "moderate", "high"}


def test_brats_predictor_rejects_wrong_channels(tmp_path):
    ckpt = tmp_path / "best.pt"
    _save_member(ckpt)
    predictor = BraTSPredictor([ckpt], num_classes=4, in_chans=4, device="cpu")
    with pytest.raises(ValueError, match="expected a"):
        predictor.predict(np.zeros((3, 32, 32), dtype=np.float32))


def test_load_brats_samples_reads_npz(tmp_path):
    np.savez(
        tmp_path / "case_1.npz",
        image=np.zeros((4, 32, 32), dtype=np.float16),
        mask=np.zeros((32, 32), dtype=np.uint8),
    )
    samples = load_brats_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0]["name"] == "case 1"
    assert samples[0]["image"].shape == (4, 32, 32)


def test_load_brats_samples_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="no sample slices"):
        load_brats_samples(tmp_path)
