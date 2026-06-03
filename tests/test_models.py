"""Tests for the model factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from medimg_uq.contract import Task
from medimg_uq.models import build_model
from medimg_uq.uq import MCDropoutSampler


def test_classification_forward_shape():
    model = build_model(
        task=Task.CLASSIFICATION, backbone="resnet18", num_classes=2, pretrained=False
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_head_has_exactly_one_dropout_for_mc_dropout():
    model = build_model(
        task=Task.CLASSIFICATION,
        backbone="resnet18",
        num_classes=2,
        pretrained=False,
        drop_rate=0.4,
    )
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0].p == 0.4
    # The explicit head dropout is what lets MC Dropout accept this model.
    MCDropoutSampler(model, n_samples=2)


def test_in_chans_is_configurable():
    model = build_model(
        task=Task.CLASSIFICATION,
        backbone="resnet18",
        num_classes=3,
        pretrained=False,
        in_chans=1,
    )
    out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 3)


def test_segmentation_model_arrives_with_brats():
    with pytest.raises(NotImplementedError, match="BraTS"):
        build_model(task=Task.SEGMENTATION, backbone="resnet18", num_classes=2, pretrained=False)
