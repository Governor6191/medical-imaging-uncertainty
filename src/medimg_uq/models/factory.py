"""Model factory.

Builds the model the task needs from a small config. Classification uses a timm
backbone as a feature extractor with an explicit dropout-plus-linear head. The
head dropout is a real ``nn.Dropout`` module, not timm's functional head dropout,
so MC Dropout can switch it on at inference without dragging batch norm into
training mode. Segmentation uses a U-Net from segmentation_models_pytorch with a
configurable encoder; its ensemble produces per-voxel uncertainty maps.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from medimg_uq.contract import Task


class ClassificationModel(nn.Module):
    """A timm backbone plus a dropout-and-linear classification head.

    Args:
        backbone: Any timm model name (for example ``resnet50``, ``tf_efficientnet_b0``,
            ``vit_small_patch16_224``).
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained backbone weights.
        drop_rate: Dropout probability in the head. Must be > 0 for MC Dropout to
            do anything.
        in_chans: Input channels (3 for RGB dermoscopy).
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        *,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # strip timm's head; we add our own
            global_pool="avg",
            in_chans=in_chans,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(
    *,
    task: Task,
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    in_chans: int = 3,
) -> nn.Module:
    """Build a model for ``task``.

    Returns logits: ``(N, num_classes)`` for classification, and (once the BraTS
    application lands) ``(N, num_classes, H, W)`` for segmentation.
    """
    if task is Task.CLASSIFICATION:
        return ClassificationModel(
            backbone,
            num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            in_chans=in_chans,
        )
    if task is Task.SEGMENTATION:
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_chans,
            classes=num_classes,
        )
    raise ValueError(f"unknown task: {task}")
