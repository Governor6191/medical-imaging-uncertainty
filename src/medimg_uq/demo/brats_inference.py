"""Slice inference for the BraTS demo.

Loads the trained BraTS Deep Ensemble (2D-slice U-Nets) and turns one
four-modality MRI slice into a predicted tumor mask plus a per-voxel uncertainty
map, rendered as the headline four-panel figure. The checkpoint location is
configurable so the same code serves a local run or a Hugging Face Space that
pulls weights from the Hub.

Segmentation does not take an arbitrary uploaded photo the way the classifier
does: the input is four co-registered, skull-stripped, intensity-normalized MRI
volumes. The demo therefore runs on a few preloaded held-out sample slices rather
than free-form uploads.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medimg_uq.calibration import metrics, segmentation_panels
from medimg_uq.contract import Task
from medimg_uq.models import build_model
from medimg_uq.uq import EnsembleSampler

# BraTS labels after the contiguous remap: 0 background, 1 necrotic and
# non-enhancing core, 2 edema, 3 enhancing tumor.
CLASS_NAMES = (
    "background",
    "necrotic / non-enhancing core",
    "edema",
    "enhancing tumor",
)
FLAIR_CHANNEL = 3  # (t1, t1ce, t2, flair); FLAIR reads the tumor well as a backdrop


def _fig_to_image(fig) -> Image.Image:
    """Render a Matplotlib figure to a PIL image for Gradio display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@dataclass
class BraTSPrediction:
    """One slice's result, ready to render."""

    figure: Image.Image  # MRI, prediction, ground truth, uncertainty panels
    classes_present: list[str]  # tumor subregions the ensemble predicted
    tumor_fraction: float  # fraction of the slice predicted as tumor
    mean_uncertainty: float  # epistemic over predicted tumor voxels, normalized to [0, 1]
    dice: float | None  # mean tumor Dice vs ground truth, when a mask is provided

    @property
    def band(self) -> str:
        if self.mean_uncertainty < 0.10:
            return "low"
        if self.mean_uncertainty < 0.30:
            return "moderate"
        return "high"


class BraTSPredictor:
    """The BraTS U-Net ensemble wrapped for one-slice-at-a-time inference."""

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        *,
        backbone: str = "resnet34",
        num_classes: int = 4,
        in_chans: int = 4,
        device: str = "cpu",
    ) -> None:
        if not checkpoint_paths:
            raise ValueError("need at least one checkpoint to load the ensemble")
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.in_chans = in_chans
        self._max_entropy = math.log(num_classes)  # epistemic ceiling for the normalization
        members = []
        for path in checkpoint_paths:
            model = build_model(
                task=Task.SEGMENTATION,
                backbone=backbone,
                num_classes=num_classes,
                pretrained=False,
                in_chans=in_chans,
            )
            state = torch.load(path, map_location=self.device)["model_state"]
            model.load_state_dict(state)
            members.append(model.to(self.device).eval())
        self.sampler = EnsembleSampler(members)

    @classmethod
    def from_dir(cls, ckpt_dir: str | Path, *, n_members: int = 3, **kwargs) -> BraTSPredictor:
        """Load member checkpoints from ``ckpt_dir/member_i/best.pt``."""
        paths = [Path(ckpt_dir) / f"member_{i}" / "best.pt" for i in range(n_members)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing ensemble checkpoints: {missing}")
        return cls([str(p) for p in paths], **kwargs)

    @torch.no_grad()
    def predict(self, image, target=None) -> BraTSPrediction:
        """Segment one ``(C, H, W)`` slice and report its per-voxel uncertainty."""
        img = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        if img.ndim != 3 or img.shape[0] != self.in_chans:
            raise ValueError(f"expected a ({self.in_chans}, H, W) slice, got {tuple(img.shape)}")
        dist = self.sampler.predict(img.unsqueeze(0).to(self.device))
        prediction = dist.prediction[0].cpu()  # (H, W)
        uncertainty = dist.epistemic[0].cpu()  # (H, W)
        probs = dist.mean_probs.cpu()  # (1, C, H, W)

        tgt = None if target is None else torch.as_tensor(np.asarray(target), dtype=torch.long)
        fig = segmentation_panels(
            image=img[FLAIR_CHANNEL],
            prediction=prediction,
            uncertainty=uncertainty,
            target=tgt,
            title="MRI (FLAIR), predicted tumor, ground truth, and per-voxel uncertainty",
        )

        tumor = prediction > 0
        present = [
            CLASS_NAMES[c] for c in range(1, self.num_classes) if bool((prediction == c).any())
        ]
        unc_region = uncertainty[tumor] if bool(tumor.any()) else uncertainty
        mean_unc = float(unc_region.mean())
        dice = None
        if tgt is not None:
            dice = float(metrics.dice_score(probs, tgt.unsqueeze(0), self.num_classes))
        return BraTSPrediction(
            figure=_fig_to_image(fig),
            classes_present=present,
            tumor_fraction=float(tumor.float().mean()),
            mean_uncertainty=min(mean_unc / self._max_entropy, 1.0),
            dice=dice,
        )
