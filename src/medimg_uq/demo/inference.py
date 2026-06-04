"""Single-image inference for the demo.

Loads the trained ISIC Deep Ensemble and turns one uploaded image into a
calibrated prediction plus an uncertainty readout. The checkpoint location is
configurable so the same code serves a local run or a Hugging Face Space that
pulls weights from the Hub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medimg_uq.contract import Task
from medimg_uq.data.isic import isic_transforms
from medimg_uq.models import build_model
from medimg_uq.uq import EnsembleSampler

CLASS_NAMES = ("benign", "malignant")
_MAX_BINARY_ENTROPY = math.log(2)  # epistemic uncertainty ceiling for two classes


@dataclass
class Prediction:
    """One image's result, ready to render."""

    probabilities: dict[str, float]  # {"benign": p, "malignant": p}
    predicted: str
    p_malignant: float
    uncertainty: float  # epistemic, in nats
    disagreement: float  # epistemic normalized to [0, 1]

    @property
    def band(self) -> str:
        if self.disagreement < 0.10:
            return "low"
        if self.disagreement < 0.30:
            return "moderate"
        return "high"


class ISICPredictor:
    """The ISIC ensemble wrapped for one-image-at-a-time inference."""

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        *,
        backbone: str = "resnet50",
        image_size: int = 320,
        drop_rate: float = 0.3,
        device: str = "cpu",
    ) -> None:
        if not checkpoint_paths:
            raise ValueError("need at least one checkpoint to load the ensemble")
        self.device = torch.device(device)
        self.transform = isic_transforms(image_size, train=False)
        members = []
        for path in checkpoint_paths:
            model = build_model(
                task=Task.CLASSIFICATION,
                backbone=backbone,
                num_classes=2,
                pretrained=False,
                drop_rate=drop_rate,
            )
            state = torch.load(path, map_location=self.device)["model_state"]
            model.load_state_dict(state)
            members.append(model.to(self.device).eval())
        self.sampler = EnsembleSampler(members)

    @classmethod
    def from_dir(cls, ckpt_dir: str | Path, *, n_members: int = 5, **kwargs) -> ISICPredictor:
        """Load member checkpoints from ``ckpt_dir/member_i/best.pt``."""
        paths = [Path(ckpt_dir) / f"member_{i}" / "best.pt" for i in range(n_members)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing ensemble checkpoints: {missing}")
        return cls([str(p) for p in paths], **kwargs)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        arr = np.asarray(image.convert("RGB"))
        tensor = self.transform(image=arr)["image"].unsqueeze(0).to(self.device)
        dist = self.sampler.predict(tensor)
        p_benign = float(dist.mean_probs[0, 0])
        p_malignant = float(dist.mean_probs[0, 1])
        epistemic = float(dist.epistemic[0])
        return Prediction(
            probabilities={"benign": p_benign, "malignant": p_malignant},
            predicted=CLASS_NAMES[int(dist.prediction[0])],
            p_malignant=p_malignant,
            uncertainty=epistemic,
            disagreement=min(epistemic / _MAX_BINARY_ENTROPY, 1.0),
        )
