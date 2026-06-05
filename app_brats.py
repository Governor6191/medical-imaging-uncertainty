"""Hugging Face Spaces entry point for the BraTS demo.

Spaces runs this file (uploaded as app.py). It resolves the ensemble checkpoints
(from a Hugging Face model repo when MEDIMG_HF_REPO is set, else a local folder),
loads the preloaded sample slices, builds the predictor, and launches the app.

Environment:
    MEDIMG_HF_REPO        Hub model repo to download member_i/best.pt from (optional)
    MEDIMG_CKPT_DIR       local checkpoint dir (default checkpoints/brats_unet)
    MEDIMG_BRATS_SAMPLES  folder of sample .npz slices (default brats_samples)
    MEDIMG_DEVICE         cpu or cuda (default cpu)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# On a Hugging Face Space the package is not pip-installed, so make the src layout
# importable. Locally this is a harmless no-op since the package is already installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medimg_uq.demo import BraTSPredictor, build_brats_demo, load_brats_samples  # noqa: E402

N_MEMBERS = 3


def _resolve_checkpoints() -> list[str]:
    repo = os.environ.get("MEDIMG_HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        return [hf_hub_download(repo, f"member_{i}/best.pt") for i in range(N_MEMBERS)]
    ckpt_dir = os.environ.get("MEDIMG_CKPT_DIR", "checkpoints/brats_unet")
    return [str(Path(ckpt_dir) / f"member_{i}" / "best.pt") for i in range(N_MEMBERS)]


samples_dir = os.environ.get("MEDIMG_BRATS_SAMPLES", "brats_samples")
predictor = BraTSPredictor(_resolve_checkpoints(), device=os.environ.get("MEDIMG_DEVICE", "cpu"))
demo = build_brats_demo(predictor, load_brats_samples(samples_dir))

if __name__ == "__main__":
    demo.launch()
