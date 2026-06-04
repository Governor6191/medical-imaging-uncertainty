"""Hugging Face Spaces entry point for the ISIC demo.

Spaces runs this file. It resolves the ensemble checkpoints (from a local folder
by default, or from a Hugging Face model repo when MEDIMG_HF_REPO is set), builds
the predictor, and launches the Gradio app.

Environment:
    MEDIMG_HF_REPO   Hub model repo to download member_i/best.pt from (optional)
    MEDIMG_CKPT_DIR  local checkpoint dir (default checkpoints/isic_resnet50)
    MEDIMG_DEVICE    cpu or cuda (default cpu)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# On a Hugging Face Space the package is not pip-installed, so make the src layout
# importable. Locally this is a harmless no-op since the package is already installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medimg_uq.demo import ISICPredictor, build_demo  # noqa: E402

N_MEMBERS = 5


def _resolve_checkpoints() -> list[str]:
    repo = os.environ.get("MEDIMG_HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        return [hf_hub_download(repo, f"member_{i}/best.pt") for i in range(N_MEMBERS)]
    ckpt_dir = os.environ.get("MEDIMG_CKPT_DIR", "checkpoints/isic_resnet50")
    return [str(Path(ckpt_dir) / f"member_{i}" / "best.pt") for i in range(N_MEMBERS)]


predictor = ISICPredictor(_resolve_checkpoints(), device=os.environ.get("MEDIMG_DEVICE", "cpu"))
demo = build_demo(predictor)

if __name__ == "__main__":
    demo.launch()
