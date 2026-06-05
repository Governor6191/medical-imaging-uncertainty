"""Deploy the BraTS demo to a Hugging Face Space.

Creates a Gradio Space, uploads the app (app_brats.py as app.py), the package
source, the BraTS-specific requirements, and the preloaded sample slices, then
points the Space at the released model repo so it pulls the weights from the Hub
at startup. Authentication comes from the ambient Hub login; no token is read or
stored here.

Usage:
    python scripts/deploy_brats_space.py \
        --space-id Governor6191/brats-brain-tumor-uncertainty \
        --model-repo Governor6191/brats-brain-tumor-uncertainty
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import gradio
from huggingface_hub import HfApi

GITHUB = "https://github.com/Governor6191/medical-imaging-uncertainty"


def space_readme(sdk_version: str) -> str:
    return f"""---
title: BraTS Brain Tumor Uncertainty
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
license: mit
---

# Brain tumor segmentation with per-voxel uncertainty

A 2D-slice U-Net Deep Ensemble for four-class brain tumor segmentation on BraTS
multi-modal MRI. It produces a tumor mask and a per-voxel uncertainty map that
lights up along the tumor boundary. Research demonstration, not a medical device.
Code: {GITHUB}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the BraTS demo to a HF Space.")
    parser.add_argument("--space-id", default="Governor6191/brats-brain-tumor-uncertainty")
    parser.add_argument("--model-repo", default="Governor6191/brats-brain-tumor-uncertainty")
    parser.add_argument("--samples-dir", type=Path, default=Path("demo_assets/brats_samples"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not any(args.samples_dir.glob("*.npz")):
        raise SystemExit(f"no sample slices (*.npz) under {args.samples_dir}")

    api = HfApi()
    api.create_repo(
        args.space_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        shutil.copy("app_brats.py", tmp / "app.py")
        shutil.copy("requirements_brats.txt", tmp / "requirements.txt")
        shutil.copytree("src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(args.samples_dir, tmp / "brats_samples")
        (tmp / "README.md").write_text(space_readme(gradio.__version__), encoding="utf-8")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=args.space_id,
            repo_type="space",
            commit_message="Deploy BraTS calibrated-uncertainty demo",
        )

    # Tell the Space which model repo to pull the ensemble weights from.
    api.add_space_variable(repo_id=args.space_id, key="MEDIMG_HF_REPO", value=args.model_repo)
    print(f"deployed: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
