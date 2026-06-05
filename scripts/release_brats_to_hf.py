"""Release the BraTS Deep Ensemble to the Hugging Face Hub.

Uploads the three member checkpoints, the training config, a manifest describing
the ensemble, and the model card. Authentication comes from the ambient Hub login
(`hf auth login` or the HF_TOKEN environment variable); no token is read or stored
by this script.

Usage:
    python scripts/release_brats_to_hf.py --repo-id Governor6191/brats-brain-tumor-uncertainty
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

GITHUB = "https://github.com/Governor6191/medical-imaging-uncertainty"


def build_manifest(comparison_path: Path, n_members: int) -> dict:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    return {
        "framework": "medimg_uq",
        "github": GITHUB,
        "task": "brain tumor segmentation (4 classes, multi-modal MRI)",
        "backbone": "unet-resnet34",
        "modalities": ["t1", "t1ce", "t2", "flair"],
        "image_size": 224,
        "num_members": n_members,
        "members": [f"member_{i}/best.pt" for i in range(n_members)],
        "test_metrics": comparison.get("test", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Release the BraTS ensemble to the HF Hub.")
    parser.add_argument("--repo-id", default="Governor6191/brats-brain-tumor-uncertainty")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/brats_unet"))
    parser.add_argument("--config", type=Path, default=Path("configs/brats_unet.yaml"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("outputs/brats_unet/comparison.json")
    )
    parser.add_argument("--card", type=Path, default=Path("MODEL_CARD_BRATS.md"))
    parser.add_argument("--n-members", type=int, default=3)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoints = [args.ckpt_dir / f"member_{i}" / "best.pt" for i in range(args.n_members)]
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for i, ckpt in enumerate(checkpoints):
            member_dir = tmp / f"member_{i}"
            member_dir.mkdir(parents=True)
            shutil.copy(ckpt, member_dir / "best.pt")
        shutil.copy(args.config, tmp / "config.yaml")
        shutil.copy(args.card, tmp / "README.md")
        manifest = build_manifest(args.comparison, args.n_members)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add BraTS Deep Ensemble checkpoints, config, manifest, and model card",
        )

    print(f"released: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
