"""Gradio app: BraTS brain tumor segmentation with per-voxel uncertainty.

Pick a held-out MRI slice and the Deep Ensemble segments the tumor and draws how
much its three members disagree at each voxel. The uncertainty is the honest part:
it concentrates along the tumor boundary and through the heterogeneous core, the
voxels where the label is genuinely ambiguous, and falls to near zero in confident
healthy tissue.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from medimg_uq.demo.brats_inference import BraTSPrediction, BraTSPredictor

HEADER = "# Brain tumor segmentation with per-voxel uncertainty"

ABOUT = (
    "A 2D-slice U-Net Deep Ensemble (three members, ResNet-34 encoder) trained on "
    "BraTS 2020 multi-modal MRI for four-class tumor segmentation. It outputs a tumor "
    "mask and a per-voxel epistemic uncertainty map, a measure of where the members "
    "disagree. On the held-out test set the ensemble reached test Dice 0.83 with a "
    "per-voxel expected calibration error of 0.0044, beating the single model on every "
    "metric. Pick a sample slice below; each is a held-out case the model never trained on."
)

DISCLAIMER = (
    "**Research demonstration, not a medical device.** This is a 2D-slice model trained on "
    "a public research dataset, not validated clinically. The sample slices are from the "
    "public BraTS 2020 dataset, preprocessed for display. Do not use this for any real "
    "diagnosis."
)


def load_brats_samples(samples_dir: str | Path) -> list[dict]:
    """Load preloaded demo slices from a folder of ``.npz`` (image + mask) files."""
    folder = Path(samples_dir)
    samples: list[dict] = []
    for npz in sorted(folder.glob("*.npz")):
        data = np.load(npz)
        label = npz.stem.replace("_", " ")
        samples.append(
            {
                "name": label,
                "image": data["image"].astype(np.float32),
                "mask": data["mask"].astype(np.int64) if "mask" in data else None,
            }
        )
    if not samples:
        raise FileNotFoundError(f"no sample slices (*.npz) found under {folder}")
    return samples


def _readout(result: BraTSPrediction) -> str:
    present = ", ".join(result.classes_present) if result.classes_present else "no tumor"
    note = {
        "low": "The three members largely agree on this mask, so it is comparatively trustworthy.",
        "moderate": "The members partly disagree near the boundary; treat the mask with care.",
        "high": "The members disagree substantially; treat this segmentation with caution.",
    }[result.band]
    lines = [
        "### Result",
        f"- Tumor subregions predicted: **{present}**",
        f"- Slice predicted as tumor: **{result.tumor_fraction:.1%}**",
        f"- Ensemble uncertainty: **{result.band}** "
        f"(mean disagreement {result.mean_uncertainty:.0%} of the way to maximum)",
    ]
    if result.dice is not None:
        lines.append(f"- Tumor Dice vs ground truth on this slice: **{result.dice:.2f}**")
    lines.append("")
    lines.append(note)
    return "\n".join(lines)


def build_brats_demo(predictor: BraTSPredictor, samples: list[dict]) -> gr.Blocks:
    """Build the Gradio app around a loaded predictor and a list of sample slices."""
    by_name = {s["name"]: s for s in samples}
    names = list(by_name)

    def segment(name):
        sample = by_name.get(name) or samples[0]
        result = predictor.predict(sample["image"], target=sample.get("mask"))
        return result.figure, _readout(result)

    with gr.Blocks(title="BraTS segmentation with per-voxel uncertainty") as demo:
        gr.Markdown(HEADER)
        gr.Markdown(ABOUT)
        with gr.Row():
            with gr.Column(scale=1):
                pick = gr.Dropdown(names, value=names[0], label="Held-out MRI slice (BraTS 2020)")
                run = gr.Button("Segment", variant="primary")
            with gr.Column(scale=3):
                figure_out = gr.Image(
                    label="MRI, prediction, ground truth, and per-voxel uncertainty"
                )
                readout_out = gr.Markdown()
        run.click(segment, inputs=pick, outputs=[figure_out, readout_out])
        pick.change(segment, inputs=pick, outputs=[figure_out, readout_out])
        demo.load(segment, inputs=pick, outputs=[figure_out, readout_out])
        gr.Markdown(DISCLAIMER)
    return demo
