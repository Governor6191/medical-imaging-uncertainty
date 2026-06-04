"""Gradio app: ISIC skin lesion classifier with calibrated uncertainty.

Drop in a dermoscopy image, get the calibrated benign/malignant probabilities
from the Deep Ensemble plus how much the five members disagree. The disagreement
is the honest part: when the members split, the model is telling you it does not
know, which is exactly what calibrated uncertainty is for.
"""

from __future__ import annotations

import gradio as gr

from medimg_uq.demo.inference import ISICPredictor, Prediction

HEADER = "# Skin lesion classifier with calibrated uncertainty"

DISCLAIMER = (
    "**Research demonstration, not a medical device.** This model was trained on a "
    "clean, balanced slice of the ISIC archive and is known to lean partly on dataset "
    "artifacts (such as colored stickers and rulers in some images). Do not use it for "
    "any real diagnosis. If you are worried about a lesion, see a dermatologist."
)

ABOUT = (
    "A ResNet-50 Deep Ensemble (five members) trained on ISIC dermoscopy images for "
    "binary benign vs malignant. It reports calibrated probabilities and the ensemble "
    "disagreement, a measure of how uncertain it is. On the held-out test set the "
    "ensemble reached AUROC 0.99 with an expected calibration error of 0.017."
)


def _readout(result: Prediction) -> str:
    if result.band == "low":
        note = "The five members largely agree, so this prediction is comparatively trustworthy."
    else:
        note = "The members disagree, so treat this prediction with caution."
    return (
        f"### Prediction: {result.predicted}\n"
        f"- Probability of malignant: **{result.p_malignant:.2f}**\n"
        f"- Ensemble uncertainty: **{result.band}** "
        f"(members disagree {result.disagreement:.0%} of the way to maximum)\n\n"
        f"{note}"
    )


def build_demo(predictor: ISICPredictor) -> gr.Blocks:
    """Build the Gradio app around a loaded predictor."""

    def classify(image):
        if image is None:
            return {}, "Upload a dermoscopy image to classify."
        result = predictor.predict(image)
        return result.probabilities, _readout(result)

    with gr.Blocks(title="ISIC classifier with calibrated uncertainty") as demo:
        gr.Markdown(HEADER)
        gr.Markdown(ABOUT)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Dermoscopy image")
                run = gr.Button("Classify", variant="primary")
            with gr.Column():
                probs_out = gr.Label(num_top_classes=2, label="Calibrated probabilities")
                readout_out = gr.Markdown()
        run.click(classify, inputs=image_in, outputs=[probs_out, readout_out])
        image_in.upload(classify, inputs=image_in, outputs=[probs_out, readout_out])
        gr.Markdown(DISCLAIMER)
    return demo
