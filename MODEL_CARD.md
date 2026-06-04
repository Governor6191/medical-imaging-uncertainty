---
license: mit
pipeline_tag: image-classification
tags:
  - medical-imaging
  - dermatology
  - skin-lesion
  - uncertainty-quantification
  - deep-ensembles
  - calibration
  - isic
---

# ISIC skin lesion classifier with calibrated uncertainty

A ResNet-50 Deep Ensemble for binary benign vs malignant classification of dermoscopy images, released as the first application of a modality-agnostic medical-imaging framework whose point is calibrated uncertainty. The code is at [Governor6191/medical-imaging-uncertainty](https://github.com/Governor6191/medical-imaging-uncertainty).

This is a research and teaching artifact, not a medical device. Do not use it for diagnosis.

## What this is

Five ResNet-50 members (ImageNet-pretrained backbone, an explicit dropout-and-linear head) trained from different seeds at 320 pixels. At inference the members are averaged into one prediction, and their disagreement gives an epistemic uncertainty estimate. A single model reports accuracy; this ensemble reports accuracy and how much it can be trusted.

## Results (held-out test set, 450 images, balanced)

| Method | Accuracy | AUROC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Single model | 0.942 | 0.990 | 0.165 | 0.084 | 0.027 |
| Deep Ensemble (K=5) | 0.951 | 0.993 | 0.118 | 0.070 | 0.017 |
| MC Dropout | 0.942 | 0.990 | 0.165 | 0.084 | 0.026 |

The ensemble leads on every metric. Accuracy and AUROC are near the ceiling on this data, so the meaningful gain is calibration: the ensemble cuts NLL by 28 percent and ECE by 37 percent over the single model. MC Dropout tracks the single model because dropout sits only in the head, so its stochastic passes are nearly identical. Full weight diversity across members does far more than head-only dropout.

## Training data

3000 images from the [ISIC Archive](https://www.isic-archive.com), balanced 1500 benign and 1500 malignant by the `metadata.clinical.diagnosis_1` label, split 70/15/15 with a seeded random stratified split so train, val, and test share a distribution.

## Intended use and limitations

- **Intended use:** a demonstration of calibrated uncertainty in medical imaging, for research and education.
- **Not for diagnosis.** It is not validated clinically and is not a medical device.
- **Dataset confounds.** ISIC dermoscopy carries artifacts (rulers, ink markings, colored stickers, vignetting) that a CNN can latch onto. The high absolute AUROC partly reflects an easier, curated, balanced subset and these confounds, not clinical-grade lesion analysis.
- **Distribution.** Performance off this distribution (other dermoscopes, skin tones under-represented in ISIC, non-dermoscopic photos) is unknown and likely worse.

## How to use

```python
from huggingface_hub import hf_hub_download
from PIL import Image
from medimg_uq.demo import ISICPredictor  # pip install from the GitHub repo

paths = [hf_hub_download("Governor6191/isic-skin-lesion-uncertainty", f"member_{i}/best.pt") for i in range(5)]
predictor = ISICPredictor(paths, device="cpu")
result = predictor.predict(Image.open("lesion.jpg"))
print(result.predicted, result.p_malignant, result.band)
```

The repo holds the five member checkpoints under `member_i/best.pt`, the training config, and a `manifest.json` describing the ensemble.

## License and citation

MIT. The dataset is from the ISIC Archive; honor its terms and per-image license. These are research models, not cleared diagnostic tools.
