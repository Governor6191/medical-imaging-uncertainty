---
license: mit
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - neuroimaging
  - brain-tumor-segmentation
  - uncertainty-quantification
  - deep-ensembles
  - calibration
  - brats
---

# BraTS brain tumor segmentation with calibrated per-voxel uncertainty

A Deep Ensemble of 2D-slice U-Nets for four-class brain tumor segmentation from multi-modal MRI, released as the second application of a modality-agnostic medical-imaging framework whose point is calibrated uncertainty. The code is at [Governor6191/medical-imaging-uncertainty](https://github.com/Governor6191/medical-imaging-uncertainty).

This is a research and teaching artifact, not a medical device. Do not use it for diagnosis.

## What this is

Three U-Net members (ResNet-34 encoder, four MRI modalities in, four classes out: background plus the three tumor subregions) trained from different seeds on 2D axial slices. At inference the members are averaged into one prediction, and their per-voxel disagreement gives an epistemic uncertainty map. A single model reports a mask; this ensemble reports a mask and how much each voxel can be trusted.

## Results (held-out test split, about 3,560 tumor slices, 178 million voxels)

| Method | Dice | IoU | Voxel acc | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Single model | 0.822 | 0.700 | 0.9926 | 0.040 | 0.0131 | 0.0059 |
| Deep Ensemble (K=3) | 0.828 | 0.708 | 0.9929 | 0.032 | 0.0116 | 0.0044 |

The ensemble leads on every metric, and the calibration gains lead: it cuts per-voxel NLL by 19 percent and ECE by 25 percent over the single model while nudging Dice and IoU up. Dice and IoU are the mean over the three tumor subclasses (background excluded), and the calibration metrics are computed per voxel. The uncertainty concentrates along the tumor boundary and through the heterogeneous core, exactly where the label is genuinely ambiguous.

## Training data

BraTS 2020 (369 patients, 368 used after dropping one case with a misnamed segmentation). Each patient is four co-registered MRI volumes (T1, T1ce, T2, FLAIR) plus an expert segmentation. Volumes are z-scored per modality over the brain, the tumor-bearing axial slices are extracted, and the split is patient-level (70/15/15) so no patient's slices leak across train, val, and test.

## Intended use and limitations

- **Intended use:** a demonstration of calibrated per-voxel uncertainty in medical image segmentation, for research and education.
- **Not for diagnosis.** It is not validated clinically and is not a medical device.
- **2D-slice model.** The Dice here is not comparable to the 3D BraTS leaderboard, which scores whole volumes on the nested whole-tumor, tumor-core, and enhancing-tumor regions. These numbers are the mean over the raw tumor subclasses.
- **Tumor-bearing slices.** Training and evaluation use slices that contain tumor, so this measures segmentation quality given that a tumor is present, not whole-volume screening.
- **Distribution.** Inputs must be BraTS-style preprocessed MRI (four co-registered, skull-stripped, intensity-normalized modalities). Performance off this distribution is unknown.

## How to use

```python
from huggingface_hub import hf_hub_download
import numpy as np
from medimg_uq.demo import BraTSPredictor  # pip install from the GitHub repo

paths = [hf_hub_download("Governor6191/brats-brain-tumor-uncertainty", f"member_{i}/best.pt") for i in range(3)]
predictor = BraTSPredictor(paths, device="cpu")

# image: a (4, H, W) array of z-scored T1, T1ce, T2, FLAIR for one axial slice
result = predictor.predict(image)
result.figure.save("prediction.png")
print(result.classes_present, result.mean_uncertainty)
```

The repo holds the three member checkpoints under `member_i/best.pt`, the training config, and a `manifest.json` describing the ensemble.

## License and citation

MIT. The dataset is the BraTS 2020 challenge data; honor its terms and cite the BraTS references. These are research models, not cleared diagnostic tools.
