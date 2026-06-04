# Medical Imaging with Calibrated Uncertainty

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#status)

A modality-agnostic medical-imaging framework whose differentiator is calibrated uncertainty. Most medical-imaging repos report accuracy and stop. This one reports calibration alongside accuracy for every model, because a model that says how sure it is matters more in a clinical setting than one that is only sometimes right and never says so.

The uncertainty method is Deep Ensembles: train K models from different seeds and aggregate. MC Dropout is included as a cheaper comparison, so the calibration claim is a measured head-to-head rather than a single method asserted on its own. Evidential deep learning is left as future work.

One core, two applications:

- **Skin lesion classification (ISIC).** Benign vs malignant from dermoscopy images. Done.
- **Brain tumor segmentation (BraTS).** Tumor masks from multi-modal MRI, with per-voxel uncertainty maps. Work in progress.

The model is the table stakes. The calibrated uncertainty and the reusable two-modality framework are the point.

## Design

The shared core is modality-agnostic. The two tasks differ in kind (one label per image vs one label per voxel), so a single trained model cannot do both. What transfers is everything around the model: the uncertainty engine, the calibration and evaluation suite, the training harness, and the demo shell. Task differences live behind a single data-adapter contract and a model factory.

```
src/medimg_uq/
  data/         adapters: a common (image, target, task) contract, ISIC and BraTS loaders
  models/       backbone factory: classification head or U-Net decoder
  uq/           Deep Ensembles engine and MC Dropout sampler (task-agnostic)
  calibration/  ECE, NLL, Brier, reliability diagrams, per-voxel calibration
  train/        config-driven loop: seeds, checkpointing, ensemble orchestration
  eval/         unified evaluation to a calibration report and figures
  demo/         app shell with two modes
configs/        one config per application
scripts/        download, train, and ensemble entry points
tests/          unit and smoke tests
```

Nothing in `uq/`, `calibration/`, `train/`, or `eval/` imports anything ISIC-specific or BraTS-specific. That separation is what makes the one-framework claim true. Every uncertainty method, single model included, runs through the same sampler interface, so a single model is just an ensemble of one and the numbers are directly comparable.

## Application 1: skin lesion classification (ISIC)

Binary benign vs malignant from dermoscopy images.

**Data.** 3000 images pulled from the [ISIC Archive](https://www.isic-archive.com) API, balanced 1500 benign and 1500 malignant, using the `metadata.clinical.diagnosis_1` benign/malignant label. Split into train, val, and test (70/15/15) with a seeded random stratified split, so the three sets are drawn from the same distribution. See `scripts/download_isic.py`.

**Model.** A ResNet-50 backbone (timm, ImageNet-pretrained) used as a feature extractor with an explicit dropout-plus-linear head, trained at 320 pixels. The head dropout is a real module rather than functional dropout, so MC Dropout can switch it on at inference without disturbing batch norm. The Deep Ensemble is five members trained from different seeds.

**Results (held-out test set, 450 images).**

| Method | Accuracy | AUROC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Single model | 0.942 | 0.990 | 0.165 | 0.084 | 0.027 |
| **Deep Ensemble (K=5)** | **0.951** | **0.993** | **0.118** | **0.070** | **0.017** |
| MC Dropout | 0.942 | 0.990 | 0.165 | 0.084 | 0.026 |

The Deep Ensemble is better on every metric. The accuracy and AUROC gains are small because both are near the ceiling on this data, but the calibration gains are real: the ensemble cuts NLL by 28 percent and ECE by 37 percent over the single model. That is the headline. A well-ranked model can still be overconfident, and ensembling is what fixes the confidence, not just the ordering.

MC Dropout barely moves off the single model. That is expected and worth stating plainly: dropout sits only in the classification head here, so the stochastic passes are nearly identical. Full weight diversity across independently trained members does far more than head-only dropout. If you want MC Dropout to compete, you need dropout deeper in the network.

**Calibration.** The reliability diagrams show the difference. Bars below the diagonal are overconfident bins (accuracy lags confidence).

![Single model reliability](docs/figures/isic_single_reliability.png)
![Deep Ensemble reliability](docs/figures/isic_ensemble_reliability.png)

*Left: single model, test ECE 0.027. Right: Deep Ensemble, test ECE 0.017. The ensemble's bars sit closer to the diagonal across the mid-confidence range.*

**Honest caveats.**

- This is a balanced 3000-image subset drawn from the earlier, well-curated part of the archive, so it is cleaner and easier than the full ISIC collection. The high absolute AUROC reflects that, not clinical-grade performance.
- ISIC dermoscopy has known confounds (rulers, ink markings, dark vignetting) that a CNN can latch onto. These numbers should be read as a calibration demonstration, not a diagnostic claim.
- The maximum calibration error (MCE) is high for both models, but it is driven by one sparse low-confidence bin with a couple of samples. ECE, which is count-weighted, is the calibration metric to trust here.

**Reproduce.**

```
python scripts/download_isic.py --out data/isic --per-class 1500 --size full
python scripts/train_ensemble.py --config configs/isic_resnet50.yaml --data-root data/isic
```

The second command trains the five members, then scores the single model, the Deep Ensemble, and MC Dropout on val and test, writing a `comparison.json` and reliability diagrams per method.

## Application 2: brain tumor segmentation (BraTS)

Planned, not yet built. BraTS is multi-modal MRI (T1, T1ce, T2, FLAIR) and dense per-voxel segmentation. It reuses the shared core with a 2D-slice U-Net, a Dice plus cross-entropy loss, and segmentation metrics (Dice, IoU). The uncertainty story is the per-voxel map: the model should be least certain at tumor boundaries. Access needs a data-use agreement, so that registration comes first on this leg.

## Status

ISIC is done. BraTS is next. These are research models, not cleared diagnostic tools.

## Setup

```
uv sync --extra dev
```

Pulls the CUDA 12.4 torch wheels (built against a local RTX 4070). Drop `--extra dev` for runtime only, or add `--extra demo` / `--extra logging` for the Gradio demo and Weights and Biases logging.

Run the tests with `pytest`.

## Calibration suite

Every model reports calibration next to accuracy. For classification that is ECE, NLL, Brier, a reliability diagram, plus AUROC and accuracy. For segmentation it is the same metrics computed per voxel, plus the uncertainty map. The metric library takes probabilities and targets and knows nothing about the model or task that produced them, so the exact same code scores a classifier and a segmenter.

## Data and license

Both datasets are public and de-identified, so no IRB is required. The ISIC Archive is openly accessible; honor its terms and the per-image license. BraTS requires a data-use agreement, honored on that leg. Code is MIT, see [LICENSE](LICENSE).
