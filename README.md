# Medical Imaging with Calibrated Uncertainty

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#status)

A modality-agnostic medical-imaging framework whose differentiator is calibrated uncertainty. Most medical-imaging repos report accuracy and stop. This one reports calibration alongside accuracy for every model, because a model that says how sure it is matters more in a clinical setting than one that is only sometimes right and never says so.

The uncertainty method is Deep Ensembles: train K models from different seeds and aggregate. MC Dropout is included as a cheaper comparison so the calibration claim is a measured head-to-head, not a single method asserted on its own.

One core, two applications:

- **Skin lesion classification (ISIC).** Benign vs malignant from dermoscopy images. Reports AUROC and accuracy with ECE, NLL, and a reliability diagram.
- **Brain tumor segmentation (BraTS).** Tumor masks from multi-modal MRI. Reports Dice and IoU with per-voxel calibration and per-voxel uncertainty maps. (Work in progress.)

The model is the table stakes. The calibrated uncertainty and the reusable two-modality framework are the point.

## Design

The shared core is modality-agnostic. The two tasks differ in kind (one label per image vs one label per voxel), so a single trained model cannot do both. What transfers is everything around the model: the uncertainty engine, the calibration and evaluation suite, the training harness, and the demo shell. Task differences live behind a single data-adapter contract and a model factory.

```
src/medimg_uq/
  data/         adapters: a common (image, target, task) contract, ISIC and BraTS loaders
  models/       backbone factory: classification head or U-Net decoder
  uq/           Deep Ensembles engine and MC Dropout sampler (task-agnostic)
  calibration/  ECE, NLL, reliability diagrams, per-voxel calibration, uncertainty maps
  train/        config-driven loop: seeds, checkpointing, logging, ensemble orchestration
  eval/         unified evaluation to a calibration report and figures
  demo/         app shell with two modes
configs/        one config per application and per uncertainty method
tests/          unit and smoke tests
```

Nothing in `uq/`, `calibration/`, `train/`, or `eval/` imports anything ISIC-specific or BraTS-specific. That separation is what makes the one-framework claim true.

## Status

Early build. The shared core and the ISIC application come first; BraTS follows. Numbers and plots land here as each piece is verified. These are research models, not cleared diagnostic tools.

## Setup

```
uv sync --extra dev
```

Pulls the CUDA 12.4 torch wheels (built against a local RTX 4070). Drop `--extra dev` if you only want the runtime, or add `--extra demo` / `--extra logging` for the Gradio demo and Weights and Biases logging.

## License

MIT. See [LICENSE](LICENSE). Dataset licenses and citations are listed per application as each one lands.
