# Calibrated uncertainty for medical imaging, and why accuracy alone stopped convincing me

Most medical imaging projects report one number: accuracy, or AUROC if they're being careful. I think that's the wrong place to stop. A model that's right 95 percent of the time but has no idea which 5 percent it got wrong is hard to trust with anything that matters. What you actually want is a model that tells you when it isn't sure.

So I built a small framework around that idea. The rule is simple: every model reports calibration next to accuracy, every time. Then I proved it on skin lesion classification, with brain tumor segmentation coming next.

## The setup

The data is ISIC dermoscopy images, framed as benign vs malignant. The model is a ResNet-50. Nothing exotic. The interesting part is what sits around the model.

For uncertainty I used Deep Ensembles: train five copies of the model from different random seeds, then average their predictions. When the five agree, you can lean on the answer. When they split, the model is telling you it doesn't know, and that disagreement is a signal you can actually use. I also included MC Dropout as a cheaper comparison, so the calibration claim is a measured head-to-head instead of one method asserted on its own.

To measure calibration I used expected calibration error (ECE) and negative log likelihood (NLL) next to the usual accuracy and AUROC. ECE asks a blunt question: when the model says it's 80 percent confident, is it actually right 80 percent of the time?

## What happened

On the held-out test set:

| Method | Accuracy | AUROC | NLL | ECE |
|---|---|---|---|---|
| Single model | 0.942 | 0.990 | 0.165 | 0.027 |
| Deep Ensemble | 0.951 | 0.993 | 0.118 | 0.017 |
| MC Dropout | 0.942 | 0.990 | 0.165 | 0.026 |

Accuracy and AUROC are near the ceiling, so they barely move. Calibration is where the ensemble earns its keep: NLL drops 28 percent and ECE drops 37 percent. The single model is already a decent ranker, but it's overconfident, and ensembling is what fixes the confidence, not the ordering.

## The parts I didn't expect, and didn't hide

Three things are worth saying plainly.

MC Dropout barely moved the needle. That surprised me until I looked at where the dropout actually lived: only in the classification head. With dropout that shallow, the stochastic passes come out nearly identical, so there's almost no diversity to read uncertainty from. Full weight diversity across independently trained members does far more than head-only dropout. That isn't a failure of the experiment, it's the experiment telling me something true.

The test numbers were near chance, until I found my own bug. My first split assigned train, val, and test by position in the download, and ISIC hands back images in ID order, which groups acquisition eras together. So I was training on one era and testing on another, and the model looked far worse than it was. A seeded random stratified split fixed it. The lesson stuck: a quiet bug in your evaluation will lie to you in whichever direction you happen not to be watching.

The 0.99 AUROC is not a clinical result, and I want to be clear about that. The data is a clean, balanced slice of ISIC, and dermoscopy images carry confounds a CNN will happily exploit: rulers, ink, colored stickers. You can see it in the worked examples, where the confident benign cases have stickers and the uncertain ones are clean dark lesions. This is a calibration demonstration, not a diagnosis.

## Why a framework, not a model

The uncertainty engine, the calibration suite, the training loop: none of it knows or cares whether it's looking at a skin photo or a brain scan. The task-specific parts live behind one data contract and one model factory. That's on purpose. The next application is BraTS brain tumor segmentation, where the same ensemble produces a per-voxel uncertainty map that should light up along tumor boundaries. Same core, different head.

That's the part I'd defend in a review: not that I trained a CNN on medical images, which is a weekend, but that the calibration rigor and the reusable two-modality design are the contribution.

## Try it

- Demo: https://huggingface.co/spaces/Governor6191/isic-skin-lesion-uncertainty
- Code: https://github.com/Governor6191/medical-imaging-uncertainty
- Weights: https://huggingface.co/Governor6191/isic-skin-lesion-uncertainty

These are research models, not medical devices. If you're worried about a lesion, see a dermatologist.
