# Task 5: Lens Finding with Class Imbalance

Binary classification: does this image contain a strong gravitational lens or not? The dataset is severely imbalanced, with roughly 1 lens for every 100 non-lens images. If you train naively on this, the model quickly learns to just predict "not a lens" every time and still gets 99% accuracy while being completely useless.

---

## Result

| Metric | Value |
|--------|-------|
| Test AUROC | **0.9852** |
| Test AUPRC | 0.7306 |
| F1 @ optimal threshold | 0.661 |
| True Positives | 113 |
| False Positives | 34 |
| False Negatives | 82 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**Model:** EfficientNet-B3 pretrained on ImageNet, with a binary classification head. Input images are 3-channel (the dataset provides color images, unlike Task 1).

**Handling imbalance: two complementary strategies**

1. **Focal loss** (alpha=0.25, gamma=2.5): a modification to standard cross-entropy that down-weights the gradient contribution from easy negatives (images the model already correctly identifies as non-lenses). This forces the model to focus its learning on the rare positive examples that are actually hard to get right.

2. **WeightedRandomSampler**: oversamples lenses during batch construction so each mini-batch has a more balanced mix of lenses and non-lenses. This stabilizes early training before focal loss has had a chance to take effect.

**Threshold selection:** The standard 0.5 probability cutoff makes no sense when classes are 100:1 imbalanced. Instead, the optimal threshold is chosen by sweeping the full ROC curve on the validation set and picking the point that maximizes F1 score.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4
- OneCycleLR scheduler, stepped once per batch
- Mixed precision (AMP)
- Test-time augmentation across 6 flips/rotations
- 30 epochs, batch size 32

---

## Reproducing

```bash
python task5/train.py \
  --data-dir task2_data \
  --epochs 30 \
  --batch-size 32 \
  --lr-backbone 5e-5 \
  --lr-head 3e-4 \
  --focal-alpha 0.25 \
  --focal-gamma 2.5 \
  --tta \
  --save-dir task5/checkpoints
```

The notebook `Task5_Lens_Finding.ipynb` covers the imbalance problem, explains focal loss vs weighted sampling, and shows precision-recall and ROC curves with threshold analysis.
