# Task 5 — Lens Finding with Class Imbalance

Binary classification: does this image contain a strong gravitational lens or not? The dataset is severely imbalanced — roughly 1 lens per 100 non-lens images — making naive training collapse to predicting the majority class.

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

**Model:** EfficientNet-B3 pretrained on ImageNet, with a binary classification head. Input images are 3-channel (the dataset provides color images unlike Task 1).

**Handling imbalance — two complementary strategies:**

1. **Focal loss** (alpha=0.25, gamma=2.5): down-weights the loss contribution from easy negatives so the model focuses on hard positives. This prevents the gradient from being dominated by the overwhelming majority class.

2. **WeightedRandomSampler**: oversamples the minority class during batch construction so each mini-batch has a more balanced mix, stabilizing early training.

**Threshold selection:** The standard 0.5 probability cutoff is inappropriate for imbalanced data. The optimal threshold is chosen by maximizing F1 on the validation set across the full ROC curve.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4
- OneCycleLR scheduler (stepped per batch)
- Mixed precision (AMP)
- Test-time augmentation: 6 flips/rotations
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
