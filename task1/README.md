# Task 1 — Multi-Class Gravitational Lens Classification

Classify strong lensing images into three substructure categories:
- **No substructure** — smooth dark matter halo
- **Sphere** — CDM subhalo (point-like substructure)
- **Vortex** — WDM axion substructure (extended)

Dataset: 30,000 training / 7,500 validation grayscale images (150x150).

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9778** |
| Test Accuracy | 89.7% |
| AUC — no substructure | 0.9842 |
| AUC — sphere | 0.9641 |
| AUC — vortex | 0.9852 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**Model:** EfficientNet-B3 pretrained on ImageNet, with the classifier head replaced by a single linear layer (3-way output). The single-channel input is replicated to 3 channels before passing through the backbone; ImageNet normalization is applied internally.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4
- Label smoothing cross-entropy (smoothing=0.05)
- Mixup augmentation (alpha=0.2)
- WeightedRandomSampler for class balance
- OneCycleLR scheduler (stepped per batch)
- Mixed precision (AMP)
- Test-time augmentation: 6 flips/rotations averaged at inference
- 50 epochs, batch size 64

---

## Reproducing

```bash
python task1/train.py \
  --data-dir task1_data \
  --epochs 50 \
  --batch-size 64 \
  --lr-backbone 5e-5 \
  --lr-head 3e-4 \
  --mixup-alpha 0.2 \
  --smoothing 0.05 \
  --tta \
  --save-dir task1/checkpoints
```

The notebook `Task1_MultiClass_Classification.ipynb` walks through the full pipeline with inline training code, loss curves, and a per-class ROC plot.
