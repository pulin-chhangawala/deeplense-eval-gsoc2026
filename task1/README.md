# Task 1: Multi-Class Gravitational Lens Classification

Classify strong lensing images into three categories based on what kind of dark matter substructure is present (or absent) in the lens:

- **No substructure**: a smooth dark matter halo with no clumps
- **Sphere**: a CDM (Cold Dark Matter) subhalo, which shows up as a compact point-like distortion
- **Vortex**: a WDM (Warm Dark Matter) axion substructure, which is more extended and diffuse

Dataset: 30,000 training / 7,500 validation grayscale images at 150x150 pixels.

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9778** |
| Test Accuracy | 89.7% |
| AUC for no substructure | 0.9842 |
| AUC for sphere | 0.9641 |
| AUC for vortex | 0.9852 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**Model:** EfficientNet-B3 pretrained on ImageNet, with the classifier head replaced by a single linear layer outputting 3 class scores. The single-channel input is replicated to 3 channels before passing through the backbone; ImageNet normalization is applied internally.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4 (the pretrained backbone needs a gentler nudge than the fresh head)
- Label smoothing cross-entropy (smoothing=0.05): softens the one-hot targets slightly to prevent overconfidence
- Mixup augmentation (alpha=0.2): blends pairs of training images and their labels to improve generalization
- WeightedRandomSampler: ensures each mini-batch sees roughly equal class representation
- OneCycleLR scheduler, stepped once per batch
- Mixed precision (AMP) for faster training
- Test-time augmentation: averages predictions across 6 flips/rotations at inference
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
