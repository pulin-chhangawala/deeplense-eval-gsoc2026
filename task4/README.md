# Task 4 — Neural Operator Classifier (Fourier Neural Operator)

Classify strong lensing images using a Fourier Neural Operator (FNO) trained entirely from scratch — no pretrained weights. The FNO operates in the frequency domain, giving each layer a global receptive field from the first layer onward.

Dataset: same 30,000/7,500 split as Task 1 (shared `task1_data/`).

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9488** |
| Test Accuracy | 83.0% |
| AUC — no substructure | 0.9699 |
| AUC — sphere | 0.9289 |
| AUC — vortex | 0.9478 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**Architecture:** `FNOClassifier` — a lift convolution projects the single input channel to a hidden dimension, followed by N FNO blocks, then a pointwise projection down to a feature vector fed into the classification head.

Each FNO block:
1. `SpectralConv2d` — applies `rfft2`, multiplies the lowest `modes` Fourier coefficients by learned complex weights, then `irfft2` back. Weights are stored as separate real/imaginary `nn.Parameter` tensors.
2. Pointwise 1x1 bypass convolution (residual path)
3. `InstanceNorm2d` + GELU activation

Configuration used: hidden=64, modes=16, 4 layers (~8.4M parameters).

Note: cuFFT requires float32 for non-power-of-2 spatial sizes (150x150), so AMP is disabled for this model. Training is still fast on a modern GPU.

**Training details:**
- Single learning rate 1e-3 with OneCycleLR (stepped per batch)
- WeightedRandomSampler for class balance
- Label smoothing cross-entropy (smoothing=0.05)
- Test-time augmentation: 6 flips/rotations
- 60 epochs, batch size 64

---

## Reproducing

```bash
python task4/train.py \
  --data-dir task1_data \
  --epochs 60 \
  --batch-size 64 \
  --lr 1e-3 \
  --hidden 64 \
  --modes 16 \
  --n-layers 4 \
  --smoothing 0.05 \
  --tta \
  --save-dir task4/checkpoints
```

The notebook `Task4_Neural_Operators.ipynb` walks through the FNO architecture, the spectral convolution math, and compares the ROC curve to the Task 1 EfficientNet baseline.
