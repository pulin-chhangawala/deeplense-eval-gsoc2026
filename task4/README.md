# Task 4: Neural Operator Classifier (Fourier Neural Operator)

Classify strong lensing images using a Fourier Neural Operator (FNO) trained entirely from scratch, with no pretrained weights. Unlike a standard CNN that looks at small patches of the image, the FNO works in the frequency domain, which means every layer can see the entire image at once (a global receptive field from layer one).

Dataset: same 30,000/7,500 split as Task 1 (shared `task1_data/`).

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9488** |
| Test Accuracy | 83.0% |
| AUC for no substructure | 0.9699 |
| AUC for sphere | 0.9289 |
| AUC for vortex | 0.9478 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**Architecture:** `FNOClassifier`

A 1x1 "lift" convolution first expands the single input channel to a wider hidden dimension. That representation then passes through N FNO blocks, a 1x1 "project" convolution squeezes it back down, and a linear classification head outputs the class scores.

Each FNO block does three things:
1. **`SpectralConv2d`**: converts the feature map to the frequency domain with `rfft2`, multiplies the lowest-frequency components by learned complex weights, then converts back with `irfft2`. This is where the global mixing happens. Weights are stored as separate real and imaginary `nn.Parameter` tensors so PyTorch can differentiate through them normally.
2. **Pointwise 1x1 bypass convolution**: a standard residual shortcut that keeps local information flowing.
3. **`InstanceNorm2d` + GELU**: normalization and a smooth nonlinearity.

Configuration: hidden=64, modes=16 (number of Fourier modes kept), 4 layers, giving about 8.4M parameters total.

> Note: cuFFT (the GPU FFT library) only supports float16 for image sizes that are powers of 2. Since these images are 150x150, AMP (mixed precision) is turned off for this model. Training is still fast on a modern GPU.

**Training details:**
- Single learning rate 1e-3 with OneCycleLR, stepped once per batch
- WeightedRandomSampler for class balance
- Label smoothing cross-entropy (smoothing=0.05)
- Test-time augmentation across 6 flips/rotations
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

The notebook `Task4_Neural_Operators.ipynb` walks through the FNO architecture, explains the spectral convolution math in plain terms, and compares the ROC curve to the Task 1 EfficientNet baseline.
