# Task 7: Physics-Guided ML / PINN

Classify strong lensing images using a network that does two things at once: it predicts the class label, and it also tries to reconstruct what the lensed image should look like given the physics of gravitational lensing. The reconstruction error gets added to the training loss, so the physics acts as a built-in regularizer that nudges the model toward representations that actually make sense in the real world.

Dataset: same 30,000/7,500 split as Task 1 (shared `task1_data/`).

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9961** |
| Test Accuracy | 96.7% |
| AUC for no substructure | 0.9966 |
| AUC for sphere | 0.9935 |
| AUC for vortex | 0.9982 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**The lensing physics**

A Singular Isothermal Sphere (SIS) is the simplest realistic model of a galaxy acting as a gravitational lens. Light from a background source gets bent as it passes the lens galaxy. The key equation is:

```
alpha(theta) = theta_E * theta / |theta|
beta = theta - alpha(theta)
```

Here `theta` is where you observe the light coming from, `beta` is where the source actually is, and `theta_E` (the Einstein radius) controls how strong the lensing is. The model learns to predict `theta_E` and the source position from the image itself.

**Architecture: `PhysicsGuidedEffNet`**

A single EfficientNet-B3 backbone extracts features from the image. Those features then split into two heads:

- **Classification head:** Dropout + Linear layer outputting 3 class scores, just like Task 1.
- **Physics head:** a small MLP that predicts 4 physical parameters from the same features:
  - `theta_E` (Einstein radius, bounded to [0, 0.40] via sigmoid)
  - `src_x`, `src_y` (source position, bounded to [-0.40, 0.40] via tanh)
  - `src_sigma` (source size, bounded to [0.05, 0.30] via sigmoid)

Those 4 parameters go into a **`LensingLayer`**, which applies the SIS equations on a precomputed coordinate grid and renders a Gaussian source through the lens to produce a reconstructed image.

**Total loss**

```
L = L_classification + lambda * L_physics
L_physics = MSE(reconstructed_image, input_image)
```

With `lambda=0.1`, the physics term is a soft nudge rather than a hard constraint. The classification head drives performance; the physics head keeps the learned features grounded in reality.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4
- OneCycleLR scheduler, stepped once per batch
- Mixup augmentation (alpha=0.2) applied to the classification loss only
- Mixed precision (AMP)
- Test-time augmentation across 6 flips/rotations
- 50 epochs, batch size 64

---

## Reproducing

```bash
python task7/train.py \
  --data-dir task1_data \
  --epochs 50 \
  --batch-size 64 \
  --lr-backbone 5e-5 \
  --lr-head 3e-4 \
  --lambda-phys 0.1 \
  --mixup-alpha 0.2 \
  --smoothing 0.05 \
  --tta \
  --save-dir task7/checkpoints
```

The notebook `Task7_Physics_Guided_ML.ipynb` derives the SIS lensing equation step by step, walks through the `LensingLayer` implementation, and shows 4-panel training curves tracking classification loss, physics residual, accuracy, and AUC.
