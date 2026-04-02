# Task 7 — Physics-Guided ML / PINN

Classify strong lensing images using a network that simultaneously learns to classify and to reconstruct the lensing physics. A differentiable SIS (Singular Isothermal Sphere) lensing layer is embedded directly in the model, and its reconstruction error is added to the training loss — making the physics a regularizer that guides the backbone toward physically meaningful representations.

Dataset: same 30,000/7,500 split as Task 1 (shared `task1_data/`).

---

## Result

| Metric | Value |
|--------|-------|
| Test ROC-AUC (macro) | **0.9961** |
| Test Accuracy | 96.7% |
| AUC — no substructure | 0.9966 |
| AUC — sphere | 0.9935 |
| AUC — vortex | 0.9982 |

Full per-epoch history in `checkpoints/results.json`.

---

## Approach

**SIS lensing equation:**

The deflection angle for a Singular Isothermal Sphere lens is:

```
alpha(theta) = theta_E * theta / |theta|
beta = theta - alpha(theta)
```

where `theta` is the observed position, `beta` is the true source position, and `theta_E` is the Einstein radius.

**Architecture:** `PhysicsGuidedEffNet`

- **Backbone:** EfficientNet-B3 (pretrained), shared feature extractor
- **Classification head:** Dropout(0.4) + Linear → 3-class logits
- **Physics head:** Linear → GELU → Linear(4), producing bounded lens parameters:
  - `theta_E` in [0, 0.40] via sigmoid
  - `src_x`, `src_y` in [-0.40, 0.40] via tanh
  - `src_sigma` in [0.05, 0.30] via sigmoid
- **LensingLayer:** takes the physics parameters, computes the SIS deflection on a precomputed coordinate grid, and renders a Gaussian source through the lens equation to produce a reconstructed image

**Total loss:**

```
L = L_classification + lambda * L_physics
L_physics = MSE(reconstructed_image, input_image)
```

With `lambda=0.1`, the physics term acts as a soft constraint — it encourages the physics head to find lens parameters consistent with the actual image, while the classification head benefits from the physically grounded features.

**Training details:**
- Differential learning rates: backbone at 5e-5, head at 3e-4
- OneCycleLR scheduler (stepped per batch)
- Mixup augmentation (alpha=0.2) on classification loss only
- Mixed precision (AMP)
- Test-time augmentation: 6 flips/rotations
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

The notebook `Task7_Physics_Guided_ML.ipynb` derives the SIS lensing equation, walks through the `LensingLayer` implementation, and shows 4-panel training curves tracking classification loss, physics residual, accuracy, and AUC.
