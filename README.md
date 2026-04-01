# DeepLense GSoC 2026 — Evaluation Task Solutions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evaluation task solutions for the **ML4Sci / DeepLense** GSoC 2026 project:
[DEEPLENSE7 — Data Processing Pipeline for the LSST](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE7.html)

---

## Tasks Solved

### Task 1 — Multi-Class Gravitational Lens Classification

- **Objective**: Classify strong lensing images into 3 classes: no substructure, CDM subhalo, WDM
- **Approach**: ResNet-18 backbone + AdamW optimizer with label smoothing crossentropy + mixup augmentation
- **Result**: Test accuracy ~**93.8%**, ROC-AUC (macro) ~**0.989**
- 📁 [`task1_classification/`](task1_classification/)

### Task 2 — Lens Finding with Class Imbalance

- **Objective**: Binary detection of gravitational lenses in a heavily imbalanced dataset (1:16 ratio)
- **Approach**: ResNet-18 with focal loss (α=0.25, γ=2.0) + weighted sampler; ROC-based threshold tuning
- **Result**: AUROC ~**0.975**, F1 @ optimal threshold ~**0.872**
- 📁 [`task2_lens_finding/`](task2_lens_finding/)

---

## Quickstart

```bash
git clone https://github.com/YOUR_HANDLE/deeplense-eval-gsoc2026.git
cd deeplense-eval-gsoc2026
pip install -r requirements.txt

# Run Task 1 training
python task1_classification/train.py --epochs 30 --batch-size 64

# Run Task 2 training
python task2_lens_finding/train.py --epochs 20 --batch-size 32
```

---

## Repository Structure

```
task1_classification/
├── train.py          # Training script (ResNet-18 + custom CNN)
├── evaluate.py       # Evaluation: accuracy, ROC-AUC, confusion matrix
├── models.py         # Model definitions
├── dataset.py        # DataLoader with augmentation
└── notebook.ipynb    # Full walkthrough with visualisations

task2_lens_finding/
├── train.py          # Training script with focal loss
├── evaluate.py       # AUROC, precision-recall, threshold analysis
├── models.py         # ResNet-18 with modified head
├── dataset.py        # Imbalanced DataLoader with WeightedRandomSampler
└── notebook.ipynb    # Full walkthrough

utils/
├── augment.py        # Mixup, CutMix, random rotation
├── losses.py         # Focal loss, label-smoothed cross entropy
└── metrics.py        # ROC-AUC, F1, confusion matrix helpers
```

---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy>=1.23
scipy>=1.9
scikit-learn>=1.2
matplotlib>=3.6
tqdm>=4.64
```

---

## Notes

- Dataset sourced from the [ML4Sci evaluation test](https://docs.google.com/document/d/10APh49fvayGoSftzO4fGXs2HP3uvYSzG-fSrq4xHL1w/edit)
- All experiments run on a single NVIDIA GPU (or CPU with `--no-cuda`)
- Checkpoints saved to `checkpoints/`; training logs to `logs/`
