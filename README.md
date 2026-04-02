# DeepLense GSoC 2026: Evaluation Tasks

Evaluation submissions for **ML4Sci / DeepLense** GSoC 2026.
All tasks use strong gravitational lensing images from the ML4Sci evaluation dataset.

---

## Results Summary

| Task | Description | Model | ROC-AUC |
|------|-------------|-------|---------|
| [Task 1](task1/) | Multi-class lens classification | EfficientNet-B3 | **0.9778** |
| [Task 4](task4/) | Neural operator classifier | Fourier Neural Operator | **0.9488** |
| [Task 5](task5/) | Lens finding (imbalanced) | EfficientNet-B3 + Focal Loss | **0.9852** |
| [Task 7](task7/) | Physics-guided classification | EfficientNet-B3 + SIS PINN | **0.9961** |

---

## Repository Structure

```
task1/       Multi-class classification (no substructure / sphere / vortex)
task4/       Fourier Neural Operator trained from scratch
task5/       Binary lens finding on a heavily imbalanced dataset
task7/       Physics-guided network with differentiable SIS lensing layer
utils/       Shared model definitions and loss functions
task1_data/  Shared dataset used by tasks 1, 4, and 7
```

Each task directory contains:
- `train.py`: standalone training script with CLI arguments
- `*.ipynb`: notebook walkthrough with explanations, training code, and result plots
- `checkpoints/results.json`: final metrics from the training run

---

## Quickstart

```bash
git clone https://github.com/pulin-chhangawala/deeplense-eval-gsoc2026.git
cd deeplense-eval-gsoc2026
pip install -r requirements.txt

# Task 1
python task1/train.py --data-dir task1_data --epochs 50 --tta

# Task 4
python task4/train.py --data-dir task1_data --epochs 60 --tta

# Task 5
python task5/train.py --data-dir task2_data --epochs 30 --tta

# Task 7
python task7/train.py --data-dir task1_data --epochs 50 --tta
```

All scripts accept `--help` for full argument listings.
Training was run on a single NVIDIA A5000.

---

## Requirements

```
torch>=2.0
torchvision>=0.15
timm>=0.9
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
```

See `requirements.txt` for the pinned environment.
