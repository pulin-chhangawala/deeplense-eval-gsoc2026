"""
task1_classification/train.py
================================
Training script for multi-class strong gravitational lensing classification.

Classes:
  0 — No substructure
  1 — CDM subhalo substructure
  2 — WDM / axion substructure

Usage
-----
  python train.py --data-dir /path/to/dataset --epochs 30 --batch-size 64

Dataset expected layout::
  data-dir/
    train/
      no_sub/    *.npy  (or *.png)
      cdm/       *.npy
      wdm/       *.npy
    val/
      no_sub/
      cdm/
      wdm/
    test/
      no_sub/
      cdm/
      wdm/

Each .npy file is a (H, W) or (C, H, W) float32 array.
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import LensClassifierResNet, LightweightLensCNN
from utils.losses import LabelSmoothingCrossEntropy


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["no_sub", "cdm", "wdm"]


class LensingDataset(Dataset):
    """
    Loads strong lensing .npy image stamps with class labels inferred
    from the subdirectory name.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples = []
        self.transform = transform
        split_dir = Path(root) / split
        for label, cls_name in enumerate(CLASS_NAMES):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing directory: {cls_dir}")
            for p in sorted(cls_dir.glob("*.npy")):
                self.samples.append((str(p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]       # (1, H, W)
        tensor = torch.from_numpy(arr)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


def get_class_weights(dataset: LensingDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted sampling."""
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-8)
    sample_weights = torch.tensor([weights[label] for _, label in dataset.samples])
    return sample_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.2):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Mixup augmentation
        if mixup_alpha > 0 and model.training:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(imgs.size(0), device=device)
            imgs   = lam * imgs + (1 - lam) * imgs[idx]
            labels_a, labels_b = labels, labels[idx]
            logits = model(imgs)
            loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total  += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)

        total_loss += loss.item() * imgs.size(0)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total  += imgs.size(0)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (total_loss / total,
            correct / total,
            np.concatenate(all_probs),
            np.concatenate(all_labels))


def compute_roc_auc(probs, labels, n_classes=3):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return auc


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 1 — Multi-class lens classification")
    p.add_argument("--data-dir",    type=str, required=True)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--model",       choices=["resnet18", "lightweight"], default="resnet18")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--smoothing",   type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--save-dir",    type=str, default="checkpoints/task1")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    train_ds = LensingDataset(args.data_dir, split="train", transform=train_tf)
    val_ds   = LensingDataset(args.data_dir, split="val")
    test_ds  = LensingDataset(args.data_dir, split="test")

    # Balanced sampling
    sample_weights = get_class_weights(train_ds)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    if args.model == "resnet18":
        model = LensClassifierResNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            pretrained=not args.no_pretrain,
        )
    else:
        model = LightweightLensCNN(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Trainable params: {n_params:,}")

    # ── Training setup ────────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mixup_alpha
        )
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_auc = compute_roc_auc(val_probs, val_labels, args.num_classes)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": float(val_auc)
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    # ── Test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"),
                                      map_location=device))
    _, test_acc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    test_auc = compute_roc_auc(test_probs, test_labels, args.num_classes)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test ROC-AUC: {test_auc:.4f}")

    # Save results
    results = {
        "test_accuracy": test_acc, "test_roc_auc": float(test_auc),
        "best_val_acc": best_val_acc, "history": history,
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
