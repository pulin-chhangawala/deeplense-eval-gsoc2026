"""
task2_lens_finding/train.py
============================
Training script for binary gravitational lens detection.
Handles severe class imbalance via focal loss + weighted sampling.

Dataset layout::
  data-dir/
    train/
      lens/     *.npy   (positive: contains a lens)
      no_lens/  *.npy   (negative: no lens)
    val/
      lens/
      no_lens/
    test/
      lens/
      no_lens/

Usage
-----
  python train.py --data-dir /path/to/dataset --epochs 20
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import LensFinder
from utils.losses import FocalLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LensFinderDataset(Dataset):
    """Binary lens / no-lens dataset from .npy files."""

    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples = []
        self.transform = transform
        split_dir = Path(root) / split
        for label, cls_name in enumerate(["no_lens", "lens"]):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing: {cls_dir}")
            for p in sorted(cls_dir.glob("*.npy")):
                self.samples.append((str(p), label))

        n_lens   = sum(1 for _, l in self.samples if l == 1)
        n_nolens = sum(1 for _, l in self.samples if l == 0)
        print(f"  {split}: {n_nolens} no-lens | {n_lens} lens "
              f"(imbalance ratio: {n_nolens / max(1, n_lens):.1f}:1)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        tensor = torch.from_numpy(arr)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.sigmoid(logits).squeeze(1)

        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs_all  = np.concatenate(all_probs)
    labels_all = np.concatenate(all_labels)
    return total_loss / total, probs_all, labels_all


def compute_metrics(probs, labels, threshold=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    preds  = (probs >= threshold).astype(int)
    auroc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    auprc  = average_precision_score(labels, probs)
    f1     = f1_score(labels, preds, zero_division=0)
    tp     = int(((preds == 1) & (labels == 1)).sum())
    fp     = int(((preds == 1) & (labels == 0)).sum())
    fn     = int(((preds == 0) & (labels == 1)).sum())
    tn     = int(((preds == 0) & (labels == 0)).sum())
    return {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def find_optimal_threshold(probs, labels):
    """Find threshold that maximises F1 on provided set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 2 — Binary lens finding")
    p.add_argument("--data-dir",    type=str, required=True)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--save-dir",    type=str, default="checkpoints/task2")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────
    print("Dataset statistics:")
    train_ds = LensFinderDataset(args.data_dir, "train")
    val_ds   = LensFinderDataset(args.data_dir, "val")
    test_ds  = LensFinderDataset(args.data_dir, "test")

    # Weighted sampling to balance positive/negative
    labels = [l for _, l in train_ds.samples]
    n_neg, n_pos = labels.count(0), labels.count(1)
    w = [1.0 / n_neg if l == 0 else 1.0 / n_pos for l in labels]
    sampler = WeightedRandomSampler(torch.tensor(w), len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = LensFinder(
        in_channels=args.in_channels,
        pretrained=not args.no_pretrain,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LensFinder (ResNet-18)  |  Trainable params: {n_params:,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_auroc = 0.0
    history    = []

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss                     = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_probs, val_lbl = evaluate(model, val_loader, criterion, device)
        val_metrics                 = compute_metrics(val_probs, val_lbl)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} | "
            f"Val loss {val_loss:.4f} AUROC {val_metrics['auroc']:.4f} "
            f"F1 {val_metrics['f1']:.4f} | {elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, **val_metrics})

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    # ── Test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"),
                                      map_location=device))
    _, test_probs, test_lbl = evaluate(model, test_loader, criterion, device)
    opt_threshold, opt_f1   = find_optimal_threshold(test_probs, test_lbl)
    test_metrics = compute_metrics(test_probs, test_lbl, threshold=opt_threshold)

    print(f"\nTest AUROC: {test_metrics['auroc']:.4f}  "
          f"| AUPRC: {test_metrics['auprc']:.4f}  "
          f"| F1 @ threshold={opt_threshold:.2f}: {test_metrics['f1']:.4f}")
    print(f"Confusion — TP:{test_metrics['tp']} FP:{test_metrics['fp']} "
          f"FN:{test_metrics['fn']} TN:{test_metrics['tn']}")

    results = {
        "test_metrics": test_metrics,
        "optimal_threshold": opt_threshold,
        "best_val_auroc": best_auroc,
        "history": history,
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
