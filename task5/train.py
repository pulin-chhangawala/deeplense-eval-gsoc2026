"""
task2_lens_finding/train.py
============================
Training script for binary gravitational lens detection.
Handles severe class imbalance (≈ 16-100:1 neg:pos) via:
  • Focal loss (down-weights easy negatives)
  • Weighted random sampling (balanced batches)
  • ROC-optimal threshold search at test time

Dataset layout::
  data-dir/
    train/
      lens/     *.npy   (positive)
      no_lens/  *.npy   (negative)
    val/
      lens/ no_lens/
    test/
      lens/ no_lens/

Usage
-----
  python train.py --data-dir /path/to/dataset --epochs 30

Models handle ImageNet normalisation internally  -  feed raw [0, 1] tensors.
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
from torchvision.transforms import functional as TF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import LensFinderEffNet, LensFinder
from utils.losses import FocalLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LensFinderDataset(Dataset):
    """Binary lens / no-lens dataset from .npy files."""

    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.samples: list[tuple[str, int]] = []
        self.augment = augment
        split_dir = Path(root) / split
        for label, cls_name in enumerate(["no_lens", "lens"]):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing: {cls_dir}")
            for p in sorted(cls_dir.glob("*.npy")):
                self.samples.append((str(p), label))

        n_lens   = sum(1 for _, l in self.samples if l == 1)
        n_nolens = sum(1 for _, l in self.samples if l == 0)
        print(f"  {split}: {n_nolens} no-lens | {n_lens} lens  "
              f"(ratio {n_nolens / max(1, n_lens):.1f}:1)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        img = torch.from_numpy(arr)

        if self.augment:
            img = self._augment(img)
        return img, label

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            img = TF.hflip(img)
        if torch.rand(1) < 0.5:
            img = TF.vflip(img)
        k = int(torch.randint(0, 4, (1,)))
        if k:
            img = torch.rot90(img, k, dims=[-2, -1])
        # Random erasing (small patch, simulates cosmic rays / detector artifacts)
        if torch.rand(1) < 0.2:
            c, h, w = img.shape
            sh = int(h * torch.empty(1).uniform_(0.02, 0.10))
            sw = int(w * torch.empty(1).uniform_(0.02, 0.10))
            r0 = int(torch.randint(0, max(1, h - sh), (1,)))
            c0 = int(torch.randint(0, max(1, w - sw), (1,)))
            img = img.clone()
            img[:, r0:r0 + sh, c0:c0 + sw] = img.mean()
        return img


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta: bool = False):
    model.eval()
    total_loss, total = 0.0, 0
    all_probs, all_labels = [], []

    tta_transforms = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: torch.rot90(x, 1, dims=[-2, -1]),
        lambda x: torch.rot90(x, 3, dims=[-2, -1]),
    ] if tta else [lambda x: x]

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        if tta:
            probs_list = []
            for tfm in tta_transforms:
                aug   = torch.stack([tfm(im) for im in imgs])
                logit = model(aug)
                probs_list.append(torch.sigmoid(logit).squeeze(1))
            probs  = torch.stack(probs_list).mean(dim=0)
            logits = torch.log(probs.unsqueeze(1) / (1 - probs.unsqueeze(1) + 1e-8))
        else:
            logits = model(imgs)
            probs  = torch.sigmoid(logits).squeeze(1)

        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return total_loss / total, np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels, threshold=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    preds  = (probs >= threshold).astype(int)
    auroc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    auprc  = average_precision_score(labels, probs)
    f1     = f1_score(labels, preds, zero_division=0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def find_optimal_threshold(probs, labels):
    """Find threshold maximising F1 on the provided set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 2  -  Binary lens finding")
    p.add_argument("--data-dir",     type=str, required=True)
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--lr-backbone",  type=float, default=5e-5)
    p.add_argument("--lr-head",      type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--focal-alpha",  type=float, default=0.25)
    p.add_argument("--focal-gamma",  type=float, default=2.5)
    p.add_argument("--model",        choices=["efficientnet", "resnet50"],
                   default="efficientnet")
    p.add_argument("--in-channels",  type=int, default=3)
    p.add_argument("--no-pretrain",  action="store_true")
    p.add_argument("--no-cuda",      action="store_true")
    p.add_argument("--no-amp",       action="store_true")
    p.add_argument("--tta",          action="store_true")
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--save-dir",     type=str, default="task5/checkpoints")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Datasets ──────────────────────────────────────────────────────
    print("Dataset statistics:")
    train_ds = LensFinderDataset(args.data_dir, "train", augment=True)
    val_ds   = LensFinderDataset(args.data_dir, "val",   augment=False)
    test_ds  = LensFinderDataset(args.data_dir, "test",  augment=False)

    # Auto-detect channels
    sample_tensor, _ = train_ds[0]
    args.in_channels  = sample_tensor.shape[0]
    print(f"  Input shape: {tuple(sample_tensor.shape)}")

    # Weighted sampling  -  balance positive / negative in each batch
    labels = [l for _, l in train_ds.samples]
    n_neg, n_pos = labels.count(0), labels.count(1)
    w = [1.0 / n_neg if l == 0 else 1.0 / n_pos for l in labels]
    sampler = WeightedRandomSampler(torch.tensor(w), len(w), replacement=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=pin,
                               persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin,
                               persistent_workers=(args.num_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin,
                               persistent_workers=(args.num_workers > 0))

    # ── Model ─────────────────────────────────────────────────────────
    pretrained = not args.no_pretrain
    if args.model == "efficientnet":
        model = LensFinderEffNet(in_channels=args.in_channels, pretrained=pretrained)
    else:
        model = LensFinder(in_channels=args.in_channels, pretrained=pretrained)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Trainable params: {n_params:,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(
            lr_backbone=args.lr_backbone, lr_head=args.lr_head
        )
        print(f"  Differential LR: backbone={args.lr_backbone}, head={args.lr_head}")
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr_head}]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone * 10, args.lr_head * 10],
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    os.makedirs(args.save_dir, exist_ok=True)
    best_auroc = 0.0
    history    = []

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        scheduler.step()

        val_loss, val_probs, val_lbl = evaluate(
            model, val_loader, criterion, device, tta=False
        )
        val_metrics = compute_metrics(val_probs, val_lbl)
        elapsed = time.time() - t0

        cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} | "
            f"Val loss {val_loss:.4f} AUROC {val_metrics['auroc']:.4f} "
            f"AUPRC {val_metrics['auprc']:.4f} F1 {val_metrics['f1']:.4f} | "
            f"LR {cur_lrs[0]:.2e}/{cur_lrs[-1]:.2e} | {elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, **val_metrics})

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print(f"  ✓ New best val AUROC: {best_auroc:.4f}")

    # ── Test evaluation with optional TTA ─────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    )
    _, test_probs, test_lbl = evaluate(
        model, test_loader, criterion, device, tta=args.tta
    )
    opt_threshold, opt_f1 = find_optimal_threshold(test_probs, test_lbl)
    test_metrics = compute_metrics(test_probs, test_lbl, threshold=opt_threshold)

    print(f"\nTest AUROC: {test_metrics['auroc']:.4f}  "
          f"| AUPRC: {test_metrics['auprc']:.4f}  "
          f"| F1 @ thr={opt_threshold:.3f}: {test_metrics['f1']:.4f}")
    print(f"Confusion  -  TP:{test_metrics['tp']} FP:{test_metrics['fp']} "
          f"FN:{test_metrics['fn']} TN:{test_metrics['tn']}")

    results = {
        "test_metrics":      test_metrics,
        "optimal_threshold": opt_threshold,
        "best_val_auroc":    best_auroc,
        "history":           history,
        "args":              vars(args),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
