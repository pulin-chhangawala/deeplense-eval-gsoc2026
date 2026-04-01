"""
task1_classification/train.py
================================
Training script for multi-class strong gravitational lensing classification.

Classes:
  0  -  No substructure   (folder: "no")
  1  -  CDM subhalo       (folder: "sphere")
  2  -  WDM / axion       (folder: "vort")

Usage
-----
  python train.py --data-dir /path/to/dataset --epochs 50

Dataset layout::
  data-dir/
    train/
      no/     *.npy
      sphere/ *.npy
      vort/   *.npy
    val/
      no/ sphere/ vort/
    test/          (optional; val used as fallback)
      no/ sphere/ vort/

Each .npy file is a (H, W) or (C, H, W) float32/float64 array in [0, 1].
Models handle ImageNet normalisation internally  -  no extra transform needed.
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
from torchvision.transforms import functional as TF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import LensClassifierEffNet, LensClassifierResNet, LightweightLensCNN
from utils.losses import LabelSmoothingCrossEntropy


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["no", "sphere", "vort"]


class LensingDataset(Dataset):
    """Loads strong lensing .npy image stamps; labels inferred from subdirectory."""

    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.samples: list[tuple[str, int]] = []
        self.augment = augment
        split_dir = Path(root) / split
        for label, cls_name in enumerate(CLASS_NAMES):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing directory: {cls_dir}")
            for p in sorted(cls_dir.glob("*.npy")):
                self.samples.append((str(p), label))

        counts = np.bincount([lbl for _, lbl in self.samples], minlength=len(CLASS_NAMES))
        print(f"  {split}: {dict(zip(CLASS_NAMES, counts))}  total={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]           # (H, W) → (1, H, W)
        img = torch.from_numpy(arr)         # (C, H, W) in [0, 1]

        if self.augment:
            img = self._augment(img)
        return img, label

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        """In-place-safe augmentation for single astronomical image tensors."""
        # Geometric
        if torch.rand(1) < 0.5:
            img = TF.hflip(img)
        if torch.rand(1) < 0.5:
            img = TF.vflip(img)
        # Random 0 / 90 / 180 / 270 degree rotation
        k = int(torch.randint(0, 4, (1,)))
        if k:
            img = torch.rot90(img, k, dims=[-2, -1])
        # Random erasing  (simulates PSF occlusion / detector artefacts)
        if torch.rand(1) < 0.3:
            c, h, w = img.shape
            sh = int(h * torch.empty(1).uniform_(0.02, 0.15))
            sw = int(w * torch.empty(1).uniform_(0.02, 0.15))
            r0 = int(torch.randint(0, h - sh + 1, (1,)))
            c0 = int(torch.randint(0, w - sw + 1, (1,)))
            img = img.clone()
            img[:, r0:r0 + sh, c0:c0 + sw] = img.mean()
        return img


def get_sample_weights(dataset: LensingDataset) -> torch.Tensor:
    counts = np.bincount([lbl for _, lbl in dataset.samples], minlength=len(CLASS_NAMES))
    class_w = 1.0 / (counts.astype(np.float64) + 1e-8)
    return torch.tensor([class_w[lbl] for _, lbl in dataset.samples])


# ─────────────────────────────────────────────────────────────────────────────
#  Mixup
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(imgs, labels, alpha: float, criterion, model):
    """Apply Mixup to a batch; return mixed loss."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    logits = model(mixed)
    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])
    return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.2,
                    scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            if mixup_alpha > 0:
                logits, loss = mixup_batch(imgs, labels, mixup_alpha, criterion, model)
            else:
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
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta: bool = False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    tta_transforms = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: torch.rot90(x, 1, dims=[-2, -1]),
        lambda x: torch.rot90(x, 2, dims=[-2, -1]),
        lambda x: torch.rot90(x, 3, dims=[-2, -1]),
    ] if tta else [lambda x: x]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if tta:
            probs_list = []
            for tfm in tta_transforms:
                aug = torch.stack([tfm(im) for im in imgs])
                logits_t = model(aug)
                probs_list.append(torch.softmax(logits_t, dim=1))
            probs = torch.stack(probs_list).mean(dim=0)
            logits = torch.log(probs + 1e-8)   # fake logits for loss
        else:
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)

        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (probs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def compute_roc_auc(probs, labels, n_classes=3):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return auc


def compute_per_class_auc(probs, labels, n_classes=3):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    aucs = {}
    for i, name in enumerate(CLASS_NAMES[:n_classes]):
        try:
            aucs[name] = float(roc_auc_score(y_bin[:, i], probs[:, i]))
        except Exception:
            aucs[name] = float("nan")
    return aucs


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 1  -  Multi-class lens classification")
    p.add_argument("--data-dir",     type=str, required=True)
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--lr-backbone",  type=float, default=5e-5,
                   help="LR for pretrained backbone layers")
    p.add_argument("--lr-head",      type=float, default=3e-4,
                   help="LR for classification head")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--model",        choices=["efficientnet", "resnet50", "lightweight"],
                   default="efficientnet")
    p.add_argument("--in-channels",  type=int, default=1)
    p.add_argument("--num-classes",  type=int, default=3)
    p.add_argument("--smoothing",    type=float, default=0.05)
    p.add_argument("--mixup-alpha",  type=float, default=0.2)
    p.add_argument("--no-pretrain",  action="store_true")
    p.add_argument("--no-cuda",      action="store_true")
    p.add_argument("--no-amp",       action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--tta",          action="store_true",
                   help="Use test-time augmentation at evaluation")
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--save-dir",     type=str, default="task1/checkpoints")
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
    train_ds = LensingDataset(args.data_dir, split="train", augment=True)
    val_ds   = LensingDataset(args.data_dir, split="val",   augment=False)
    try:
        test_ds = LensingDataset(args.data_dir, split="test", augment=False)
    except FileNotFoundError:
        print("  Test split not found; using val for final evaluation")
        test_ds = val_ds

    # Auto-detect input channels
    sample_tensor, _ = train_ds[0]
    args.in_channels = sample_tensor.shape[0]
    print(f"  Input shape: {tuple(sample_tensor.shape)}")

    sample_weights = get_sample_weights(train_ds)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

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
        model = LensClassifierEffNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            pretrained=pretrained,
        )
    elif args.model == "resnet50":
        model = LensClassifierResNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            pretrained=pretrained,
        )
    else:
        model = LightweightLensCNN(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Trainable params: {n_params:,}")

    # ── Optimiser  -  differential LR for backbone vs head ──────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(
            lr_backbone=args.lr_backbone, lr_head=args.lr_head
        )
        print(f"  Differential LR: backbone={args.lr_backbone}, head={args.lr_head}")
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr_head}]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # OneCycleLR: fast warm-up → smooth cosine decay
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

    best_val_auc = 0.0
    history = []

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_alpha=args.mixup_alpha, scaler=scaler
        )
        scheduler.step()   # OneCycleLR steps per batch inside train_one_epoch;
        #                    call again here is harmless (already finished epoch)

        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device, tta=False
        )
        val_auc = compute_roc_auc(val_probs, val_labels, args.num_classes)
        elapsed = time.time() - t0

        cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f} | "
            f"LR {cur_lrs[0]:.2e}/{cur_lrs[-1]:.2e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": float(val_auc)
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print(f"  ✓ New best val AUC: {best_val_auc:.4f}")

    # ── Test evaluation with TTA ───────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    )
    _, test_acc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device, tta=args.tta
    )
    test_auc = compute_roc_auc(test_probs, test_labels, args.num_classes)
    per_class = compute_per_class_auc(test_probs, test_labels, args.num_classes)

    print(f"\nTest accuracy: {test_acc:.4f}  |  Test ROC-AUC (macro): {test_auc:.4f}")
    print("Per-class AUC:", {k: f"{v:.4f}" for k, v in per_class.items()})

    results = {
        "test_accuracy": test_acc,
        "test_roc_auc":  float(test_auc),
        "per_class_auc": per_class,
        "best_val_auc":  best_val_auc,
        "history":       history,
        "args":          vars(args),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
