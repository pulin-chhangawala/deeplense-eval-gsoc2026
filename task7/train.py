"""
task7/train.py
================================
Physics-guided EfficientNet-B3 classifier for multi-class gravitational lens
classification. Uses the same dataset as Task 1 for direct comparison.

Architecture: EfficientNet-B3 backbone shared between two heads.
  1. Classification head  -- standard 3-class softmax.
  2. Physics head         -- predicts SIS lens parameters [theta_E, src_x, src_y, src_sigma].
     A differentiable lensing layer (LensingLayer) applies the gravitational lensing
     equation to reconstruct the observed image from predicted parameters.
     The reconstruction MSE is used as a physics residual loss term.

Total loss = LabelSmoothingCE(logits, labels) + lambda_phys * MSE(reconstructed, input)

The physics loss does two things:
  - Forces the backbone to learn features consistent with lensing physics.
  - Acts as a regulariser: "no substructure" images reconstruct well under a pure SIS
    model, while substructure images cannot, so the residuals carry class-relevant signal.

Usage
-----
  python task7/train.py --data-dir task1_data --epochs 50
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models import PhysicsGuidedEffNet
from utils.losses import LabelSmoothingCrossEntropy


CLASS_NAMES = ["no", "sphere", "vort"]


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset  (identical to Task 1)
# ─────────────────────────────────────────────────────────────────────────────

class LensingDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.samples = []
        self.augment = augment
        split_dir = Path(root) / split
        for label, cls_name in enumerate(CLASS_NAMES):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing: {cls_dir}")
            for p in sorted(cls_dir.glob("*.npy")):
                self.samples.append((str(p), label))
        counts = np.bincount([l for _, l in self.samples], minlength=3)
        print(f"  {split}: {dict(zip(CLASS_NAMES, counts))}  total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        img = torch.from_numpy(arr)
        if self.augment:
            img = self._augment(img)
        return img, label

    @staticmethod
    def _augment(img):
        if torch.rand(1) < 0.5:
            img = TF.hflip(img)
        if torch.rand(1) < 0.5:
            img = TF.vflip(img)
        k = int(torch.randint(0, 4, (1,)))
        if k:
            img = torch.rot90(img, k, dims=[-2, -1])
        if torch.rand(1) < 0.3:
            c, h, w = img.shape
            sh = int(h * torch.empty(1).uniform_(0.02, 0.15))
            sw = int(w * torch.empty(1).uniform_(0.02, 0.15))
            r0 = int(torch.randint(0, h - sh + 1, (1,)))
            c0 = int(torch.randint(0, w - sw + 1, (1,)))
            img = img.clone()
            img[:, r0:r0 + sh, c0:c0 + sw] = img.mean()
        return img


def get_sample_weights(dataset):
    counts  = np.bincount([l for _, l in dataset.samples], minlength=3)
    class_w = 1.0 / (counts.astype(np.float64) + 1e-8)
    return torch.tensor([class_w[l] for _, l in dataset.samples])


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scheduler, lambda_phys=0.1, mixup_alpha=0.2, scaler=None):
    model.train()
    total_loss, cls_loss_sum, phys_loss_sum = 0.0, 0.0, 0.0
    correct, total = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            # Optional mixup
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(imgs.size(0), device=device)
                imgs_mix = lam * imgs + (1 - lam) * imgs[idx]
                logits, reconstructed = model(imgs_mix)
                cls_loss = (lam * criterion(logits, labels) +
                            (1 - lam) * criterion(logits, labels[idx]))
            else:
                logits, reconstructed = model(imgs)
                cls_loss = criterion(logits, labels)

            phys_loss  = F.mse_loss(reconstructed, imgs)
            loss       = cls_loss + lambda_phys * phys_loss

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

        scheduler.step()
        total_loss    += loss.item()     * imgs.size(0)
        cls_loss_sum  += cls_loss.item() * imgs.size(0)
        phys_loss_sum += phys_loss.item()* imgs.size(0)
        correct       += (logits.argmax(1) == labels).sum().item()
        total         += imgs.size(0)

    n = total
    return total_loss / n, cls_loss_sum / n, phys_loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta=False):
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
            for t in tta_transforms:
                aug = torch.stack([t(im) for im in imgs])
                logits_t, _ = model(aug)
                probs_list.append(torch.softmax(logits_t, dim=1))
            probs  = torch.stack(probs_list).mean(0)
            logits = torch.log(probs + 1e-8)
        else:
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
        cls_loss = criterion(logits, labels)
        total_loss += cls_loss.item() * imgs.size(0)
        correct    += (probs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(all_probs), np.concatenate(all_labels)


def compute_roc_auc(probs, labels, n_classes=3):
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        return float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def compute_per_class_auc(probs, labels, n_classes=3):
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    out = {}
    for i, name in enumerate(CLASS_NAMES[:n_classes]):
        try:
            out[name] = float(roc_auc_score(y_bin[:, i], probs[:, i]))
        except Exception:
            out[name] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 7: Physics-guided lens classifier")
    p.add_argument("--data-dir",     type=str,   required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr-backbone",  type=float, default=5e-5)
    p.add_argument("--lr-head",      type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoothing",    type=float, default=0.05)
    p.add_argument("--mixup-alpha",  type=float, default=0.2)
    p.add_argument("--lambda-phys",  type=float, default=0.1,
                   help="Weight for the physics residual loss term")
    p.add_argument("--no-pretrain",  action="store_true")
    p.add_argument("--no-cuda",      action="store_true")
    p.add_argument("--no-amp",       action="store_true")
    p.add_argument("--tta",          action="store_true")
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--save-dir",     type=str,   default="task7/checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Host: {os.uname().nodename}")
    print(f"Device: {device}  |  AMP: {use_amp}")

    train_ds = LensingDataset(args.data_dir, "train", augment=True)
    val_ds   = LensingDataset(args.data_dir, "val",   augment=False)
    try:
        test_ds = LensingDataset(args.data_dir, "test", augment=False)
    except FileNotFoundError:
        print("  No test split; using val")
        test_ds = val_ds

    sample_tensor, _ = train_ds[0]
    in_channels = sample_tensor.shape[0]
    img_size    = sample_tensor.shape[-1]
    print(f"  Input shape: {tuple(sample_tensor.shape)}")

    sampler = WeightedRandomSampler(get_sample_weights(train_ds),
                                    len(train_ds.samples), replacement=True)
    pin = device.type == "cuda"
    kw  = dict(num_workers=args.num_workers, pin_memory=pin,
               persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, **kw)

    model = PhysicsGuidedEffNet(
        in_channels=in_channels, num_classes=3,
        img_size=img_size, pretrained=not args.no_pretrain,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: PhysicsGuidedEffNet  |  Params: {n_params:,}")
    print(f"  lambda_phys={args.lambda_phys}  mixup_alpha={args.mixup_alpha}")

    criterion    = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    param_groups = model.get_param_groups(lr_backbone=args.lr_backbone, lr_head=args.lr_head)
    optimizer    = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler    = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone * 10, args.lr_head * 10],
        steps_per_epoch=len(train_loader), epochs=args.epochs,
        pct_start=0.1, anneal_strategy="cos",
        div_factor=25.0, final_div_factor=1e4,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_auc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_cls, tr_phys, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler, lambda_phys=args.lambda_phys, mixup_alpha=args.mixup_alpha, scaler=scaler,
        )
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device, tta=False
        )
        val_auc = compute_roc_auc(val_probs, val_labels)
        elapsed = time.time() - t0

        cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} (cls {tr_cls:.4f} phys {tr_phys:.4f}) acc {tr_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f} | "
            f"LR {cur_lrs[0]:.2e}/{cur_lrs[-1]:.2e} | {elapsed:.1f}s",
            flush=True,
        )
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_cls_loss": tr_cls,
            "train_phys_loss": tr_phys, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": float(val_auc),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print(f"  New best val AUC: {best_val_auc:.4f}")

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"),
                                     map_location=device))
    _, test_acc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device, tta=args.tta
    )
    test_auc  = compute_roc_auc(test_probs, test_labels)
    per_class = compute_per_class_auc(test_probs, test_labels)

    print(f"\nTest accuracy: {test_acc:.4f}  |  Test ROC-AUC (macro): {test_auc:.4f}")
    print("Per-class AUC:", {k: f"{v:.4f}" for k, v in per_class.items()})

    results = {
        "test_accuracy": test_acc, "test_roc_auc": float(test_auc),
        "per_class_auc": per_class, "best_val_auc": best_val_auc,
        "history": history, "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
