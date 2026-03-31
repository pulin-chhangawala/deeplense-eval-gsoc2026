"""
models.py  —  DeepLense GSoC 2026 Evaluation Tasks
====================================================
Model architectures for gravitational lens classification and finding.

Includes:
- LensClassifierResNet  : ResNet-18 backbone, multi-class head
- LensFinder            : ResNet-18 backbone, binary head
- LightweightLensCNN    : Small custom CNN for ablation / CPU benchmarking
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAdapter(nn.Module):
    """Adapt an arbitrary number of input channels to a pretrained backbone."""
    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Task 1 — Multi-class classifier
# ─────────────────────────────────────────────────────────────────────────────

class LensClassifierResNet(nn.Module):
    """
    ResNet-18 backbone fine-tuned for multi-class strong lensing
    substructure classification.

    Classes
    -------
    0 : No substructure
    1 : CDM subhalo substructure
    2 : WDM / axion substructure

    Parameters
    ----------
    num_classes  : int     Number of output classes (default: 3).
    in_channels  : int     Input channels — 1 (single band) or 3 (g,r,i).
    pretrained   : bool    Use ImageNet pre-trained weights.
    dropout      : float   Dropout probability before the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Channel adapter if not 3-channel input
        if in_channels != 3:
            self.channel_adapter: Optional[nn.Module] = ChannelAdapter(in_channels, 3)
        else:
            self.channel_adapter = None

        # Replace final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities."""
        return F.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Task 2 — Binary lens finder
# ─────────────────────────────────────────────────────────────────────────────

class LensFinder(nn.Module):
    """
    Binary lens detection model: ResNet-18 backbone with a single sigmoid
    output node, trained with focal loss to handle class imbalance.

    Parameters
    ----------
    in_channels : int   Input channels.
    pretrained  : bool  Use ImageNet pre-trained weights.
    dropout     : float Dropout before classifier head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        if in_channels != 3:
            self.channel_adapter: Optional[nn.Module] = ChannelAdapter(in_channels, 3)
        else:
            self.channel_adapter = None

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (shape: [B, 1])."""
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return lens probability (shape: [B])."""
        return torch.sigmoid(self.forward(x)).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight CNN (ablation / CPU baseline)
# ─────────────────────────────────────────────────────────────────────────────

class LightweightLensCNN(nn.Module):
    """
    Small 6-layer convolutional network for quick ablation studies
    and CPU-only environments.

    Parameters
    ----------
    num_classes : int  Output classes (1 for binary, ≥2 for multi-class).
    in_channels : int  Input channels.
    img_size    : int  Spatial size of input (assumed square).
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        img_size: int = 64,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),           nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),   nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),   nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),   nn.Dropout2d(0.2),
        )
        flat_size = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
