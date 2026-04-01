"""
models.py  —  DeepLense GSoC 2026 Evaluation Tasks
====================================================
Model architectures for gravitational lens classification and finding.

Includes:
- LensClassifierEffNet  : EfficientNet-B3 backbone, multi-class head  (primary)
- LensFinderEffNet      : EfficientNet-B3 backbone, binary head        (primary)
- LensClassifierResNet  : ResNet-50 backbone, multi-class head         (fallback)
- LensFinder            : ResNet-50 backbone, binary head              (fallback)
- LightweightLensCNN    : Small custom CNN for ablation / CPU baseline

All models handle their own ImageNet normalization internally so the
data pipeline can pass raw [0, 1] tensors without any extra transforms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _imagenet_buffers(in_channels: int):
    """Return (mean, std) tensors shaped for broadcasting over (B, C, H, W)."""
    if in_channels == 1:
        m = torch.tensor([0.449]).view(1, 1, 1, 1)
        s = torch.tensor([0.226]).view(1, 1, 1, 1)
    else:
        m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return m, s


def _adapt_first_conv(orig_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Replace the first conv layer to accept `in_channels` instead of 3.
    When pretrained weights exist, initialise by averaging across the RGB dim
    so the output scale is preserved.
    """
    new_conv = nn.Conv2d(
        in_channels,
        orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=(orig_conv.bias is not None),
    )
    if orig_conv.weight.shape[1] == 3:
        # Average pretrained RGB weights → single (or multi) channel
        avg_w = orig_conv.weight.data.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
        new_conv.weight.data = avg_w.repeat(1, in_channels, 1, 1) / in_channels
    if orig_conv.bias is not None and new_conv.bias is not None:
        new_conv.bias.data = orig_conv.bias.data.clone()
    return new_conv


# ─────────────────────────────────────────────────────────────────────────────
#  Primary: EfficientNet-B3 models
# ─────────────────────────────────────────────────────────────────────────────

class LensClassifierEffNet(nn.Module):
    """
    EfficientNet-B3 for multi-class strong lensing substructure classification.

    Handles its own ImageNet normalisation internally; feed raw [0, 1] tensors.

    Parameters
    ----------
    num_classes : int   Number of output classes (default: 3).
    in_channels : int   Input channels — 1 (single band) or 3 (multi-band).
    pretrained  : bool  Load ImageNet pretrained weights.
    dropout     : float Dropout probability in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Adapt first conv for non-RGB input
        if in_channels != 3:
            backbone.features[0][0] = _adapt_first_conv(
                backbone.features[0][0], in_channels
            )

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone
        self.num_classes = num_classes

        m, s = _imagenet_buffers(in_channels)
        self.register_buffer("_norm_mean", m)
        self.register_buffer("_norm_std", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._norm_mean) / self._norm_std
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def get_param_groups(
        self, lr_backbone: float = 5e-5, lr_head: float = 3e-4
    ) -> list[dict]:
        """Differential learning rates: lower for pretrained backbone, higher for head."""
        head_ids = {id(p) for p in self.backbone.classifier.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": list(self.backbone.classifier.parameters()), "lr": lr_head},
        ]


class LensFinderEffNet(nn.Module):
    """
    EfficientNet-B3 binary lens detector.
    Outputs raw logits (shape: [B, 1]); apply sigmoid for probabilities.

    Parameters
    ----------
    in_channels : int   Input channels (3 for multi-band telescope images).
    pretrained  : bool  Load ImageNet pretrained weights.
    dropout     : float Dropout before classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        if in_channels != 3:
            backbone.features[0][0] = _adapt_first_conv(
                backbone.features[0][0], in_channels
            )

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

        m, s = _imagenet_buffers(in_channels)
        self.register_buffer("_norm_mean", m)
        self.register_buffer("_norm_std", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._norm_mean) / self._norm_std
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)).squeeze(1)

    def get_param_groups(
        self, lr_backbone: float = 5e-5, lr_head: float = 3e-4
    ) -> list[dict]:
        head_ids = {id(p) for p in self.backbone.classifier.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": list(self.backbone.classifier.parameters()), "lr": lr_head},
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback: ResNet-50 models
# ─────────────────────────────────────────────────────────────────────────────

class LensClassifierResNet(nn.Module):
    """
    ResNet-50 backbone fine-tuned for multi-class strong lensing classification.
    Internal ImageNet normalisation; feed raw [0, 1] tensors.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        if in_channels != 3:
            backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone
        self.num_classes = num_classes

        m, s = _imagenet_buffers(in_channels)
        self.register_buffer("_norm_mean", m)
        self.register_buffer("_norm_std", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._norm_mean) / self._norm_std
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def get_param_groups(
        self, lr_backbone: float = 5e-5, lr_head: float = 3e-4
    ) -> list[dict]:
        head_ids = {id(p) for p in self.backbone.fc.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": list(self.backbone.fc.parameters()), "lr": lr_head},
        ]


class LensFinder(nn.Module):
    """ResNet-50 binary lens detector."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        if in_channels != 3:
            backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.backbone = backbone

        m, s = _imagenet_buffers(in_channels)
        self.register_buffer("_norm_mean", m)
        self.register_buffer("_norm_std", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._norm_mean) / self._norm_std
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)).squeeze(1)

    def get_param_groups(
        self, lr_backbone: float = 5e-5, lr_head: float = 3e-4
    ) -> list[dict]:
        head_ids = {id(p) for p in self.backbone.fc.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": list(self.backbone.fc.parameters()), "lr": lr_head},
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight CNN (ablation / CPU baseline)
# ─────────────────────────────────────────────────────────────────────────────

class LightweightLensCNN(nn.Module):
    """
    Small 6-layer convolutional network for quick ablation studies
    and CPU-only environments.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        img_size: int = 64,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),           nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),   nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),   nn.Dropout2d(0.1),
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
