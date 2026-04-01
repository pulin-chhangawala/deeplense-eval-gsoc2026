"""
utils/losses.py  -  Custom loss functions for imbalanced lensing datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017).

    Parameters
    ----------
    alpha : float   Weighting factor for positive class (default: 0.25).
    gamma : float   Focusing parameter; 0 = standard BCE (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1) raw model outputs (before sigmoid).
        targets : (B,) binary labels {0, 1}.
        """
        targets = targets.float().view(-1, 1)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing for multi-class classification.

    Parameters
    ----------
    smoothing : float   Smoothing factor in [0, 1). Default: 0.1.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_prob  = F.log_softmax(logits, dim=-1)
        # Hard targets
        nll  = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Soft (uniform) targets
        soft = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * soft
        return loss.mean()
