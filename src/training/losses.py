"""Loss functions for PSF regression training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss for regression targets."""
    return F.mse_loss(predictions, targets)


def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss for robust regression training."""
    return F.smooth_l1_loss(predictions, targets)
