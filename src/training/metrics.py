"""Evaluation metrics for PSF regression tasks."""

from __future__ import annotations

import torch


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average absolute error over the last dimension."""
    return torch.mean(torch.abs(predictions - targets))


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average squared error over the last dimension."""
    return torch.mean(torch.square(predictions - targets))


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """R-squared score for regression outputs."""
    residual_sum_of_squares = torch.sum(torch.square(targets - predictions))
    total_sum_of_squares = torch.sum(torch.square(targets - torch.mean(targets, dim=0)))
    return 1.0 - residual_sum_of_squares / total_sum_of_squares.clamp_min(1e-8)
