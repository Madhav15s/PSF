"""Gaussian basis kernel for EO/IR point-spread functions."""

from __future__ import annotations

import torch


def _coordinate_grid(size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")

    coordinates = torch.arange(size, dtype=torch.float32, device=device)
    coordinates = coordinates - (size - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return grid_x, grid_y


def gaussian_kernel(
    size: int = 65,
    sigma: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a centered, normalized two-dimensional Gaussian kernel.

    Purpose:
        Models diffraction-limited or approximately isotropic optical blur.

    Args:
        size: Odd spatial kernel size in pixels.
        sigma: Gaussian standard deviation in pixels.
        device: Torch device on which to allocate the kernel.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and sum one.

    Raises:
        ValueError: If ``size`` is not positive and odd, or if ``sigma`` is not
            strictly positive.

    Mathematical description:
        ``G(x, y) = exp(-(x^2 + y^2) / (2 sigma^2))``, normalized so that
        ``sum(G) = 1``.

    Example:
        ``kernel = gaussian_kernel(size=65, sigma=2.0, device="cuda")``
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")

    grid_x, grid_y = _coordinate_grid(size=size, device=device)
    sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    squared_radius = grid_x.square() + grid_y.square()
    kernel = torch.exp(-squared_radius / (2.0 * sigma_tensor.square()))
    return kernel / kernel.sum()
