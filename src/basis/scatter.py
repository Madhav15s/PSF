"""Diffuse scatter basis kernel for EO/IR point-spread functions."""

from __future__ import annotations

import torch

from basis.gaussian import _coordinate_grid


def scatter_kernel(
    size: int = 65,
    radius: float = 15.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a smooth low-frequency scatter halo kernel.

    Purpose:
        Models broad-angle stray light and atmospheric or optical scattering.

    Args:
        size: Odd spatial kernel size in pixels.
        radius: Characteristic halo falloff radius in pixels.
        device: Torch device on which to allocate the kernel.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and sum one.

    Raises:
        ValueError: If ``size`` is invalid or ``radius`` is not positive.

    Mathematical description:
        ``K(r) = 1 / (1 + (r / radius)^2)^2`` where
        ``r = sqrt(x^2 + y^2)``, normalized so that ``sum(K) = 1``.

    Example:
        ``kernel = scatter_kernel(size=65, radius=18.0)``
    """
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive.")

    grid_x, grid_y = _coordinate_grid(size=size, device=device)
    radius_tensor = torch.as_tensor(radius, dtype=torch.float32, device=device)
    radial_distance = torch.sqrt(grid_x.square() + grid_y.square() + 1.0e-12)
    kernel = 1.0 / (1.0 + (radial_distance / radius_tensor).square()).square()
    return kernel / kernel.sum()
