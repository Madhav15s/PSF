"""Ring reflection basis kernel for EO/IR point-spread functions."""

from __future__ import annotations

import torch

from basis.gaussian import _coordinate_grid


def ring_kernel(
    size: int = 65,
    radius: float = 12.0,
    width: float = 3.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a normalized circular ring reflection kernel.

    Purpose:
        Models annular reflections caused by aperture, filter, or lens effects.

    Args:
        size: Odd spatial kernel size in pixels.
        radius: Ring radius in pixels.
        width: Ring Gaussian width in pixels.
        device: Torch device on which to allocate the kernel.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and sum one.

    Raises:
        ValueError: If ``size`` is invalid, ``radius`` is negative, or
            ``width`` is not positive.

    Mathematical description:
        ``K(x, y) = exp(-((sqrt(x^2 + y^2) - radius)^2) / (2 width^2))``,
        normalized so that ``sum(K) = 1``.

    Example:
        ``kernel = ring_kernel(radius=14.0, width=2.5, device="cpu")``
    """
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if width <= 0.0:
        raise ValueError("width must be strictly positive.")

    grid_x, grid_y = _coordinate_grid(size=size, device=device)
    radius_tensor = torch.as_tensor(radius, dtype=torch.float32, device=device)
    width_tensor = torch.as_tensor(width, dtype=torch.float32, device=device)
    radial_distance = torch.sqrt(grid_x.square() + grid_y.square() + 1.0e-12)
    kernel = torch.exp(-((radial_distance - radius_tensor).square()) / (2.0 * width_tensor.square()))
    return kernel / kernel.sum()
