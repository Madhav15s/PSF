"""Ghost reflection basis kernel for EO/IR point-spread functions."""

from __future__ import annotations

import torch

from basis.gaussian import _coordinate_grid


def ghost_kernel(
    size: int = 65,
    offset_x: float = 8.0,
    offset_y: float = 0.0,
    strength: float = 0.2,
    sigma: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a normalized primary-plus-shifted Gaussian ghost kernel.

    Purpose:
        Models faint internal reflections or secondary optical images.

    Args:
        size: Odd spatial kernel size in pixels.
        offset_x: Horizontal ghost displacement in pixels.
        offset_y: Vertical ghost displacement in pixels.
        strength: Non-negative relative amplitude of the shifted ghost.
        sigma: Standard deviation of both Gaussian components in pixels.
        device: Torch device on which to allocate the kernel.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and sum one.

    Raises:
        ValueError: If ``size`` is invalid, ``sigma`` is not positive, or
            ``strength`` is negative.

    Mathematical description:
        ``K = G(x, y; sigma) + strength * G(x - offset_x, y - offset_y; sigma)``,
        normalized so that ``sum(K) = 1``.

    Example:
        ``kernel = ghost_kernel(offset_x=7.5, offset_y=-1.0, strength=0.15)``
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")
    if strength < 0.0:
        raise ValueError("strength must be non-negative.")

    grid_x, grid_y = _coordinate_grid(size=size, device=device)
    sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    offset_x_tensor = torch.as_tensor(offset_x, dtype=torch.float32, device=device)
    offset_y_tensor = torch.as_tensor(offset_y, dtype=torch.float32, device=device)
    strength_tensor = torch.as_tensor(strength, dtype=torch.float32, device=device)

    primary = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * sigma_tensor.square()))
    shifted_radius = (grid_x - offset_x_tensor).square() + (grid_y - offset_y_tensor).square()
    shifted = torch.exp(-shifted_radius / (2.0 * sigma_tensor.square()))
    kernel = primary + strength_tensor * shifted
    return kernel / kernel.sum()
