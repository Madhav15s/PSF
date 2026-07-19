"""Directional motion basis kernel for EO/IR point-spread functions."""

from __future__ import annotations

import math

import torch

from basis.gaussian import _coordinate_grid


def motion_kernel(
    size: int = 65,
    length: float = 10.0,
    angle_deg: float = 45.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a differentiable directional motion blur kernel.

    Purpose:
        Models linear smear caused by platform motion, target motion, or sensor
        integration during motion.

    Args:
        size: Odd spatial kernel size in pixels.
        length: Approximate full motion extent in pixels.
        angle_deg: Motion direction in degrees counter-clockwise from +x.
        device: Torch device on which to allocate the kernel.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and sum one.

    Raises:
        ValueError: If ``size`` is invalid or ``length`` is not positive.

    Mathematical description:
        The coordinate grid is rotated into parallel and perpendicular axes.
        A smooth rectangular window along the motion axis is multiplied by a
        narrow Gaussian across the motion axis, then normalized.

    Example:
        ``kernel = motion_kernel(length=16.0, angle_deg=30.0, device="cuda")``
    """
    if length <= 0.0:
        raise ValueError("length must be strictly positive.")

    grid_x, grid_y = _coordinate_grid(size=size, device=device)
    angle = torch.as_tensor(math.radians(angle_deg), dtype=torch.float32, device=device)
    length_tensor = torch.as_tensor(length, dtype=torch.float32, device=device)
    parallel = grid_x * torch.cos(angle) + grid_y * torch.sin(angle)
    perpendicular = -grid_x * torch.sin(angle) + grid_y * torch.cos(angle)

    edge_softness = torch.clamp(length_tensor / 12.0, min=torch.tensor(0.75, device=device))
    half_length = length_tensor / 2.0
    along_track = torch.sigmoid((parallel + half_length) / edge_softness)
    along_track = along_track * torch.sigmoid((half_length - parallel) / edge_softness)

    cross_sigma = torch.clamp(length_tensor / 18.0, min=torch.tensor(0.75, device=device))
    cross_track = torch.exp(-perpendicular.square() / (2.0 * cross_sigma.square()))
    kernel = along_track * cross_track
    return kernel / kernel.sum()
