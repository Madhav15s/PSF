"""Batched PSF generator from weighted basis kernels."""

from __future__ import annotations

import torch


def generate_psf(weights: torch.Tensor, kernels: list[torch.Tensor]) -> torch.Tensor:
    """Generate normalized PSFs from basis weights.

    Purpose:
        Combines a bank of physically meaningful basis kernels into one PSF per
        batch element.

    Args:
        weights: Tensor with shape ``(batch_size, num_kernels)``.
        kernels: List of ``num_kernels`` tensors, each with shape ``(H, W)``.

    Returns:
        Tensor with shape ``(batch_size, 1, H, W)``. Each PSF is normalized to
        have spatial sum one.

    Raises:
        ValueError: If inputs are empty, shape-incompatible, or would normalize
            by a near-zero PSF sum.

    Mathematical description:
        ``PSF_b = sum_k weights[b, k] * basis_k`` followed by
        ``PSF_b = PSF_b / sum_ij PSF_b[i, j]``.

    Example:
        ``psf = generate_psf(torch.ones(1, 5), kernels)``
    """
    if weights.ndim != 2:
        raise ValueError("weights must have shape (batch_size, num_kernels).")
    if not kernels:
        raise ValueError("kernels must contain at least one basis kernel.")
    if weights.shape[1] != len(kernels):
        raise ValueError("weights second dimension must match number of kernels.")

    reference_shape = kernels[0].shape
    if len(reference_shape) != 2:
        raise ValueError("each kernel must have shape (H, W).")

    prepared_kernels = []
    for kernel in kernels:
        if kernel.shape != reference_shape:
            raise ValueError("all kernels must have the same spatial shape.")
        prepared_kernels.append(kernel.to(device=weights.device, dtype=weights.dtype))

    kernel_bank = torch.stack(prepared_kernels, dim=0)
    psf = torch.einsum("bk,khw->bhw", weights, kernel_bank)
    normalizer = psf.sum(dim=(-2, -1), keepdim=True)
    if torch.any(torch.isclose(normalizer, torch.zeros_like(normalizer), atol=1.0e-12)):
        raise ValueError("cannot normalize PSF with near-zero spatial sum.")

    return (psf / normalizer).unsqueeze(1)
