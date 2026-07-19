"""Tests for the batched PSF generator."""

import pytest
import torch

from basis import gaussian_kernel, ghost_kernel, motion_kernel, ring_kernel, scatter_kernel
from generator import generate_psf


def _kernels(device: torch.device | str = "cpu") -> list[torch.Tensor]:
    return [
        gaussian_kernel(device=device),
        ghost_kernel(device=device),
        scatter_kernel(device=device),
        ring_kernel(device=device),
        motion_kernel(device=device),
    ]


def test_generate_psf_shape_and_normalization() -> None:
    weights = torch.tensor([[1.0, 0.4, 0.2, 0.1, 0.3], [0.5, 0.5, 0.5, 0.5, 0.5]])
    psf = generate_psf(weights=weights, kernels=_kernels())

    assert psf.shape == (2, 1, 65, 65)
    assert torch.allclose(psf.sum(dim=(-2, -1)), torch.ones(2, 1), atol=1.0e-5)
    assert not torch.isnan(psf).any()
    assert not torch.isinf(psf).any()


def test_generate_psf_is_differentiable() -> None:
    weights = torch.tensor([[1.0, 0.4, 0.2, 0.1, 0.3]], requires_grad=True)
    psf = generate_psf(weights=weights, kernels=_kernels())
    loss = psf.square().sum()
    loss.backward()

    assert weights.grad is not None
    assert weights.grad.shape == weights.shape
    assert not torch.isnan(weights.grad).any()


def test_generate_psf_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        generate_psf(weights=torch.ones(5), kernels=_kernels())
    with pytest.raises(ValueError):
        generate_psf(weights=torch.ones(1, 4), kernels=_kernels())
    with pytest.raises(ValueError):
        generate_psf(weights=torch.zeros(1, 5), kernels=_kernels())


def test_generate_psf_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return

    weights = torch.ones(1, 5, device="cuda")
    psf = generate_psf(weights=weights, kernels=_kernels(device="cuda"))
    assert psf.is_cuda
    assert torch.allclose(psf.sum(dim=(-2, -1)).cpu(), torch.ones(1, 1), atol=1.0e-5)
