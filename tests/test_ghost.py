"""Tests for the ghost reflection basis kernel."""

import pytest
import torch

from basis.ghost import ghost_kernel


def test_ghost_kernel_properties() -> None:
    kernel = ghost_kernel(size=65, offset_x=8.0, offset_y=0.0, strength=0.2, sigma=2.0)

    assert kernel.shape == (65, 65)
    assert kernel.dtype == torch.float32
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1.0e-5)
    assert not torch.isnan(kernel).any()
    assert not torch.isinf(kernel).any()
    assert torch.all(kernel >= 0.0)
    assert kernel[32, 32] > 0.0


def test_ghost_kernel_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ghost_kernel(strength=-0.1)
    with pytest.raises(ValueError):
        ghost_kernel(sigma=0.0)


def test_ghost_kernel_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return

    kernel = ghost_kernel(device="cuda")
    assert kernel.is_cuda
    assert torch.isclose(kernel.sum().cpu(), torch.tensor(1.0), atol=1.0e-5)
