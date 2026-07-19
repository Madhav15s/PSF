"""Tests for the ring reflection basis kernel."""

import pytest
import torch

from basis.ring import ring_kernel


def test_ring_kernel_properties() -> None:
    kernel = ring_kernel(size=65, radius=12.0, width=3.0)

    assert kernel.shape == (65, 65)
    assert kernel.dtype == torch.float32
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1.0e-5)
    assert not torch.isnan(kernel).any()
    assert not torch.isinf(kernel).any()
    assert torch.all(kernel >= 0.0)
    assert kernel[32, 32] > 0.0


def test_ring_kernel_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ring_kernel(radius=-1.0)
    with pytest.raises(ValueError):
        ring_kernel(width=0.0)


def test_ring_kernel_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return

    kernel = ring_kernel(device="cuda")
    assert kernel.is_cuda
    assert torch.isclose(kernel.sum().cpu(), torch.tensor(1.0), atol=1.0e-5)
