"""Tests for the Gaussian basis kernel."""

import pytest
import torch

from basis.gaussian import gaussian_kernel


def test_gaussian_kernel_properties() -> None:
    kernel = gaussian_kernel(size=65, sigma=2.0)

    assert kernel.shape == (65, 65)
    assert kernel.dtype == torch.float32
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1.0e-5)
    assert not torch.isnan(kernel).any()
    assert not torch.isinf(kernel).any()
    assert torch.all(kernel >= 0.0)
    assert kernel[32, 32] == kernel.max()


def test_gaussian_kernel_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        gaussian_kernel(size=64)
    with pytest.raises(ValueError):
        gaussian_kernel(sigma=0.0)


def test_gaussian_kernel_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return

    kernel = gaussian_kernel(device="cuda")
    assert kernel.is_cuda
    assert torch.isclose(kernel.sum().cpu(), torch.tensor(1.0), atol=1.0e-5)
