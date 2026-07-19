"""Tests for the directional motion basis kernel."""

import pytest
import torch

from basis.motion import motion_kernel


def test_motion_kernel_properties() -> None:
    kernel = motion_kernel(size=65, length=10.0, angle_deg=45.0)

    assert kernel.shape == (65, 65)
    assert kernel.dtype == torch.float32
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1.0e-5)
    assert not torch.isnan(kernel).any()
    assert not torch.isinf(kernel).any()
    assert torch.all(kernel >= 0.0)
    assert kernel[32, 32] > 0.0


def test_motion_kernel_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        motion_kernel(length=0.0)
    with pytest.raises(ValueError):
        motion_kernel(size=64)


def test_motion_kernel_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        return

    kernel = motion_kernel(device="cuda")
    assert kernel.is_cuda
    assert torch.isclose(kernel.sum().cpu(), torch.tensor(1.0), atol=1.0e-5)
