"""Tests for deterministic PSF parameter sampling."""

from __future__ import annotations

import pytest

from dataset.parameter_sampler import FloatRange, ParameterRanges, ParameterSampler


def test_parameter_sampler_is_reproducible_and_normalizes_weights() -> None:
    sampler_a = ParameterSampler(seed=7)
    sampler_b = ParameterSampler(seed=7)

    params_a = sampler_a.sample()
    params_b = sampler_b.sample()

    assert params_a == params_b
    assert len(params_a.weights) == 5
    assert abs(sum(params_a.weights) - 1.0) < 1.0e-6
    assert params_a.gaussian_sigma >= 0.8
    assert params_a.motion_length <= 28.0


def test_parameter_ranges_reject_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        FloatRange(2.0, 1.0)
