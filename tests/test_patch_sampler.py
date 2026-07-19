"""Tests for deterministic patch sampling."""

from __future__ import annotations

import torch

from dataset.patch_sampler import PatchSampler


def test_patch_sampler_extracts_expected_shape_and_is_reproducible() -> None:
    image = torch.arange(0, 10000, dtype=torch.float32).reshape(100, 100) / 10000.0

    sampler_a = PatchSampler(64, seed=42)
    sampler_b = PatchSampler(64, seed=42)

    patch_a = sampler_a.sample(image)
    patch_b = sampler_b.sample(image)

    assert patch_a.shape == (64, 64)
    assert torch.equal(patch_a, patch_b)
    assert patch_a.dtype == torch.float32
