"""Random parameter sampling for synthetic EO/IR PSF generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FloatRange:
    """Closed numeric range used for uniform sampling."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate range bounds."""
        if self.minimum > self.maximum:
            raise ValueError("range minimum must be less than or equal to maximum.")


@dataclass(frozen=True)
class ParameterRanges:
    """Configurable sampling ranges for Sprint 2 synthetic PSF parameters."""

    gaussian_sigma: FloatRange = FloatRange(0.8, 6.0)
    ghost_offset_x: FloatRange = FloatRange(-16.0, 16.0)
    ghost_offset_y: FloatRange = FloatRange(-16.0, 16.0)
    ghost_strength: FloatRange = FloatRange(0.0, 0.6)
    ghost_sigma: FloatRange = FloatRange(0.8, 6.0)
    scatter_radius: FloatRange = FloatRange(4.0, 28.0)
    ring_radius: FloatRange = FloatRange(2.0, 28.0)
    ring_width: FloatRange = FloatRange(0.8, 8.0)
    motion_angle: FloatRange = FloatRange(-180.0, 180.0)
    motion_length: FloatRange = FloatRange(1.0, 28.0)
    basis_weight: FloatRange = FloatRange(0.0, 1.0)


@dataclass(frozen=True)
class SampledParameters:
    """One sampled PSF parameter set and normalized basis weight vector."""

    weights: list[float]
    gaussian_sigma: float
    ghost_offset_x: float
    ghost_offset_y: float
    ghost_strength: float
    ghost_sigma: float
    scatter_radius: float
    ring_radius: float
    ring_width: float
    motion_angle: float
    motion_length: float


class ParameterSampler:
    """Samples reproducible synthetic PSF parameters."""

    def __init__(self, ranges: ParameterRanges | None = None, seed: int | None = None) -> None:
        """Initialize the parameter sampler.

        Args:
            ranges: Optional custom parameter ranges.
            seed: Optional random seed for deterministic sampling.
        """
        self.ranges = ranges if ranges is not None else ParameterRanges()
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self) -> SampledParameters:
        """Sample one normalized parameter set.

        Returns:
            A ``SampledParameters`` instance with five normalized basis weights.

        Raises:
            ValueError: If sampled basis weights sum to zero.

        Example:
            ``params = ParameterSampler(seed=42).sample()``
        """
        raw_weights = torch.tensor(
            [self._uniform(self.ranges.basis_weight) for _ in range(5)],
            dtype=torch.float32,
        )
        weight_sum = raw_weights.sum()
        if torch.isclose(weight_sum, torch.tensor(0.0)):
            raise ValueError("basis weights must not sum to zero.")
        weights = (raw_weights / weight_sum).tolist()

        return SampledParameters(
            weights=[float(weight) for weight in weights],
            gaussian_sigma=self._uniform(self.ranges.gaussian_sigma),
            ghost_offset_x=self._uniform(self.ranges.ghost_offset_x),
            ghost_offset_y=self._uniform(self.ranges.ghost_offset_y),
            ghost_strength=self._uniform(self.ranges.ghost_strength),
            ghost_sigma=self._uniform(self.ranges.ghost_sigma),
            scatter_radius=self._uniform(self.ranges.scatter_radius),
            ring_radius=self._uniform(self.ranges.ring_radius),
            ring_width=self._uniform(self.ranges.ring_width),
            motion_angle=self._uniform(self.ranges.motion_angle),
            motion_length=self._uniform(self.ranges.motion_length),
        )

    def _uniform(self, value_range: FloatRange) -> float:
        sample = torch.rand(1, generator=self.generator).item()
        return float(value_range.minimum + (value_range.maximum - value_range.minimum) * sample)
