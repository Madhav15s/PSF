"""Metadata model and JSON serialization for generated EO/IR PSF samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset.parameter_sampler import SampledParameters


@dataclass(frozen=True)
class SampleMetadata:
    """Serializable metadata for one synthetic dataset sample."""

    sample_id: str
    weights: list[float]
    gaussian_sigma: float
    ghost_offset_x: float
    ghost_offset_y: float
    ghost_strength: float
    scatter_radius: float
    ring_radius: float
    ring_width: float
    motion_angle: float
    motion_length: float
    kernel_size: int
    patch_size: int
    source_image: str
    random_seed: int | None


def create_metadata(
    sample_id: str,
    parameters: SampledParameters,
    kernel_size: int,
    patch_size: int,
    source_image: Path,
    random_seed: int | None,
) -> SampleMetadata:
    """Create metadata for a generated sample.

    Args:
        sample_id: Zero-padded sample identifier.
        parameters: Sampled PSF parameters and normalized weights.
        kernel_size: Generated PSF kernel size.
        patch_size: Saved sharp and blurred patch size.
        source_image: Source image path used for sampling.
        random_seed: Dataset generation seed, if supplied.

    Returns:
        A ``SampleMetadata`` value ready for JSON serialization.

    Example:
        ``metadata = create_metadata("000001", params, 65, 128, path, 42)``
    """
    return SampleMetadata(
        sample_id=sample_id,
        weights=parameters.weights,
        gaussian_sigma=parameters.gaussian_sigma,
        ghost_offset_x=parameters.ghost_offset_x,
        ghost_offset_y=parameters.ghost_offset_y,
        ghost_strength=parameters.ghost_strength,
        scatter_radius=parameters.scatter_radius,
        ring_radius=parameters.ring_radius,
        ring_width=parameters.ring_width,
        motion_angle=parameters.motion_angle,
        motion_length=parameters.motion_length,
        kernel_size=kernel_size,
        patch_size=patch_size,
        source_image=str(source_image),
        random_seed=random_seed,
    )


def write_metadata(metadata: SampleMetadata, output_path: Path) -> None:
    """Write sample metadata as formatted JSON.

    Args:
        metadata: Metadata to serialize.
        output_path: Destination JSON path.

    Raises:
        OSError: If the file cannot be written.

    Example:
        ``write_metadata(metadata, Path("metadata/000001.json"))``
    """
    output_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
