"""Synthetic dataset generator for EO/IR PSF estimation experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from basis import gaussian_kernel, ghost_kernel, motion_kernel, ring_kernel, scatter_kernel
from dataset.image_loader import load_grayscale_images
from dataset.metadata import create_metadata, write_metadata
from dataset.parameter_sampler import ParameterRanges, ParameterSampler, SampledParameters
from dataset.patch_sampler import PatchSampler
from generator import generate_psf

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetGeneratorConfig:
    """Configuration for synthetic dataset generation."""

    input_dir: Path
    output_dir: Path
    samples: int
    input_dirs: tuple[Path, ...] = ()
    patch_size: int = 128
    kernel_size: int = 65
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int | None = None
    parameter_ranges: ParameterRanges = ParameterRanges()
    device: str = "cpu"
    show_progress: bool = True
    resume: bool = True


@dataclass(frozen=True)
class GeneratedSample:
    """In-memory representation of one generated dataset sample."""

    sharp_patch: torch.Tensor
    blurred_patch: torch.Tensor
    psf: torch.Tensor
    metadata: object


@dataclass(frozen=True)
class GenerationSummary:
    """Summary returned after dataset generation."""

    requested_samples: int
    generated_samples: int
    skipped_existing_samples: int
    interrupted: bool
    output_dir: Path


class SyntheticDatasetGenerator:
    """Builds synthetic sharp/blurred/PSF datasets from sharp image folders."""

    def __init__(self, config: DatasetGeneratorConfig) -> None:
        """Initialize a deterministic synthetic dataset generator.

        Args:
            config: Dataset generation configuration.

        Raises:
            ValueError: If split fractions or sample counts are invalid.
        """
        _validate_config(config)
        self.config = config
        self.device = torch.device(config.device)
        self.input_dirs = config.input_dirs if config.input_dirs else (config.input_dir,)
        self.patch_sampler = PatchSampler(config.patch_size, seed=config.random_seed)
        self.parameter_sampler = ParameterSampler(config.parameter_ranges, seed=config.random_seed)
        self.image_choice_generator = torch.Generator(device="cpu")
        if config.random_seed is not None:
            self.image_choice_generator.manual_seed(config.random_seed)

    def generate(self) -> GenerationSummary:
        """Generate and save the configured synthetic dataset.

        Returns:
            Summary with generated, skipped, and interruption status.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            ValueError: If source images are invalid or too small.

        Example:
            ``summary = SyntheticDatasetGenerator(config).generate()``
        """
        source_images = load_grayscale_images(self.input_dirs, device=self.device)
        split_counts = compute_split_counts(
            samples=self.config.samples,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
        )
        create_output_directories(self.config.output_dir)

        generated = 0
        skipped = 0
        interrupted = False
        try:
            for sample_index in range(self.config.samples):
                split_name = split_for_index(sample_index, split_counts)
                sample_id = f"{sample_index + 1:06d}"
                paths = _sample_paths(self.config.output_dir, split_name, sample_id)
                if self.config.resume and all(path.exists() for path in paths.values()):
                    skipped += 1
                    self._progress(sample_index + 1)
                    continue

                source_path, image = self._choose_source_image(source_images)
                sample = self.generate_sample(
                    image=image,
                    source_image=source_path,
                    sample_id=sample_id,
                )
                _save_sample(sample=sample, paths=paths)
                generated += 1
                self._progress(sample_index + 1)
        except KeyboardInterrupt:
            interrupted = True
            print("\nGeneration interrupted. Completed samples remain on disk.", file=sys.stderr)

        self.write_manifest_files()
        if self.config.show_progress:
            print()
        return GenerationSummary(
            requested_samples=self.config.samples,
            generated_samples=generated,
            skipped_existing_samples=skipped,
            interrupted=interrupted,
            output_dir=self.config.output_dir,
        )

    def generate_sample(
        self,
        image: torch.Tensor,
        source_image: Path,
        sample_id: str,
    ) -> GeneratedSample:
        """Generate one synthetic sharp/blurred/PSF sample.

        Args:
            image: Source sharp image tensor with shape ``(H, W)``.
            source_image: Path to the source image.
            sample_id: Zero-padded sample identifier.

        Returns:
            Generated sample containing sharp patch, blurred patch, PSF, and
            metadata.

        Raises:
            ValueError: If patch sampling or PSF generation fails.

        Example:
            ``sample = generator.generate_sample(image, path, "000001")``
        """
        sharp_patch = self.patch_sampler.sample(image)
        parameters = self.parameter_sampler.sample()
        psf = build_psf(parameters=parameters, kernel_size=self.config.kernel_size, device=self.device)
        blurred_patch = blur_patch(sharp_patch, psf)
        metadata = create_metadata(
            sample_id=sample_id,
            parameters=parameters,
            kernel_size=self.config.kernel_size,
            patch_size=self.config.patch_size,
            source_image=source_image,
            random_seed=self.config.random_seed,
        )
        return GeneratedSample(
            sharp_patch=sharp_patch,
            blurred_patch=blurred_patch,
            psf=psf[0, 0],
            metadata=metadata,
        )

    def _choose_source_image(self, source_images: list[tuple[Path, torch.Tensor]]) -> tuple[Path, torch.Tensor]:
        image_index = int(
            torch.randint(0, len(source_images), (1,), generator=self.image_choice_generator).item()
        )
        return source_images[image_index]

    def write_manifest_files(self) -> None:
        """Write train/validation/test manifest CSV files from the discovered sample inventory.

        The original bug was that manifests were derived from a simple sample-index
        split assignment rather than the files actually present on disk. That made
        manifest generation brittle for partial runs, retries, and rebuilds, and it
        could leave val/test manifests empty even when samples existed. We now
        discover generated samples from the output tree, shuffle them with a fixed
        seed, assign splits from that inventory, validate the resulting split sizes,
        and only then write the manifests.
        """
        sample_rows = self._discover_generated_samples()
        if not sample_rows:
            raise FileNotFoundError(f"no generated samples found in: {self.config.output_dir}")

        total_samples = len(sample_rows)
        split_counts = compute_split_counts(
            samples=total_samples,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
        )
        self._validate_manifest_splits(total_samples=total_samples, split_counts=split_counts)

        shuffled_rows = self._shuffle_sample_rows(sample_rows)
        split_rows = self._assign_sample_rows(shuffled_rows, split_counts)

        for split_name in SPLIT_NAMES:
            manifest_rows = [
                {
                    "sample_id": sample_row["sample_id"],
                    "sharp_path": sample_row["sharp_path"].relative_to(self.config.output_dir).as_posix(),
                    "blurred_path": sample_row["blurred_path"].relative_to(self.config.output_dir).as_posix(),
                    "psf_path": sample_row["psf_path"].relative_to(self.config.output_dir).as_posix(),
                    "metadata_path": sample_row["metadata_path"].relative_to(self.config.output_dir).as_posix(),
                }
                for sample_row in split_rows[split_name]
            ]

            manifest_path = self.config.output_dir / f"{split_name}.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "sharp_path", "blurred_path", "psf_path", "metadata_path"],
                )
                writer.writeheader()
                writer.writerows(manifest_rows)

        self._print_manifest_summary(total_samples=total_samples, split_counts=split_counts, split_rows=split_rows)

    def rebuild_manifest_files(self) -> None:
        """Rebuild train/val/test manifests from existing generated samples only."""
        self.write_manifest_files()

    def _discover_generated_samples(self) -> list[dict[str, Path]]:
        discovered: dict[str, dict[str, Path]] = {}
        duplicate_ids: list[str] = []
        # Prefer earlier splits (train, then val, then test) when the same
        # sample_id appears in multiple folders after resumed/reconfigured runs.
        for split_name in SPLIT_NAMES:
            split_root = self.config.output_dir / split_name
            sharp_dir = split_root / "sharp"
            blurred_dir = split_root / "blurred"
            psf_dir = split_root / "psf"
            metadata_dir = split_root / "metadata"
            if not sharp_dir.exists():
                continue

            sharp_stems = _list_stems(sharp_dir, ".png")
            blurred_stems = _list_stems(blurred_dir, ".png")
            psf_stems = _list_stems(psf_dir, ".npy")
            metadata_stems = _list_stems(metadata_dir, ".json")
            complete_ids = sorted(sharp_stems & blurred_stems & psf_stems & metadata_stems)
            print(f"Discovered {len(complete_ids)} complete samples under {split_name}/")

            for sample_id in complete_ids:
                if sample_id in discovered:
                    duplicate_ids.append(sample_id)
                    continue
                discovered[sample_id] = {
                    "sample_id": sample_id,
                    "sharp_path": sharp_dir / f"{sample_id}.png",
                    "blurred_path": blurred_dir / f"{sample_id}.png",
                    "psf_path": psf_dir / f"{sample_id}.npy",
                    "metadata_path": metadata_dir / f"{sample_id}.json",
                }

        if duplicate_ids:
            unique_duplicates = sorted(set(duplicate_ids))
            print(
                f"Warning: skipped {len(unique_duplicates)} duplicate sample_id(s) "
                f"found across split folders (kept first occurrence). "
                f"Examples: {unique_duplicates[:5]}"
            )
        return [discovered[sample_id] for sample_id in sorted(discovered)]

    def _shuffle_sample_rows(self, sample_rows: list[dict[str, Path]]) -> list[dict[str, Path]]:
        if not sample_rows:
            return []
        seed = self.config.random_seed if self.config.random_seed is not None else 42
        shuffled = list(sample_rows)
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffled)
        return shuffled

    def _assign_sample_rows(
        self,
        sample_rows: list[dict[str, Path]],
        split_counts: dict[str, int],
    ) -> dict[str, list[dict[str, Path]]]:
        split_rows = {split_name: [] for split_name in SPLIT_NAMES}
        index = 0
        for split_name in ("train", "val", "test"):
            count = split_counts[split_name]
            split_rows[split_name] = sample_rows[index : index + count]
            index += count
        return split_rows

    def _validate_manifest_splits(self, total_samples: int, split_counts: dict[str, int]) -> None:
        if total_samples <= 0:
            raise ValueError("total samples must be positive.")
        assigned = sum(split_counts[name] for name in SPLIT_NAMES)
        if assigned != total_samples:
            raise ValueError(
                f"split counts must sum to total samples: got {assigned}, expected {total_samples}."
            )
        if split_counts["train"] == 0 and self.config.train_split > 0.0:
            raise ValueError("train split is empty but train_split was requested to be non-zero.")
        if split_counts["val"] == 0 and self.config.val_split > 0.0:
            raise ValueError("validation split is empty but val_split was requested to be non-zero.")
        if split_counts["test"] == 0 and self.config.test_split > 0.0:
            raise ValueError("test split is empty but test_split was requested to be non-zero.")

    def _print_manifest_summary(
        self,
        total_samples: int,
        split_counts: dict[str, int],
        split_rows: dict[str, list[dict[str, Path]]],
    ) -> None:
        print("\nDataset manifest summary")
        print(f"Total samples: {total_samples}")
        for split_name in SPLIT_NAMES:
            count = len(split_rows[split_name])
            percentage = (count / total_samples * 100.0) if total_samples else 0.0
            print(f"{split_name.title()}: {count} samples ({percentage:.2f}%)")

    def _progress(self, completed: int) -> None:
        if not self.config.show_progress:
            return
        width = 30
        fraction = completed / max(self.config.samples, 1)
        filled = int(width * fraction)
        bar = "#" * filled + "." * (width - filled)
        print(f"\r[{bar}] {completed}/{self.config.samples}", end="", flush=True)


def compute_split_counts(
    samples: int,
    train_split: float,
    val_split: float,
    test_split: float,
) -> dict[str, int]:
    """Compute train/validation/test sample counts.

    Args:
        samples: Total number of samples.
        train_split: Train fraction.
        val_split: Validation fraction.
        test_split: Test fraction.

    Returns:
        Dictionary keyed by ``train``, ``val``, and ``test``.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        ``counts = compute_split_counts(100, 0.8, 0.1, 0.1)``
    """
    if samples <= 0:
        raise ValueError("samples must be positive.")
    _validate_splits(train_split, val_split, test_split)
    train_count = int(samples * train_split)
    val_count = int(samples * val_split)
    test_count = samples - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def split_for_index(sample_index: int, split_counts: dict[str, int]) -> str:
    """Return split name for a zero-based sample index.

    Args:
        sample_index: Zero-based sample index.
        split_counts: Counts from ``compute_split_counts``.

    Returns:
        ``train``, ``val``, or ``test``.

    Raises:
        ValueError: If ``sample_index`` is negative.

    Example:
        ``split = split_for_index(12, counts)``
    """
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    train_end = split_counts["train"]
    val_end = train_end + split_counts["val"]
    if sample_index < train_end:
        return "train"
    if sample_index < val_end:
        return "val"
    return "test"


def create_output_directories(output_dir: Path) -> None:
    """Create the required dataset output directory structure.

    Args:
        output_dir: Dataset root directory.

    Raises:
        OSError: If directories cannot be created.

    Example:
        ``create_output_directories(Path("dataset"))``
    """
    for split_name in SPLIT_NAMES:
        for subdirectory in ("sharp", "blurred", "psf", "metadata"):
            (output_dir / split_name / subdirectory).mkdir(parents=True, exist_ok=True)


def build_psf(
    parameters: SampledParameters,
    kernel_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a normalized PSF from sampled Sprint 2 parameters.

    Args:
        parameters: Sampled parameters and normalized weights.
        kernel_size: Odd PSF kernel size.
        device: Torch device for computation.

    Returns:
        PSF tensor with shape ``(1, 1, kernel_size, kernel_size)``.

    Raises:
        ValueError: If Sprint 0 kernel or generator validation fails.

    Example:
        ``psf = build_psf(parameters, 65)``
    """
    kernels = [
        gaussian_kernel(size=kernel_size, sigma=parameters.gaussian_sigma, device=device),
        ghost_kernel(
            size=kernel_size,
            offset_x=parameters.ghost_offset_x,
            offset_y=parameters.ghost_offset_y,
            strength=parameters.ghost_strength,
            sigma=parameters.ghost_sigma,
            device=device,
        ),
        scatter_kernel(size=kernel_size, radius=parameters.scatter_radius, device=device),
        ring_kernel(
            size=kernel_size,
            radius=parameters.ring_radius,
            width=parameters.ring_width,
            device=device,
        ),
        motion_kernel(
            size=kernel_size,
            length=parameters.motion_length,
            angle_deg=parameters.motion_angle,
            device=device,
        ),
    ]
    weights = torch.tensor([parameters.weights], dtype=torch.float32, device=device)
    return generate_psf(weights=weights, kernels=kernels)


def blur_patch(sharp_patch: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Convolve a sharp patch with a PSF.

    Args:
        sharp_patch: Tensor with shape ``(H, W)``.
        psf: Tensor with shape ``(1, 1, K, K)``.

    Returns:
        Blurred patch tensor with shape ``(H, W)`` and values clamped to
        ``[0, 1]``.

    Raises:
        ValueError: If inputs have invalid shapes.

    Example:
        ``blurred = blur_patch(sharp_patch, psf)``
    """
    if sharp_patch.ndim != 2:
        raise ValueError("sharp_patch must have shape (H, W).")
    if psf.ndim != 4 or psf.shape[:2] != (1, 1):
        raise ValueError("psf must have shape (1, 1, K, K).")
    kernel_height, kernel_width = psf.shape[-2:]
    if kernel_height != kernel_width or kernel_height % 2 == 0:
        raise ValueError("psf must be square with an odd spatial size.")

    padding = kernel_height // 2
    patch_batch = sharp_patch.unsqueeze(0).unsqueeze(0)
    padding_mode = "reflect" if min(sharp_patch.shape) > padding else "replicate"
    padded = functional.pad(patch_batch, (padding, padding, padding, padding), mode=padding_mode)
    blurred = functional.conv2d(padded, torch.flip(psf, dims=(-2, -1)))
    return torch.clamp(blurred.squeeze(0).squeeze(0), 0.0, 1.0)


def _validate_config(config: DatasetGeneratorConfig) -> None:
    if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    compute_split_counts(
        samples=config.samples,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
    )


def _validate_splits(train_split: float, val_split: float, test_split: float) -> None:
    splits = (train_split, val_split, test_split)
    if any(split < 0.0 for split in splits):
        raise ValueError("split fractions must be non-negative.")
    if not np.isclose(sum(splits), 1.0, atol=1.0e-8):
        raise ValueError("train, validation, and test splits must sum to 1.0.")


def _sample_paths(output_dir: Path, split_name: str, sample_id: str) -> dict[str, Path]:
    split_root = output_dir / split_name
    return {
        "sharp": split_root / "sharp" / f"{sample_id}.png",
        "blurred": split_root / "blurred" / f"{sample_id}.png",
        "psf": split_root / "psf" / f"{sample_id}.npy",
        "metadata": split_root / "metadata" / f"{sample_id}.json",
    }


def _list_stems(directory: Path, suffix: str) -> set[str]:
    """Return file stems for ``suffix`` entries using a fast directory scan."""
    if not directory.exists():
        return set()
    suffix_lower = suffix.lower()
    stems: set[str] = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if name.lower().endswith(suffix_lower):
                stems.add(name[: -len(suffix)])
    return stems


def _save_sample(sample: GeneratedSample, paths: dict[str, Path]) -> None:
    _save_image(sample.sharp_patch, paths["sharp"])
    _save_image(sample.blurred_patch, paths["blurred"])
    np.save(paths["psf"], sample.psf.detach().cpu().numpy().astype(np.float32))
    write_metadata(sample.metadata, paths["metadata"])


def _save_image(image: torch.Tensor, output_path: Path) -> None:
    image_array = image.detach().cpu().float().numpy()
    image_uint8 = np.uint8(np.clip(image_array, 0.0, 1.0) * 255.0)
    if not cv2.imwrite(str(output_path), image_uint8):
        raise OSError(f"failed to write image: {output_path}")
