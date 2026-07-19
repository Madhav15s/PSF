"""Image loading utilities for synthetic EO/IR PSF datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import torch

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def discover_image_files(input_dir: Path | Iterable[Path]) -> list[Path]:
    """Discover supported image files recursively from one or more roots.

    Purpose:
        Finds candidate sharp grayscale images for synthetic dataset generation
        from arbitrary source collections.

    Args:
        input_dir: Directory or directories to traverse recursively.

    Returns:
        Sorted list of supported image paths.

    Raises:
        FileNotFoundError: If any input directory does not exist.
        NotADirectoryError: If any input path is not a directory.
        ValueError: If no supported images are found.

    Example:
        ``paths = discover_image_files([Path("dataset_a"), Path("dataset_b")])``
    """
    roots = [input_dir] if isinstance(input_dir, Path) else list(input_dir)
    if not roots:
        raise ValueError("at least one input directory is required.")

    image_paths: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"input directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"input path is not a directory: {root}")

        image_paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    if not image_paths:
        raise ValueError(f"no supported image files found in: {roots}")
    return sorted(image_paths)


def load_grayscale_image(image_path: Path, device: torch.device | str = "cpu") -> torch.Tensor:
    """Load one image as a normalized grayscale Torch tensor.

    Purpose:
        Validates and decodes a sharp source image for patch sampling.

    Args:
        image_path: Path to a supported image file.
        device: Torch device for the returned tensor.

    Returns:
        ``torch.float32`` tensor with shape ``(H, W)`` and values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the extension is unsupported or decoding fails.

    Example:
        ``image = load_grayscale_image(Path("img001.png"))``
    """
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {image_path.suffix}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"corrupted or unreadable image: {image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"expected a grayscale-compatible image: {image_path}")

    image_array = image.astype(np.float32)
    max_value = _dtype_max_value(image)
    if max_value <= 0.0:
        raise ValueError(f"invalid image dynamic range: {image_path}")
    image_array = np.clip(image_array / max_value, 0.0, 1.0)
    return torch.from_numpy(image_array).to(device=device, dtype=torch.float32)


def load_grayscale_images(
    input_dir: Path | Iterable[Path],
    device: torch.device | str = "cpu",
) -> list[tuple[Path, torch.Tensor]]:
    """Load all supported images from one or more directories recursively.

    Purpose:
        Provides the validated source image bank used by the dataset generator.

    Args:
        input_dir: Directory or directories containing sharp source images.
        device: Torch device for returned tensors.

    Returns:
        List of ``(path, image_tensor)`` tuples.

    Raises:
        ValueError: If any candidate image is corrupted or unsupported.

    Example:
        ``images = load_grayscale_images([Path("dataset_a"), Path("dataset_b")])``
    """
    return [(path, load_grayscale_image(path, device=device)) for path in discover_image_files(input_dir)]


def _dtype_max_value(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    maximum = float(np.max(image))
    return maximum if maximum > 1.0 else 1.0
