"""Random patch sampling for synthetic EO/IR PSF datasets."""

from __future__ import annotations

import torch

SUPPORTED_PATCH_SIZES = frozenset({64, 128, 256})


class PatchSampler:
    """Randomly extracts square patches from grayscale images."""

    def __init__(self, patch_size: int, seed: int | None = None) -> None:
        """Initialize a deterministic patch sampler.

        Args:
            patch_size: Square patch size. Must be one of ``64``, ``128``, or
                ``256``.
            seed: Optional random seed for reproducible sampling.

        Raises:
            ValueError: If ``patch_size`` is unsupported.
        """
        if patch_size not in SUPPORTED_PATCH_SIZES:
            raise ValueError("patch_size must be one of 64, 128, or 256.")
        self.patch_size = patch_size
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, image: torch.Tensor) -> torch.Tensor:
        """Extract one random square patch.

        Args:
            image: Grayscale tensor with shape ``(H, W)``.

        Returns:
            Tensor with shape ``(patch_size, patch_size)``.

        Raises:
            ValueError: If the image is not two-dimensional or is too small.

        Example:
            ``patch = PatchSampler(128, seed=42).sample(image)``
        """
        if image.ndim != 2:
            raise ValueError("image must have shape (H, W).")
        height, width = image.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("image is smaller than the requested patch size.")

        max_row = height - self.patch_size
        max_column = width - self.patch_size
        row = int(torch.randint(0, max_row + 1, (1,), generator=self.generator).item())
        column = int(torch.randint(0, max_column + 1, (1,), generator=self.generator).item())
        return image[row : row + self.patch_size, column : column + self.patch_size].clone()
