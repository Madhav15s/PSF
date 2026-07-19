"""Matplotlib visualization helpers for PSF basis experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _to_numpy(image: torch.Tensor) -> object:
    return image.detach().cpu().float().numpy()


def plot_kernel(kernel: torch.Tensor, title: str = "Basis Kernel") -> tuple[Figure, Axes]:
    """Plot a single basis kernel.

    Purpose:
        Provides a compact inspection view for one normalized basis kernel.

    Args:
        kernel: Tensor with shape ``(H, W)``.
        title: Display title.

    Returns:
        Matplotlib figure and axes containing the plot.

    Raises:
        ValueError: If ``kernel`` is not two-dimensional.

    Mathematical description:
        The function visualizes the scalar kernel values directly with no
        additional normalization.

    Example:
        ``figure, axes = plot_kernel(kernel, title="Gaussian")``
    """
    if kernel.ndim != 2:
        raise ValueError("kernel must have shape (H, W).")

    figure, axes = plt.subplots(figsize=(4, 4))
    image = axes.imshow(_to_numpy(kernel), cmap="magma")
    axes.set_title(title)
    axes.axis("off")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    return figure, axes


def plot_kernel_bank(kernels: list[torch.Tensor], names: list[str]) -> tuple[Figure, list[Axes]]:
    """Plot a bank of basis kernels side by side.

    Purpose:
        Shows every physically motivated basis kernel used by Experiment 0.

    Args:
        kernels: List of tensors with shape ``(H, W)``.
        names: One display name per kernel.

    Returns:
        Matplotlib figure and axes list.

    Raises:
        ValueError: If the kernel and name counts differ or the bank is empty.

    Mathematical description:
        Each basis kernel is visualized as its sampled two-dimensional function.

    Example:
        ``figure, axes = plot_kernel_bank(kernels, ["Gaussian", "Ghost"])``
    """
    if not kernels:
        raise ValueError("kernels must not be empty.")
    if len(kernels) != len(names):
        raise ValueError("kernels and names must have the same length.")

    figure, axes_array = plt.subplots(1, len(kernels), figsize=(3 * len(kernels), 3))
    axes_list = list(axes_array if len(kernels) > 1 else [axes_array])
    for axes, kernel, name in zip(axes_list, kernels, names, strict=True):
        if kernel.ndim != 2:
            raise ValueError("each kernel must have shape (H, W).")
        axes.imshow(_to_numpy(kernel), cmap="magma")
        axes.set_title(name)
        axes.axis("off")
    figure.tight_layout()
    return figure, axes_list


def plot_psf(psf: torch.Tensor, title: str = "Generated PSF") -> tuple[Figure, Axes]:
    """Plot a generated PSF.

    Purpose:
        Inspects the weighted, normalized PSF produced by the generator.

    Args:
        psf: Tensor with shape ``(H, W)`` or ``(1, H, W)``.
        title: Display title.

    Returns:
        Matplotlib figure and axes containing the plot.

    Raises:
        ValueError: If ``psf`` is not two- or three-dimensional as documented.

    Mathematical description:
        The function visualizes the generated PSF values directly.

    Example:
        ``figure, axes = plot_psf(psf[0, 0])``
    """
    if psf.ndim == 3 and psf.shape[0] == 1:
        psf = psf[0]
    if psf.ndim != 2:
        raise ValueError("psf must have shape (H, W) or (1, H, W).")
    return plot_kernel(psf, title=title)


def plot_blur_comparison(
    original: torch.Tensor,
    blurred: torch.Tensor,
    difference: torch.Tensor,
) -> tuple[Figure, list[Axes]]:
    """Plot original, blurred, and difference images.

    Purpose:
        Visualizes the image-domain effect of a generated PSF.

    Args:
        original: Grayscale image tensor with shape ``(H, W)``.
        blurred: Blurred grayscale image tensor with shape ``(H, W)``.
        difference: Difference image tensor with shape ``(H, W)``.

    Returns:
        Matplotlib figure and axes list.

    Raises:
        ValueError: If the inputs do not share the same two-dimensional shape.

    Mathematical description:
        The displayed difference is the caller-provided pixelwise residual,
        commonly ``abs(original - blurred)``.

    Example:
        ``figure, axes = plot_blur_comparison(image, blurred, abs_diff)``
    """
    if original.ndim != 2 or blurred.shape != original.shape or difference.shape != original.shape:
        raise ValueError("all images must have the same two-dimensional shape.")

    figure, axes_array = plt.subplots(1, 3, figsize=(12, 4))
    axes_list = list(axes_array)
    for axes, image, title, cmap in [
        (axes_list[0], original, "Original", "gray"),
        (axes_list[1], blurred, "Blurred", "gray"),
        (axes_list[2], difference, "Absolute Difference", "inferno"),
    ]:
        axes.imshow(_to_numpy(image), cmap=cmap)
        axes.set_title(title)
        axes.axis("off")
    figure.tight_layout()
    return figure, axes_list
