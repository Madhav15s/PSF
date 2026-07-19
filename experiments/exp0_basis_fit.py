"""Interactive Experiment 0 basis-fit application.

This script lets a user load a real grayscale EO/IR image, tune physically
motivated basis-kernel parameters and weights, inspect the generated PSF and
blurred image in real time, view quantitative diagnostics, and save the chosen
state for later experiments.

It intentionally does not implement CNNs, dataset generation, training, or any
machine-learning pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from basis import (  # noqa: E402
    gaussian_kernel,
    ghost_kernel,
    motion_kernel,
    ring_kernel,
    scatter_kernel,
)
from generator import generate_psf  # noqa: E402


@dataclass(frozen=True)
class BasisParameters:
    """Mutable UI state represented as an immutable value object."""

    gaussian_weight: float
    ghost_weight: float
    scatter_weight: float
    ring_weight: float
    motion_weight: float
    gaussian_sigma: float
    ghost_offset_x: float
    ghost_offset_y: float
    ghost_strength: float
    ghost_sigma: float
    scatter_radius: float
    ring_radius: float
    ring_width: float
    motion_length: float
    motion_angle_deg: float


@dataclass(frozen=True)
class ExperimentState:
    """Computed state for one interactive update."""

    kernels: list[torch.Tensor]
    kernel_names: list[str]
    weights: torch.Tensor
    psf: torch.Tensor
    blurred: torch.Tensor
    difference: torch.Tensor
    horizontal_edge_profile_original: torch.Tensor
    horizontal_edge_profile_blurred: torch.Tensor
    vertical_edge_profile_original: torch.Tensor
    vertical_edge_profile_blurred: torch.Tensor


def load_grayscale_image(
    image_path: Path,
    device: torch.device,
    max_side: int = 768,
) -> torch.Tensor:
    """Load an image as a normalized grayscale Torch tensor.

    Purpose:
        Reads a real EO/IR image from disk and converts it into the ``(H, W)``
        floating-point representation used by Experiment 0.

    Args:
        image_path: Path to an image readable by OpenCV.
        device: Torch device for the returned tensor.
        max_side: Maximum display-side length after optional downsampling.

    Returns:
        A ``torch.float32`` tensor with shape ``(H, W)`` and values in
        ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If OpenCV cannot decode the image or ``max_side`` is
            invalid.

    Mathematical description:
        Pixel intensities are min-max normalized as
        ``(image - min(image)) / (max(image) - min(image))``.

    Example:
        ``image = load_grayscale_image(Path("frame.png"), torch.device("cpu"))``
    """
    if max_side <= 0:
        raise ValueError("max_side must be positive.")
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = _resize_if_needed(image=image, max_side=max_side)
    normalized = _normalize_array(image.astype(np.float32))
    return torch.from_numpy(normalized).to(device=device, dtype=torch.float32)


def create_sample_image(device: torch.device, size: int = 384) -> torch.Tensor:
    """Create a deterministic grayscale calibration image.

    Purpose:
        Gives the application a runnable initial state when no image is supplied.
        Users can replace it with a real EO/IR image using the Load Image button.

    Args:
        device: Torch device for the returned tensor.
        size: Square image size in pixels.

    Returns:
        A ``torch.float32`` tensor with shape ``(size, size)`` and values in
        ``[0, 1]``.

    Raises:
        ValueError: If ``size`` is too small to contain the calibration pattern.

    Mathematical description:
        Combines smooth gradients, point targets, hot rectangles, and sinusoidal
        texture to expose different blur behaviors.

    Example:
        ``image = create_sample_image(torch.device("cpu"), size=384)``
    """
    if size < 96:
        raise ValueError("size must be at least 96 pixels.")

    coordinates = torch.linspace(0.0, 1.0, size, device=device)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = 0.15 + 0.25 * grid_x + 0.10 * grid_y
    image = image + 0.08 * torch.sin(28.0 * grid_x) * torch.cos(20.0 * grid_y)

    image[size // 5 : size // 5 + 42, size // 7 : size // 7 + 120] = 0.88
    image[size // 2 : size // 2 + 70, size // 2 - 35 : size // 2 + 45] = 0.70

    point_targets = [
        (size // 3, size // 3),
        (size // 4, 3 * size // 4),
        (3 * size // 4, size // 4),
        (2 * size // 3, 2 * size // 3),
    ]
    for row, column in point_targets:
        image[row - 2 : row + 3, column - 2 : column + 3] = 1.0

    return torch.clamp(image, 0.0, 1.0)


def build_basis_kernels(
    parameters: BasisParameters,
    kernel_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[str]]:
    """Build all Experiment 0 basis kernels from UI parameters.

    Purpose:
        Keeps the interactive application tied to the reusable basis-kernel API.

    Args:
        parameters: Current basis and weight parameters.
        kernel_size: Odd PSF kernel size in pixels.
        device: Torch device on which to allocate kernels.

    Returns:
        A kernel list and matching display-name list.

    Raises:
        ValueError: Propagated from individual basis functions when a parameter
            is outside its valid range.

    Mathematical description:
        Constructs the Gaussian, ghost, scatter, ring, and motion basis
        functions sampled on the same centered grid.

    Example:
        ``kernels, names = build_basis_kernels(params, 65, torch.device("cpu"))``
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
            angle_deg=parameters.motion_angle_deg,
            device=device,
        ),
    ]
    return kernels, ["Gaussian", "Ghost", "Scatter", "Ring", "Motion"]


def apply_psf_blur(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Blur a grayscale image with a generated PSF.

    Purpose:
        Applies the currently generated PSF to the loaded image for immediate
        visual comparison.

    Args:
        image: Grayscale image tensor with shape ``(H, W)``.
        psf: PSF tensor with shape ``(1, 1, K, K)``.

    Returns:
        Blurred image tensor with shape ``(H, W)``.

    Raises:
        ValueError: If image or PSF shapes are invalid.

    Mathematical description:
        Computes the two-dimensional convolution ``image * psf`` using reflected
        boundary conditions when image dimensions support reflection padding.

    Example:
        ``blurred = apply_psf_blur(image, psf)``
    """
    if image.ndim != 2:
        raise ValueError("image must have shape (H, W).")
    if psf.ndim != 4 or psf.shape[0] != 1 or psf.shape[1] != 1:
        raise ValueError("psf must have shape (1, 1, K, K).")

    kernel_size = int(psf.shape[-1])
    if psf.shape[-2] != kernel_size or kernel_size % 2 == 0:
        raise ValueError("psf must be square with an odd spatial size.")

    padding = kernel_size // 2
    image_batch = image.unsqueeze(0).unsqueeze(0)
    padding_mode = "reflect" if min(image.shape) > padding else "replicate"
    padded = functional.pad(image_batch, (padding, padding, padding, padding), mode=padding_mode)
    flipped_psf = torch.flip(psf, dims=(-2, -1))
    blurred = functional.conv2d(padded, flipped_psf)
    return torch.clamp(blurred.squeeze(0).squeeze(0), 0.0, 1.0)


def compute_edge_profiles(
    original: torch.Tensor,
    blurred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Sobel edge profiles for original and blurred images.

    Purpose:
        Quantifies how the generated PSF reduces edge energy along image rows
        and columns.

    Args:
        original: Original grayscale image tensor with shape ``(H, W)``.
        blurred: Blurred grayscale image tensor with shape ``(H, W)``.

    Returns:
        Horizontal original profile, horizontal blurred profile, vertical
        original profile, and vertical blurred profile.

    Raises:
        ValueError: If the images do not share a two-dimensional shape.

    Mathematical description:
        Computes Sobel gradient magnitude ``sqrt(gx^2 + gy^2)`` for each image.
        Horizontal profiles average edge magnitude over rows by column; vertical
        profiles average edge magnitude over columns by row.

    Example:
        ``h0, h1, v0, v1 = compute_edge_profiles(image, blurred)``
    """
    if original.ndim != 2 or blurred.shape != original.shape:
        raise ValueError("original and blurred must share shape (H, W).")

    original_edges = _sobel_magnitude(original)
    blurred_edges = _sobel_magnitude(blurred)
    horizontal_original = original_edges.mean(dim=0)
    horizontal_blurred = blurred_edges.mean(dim=0)
    vertical_original = original_edges.mean(dim=1)
    vertical_blurred = blurred_edges.mean(dim=1)
    return horizontal_original, horizontal_blurred, vertical_original, vertical_blurred


def compute_experiment_state(
    image: torch.Tensor,
    parameters: BasisParameters,
    kernel_size: int,
    device: torch.device,
) -> ExperimentState:
    """Compute kernels, PSF, blurred image, difference image, and diagnostics.

    Purpose:
        Provides a side-effect-free computation path for the GUI and tests.

    Args:
        image: Grayscale image tensor with shape ``(H, W)``.
        parameters: Current basis parameters and weights.
        kernel_size: Odd PSF size in pixels.
        device: Torch device on which to compute.

    Returns:
        Fully computed Experiment 0 state.

    Raises:
        ValueError: If inputs are invalid or PSF normalization fails.

    Mathematical description:
        Builds basis functions, forms ``PSF = sum_k w_k B_k``, normalizes the
        PSF, convolves the image, then computes absolute residuals and Sobel
        edge profiles.

    Example:
        ``state = compute_experiment_state(image, params, 65, torch.device("cpu"))``
    """
    kernels, kernel_names = build_basis_kernels(
        parameters=parameters,
        kernel_size=kernel_size,
        device=device,
    )
    weights = torch.tensor(
        [[
            parameters.gaussian_weight,
            parameters.ghost_weight,
            parameters.scatter_weight,
            parameters.ring_weight,
            parameters.motion_weight,
        ]],
        dtype=torch.float32,
        device=device,
    )
    psf = generate_psf(weights=weights, kernels=kernels)
    blurred = apply_psf_blur(image=image, psf=psf)
    difference = torch.abs(image - blurred)
    profiles = compute_edge_profiles(original=image, blurred=blurred)
    return ExperimentState(
        kernels=kernels,
        kernel_names=kernel_names,
        weights=weights,
        psf=psf,
        blurred=blurred,
        difference=difference,
        horizontal_edge_profile_original=profiles[0],
        horizontal_edge_profile_blurred=profiles[1],
        vertical_edge_profile_original=profiles[2],
        vertical_edge_profile_blurred=profiles[3],
    )


def save_experiment_state(
    state: ExperimentState,
    parameters: BasisParameters,
    output_dir: Path,
    source_image: Path | None,
) -> Path:
    """Save selected weights, basis parameters, and generated PSF.

    Purpose:
        Persists the selected Experiment 0 state for later restoration or
        repeatability experiments.

    Args:
        state: Computed experiment state to save.
        parameters: Current UI basis parameters and weights.
        output_dir: Destination directory.
        source_image: Optional path to the loaded EO/IR source image.

    Returns:
        Path to the compressed ``.npz`` state file.

    Raises:
        OSError: If the output directory cannot be created or files cannot be
            written.

    Mathematical description:
        Saves sampled arrays exactly as computed by the generator; no additional
        normalization is applied during serialization.

    Example:
        ``path = save_experiment_state(state, params, Path("outputs"), image_path)``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_path = output_dir / f"exp0_basis_fit_{timestamp}.npz"
    psf_path = output_dir / f"exp0_generated_psf_{timestamp}.npy"
    metadata_path = output_dir / f"exp0_basis_fit_{timestamp}.json"
    preview_path = output_dir / f"exp0_generated_psf_{timestamp}.png"

    psf_array = state.psf.detach().cpu().numpy()[0, 0]
    weights_array = state.weights.detach().cpu().numpy()[0]
    kernel_arrays = np.stack([kernel.detach().cpu().numpy() for kernel in state.kernels], axis=0)
    metadata = {
        "basis_names": state.kernel_names,
        "basis_parameters": asdict(parameters),
        "weights": {
            name: float(value)
            for name, value in zip(state.kernel_names, weights_array, strict=True)
        },
        "psf_sum": float(psf_array.sum()),
        "source_image": str(source_image) if source_image is not None else None,
    }

    np.save(psf_path, psf_array)
    np.savez_compressed(
        state_path,
        psf=psf_array,
        weights=weights_array,
        kernels=kernel_arrays,
        basis_names=np.array(state.kernel_names),
        metadata=json.dumps(metadata, indent=2),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preview = psf_array / max(float(psf_array.max()), 1.0e-12)
    cv2.imwrite(str(preview_path), np.uint8(np.clip(preview, 0.0, 1.0) * 255.0))
    return state_path


class InteractiveBasisFitApp:
    """Matplotlib application for Experiment 0 basis-kernel exploration."""

    def __init__(
        self,
        image_path: Path | None,
        device: torch.device,
        kernel_size: int,
        output_dir: Path,
    ) -> None:
        """Initialize the interactive Experiment 0 application.

        Args:
            image_path: Optional initial EO/IR image path.
            device: Torch device for computation.
            kernel_size: Odd PSF kernel size.
            output_dir: Directory used by the Save State button.
        """
        self.device = device
        self.kernel_size = kernel_size
        self.output_dir = output_dir
        self.source_image = image_path
        self.image = (
            load_grayscale_image(image_path, device=device)
            if image_path is not None
            else create_sample_image(device=device)
        )
        self.state: ExperimentState | None = None
        self.figure = plt.figure(figsize=(16, 10))
        self.image_artists: dict[str, AxesImage] = {}
        self.basis_artists: list[AxesImage] = []
        self.metric_text: Text | None = None
        self.sliders: dict[str, Slider] = {}
        self._build_layout()
        self.refresh()

    def show(self) -> None:
        """Display the interactive Matplotlib application."""
        plt.show()

    def refresh(self) -> None:
        """Recompute and redraw every visible Experiment 0 output."""
        parameters = self._current_parameters()
        self.state = compute_experiment_state(
            image=self.image,
            parameters=parameters,
            kernel_size=self.kernel_size,
            device=self.device,
        )
        self._update_images()
        self._update_profiles()
        self._update_metrics(parameters)
        self._print_metrics(parameters)
        self.figure.canvas.draw_idle()

    def _build_layout(self) -> None:
        grid = self.figure.add_gridspec(
            nrows=3,
            ncols=5,
            left=0.04,
            right=0.98,
            top=0.94,
            bottom=0.34,
            hspace=0.42,
            wspace=0.28,
        )
        axes = {
            "original": self.figure.add_subplot(grid[0, 0]),
            "psf": self.figure.add_subplot(grid[0, 1]),
            "blurred": self.figure.add_subplot(grid[0, 2]),
            "difference": self.figure.add_subplot(grid[0, 3]),
            "metrics": self.figure.add_subplot(grid[0, 4]),
            "profile_x": self.figure.add_subplot(grid[2, 0:3]),
            "profile_y": self.figure.add_subplot(grid[2, 3:5]),
        }
        basis_axes = [self.figure.add_subplot(grid[1, column]) for column in range(5)]

        for key in ["original", "psf", "blurred", "difference"]:
            axes[key].axis("off")
        axes["metrics"].axis("off")
        axes["original"].set_title("Original Image")
        axes["psf"].set_title("Generated PSF")
        axes["blurred"].set_title("Blurred Image")
        axes["difference"].set_title("Absolute Difference")

        blank_image = np.zeros((16, 16), dtype=np.float32)
        self.image_artists["original"] = axes["original"].imshow(blank_image, cmap="gray", vmin=0.0, vmax=1.0)
        self.image_artists["psf"] = axes["psf"].imshow(blank_image, cmap="magma")
        self.image_artists["blurred"] = axes["blurred"].imshow(blank_image, cmap="gray", vmin=0.0, vmax=1.0)
        self.image_artists["difference"] = axes["difference"].imshow(blank_image, cmap="inferno", vmin=0.0)

        for axis, title in zip(basis_axes, ["Gaussian", "Ghost", "Scatter", "Ring", "Motion"], strict=True):
            axis.set_title(title)
            axis.axis("off")
            self.basis_artists.append(axis.imshow(blank_image, cmap="magma"))

        self.metric_text = axes["metrics"].text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        self._profile_axes = (axes["profile_x"], axes["profile_y"])
        self._create_sliders()
        self._create_buttons()

    def _create_sliders(self) -> None:
        slider_specs = [
            ("gaussian_weight", "W Gaussian", 0.0, 2.0, 1.0),
            ("ghost_weight", "W Ghost", 0.0, 2.0, 0.4),
            ("scatter_weight", "W Scatter", 0.0, 2.0, 0.25),
            ("ring_weight", "W Ring", 0.0, 2.0, 0.15),
            ("motion_weight", "W Motion", 0.0, 2.0, 0.3),
            ("gaussian_sigma", "Gaussian sigma", 0.5, 12.0, 2.0),
            ("ghost_offset_x", "Ghost dx", -24.0, 24.0, 8.0),
            ("ghost_offset_y", "Ghost dy", -24.0, 24.0, 0.0),
            ("ghost_strength", "Ghost strength", 0.0, 1.0, 0.2),
            ("ghost_sigma", "Ghost sigma", 0.5, 12.0, 2.0),
            ("scatter_radius", "Scatter radius", 2.0, 32.0, 15.0),
            ("ring_radius", "Ring radius", 0.0, 32.0, 12.0),
            ("ring_width", "Ring width", 0.5, 12.0, 3.0),
            ("motion_length", "Motion length", 1.0, 32.0, 10.0),
            ("motion_angle_deg", "Motion angle", -180.0, 180.0, 45.0),
        ]
        left_x = 0.09
        right_x = 0.58
        width = 0.34
        height = 0.016
        top_y = 0.285
        row_gap = 0.032

        for index, (key, label, minimum, maximum, initial) in enumerate(slider_specs):
            column = 0 if index < 8 else 1
            row = index if column == 0 else index - 8
            x_position = left_x if column == 0 else right_x
            y_position = top_y - row * row_gap
            axis = self.figure.add_axes([x_position, y_position, width, height])
            slider = Slider(
                ax=axis,
                label=label,
                valmin=minimum,
                valmax=maximum,
                valinit=initial,
                valfmt="%0.3f",
            )
            slider.on_changed(self._on_slider_changed)
            self.sliders[key] = slider

    def _create_buttons(self) -> None:
        load_axis = self.figure.add_axes([0.09, 0.015, 0.14, 0.04])
        save_axis = self.figure.add_axes([0.25, 0.015, 0.14, 0.04])
        reset_axis = self.figure.add_axes([0.41, 0.015, 0.14, 0.04])
        load_button = Button(load_axis, "Load Image")
        save_button = Button(save_axis, "Save State")
        reset_button = Button(reset_axis, "Reset Sliders")
        load_button.on_clicked(self._on_load_clicked)
        save_button.on_clicked(self._on_save_clicked)
        reset_button.on_clicked(self._on_reset_clicked)
        self._buttons = [load_button, save_button, reset_button]

    def _current_parameters(self) -> BasisParameters:
        return BasisParameters(
            gaussian_weight=float(self.sliders["gaussian_weight"].val),
            ghost_weight=float(self.sliders["ghost_weight"].val),
            scatter_weight=float(self.sliders["scatter_weight"].val),
            ring_weight=float(self.sliders["ring_weight"].val),
            motion_weight=float(self.sliders["motion_weight"].val),
            gaussian_sigma=float(self.sliders["gaussian_sigma"].val),
            ghost_offset_x=float(self.sliders["ghost_offset_x"].val),
            ghost_offset_y=float(self.sliders["ghost_offset_y"].val),
            ghost_strength=float(self.sliders["ghost_strength"].val),
            ghost_sigma=float(self.sliders["ghost_sigma"].val),
            scatter_radius=float(self.sliders["scatter_radius"].val),
            ring_radius=float(self.sliders["ring_radius"].val),
            ring_width=float(self.sliders["ring_width"].val),
            motion_length=float(self.sliders["motion_length"].val),
            motion_angle_deg=float(self.sliders["motion_angle_deg"].val),
        )

    def _update_images(self) -> None:
        if self.state is None:
            raise RuntimeError("state must be computed before updating images.")

        self.image_artists["original"].set_data(_to_numpy(self.image))
        self.image_artists["blurred"].set_data(_to_numpy(self.state.blurred))
        self.image_artists["difference"].set_data(_to_numpy(self.state.difference))
        self.image_artists["difference"].set_clim(vmin=0.0, vmax=max(float(self.state.difference.max()), 1.0e-6))

        psf_image = self.state.psf[0, 0]
        self.image_artists["psf"].set_data(_to_numpy(psf_image))
        self.image_artists["psf"].set_clim(vmin=0.0, vmax=max(float(psf_image.max()), 1.0e-12))

        for artist, kernel in zip(self.basis_artists, self.state.kernels, strict=True):
            artist.set_data(_to_numpy(kernel))
            artist.set_clim(vmin=0.0, vmax=max(float(kernel.max()), 1.0e-12))

    def _update_profiles(self) -> None:
        if self.state is None:
            raise RuntimeError("state must be computed before updating profiles.")

        horizontal_axis, vertical_axis = self._profile_axes
        horizontal_axis.clear()
        vertical_axis.clear()
        horizontal_axis.plot(_to_numpy(self.state.horizontal_edge_profile_original), label="Original")
        horizontal_axis.plot(_to_numpy(self.state.horizontal_edge_profile_blurred), label="Blurred")
        vertical_axis.plot(_to_numpy(self.state.vertical_edge_profile_original), label="Original")
        vertical_axis.plot(_to_numpy(self.state.vertical_edge_profile_blurred), label="Blurred")
        horizontal_axis.set_title("Horizontal Edge Profile")
        vertical_axis.set_title("Vertical Edge Profile")
        horizontal_axis.set_xlabel("Column")
        vertical_axis.set_xlabel("Row")
        horizontal_axis.set_ylabel("Mean Sobel magnitude")
        vertical_axis.set_ylabel("Mean Sobel magnitude")
        horizontal_axis.grid(True, alpha=0.25)
        vertical_axis.grid(True, alpha=0.25)
        horizontal_axis.legend(loc="upper right")
        vertical_axis.legend(loc="upper right")

    def _update_metrics(self, parameters: BasisParameters) -> None:
        if self.state is None or self.metric_text is None:
            raise RuntimeError("state and metric text must exist before updating metrics.")

        weights = self.state.weights[0].detach().cpu().tolist()
        kernel_sums = [float(kernel.sum()) for kernel in self.state.kernels]
        psf_sum = float(self.state.psf.sum())
        text = [
            "Experiment 0 Metrics",
            "",
            f"PSF sum: {psf_sum:.8f}",
            f"PSF min/max: {float(self.state.psf.min()):.3e} / {float(self.state.psf.max()):.3e}",
            f"Mean abs diff: {float(self.state.difference.mean()):.6f}",
            f"Max abs diff: {float(self.state.difference.max()):.6f}",
            "",
            "Weights",
        ]
        text.extend(
            f"{name:<8}: {value:.4f}"
            for name, value in zip(self.state.kernel_names, weights, strict=True)
        )
        text.extend(["", "Kernel sums"])
        text.extend(
            f"{name:<8}: {value:.8f}"
            for name, value in zip(self.state.kernel_names, kernel_sums, strict=True)
        )
        text.extend(["", f"Source: {self.source_image if self.source_image else 'sample image'}"])
        self.metric_text.set_text("\n".join(text))
        self.figure.suptitle(
            "Experiment 0 Interactive Basis-Fit Tool",
            fontsize=14,
        )
        _ = parameters

    def _print_metrics(self, parameters: BasisParameters) -> None:
        if self.state is None:
            raise RuntimeError("state must exist before printing metrics.")

        weights = {
            name: float(value)
            for name, value in zip(
                self.state.kernel_names,
                self.state.weights[0].detach().cpu().tolist(),
                strict=True,
            )
        }
        kernel_sums = {
            name: float(kernel.sum())
            for name, kernel in zip(self.state.kernel_names, self.state.kernels, strict=True)
        }
        print(
            json.dumps(
                {
                    "weights": weights,
                    "kernel_normalization": kernel_sums,
                    "psf_normalization": float(self.state.psf.sum()),
                    "parameters": asdict(parameters),
                },
                indent=2,
            )
        )

    def _on_slider_changed(self, _: float) -> None:
        self.refresh()

    def _on_load_clicked(self, _: object) -> None:
        selected_path = _choose_image_file()
        if selected_path is None:
            return
        self.image = load_grayscale_image(selected_path, device=self.device)
        self.source_image = selected_path
        self.refresh()

    def _on_save_clicked(self, _: object) -> None:
        if self.state is None:
            raise RuntimeError("state must exist before saving.")
        saved_path = save_experiment_state(
            state=self.state,
            parameters=self._current_parameters(),
            output_dir=self.output_dir,
            source_image=self.source_image,
        )
        print(f"Saved Experiment 0 state: {saved_path}")

    def _on_reset_clicked(self, _: object) -> None:
        for slider in self.sliders.values():
            slider.reset()


def run_noninteractive_smoke(
    image_path: Path | None,
    device: torch.device,
    kernel_size: int,
) -> ExperimentState:
    """Run one Experiment 0 computation without opening the GUI.

    Purpose:
        Supports command-line validation and automated tests of the app path.

    Args:
        image_path: Optional image path to load.
        device: Torch computation device.
        kernel_size: Odd PSF size.

    Returns:
        Computed Experiment 0 state.

    Raises:
        ValueError: If computation inputs are invalid.

    Mathematical description:
        Uses the same basis generation, PSF normalization, convolution, and
        metric computation as the interactive app.

    Example:
        ``state = run_noninteractive_smoke(None, torch.device("cpu"), 65)``
    """
    image = (
        load_grayscale_image(image_path, device=device)
        if image_path is not None
        else create_sample_image(device=device, size=192)
    )
    parameters = BasisParameters(
        gaussian_weight=1.0,
        ghost_weight=0.4,
        scatter_weight=0.25,
        ring_weight=0.15,
        motion_weight=0.3,
        gaussian_sigma=2.0,
        ghost_offset_x=8.0,
        ghost_offset_y=0.0,
        ghost_strength=0.2,
        ghost_sigma=2.0,
        scatter_radius=15.0,
        ring_radius=12.0,
        ring_width=3.0,
        motion_length=10.0,
        motion_angle_deg=45.0,
    )
    state = compute_experiment_state(
        image=image,
        parameters=parameters,
        kernel_size=kernel_size,
        device=device,
    )
    print(
        json.dumps(
            {
                "weights": state.weights[0].detach().cpu().tolist(),
                "kernel_normalization": [float(kernel.sum()) for kernel in state.kernels],
                "psf_normalization": float(state.psf.sum()),
                "mean_abs_difference": float(state.difference.mean()),
            },
            indent=2,
        )
    )
    return state


def parse_args() -> argparse.Namespace:
    """Parse Experiment 0 command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Experiment 0 PSF basis-fit tool.")
    parser.add_argument("--image", type=Path, default=None, help="Optional EO/IR image to load at startup.")
    parser.add_argument("--kernel-size", type=int, default=65, help="Odd PSF kernel size.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for saved states.")
    parser.add_argument("--no-show", action="store_true", help="Run one computation without opening the GUI.")
    return parser.parse_args()


def main() -> None:
    """Run the interactive Experiment 0 application."""
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd integer.")

    if args.no_show:
        run_noninteractive_smoke(
            image_path=args.image,
            device=device,
            kernel_size=args.kernel_size,
        )
        return

    app = InteractiveBasisFitApp(
        image_path=args.image,
        device=device,
        kernel_size=args.kernel_size,
        output_dir=args.output_dir,
    )
    app.show()


def _normalize_array(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum - minimum <= 1.0e-12:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - minimum) / (maximum - minimum)).astype(np.float32)


def _resize_if_needed(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image
    scale = max_side / float(longest_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    )
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    )
    kernels = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)
    image_batch = image.unsqueeze(0).unsqueeze(0)
    padded = functional.pad(image_batch, (1, 1, 1, 1), mode="replicate")
    gradients = functional.conv2d(padded, kernels)
    return torch.sqrt(gradients[:, 0].square() + gradients[:, 1].square() + 1.0e-12).squeeze(0)


def _choose_image_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Load EO/IR image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


if __name__ == "__main__":
    main()
