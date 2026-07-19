"""Tests for the non-GUI parts of the Experiment 0 application."""

from pathlib import Path

import torch

from experiments.exp0_basis_fit import (
    BasisParameters,
    compute_edge_profiles,
    compute_experiment_state,
    create_sample_image,
    save_experiment_state,
)


def _parameters() -> BasisParameters:
    return BasisParameters(
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


def test_compute_experiment_state_outputs_expected_shapes() -> None:
    device = torch.device("cpu")
    image = create_sample_image(device=device, size=128)
    state = compute_experiment_state(
        image=image,
        parameters=_parameters(),
        kernel_size=65,
        device=device,
    )

    assert len(state.kernels) == 5
    assert state.psf.shape == (1, 1, 65, 65)
    assert state.blurred.shape == image.shape
    assert state.difference.shape == image.shape
    assert torch.isclose(state.psf.sum(), torch.tensor(1.0), atol=1.0e-5)
    assert torch.all(state.difference >= 0.0)


def test_compute_edge_profiles_outputs_one_profile_per_axis() -> None:
    image = create_sample_image(device=torch.device("cpu"), size=128)
    blurred = image.clone()

    horizontal_original, horizontal_blurred, vertical_original, vertical_blurred = compute_edge_profiles(
        original=image,
        blurred=blurred,
    )

    assert horizontal_original.shape == (128,)
    assert horizontal_blurred.shape == (128,)
    assert vertical_original.shape == (128,)
    assert vertical_blurred.shape == (128,)


def test_save_experiment_state_writes_npz_and_sidecars(tmp_path: Path) -> None:
    device = torch.device("cpu")
    image = create_sample_image(device=device, size=128)
    parameters = _parameters()
    state = compute_experiment_state(
        image=image,
        parameters=parameters,
        kernel_size=65,
        device=device,
    )

    saved_path = save_experiment_state(
        state=state,
        parameters=parameters,
        output_dir=tmp_path,
        source_image=None,
    )

    assert saved_path.exists()
    assert list(tmp_path.glob("exp0_generated_psf_*.npy"))
    assert list(tmp_path.glob("exp0_basis_fit_*.json"))
    assert list(tmp_path.glob("exp0_generated_psf_*.png"))
