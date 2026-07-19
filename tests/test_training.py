"""Smoke tests for the training subsystem."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image

from models.resnet18_regressor import ResNet18Regressor
from training.dataset import PSFDataset


def test_psf_dataset_reads_manifest_and_targets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "metadata").mkdir(parents=True, exist_ok=True)

    image_path = data_dir / "train" / "sample.png"
    Image.fromarray((255 * torch.ones(64, 64, 3, dtype=torch.uint8)).numpy()).save(image_path)

    metadata = {
        "sample_id": "000001",
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gaussian_sigma": 1.0,
        "ghost_offset_x": 0.0,
        "ghost_offset_y": 0.0,
        "ghost_strength": 0.1,
        "scatter_radius": 1.0,
        "ring_radius": 2.0,
        "ring_width": 0.5,
        "motion_angle": 10.0,
        "motion_length": 3.0,
        "kernel_size": 65,
        "patch_size": 128,
        "source_image": "source.png",
        "random_seed": 42,
    }
    metadata_path = data_dir / "metadata" / "sample.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    manifest_path = data_dir / "train.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sharp_path", "blurred_path", "psf_path", "metadata_path"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "000001",
                "sharp_path": "train/sample.png",
                "blurred_path": "train/sample.png",
                "psf_path": "train/sample.npy",
                "metadata_path": "metadata/sample.json",
            }
        )

    dataset = PSFDataset(root_dir=data_dir, manifest_path=manifest_path, image_key="blurred_path", target_dim=5)
    sample_image, sample_target = dataset[0]

    assert sample_image.shape == (3, 64, 64)
    assert sample_target.shape == (5,)
    assert torch.allclose(sample_target, torch.tensor(metadata["weights"], dtype=torch.float32))


def test_resnet18_regressor_outputs_requested_dimension() -> None:
    model = ResNet18Regressor(num_outputs=7, pretrained=False, dropout=0.1)
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)

    assert outputs.shape == (2, 7)
