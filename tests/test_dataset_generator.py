"""Tests for synthetic dataset generation and manifest export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.dataset_generator import (
    DatasetGeneratorConfig,
    SyntheticDatasetGenerator,
    compute_split_counts,
    create_output_directories,
    split_for_index,
)
from dataset.image_loader import load_grayscale_images
from dataset.parameter_sampler import ParameterRanges


def _write_image(output_path: Path, image_array: np.ndarray) -> None:
    image_uint8 = np.uint8(np.clip(image_array, 0.0, 1.0) * 255.0)
    assert cv2.imwrite(str(output_path), image_uint8)


def _create_input_dir(base_dir: Path) -> Path:
    input_dir = base_dir / "input"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)

    for index, image_array in enumerate(
        [
            np.ones((256, 256), dtype=np.float32),
            np.linspace(0.0, 1.0, num=256 * 256, dtype=np.float32).reshape(256, 256),
        ]
    ):
        _write_image(nested_dir / f"img{index}.png", image_array)

    return input_dir


def test_dataset_generation_creates_outputs_and_manifest(tmp_path: Path) -> None:
    input_dir = _create_input_dir(tmp_path)
    output_dir = tmp_path / "dataset"

    config = DatasetGeneratorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        samples=6,
        patch_size=64,
        kernel_size=5,
        train_split=0.5,
        val_split=0.25,
        test_split=0.25,
        random_seed=7,
        parameter_ranges=ParameterRanges(),
        device="cpu",
        show_progress=False,
        resume=False,
    )

    summary = SyntheticDatasetGenerator(config).generate()

    assert summary.generated_samples == 6
    assert summary.skipped_existing_samples == 0
    assert summary.output_dir == output_dir

    create_output_directories(output_dir)
    assert (output_dir / "train" / "sharp").exists()
    assert (output_dir / "train" / "blurred").exists()
    assert (output_dir / "train" / "psf").exists()
    assert (output_dir / "train" / "metadata").exists()

    counts = compute_split_counts(6, 0.5, 0.25, 0.25)
    assert counts == {"train": 3, "val": 1, "test": 2}
    assert split_for_index(0, counts) == "train"
    assert split_for_index(3, counts) == "val"
    assert split_for_index(4, counts) == "test"

    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / f"{split_name}.csv"
        assert manifest_path.exists()
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == counts[split_name]

        for row in rows:
            assert row["sample_id"]
            assert row["sharp_path"].endswith(".png")
            assert row["blurred_path"].endswith(".png")
            assert row["psf_path"].endswith(".npy")
            assert row["metadata_path"].endswith(".json")
            assert (output_dir / row["sharp_path"]).exists()
            assert (output_dir / row["blurred_path"]).exists()
            assert (output_dir / row["psf_path"]).exists()
            assert (output_dir / row["metadata_path"]).exists()

            metadata = json.loads((output_dir / row["metadata_path"]).read_text(encoding="utf-8"))
            assert metadata["sample_id"] == row["sample_id"]
            assert metadata["patch_size"] == 64
            assert metadata["kernel_size"] == 5
            assert metadata["random_seed"] == 7


def test_load_grayscale_images_discovers_nested_images(tmp_path: Path) -> None:
    input_dir = _create_input_dir(tmp_path)

    images = load_grayscale_images(input_dir)

    assert len(images) == 2
    for _, image in images:
        assert image.ndim == 2
        assert image.min() >= 0.0
        assert image.max() <= 1.0


def test_load_grayscale_images_accepts_multiple_input_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "dataset_a"
    second_root = tmp_path / "dataset_b"
    first_root.mkdir(parents=True, exist_ok=True)
    second_root.mkdir(parents=True, exist_ok=True)

    _write_image(first_root / "img_a.png", np.ones((64, 64), dtype=np.float32))
    _write_image(second_root / "img_b.png", np.linspace(0.0, 1.0, num=64 * 64, dtype=np.float32).reshape(64, 64))

    images = load_grayscale_images([first_root, second_root])

    assert len(images) == 2
    assert {path.parent.name for path, _ in images} == {"dataset_a", "dataset_b"}


def test_invalid_dataset_config_rejected() -> None:
    with pytest.raises(ValueError):
        compute_split_counts(0, 0.8, 0.1, 0.1)

    with pytest.raises(ValueError):
        compute_split_counts(10, 0.8, 0.1, 0.05)


def test_rebuild_manifests_from_existing_samples_is_deterministic(tmp_path: Path) -> None:
    input_dir = _create_input_dir(tmp_path)
    output_dir = tmp_path / "dataset"

    config = DatasetGeneratorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        samples=10,
        patch_size=64,
        kernel_size=5,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
        parameter_ranges=ParameterRanges(),
        device="cpu",
        show_progress=False,
        resume=False,
    )
    SyntheticDatasetGenerator(config).generate()

    rebuild_config = DatasetGeneratorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        samples=1,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
        show_progress=False,
    )
    generator = SyntheticDatasetGenerator(rebuild_config)
    generator.rebuild_manifest_files()

    manifests = {}
    for split_name in ("train", "val", "test"):
        with (output_dir / f"{split_name}.csv").open(encoding="utf-8", newline="") as handle:
            manifests[split_name] = list(csv.DictReader(handle))

    assert len(manifests["train"]) == 8
    assert len(manifests["val"]) == 1
    assert len(manifests["test"]) == 1

    all_ids = [row["sample_id"] for rows in manifests.values() for row in rows]
    assert len(all_ids) == 10
    assert len(set(all_ids)) == 10

    for rows in manifests.values():
        for row in rows:
            assert (output_dir / row["sharp_path"]).exists()
            assert (output_dir / row["blurred_path"]).exists()
            assert (output_dir / row["psf_path"]).exists()
            assert (output_dir / row["metadata_path"]).exists()

    # Deterministic rebuild with the same seed yields identical manifests.
    generator.rebuild_manifest_files()
    for split_name in ("train", "val", "test"):
        with (output_dir / f"{split_name}.csv").open(encoding="utf-8", newline="") as handle:
            rebuilt = list(csv.DictReader(handle))
        assert [row["sample_id"] for row in rebuilt] == [row["sample_id"] for row in manifests[split_name]]


def test_rebuild_manifests_prefers_first_split_on_duplicate_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    create_output_directories(output_dir)

    # Simulate a pilot leftover under val/ and a later full-run copy under train/.
    for split_name, sample_id in (("train", "000081"), ("val", "000081")):
        sharp = output_dir / split_name / "sharp" / f"{sample_id}.png"
        blurred = output_dir / split_name / "blurred" / f"{sample_id}.png"
        psf = output_dir / split_name / "psf" / f"{sample_id}.npy"
        metadata = output_dir / split_name / "metadata" / f"{sample_id}.json"
        _write_image(sharp, np.ones((8, 8), dtype=np.float32))
        _write_image(blurred, np.ones((8, 8), dtype=np.float32))
        np.save(psf, np.ones((5, 5), dtype=np.float32))
        metadata.write_text(json.dumps({"sample_id": sample_id, "weights": [1, 0, 0, 0, 0]}), encoding="utf-8")

    config = DatasetGeneratorConfig(
        input_dir=tmp_path,
        output_dir=output_dir,
        samples=1,
        train_split=1.0,
        val_split=0.0,
        test_split=0.0,
        random_seed=0,
        show_progress=False,
    )
    SyntheticDatasetGenerator(config).rebuild_manifest_files()

    with (output_dir / "train.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["sharp_path"].startswith("train/")
