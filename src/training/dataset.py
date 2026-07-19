"""Manifest-driven dataset for PSF basis-weight regression."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PSFDataset(Dataset):
    """Load blurred images and metadata-defined regression targets from CSV manifests."""

    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path,
        image_key: str = "blurred_path",
        target_dim: int = 5,
        transform: transforms.Compose | None = None,
        target_transform: callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        self.image_key = image_key
        self.target_dim = target_dim
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.target_transform = target_transform
        self.samples = self._load_manifest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, metadata_path = self.samples[index]
        image = self._load_image(image_path)
        target = self._load_target(metadata_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _load_manifest(self) -> list[tuple[Path, Path]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest does not exist: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        samples: list[tuple[Path, Path]] = []
        for row in rows:
            if not row:
                continue
            image_path = self._resolve_path(row.get(self.image_key, ""))
            metadata_path = self._resolve_path(row.get("metadata_path", ""))
            if image_path is None or metadata_path is None:
                continue
            samples.append((image_path, metadata_path))
        return samples

    def _load_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        return image

    def _load_target(self, metadata_path: Path) -> torch.Tensor:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        weights = metadata.get("weights", [])
        if not isinstance(weights, list):
            raise ValueError(f"metadata weights must be a list: {metadata_path}")

        tensor = torch.tensor(weights[: self.target_dim], dtype=torch.float32)
        if tensor.numel() < self.target_dim:
            tensor = torch.cat(
                [tensor, torch.zeros(self.target_dim - tensor.numel(), dtype=torch.float32)]
            )
        return tensor

    def _resolve_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if (self.root_dir / candidate).exists():
            return self.root_dir / candidate
        if self.manifest_path.parent / candidate:
            try:
                path = (self.manifest_path.parent / candidate).resolve()
                if path.exists():
                    return path
            except OSError:
                return None
        return None
