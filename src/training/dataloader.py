"""Data-loader helpers for PSF regression experiments."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from training.dataset import PSFDataset


def build_dataloaders(
    root_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_dim: int = 5,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and validation loaders from manifest files."""
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = PSFDataset(
        root_dir=root_dir,
        manifest_path=train_manifest,
        target_dim=target_dim,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if val_manifest is None:
        return train_loader, None

    val_dataset = PSFDataset(
        root_dir=root_dir,
        manifest_path=val_manifest,
        target_dim=target_dim,
        transform=train_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
