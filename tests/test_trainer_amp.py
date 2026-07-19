"""Verification helpers and smoke checks for trainer AMP compatibility."""

from __future__ import annotations

from pathlib import Path

import torch

from training.trainer import Trainer, TrainerConfig, _build_grad_scaler


def test_build_grad_scaler_constructs_without_error() -> None:
    scaler = _build_grad_scaler(enabled=False)
    assert scaler is not None


def test_trainer_amp_path_uses_torch_amp_autocast(tmp_path: Path) -> None:
    config = TrainerConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        patience=1,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        device="cpu",
        amp=True,
        target_dim=5,
        num_workers=0,
    )
    trainer = Trainer(config)
    assert trainer.scaler is None  # AMP only activates on CUDA
    assert trainer.amp_device_type == "cpu"

    images = torch.randn(2, 3, 64, 64)
    targets = torch.randn(2, 5)
    loader = torch.utils.data.DataLoader(list(zip(images, targets)), batch_size=2)
    results = trainer.fit(loader, None)
    assert "history" in results
    assert (tmp_path / "checkpoints" / "checkpoint_epoch_001.pt").exists()
