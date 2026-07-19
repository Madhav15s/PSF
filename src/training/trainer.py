"""Training loop and checkpointing utilities for PSF regression."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.resnet18_regressor import ResNet18Regressor
from training.losses import mse_loss
from training.metrics import mean_absolute_error


@dataclass
class TrainerConfig:
    """Configuration for the PSF regression trainer."""

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    checkpoint_dir: str | Path = "artifacts/checkpoints"
    log_dir: str | Path = "artifacts/logs"
    device: str = "cpu"
    seed: int = 42
    target_dim: int = 5
    num_workers: int = 0
    amp: bool = False
    pretrained: bool = False
    dropout: float = 0.0


class Trainer:
    """Simple but reusable training loop for regression models."""

    def __init__(self, config: TrainerConfig, model: nn.Module | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model or ResNet18Regressor(
            num_outputs=config.target_dim,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda" and config.amp) if self.device.type == "cuda" and config.amp else None
        self.loss_fn = mse_loss
        self.metric_fn = mean_absolute_error
        self.best_state = None
        self.best_score = float("inf")
        self.epochs_without_improvement = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        """Train the model for the configured number of epochs."""
        self._set_seed(self.config.seed)
        history: list[dict[str, float]] = []
        checkpoint_root = Path(self.config.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss, train_metric = self._train_epoch(train_loader, epoch)
            val_loss = None
            val_metric = None
            if val_loader is not None:
                val_loss, val_metric = self._evaluate(val_loader)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_mae": float(train_metric),
                    "val_loss": float(val_loss) if val_loss is not None else float("nan"),
                    "val_mae": float(val_metric) if val_metric is not None else float("nan"),
                }
            )

            self._save_checkpoint(epoch + 1, train_loss, val_loss)
            checkpoint_name = f"checkpoint_epoch_{epoch + 1:03d}.pt"
            val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "n/a"
            val_mae_str = f"{val_metric:.6f}" if val_metric is not None else "n/a"
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"train_loss={train_loss:.6f}, train_mae={train_metric:.6f}, "
                f"val_loss={val_loss_str}, val_mae={val_mae_str}"
            )
            print(f"Saved checkpoint: {Path(self.config.checkpoint_dir) / checkpoint_name}")

            current_score = float(val_loss if val_loss is not None else train_loss)
            if self._is_improvement(current_score):
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_score = current_score
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.patience:
                print("Early stopping triggered.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {"history": history, "best_score": self.best_score}

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        running_loss = 0.0
        running_metric = 0.0
        samples_seen = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=True):
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            running_metric += float(self.metric_fn(predictions, targets).item()) * images.size(0)
            samples_seen += images.size(0)

        mean_loss = torch.tensor(running_loss / max(samples_seen, 1), dtype=torch.float32)
        mean_metric = torch.tensor(running_metric / max(samples_seen, 1), dtype=torch.float32)
        return mean_loss, mean_metric

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0
        samples_seen = 0
        for images, targets in val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(images)
            loss = self.loss_fn(predictions, targets)
            running_loss += float(loss.item()) * images.size(0)
            running_metric += float(self.metric_fn(predictions, targets).item()) * images.size(0)
            samples_seen += images.size(0)

        mean_loss = torch.tensor(running_loss / max(samples_seen, 1), dtype=torch.float32)
        mean_metric = torch.tensor(running_metric / max(samples_seen, 1), dtype=torch.float32)
        return mean_loss, mean_metric

    def _save_checkpoint(self, epoch: int, train_loss: torch.Tensor, val_loss: torch.Tensor | None) -> None:
        checkpoint_root = Path(self.config.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / f"checkpoint_epoch_{epoch:03d}.pt"
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "config": self.config.__dict__,
        }
        torch.save(payload, checkpoint_path)

    def _is_improvement(self, score: float) -> bool:
        improved = score < self.best_score
        return improved

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
