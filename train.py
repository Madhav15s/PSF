"""CLI entry point for training a PSF basis-weight regressor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.dataloader import build_dataloaders
from training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PSF basis-weight regressor")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    trainer_kwargs = {
        key: config_data[key]
        for key in TrainerConfig.__dataclass_fields__
        if key in config_data
    }
    trainer_config = TrainerConfig(**trainer_kwargs)
    train_loader, val_loader = build_dataloaders(
        root_dir=config_data.get("root_dir", "dataset/generated"),
        train_manifest=config_data.get("train_manifest", "dataset/generated/train.csv"),
        val_manifest=config_data.get("val_manifest"),
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
        target_dim=trainer_config.target_dim,
    )

    trainer = Trainer(trainer_config)
    print(f"Loaded training config from: {args.config}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}")
    print("Starting training...")
    results = trainer.fit(train_loader, val_loader)
    print("Training complete.")
    print(f"Best score: {results['best_score']}")


if __name__ == "__main__":
    main()
