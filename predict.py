"""CLI entry point for running inference with a trained PSF regressor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.resnet18_regressor import ResNet18Regressor
from training.dataset import PSFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained PSF regressor")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("dataset/generated/val.csv"))
    parser.add_argument("--root-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    target_dim = int(config.get("target_dim", 5))
    model = ResNet18Regressor(num_outputs=target_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = PSFDataset(root_dir=args.root_dir, manifest_path=args.manifest, target_dim=target_dim)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in loader:
            predictions.append(model(images))
    outputs = torch.cat(predictions, dim=0)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outputs.tolist()), encoding="utf-8")
    else:
        print(outputs)


if __name__ == "__main__":
    main()
