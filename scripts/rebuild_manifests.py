#!/usr/bin/env python3
"""Rebuild train/val/test manifests from an existing generated dataset."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.dataset_generator import DatasetGeneratorConfig, SyntheticDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild train/val/test CSV manifests by scanning existing generated "
            "samples. Does not regenerate images, PSFs, or metadata."
        )
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        help="Dataset root containing train/val/test folders and manifests.",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for deterministic splits.")
    parser.add_argument(
        "--verify-paths",
        action="store_true",
        help="After rewriting manifests, verify every referenced path exists.",
    )
    return parser.parse_args()


def _verify_manifest_paths(output_dir: Path) -> None:
    missing = 0
    checked = 0
    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        # Group referenced relative paths by parent directory for batched existence checks.
        by_dir: dict[Path, set[str]] = {}
        for row in rows:
            for key in ("sharp_path", "blurred_path", "psf_path", "metadata_path"):
                relative = Path(row[key])
                by_dir.setdefault(output_dir / relative.parent, set()).add(relative.name)

        existing_by_dir: dict[Path, set[str]] = {}
        for directory, names in by_dir.items():
            present: set[str] = set()
            if directory.exists():
                with os.scandir(directory) as entries:
                    present = {entry.name for entry in entries if entry.is_file()}
            existing_by_dir[directory] = present
            for name in names:
                checked += 1
                if name not in present:
                    missing += 1
                    if missing <= 10:
                        print(f"Missing path ({split_name}): {directory / name}", file=sys.stderr)

    if missing:
        raise FileNotFoundError(f"{missing}/{checked} manifest paths are missing.")
    print(f"Verified {checked} manifest paths exist.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")

    # samples is required by DatasetGeneratorConfig validation but unused for rebuild;
    # inventory size is discovered from disk.
    config = DatasetGeneratorConfig(
        input_dir=output_dir,
        output_dir=output_dir,
        samples=1,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        show_progress=False,
        resume=True,
    )
    generator = SyntheticDatasetGenerator(config)
    generator.rebuild_manifest_files()

    if args.verify_paths:
        _verify_manifest_paths(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
