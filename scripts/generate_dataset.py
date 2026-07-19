#!/usr/bin/env python3
"""CLI for generating synthetic EO/IR PSF datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.dataset_generator import DatasetGeneratorConfig, SyntheticDatasetGenerator
from dataset.parameter_sampler import FloatRange, ParameterRanges


def _parse_parameter_ranges(raw_value: str | None) -> ParameterRanges:
    if raw_value is None:
        return ParameterRanges()

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("--parameter-ranges must be a JSON object.")

    def _coerce(name: str, default: tuple[float, float]) -> FloatRange:
        value = parsed.get(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"parameter range '{name}' must be a two-item list.")
        minimum, maximum = value
        return FloatRange(float(minimum), float(maximum))

    return ParameterRanges(
        gaussian_sigma=_coerce("gaussian_sigma", (0.8, 6.0)),
        ghost_offset_x=_coerce("ghost_offset_x", (-16.0, 16.0)),
        ghost_offset_y=_coerce("ghost_offset_y", (-16.0, 16.0)),
        ghost_strength=_coerce("ghost_strength", (0.0, 0.6)),
        ghost_sigma=_coerce("ghost_sigma", (0.8, 6.0)),
        scatter_radius=_coerce("scatter_radius", (4.0, 28.0)),
        ring_radius=_coerce("ring_radius", (2.0, 28.0)),
        ring_width=_coerce("ring_width", (0.8, 8.0)),
        motion_angle=_coerce("motion_angle", (-180.0, 180.0)),
        motion_length=_coerce("motion_length", (1.0, 28.0)),
        basis_weight=_coerce("basis_weight", (0.0, 1.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic EO/IR PSF datasets.")
    parser.add_argument(
        "--input",
        dest="input_dirs",
        action="append",
        required=True,
        help="Folder containing sharp source images. Repeat to include multiple source roots.",
    )
    parser.add_argument("--output", dest="output_dir", required=True, help="Folder for generated dataset outputs.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of samples assigned to training.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction assigned to validation.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction assigned to testing.")
    parser.add_argument("--patch-size", type=int, default=128, choices=[64, 128, 256], help="Square patch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--kernel-size", type=int, default=65, help="Odd kernel size for generated PSFs.")
    parser.add_argument(
        "--parameter-ranges",
        default=None,
        help="Optional JSON object mapping parameter names to [min, max] ranges.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device to use for generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parameter_ranges = _parse_parameter_ranges(args.parameter_ranges)
    config = DatasetGeneratorConfig(
        input_dir=Path(args.input_dirs[0]),
        output_dir=Path(args.output_dir),
        input_dirs=tuple(Path(path) for path in args.input_dirs),
        samples=args.samples,
        patch_size=args.patch_size,
        kernel_size=args.kernel_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        parameter_ranges=parameter_ranges,
        device=args.device,
        show_progress=True,
        resume=True,
    )
    summary = SyntheticDatasetGenerator(config).generate()
    print(
        f"Generated {summary.generated_samples}/{summary.requested_samples} samples in {summary.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
