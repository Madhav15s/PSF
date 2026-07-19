# Physics-Constrained EO/IR PSF Basis Experiments

This repository contains Sprint 0 and Sprint 2 tooling for exploring whether a
small bank of physically motivated basis kernels can represent blur observed in
EO/IR imagery.

Sprint 0 provides differentiable Torch kernels, a batched PSF generator,
visualization helpers, and an interactive Experiment 0 application. Sprint 2 adds
synthetic dataset generation, deterministic patch and parameter sampling, and a
manifest-based dataset export pipeline.

## Sprint 2 Overview

The dataset pipeline generates sharp patches, blurred patches, and PSF kernels
from a folder of input grayscale images. It writes output files into a
train/validation/test directory structure and creates CSV manifests that can be
consumed by future dataloaders.

## Dataset Generation Pipeline

The generation workflow uses the existing Sprint 0 PSF kernels and generator,
then layers them with deterministic sampling utilities for patch extraction and
PSF parameter generation.

## Folder Structure

```text
dataset/
  train/
    sharp/
    blurred/
    psf/
    metadata/
  val/
    sharp/
    blurred/
    psf/
    metadata/
  test/
    sharp/
    blurred/
    psf/
    metadata/
  train.csv
  val.csv
  test.csv
```

## Run Experiment 0

```bash
python experiments/exp0_basis_fit.py --image path/to/eoir_image.png
```

The app can also be launched without an image path and then pointed at a real
grayscale EO/IR image using the **Load Image** button.

## Generate a Dataset

```bash
python scripts/generate_dataset.py \
  --input input/sharp_images \
  --output dataset \
  --samples 1000 \
  --patch-size 128 \
  --seed 42
```

## Train a PSF Regression Model

The repository now includes a manifest-driven training stack that reads the generated CSV manifests, loads blurred patches, and regresses the normalized PSF basis weights from metadata.

```bash
python train.py --config config/train.yaml
```

The default configuration expects generated manifests at dataset/generated/train.csv and dataset/generated/val.csv. To run inference with a saved checkpoint:

```bash
python predict.py --checkpoint artifacts/checkpoints/checkpoint_epoch_001.pt --manifest dataset/generated/val.csv
```

## Configuration Options

The CLI accepts:

- `--input`: source folder of grayscale sharp images; repeat the option to include multiple source roots or arbitrary dataset folders
- `--output`: destination root for the generated dataset
- `--samples`: number of samples to generate
- `--train-split`, `--val-split`, `--test-split`: split fractions
- `--patch-size`: square patch size, one of 64, 128, or 256
- `--seed`: random seed for deterministic sampling
- `--kernel-size`: odd PSF kernel size
- `--parameter-ranges`: optional JSON object to override PSF parameter ranges
- `--device`: torch device to use for generation

## Example Commands

```bash
python scripts/generate_dataset.py --help
python scripts/generate_dataset.py --input input/sharp_images --output dataset --samples 10 --patch-size 64 --seed 7
```

## Run Tests

```bash
python -m pytest
```
