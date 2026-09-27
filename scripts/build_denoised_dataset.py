#!/usr/bin/env python3
"""Writes a copy of the detection dataset with every frame denoised (Phase 4).

The detector learns what a victim looks like in the frames it is trained on.
Trained on smoky frames, it learns "a victim seen through smoke" — and once
the denoiser sits in front of it, it is never shown smoke again. Measured,
that shift costs more than the denoiser gains
(docs/architecture/phase4-denoising.md). The fix is to train the detector on
what it will actually see:

    python scripts/build_denoised_dataset.py --weights models/autoencoder/sentry/best.pt
    python scripts/train_yolo.py --dataset data/denoised --name denoised
    python scripts/evaluate_denoiser.py --detector models/yolo/labelfix/weights/best.pt \\
        --denoised-detector models/yolo/denoised/weights/best.pt

Labels and the train/validation split are copied unchanged: denoising moves
pixels, never objects, and validation missions stay held out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder
from sentry_ai.training.dataset import DatasetLayout
from sentry_ai.training.denoising import write_denoised_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Denoise every degraded frame of the source dataset into a new one."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    denoiser = ConvDenoisingAutoencoder.from_checkpoint(
        loader.resolve(args.weights), device=args.device
    )
    output = loader.resolve(args.output)
    stats = write_denoised_dataset(
        source=DatasetLayout(root=loader.resolve(args.source)),
        target=DatasetLayout(root=output),
        denoiser=denoiser,
    )

    print(f"Denoised dataset written to {output}")
    for label, value in stats.as_display_rows():
        print(f"  {label:<8}{value} frames")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise the detection dataset.")
    parser.add_argument(
        "--weights",
        default="models/autoencoder/sentry/best.pt",
        help="Denoiser checkpoint (default: models/autoencoder/sentry/best.pt).",
    )
    parser.add_argument(
        "--source", default="data/synthetic", help="Dataset to read (default: data/synthetic)."
    )
    parser.add_argument(
        "--output", default="data/denoised", help="Dataset to write (default: data/denoised)."
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
