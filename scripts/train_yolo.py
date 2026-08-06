#!/usr/bin/env python3
"""Composition root for fine-tuning the YOLOv8n detector (Unit II).

Reads ``configs/training/yolo.yaml``, trains against the dataset that
``scripts/build_dataset.py`` produced, and reports validation mAP plus
where the weights landed.

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 5 --name quick
    python scripts/train_yolo.py --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import YoloTrainingConfig
from sentry_ai.training.yolo import TrainingOutcome, YoloTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, fine-tune the detector, and report the result."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    config = _apply_overrides(loader.load_yolo_config(args.config), args, loader)
    trainer = YoloTrainer(config)

    device = trainer.resolve_device()
    if device == "cpu":
        # Said up front rather than discovered an hour in.
        logger.warning(
            "Training on CPU — this is far slower than a GPU. Install a CUDA "
            "build of Torch to use the graphics card."
        )

    outcome = trainer.train(run_name=args.name)
    _report(outcome, device)
    return 0


def _apply_overrides(
    config: YoloTrainingConfig, args: argparse.Namespace, loader: ConfigLoader
) -> YoloTrainingConfig:
    """Let command-line flags win over the config file, for quick experiments."""
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.dataset is not None:
        config = replace(config, dataset_dir=loader.resolve(args.dataset))
    return config


def _report(outcome: TrainingOutcome, device: str) -> None:
    """Print the headline metrics and where everything was written."""
    print(f"Training complete on {device}")
    for label, value in outcome.as_display_rows():
        print(f"  {label:<12}{value}")
    print(f"  {'curves':<12}{outcome.run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on synthetic frames.")
    parser.add_argument(
        "--config",
        default="configs/training/yolo.yaml",
        help="Path to the YOLO training config (default: configs/training/yolo.yaml).",
    )
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    parser.add_argument("--dataset", help="Override the dataset directory.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
