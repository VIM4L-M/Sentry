#!/usr/bin/env python3
"""Composition root for training the denoising autoencoder (Unit IV).

Reads ``configs/training/autoencoder.yaml`` for the network and
``configs/sensors.yaml`` for the corruption it learns to undo, trains on the
dataset ``scripts/build_dataset.py`` produced, and reports validation PSNR
against the degraded input's own PSNR.

Usage:
    python scripts/train_autoencoder.py
    python scripts/train_autoencoder.py --epochs 5 --name quick
    python scripts/train_autoencoder.py --device cpu --workers 0
    python scripts/train_autoencoder.py --no-skip --name bottleneck   # ablation
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AutoencoderTrainingConfig
from sentry_ai.training.denoising import AutoencoderTrainer, DenoiserTrainingOutcome

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, train the denoiser, and report the result."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    config = _apply_overrides(loader.load_autoencoder_config(args.config), args, loader)
    degradation = loader.load_sensor_config(args.sensors).degradation
    trainer = AutoencoderTrainer(config, degradation)

    device = trainer.resolve_device()
    if device == "cpu":
        # Said up front rather than discovered an hour in.
        logger.warning(
            "Training on CPU — this is far slower than a GPU. Install a CUDA "
            "build of Torch to use the graphics card."
        )

    outcome = trainer.train(run_name=args.name, on_epoch=_print_epoch)
    _report(outcome, device)
    return 0


def _apply_overrides(
    config: AutoencoderTrainingConfig, args: argparse.Namespace, loader: ConfigLoader
) -> AutoencoderTrainingConfig:
    """Let command-line flags win over the config file, for quick experiments."""
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.workers is not None:
        config = replace(config, num_workers=args.workers)
    if args.base_channels is not None:
        config = replace(config, base_channels=args.base_channels)
    if args.no_skip:
        config = replace(config, skip_connections=False)
    if args.dataset is not None:
        config = replace(config, dataset_dir=loader.resolve(args.dataset))
    return config


def _print_epoch(row: dict[str, float]) -> None:
    """One line per epoch, so a long run visibly makes progress."""
    print(
        f"epoch {int(row['epoch']):>3}  train {row['train_loss']:.5f}  "
        f"val {row['val_loss']:.5f}  PSNR {row['val_psnr']:.2f} dB "
        f"(input {row['baseline_psnr']:.2f})",
        flush=True,
    )


def _report(outcome: DenoiserTrainingOutcome, device: str) -> None:
    """Print the headline numbers and where everything was written."""
    print(f"\nTraining complete on {device}")
    for label, value in outcome.as_display_rows():
        print(f"  {label:<12}{value}")
    print(f"  {'log':<12}{outcome.run_dir / 'metrics.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the denoising autoencoder.")
    parser.add_argument(
        "--config",
        default="configs/training/autoencoder.yaml",
        help="Autoencoder config (default: configs/training/autoencoder.yaml).",
    )
    parser.add_argument(
        "--sensors",
        default="configs/sensors.yaml",
        help="Sensors config whose degradation section defines the corruption.",
    )
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    parser.add_argument("--workers", type=int, help="Override the DataLoader worker count.")
    parser.add_argument("--base-channels", type=int, help="Override the network width.")
    parser.add_argument(
        "--no-skip", action="store_true", help="Train without skip connections (ablation)."
    )
    parser.add_argument("--dataset", help="Override the dataset directory.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
