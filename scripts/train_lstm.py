#!/usr/bin/env python3
"""Composition root for training the behaviour LSTM (Unit III).

Reads ``configs/training/lstm.yaml``, trains on the trajectories
``scripts/record_trajectories.py`` wrote, and reports validation macro-F1
next to the two baselines milestone M5 has to beat.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 5 --name quick
    python scripts/train_lstm.py --no-class-weighting --name unweighted   # ablation
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import LstmTrainingConfig
from sentry_ai.training.motion import LstmTrainer, LstmTrainingOutcome

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, train the LSTM, and report it against the baselines."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    config = _apply_overrides(loader.load_lstm_config(args.config), args, loader)
    trainer = LstmTrainer(config)
    outcome = trainer.train(run_name=args.name, on_epoch=_print_epoch)
    _report(outcome, trainer.resolve_device())
    return 0


def _apply_overrides(
    config: LstmTrainingConfig, args: argparse.Namespace, loader: ConfigLoader
) -> LstmTrainingConfig:
    """Let command-line flags win over the config file, for quick experiments."""
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.hidden_size is not None:
        config = replace(config, hidden_size=args.hidden_size)
    if args.no_class_weighting:
        config = replace(config, class_weighting=False)
    if args.trajectories is not None:
        config = replace(config, trajectories_dir=loader.resolve(args.trajectories))
    return config


def _print_epoch(row: dict[str, float]) -> None:
    print(
        f"epoch {int(row['epoch']):>3}  train {row['train_loss']:.4f}  "
        f"val {row['val_loss']:.4f}  acc {row['val_accuracy']:.4f}  "
        f"macro-F1 {row['val_macro_f1']:.4f}",
        flush=True,
    )


def _report(outcome: LstmTrainingOutcome, device: str) -> None:
    """The model next to both baselines, and whether M5 is met."""
    print(
        f"\nTraining complete on {device}: "
        f"{outcome.epochs_run} epochs, best {outcome.best_epoch}"
    )
    print(f"  {'':<14}{'accuracy':>9}  {'macro-F1':>9}")
    for name, report in (*outcome.baselines.items(), ("LSTM", outcome.report)):
        print(f"  {name:<14}{report.accuracy:>9.4f}  {report.macro_f1:>9.4f}")
    print("\nLSTM per class")
    for label, value in outcome.report.as_display_rows()[2:]:
        print(f"  {label:<12}{value}")
    verdict = "MET" if outcome.beats_baselines else "NOT MET"
    print(f"\nM5 (beats every baseline on macro-F1): {verdict}")
    print(f"  weights     {outcome.weights_path}")
    print(f"  log         {outcome.run_dir / 'metrics.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the behaviour LSTM.")
    parser.add_argument(
        "--config",
        default="configs/training/lstm.yaml",
        help="LSTM config (default: configs/training/lstm.yaml).",
    )
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    parser.add_argument("--hidden-size", type=int, help="Override the LSTM width.")
    parser.add_argument(
        "--no-class-weighting",
        action="store_true",
        help="Train with an unweighted loss (ablation).",
    )
    parser.add_argument("--trajectories", help="Override the trajectories directory.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
