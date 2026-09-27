#!/usr/bin/env python3
"""Scores the behaviour LSTM against its baselines — twice (milestone M5).

1. **All held-out windows.** Validation missions share no ticks with
   training, but they drive the same few streets, so most of their windows
   repeat a training window exactly. This number is fair for the city the
   model will actually run in, and it flatters generalisation.
2. **Novel windows only** — the held-out windows whose exact sequence of
   positions and headings never occurred in training. This is the honest
   test of whether the model learned to predict behaviour or memorised the
   streets. It is a small set; its size is printed next to it.

Every predictor is scored on exactly the same windows, and the LSTM is
scored through ``IMotionPredictor`` — the same port the live pipeline will
use — not through the trainer's batched shortcut.

Usage:
    python scripts/evaluate_lstm.py
    python scripts/evaluate_lstm.py --weights models/lstm/sentry/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor
from sentry_ai.training.motion import (
    ClassificationReport,
    LabelledWindow,
    labelled_windows,
    majority_class,
    novel_windows,
    score_majority,
    score_persistence,
    score_predictor,
)
from sentry_ai.training.trajectories import TrajectorySet

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Score the LSTM and both baselines on all, then novel, held-out windows."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    config = loader.load_lstm_config(args.config)
    weights = (
        loader.resolve(args.weights) if args.weights else config.runs_dir / args.run / "best.pt"
    )
    predictor = LstmMotionPredictor.from_checkpoint(weights, device=args.device)
    rule = (config.window, config.horizon, config.cone_degrees)

    train = labelled_windows(
        TrajectorySet.load(config.trajectories_dir / "train.json").trajectories, *rule
    )
    held_out = labelled_windows(
        TrajectorySet.load(config.trajectories_dir / "val.json").trajectories, *rule
    )
    novel = novel_windows(held_out, train)
    majority = majority_class(train)

    print(f"Behaviour LSTM {weights}")
    print(f"  predicting {config.horizon} ticks ahead from {config.window}\n")
    for title, windows in (("All held-out windows", held_out), ("Novel windows only", novel)):
        share = len(windows) / len(held_out)
        print(f"{title} — {len(windows)} ({share:.1%} of held-out)")
        if not windows:
            print("  none\n")
            continue
        _print_table(
            {
                "majority": score_majority(windows, majority),
                "persistence": score_persistence(windows, config.horizon, config.cone_degrees),
                "LSTM": score_predictor(predictor, windows),
            },
            windows,
        )
    return 0


def _print_table(reports: dict[str, ClassificationReport], windows: list[LabelledWindow]) -> None:
    print(f"  {'':<13}{'accuracy':>9}  {'macro-F1':>9}")
    for name, report in reports.items():
        print(f"  {name:<13}{report.accuracy:>9.4f}  {report.macro_f1:>9.4f}")
    lstm = reports["LSTM"]
    best_baseline = max(r.macro_f1 for name, r in reports.items() if name != "LSTM")
    verdict = "beats" if lstm.macro_f1 > best_baseline else "does NOT beat"
    print(f"  -> LSTM {verdict} the best baseline on macro-F1")
    for label, value in lstm.as_display_rows()[2:]:
        print(f"     {label:<12}{value}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the behaviour LSTM.")
    parser.add_argument(
        "--config",
        default="configs/training/lstm.yaml",
        help="LSTM config (default: configs/training/lstm.yaml).",
    )
    parser.add_argument("--weights", help="Checkpoint to score (default: the run's best.pt).")
    parser.add_argument("--run", default="sentry", help="Training run name to score.")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
