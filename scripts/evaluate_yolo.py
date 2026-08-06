#!/usr/bin/env python3
"""Scores a trained detector against the held-out validation missions.

Reports per class, not just overall. The synthetic dataset runs roughly
twelve obstacles to every victim, so a model that never found a single
victim would still post a respectable overall mAP — and victims are the one
class the mission actually depends on.

Usage:
    python scripts/evaluate_yolo.py
    python scripts/evaluate_yolo.py --weights models/yolo/sentry/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import YoloTrainingConfig
from sentry_ai.training.yolo import EvaluationReport, YoloEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Below this, the detector is not usable for the command center: it would
#: be missing more trapped people than it finds.
VICTIM_RECALL_FLOOR = 0.60

logger = get_logger(__name__)


def main() -> int:
    """Evaluate a checkpoint and print per-class metrics."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    config = loader.load_yolo_config(args.config)
    weights = _resolve_weights(config, args, loader)
    report = YoloEvaluator(config).evaluate(weights, run_name=args.name)

    _report(report, weights)
    return 0


def _resolve_weights(
    config: YoloTrainingConfig, args: argparse.Namespace, loader: ConfigLoader
) -> Path:
    """The checkpoint to score, defaulting to the last training run's best."""
    if args.weights:
        return loader.resolve(args.weights)
    return config.runs_dir / args.run / "weights" / "best.pt"


def _report(report: EvaluationReport, weights: Path) -> None:
    """Print the metrics, and say plainly whether victims are being found."""
    print(f"Evaluated {weights}")
    for label, value in report.as_display_rows():
        print(f"  {label:<16}{value}")

    victim = report.metrics_for("victim")
    if victim is None:
        print("\nNo victim instances in the validation split — nothing to judge.")
        return
    verdict = "OK" if victim.recall >= VICTIM_RECALL_FLOOR else "TOO LOW"
    print(
        f"\nVictim recall {victim.recall:.1%} ({verdict}; "
        f"floor is {VICTIM_RECALL_FLOOR:.0%}) — this is the number that decides "
        f"whether the command center can trust the detector."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO detector.")
    parser.add_argument(
        "--config",
        default="configs/training/yolo.yaml",
        help="Path to the YOLO training config (default: configs/training/yolo.yaml).",
    )
    parser.add_argument("--weights", help="Checkpoint to evaluate (default: the run's best.pt).")
    parser.add_argument("--run", default="sentry", help="Training run name to score.")
    parser.add_argument("--name", default="eval", help="Subdirectory for evaluation artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
