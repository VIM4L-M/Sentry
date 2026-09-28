#!/usr/bin/env python3
"""Composition root for decision fusion (Unit I, milestone M7).

Two steps:

1. ``--record``: drive missions with the full pipeline — DQN, LSTM, onboard
   camera and detector — on a command-center map that lags reality by
   ``belief_lag`` refreshes, and record every tick with its safe label.
2. Train the MLP on everything, and the same MLP restricted to one signal
   group at a time, then report all of them next to two untrained
   references: the DQN as-is and the hand-written camera-veto rule.

M7 is met when fusion's validation macro-F1 beats every single-signal model.
Mission-level results — crashes avoided — are ``scripts/evaluate_fusion.py``.

Usage:
    python scripts/train_fusion.py --record          # first time: record, then train
    python scripts/train_fusion.py                   # retrain on the recorded samples
    python scripts/train_fusion.py --record --detector models/yolo/labelfix/weights/best.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.decision.fusion import CameraVetoFusion, FusionArchitecture
from sentry_ai.training.fusion import (
    ABLATIONS,
    DecisionReport,
    FusionSamples,
    assemble_pipeline,
    record_samples,
    score_reference,
    train_model,
)
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Record (optionally), train every model, and report M7."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = loader.load_fusion_config(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    train_path = config.data_dir / "train.npz"
    val_path = config.data_dir / "val.npz"

    if args.record:
        _record(loader, app_config, config, args, train_path, val_path)
    train = FusionSamples.load(train_path)
    val = FusionSamples.load(val_path)
    print(
        f"\nSamples: {len(train):,} train ({train.vetoes} unsafe moves), "
        f"{len(val):,} val ({val.vetoes} unsafe moves)\n",
        flush=True,
    )

    reports: dict[str, DecisionReport] = {
        "DQN as-is (no fusion)": score_reference(val, None),
        "camera-veto rule": score_reference(val, CameraVetoFusion()),
    }
    for name, groups in ABLATIONS.items():
        architecture = FusionArchitecture(
            groups=groups, hidden_sizes=config.hidden_sizes, dropout=config.dropout
        )
        run_dir = config.runs_dir / args.name / name.replace(" ", "_")
        reports[name] = train_model(name, architecture, train, val, config, run_dir).report

    _print(reports)
    return 0


def _record(
    loader: ConfigLoader,
    app_config: object,
    config: FusionTrainingConfig,
    args: argparse.Namespace,
    train_path: Path,
    val_path: Path,
) -> None:
    pipeline = assemble_pipeline(
        loader,
        loader.resolve(args.sensors),
        dqn_weights=loader.resolve(args.dqn),
        lstm_weights=loader.resolve(args.lstm),
        detector_weights=loader.resolve(args.detector),
        device=config.device if config.device != "auto" else "cpu",
        seed=config.seed,
    )
    factory = MissionFactory.from_app_config(loader, app_config)  # type: ignore[arg-type]
    for path, seeds in ((train_path, config.training_seeds), (val_path, config.validation_seeds)):
        print(f"Recording {len(seeds)} missions -> {path}", flush=True)
        samples = record_samples(factory, pipeline, seeds, config.belief_lag, config.max_ticks)
        samples.save(path)


def _print(reports: dict[str, DecisionReport]) -> None:
    print(f"  {'':<24}{'accuracy':>9}{'macro-F1':>10}{'crashes vetoed':>16}{'needless stops':>16}")
    for name, report in reports.items():
        print(
            f"  {name:<24}{report.accuracy:>9.4f}{report.macro_f1:>10.4f}"
            f"{_pct(report.veto_recall):>16}{_pct(report.false_veto_rate):>16}"
        )
    fusion = reports["fusion"].macro_f1
    singles = {name: reports[name].macro_f1 for name in ABLATIONS if name != "fusion"}
    verdict = "MET" if all(fusion > score for score in singles.values()) else "NOT MET"
    best_single = max(singles, key=singles.__getitem__)
    print(
        f"\nM7 (fusion beats every single signal on macro-F1): {verdict} — "
        f"{fusion:.4f} vs best single signal '{best_single}' {singles[best_single]:.4f}"
    )
    rule = reports["camera-veto rule"].macro_f1
    print(
        f"Against the hand-written camera-veto rule: {fusion:.4f} vs {rule:.4f} "
        f"({'the MLP is better' if fusion > rule else 'the rule is as good or better'})"
    )


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record data for and train decision fusion.")
    parser.add_argument(
        "--config",
        default="configs/training/fusion.yaml",
        help="Fusion config (default: configs/training/fusion.yaml).",
    )
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--sensors", default="configs/sensors.yaml", help="Camera config.")
    parser.add_argument("--record", action="store_true", help="Record fresh samples first.")
    parser.add_argument("--dqn", default="models/dqn/sentry/best.zip", help="DQN weights.")
    parser.add_argument("--lstm", default="models/lstm/dqn/best.pt", help="LSTM weights.")
    parser.add_argument(
        "--detector",
        default="models/yolo/labelfix/weights/best.pt",
        help="YOLO weights for the onboard camera (trained on degraded frames).",
    )
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
