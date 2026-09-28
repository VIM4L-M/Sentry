#!/usr/bin/env python3
"""Composition root for training the decision-fusion MLP (Unit I, Phase 7).

Collects ticks from missions whose command-center map lags the world (see
``configs/training/fusion.yaml``), trains the fusion network on every
signal and on each signal alone, and reports milestone M7: fusion beats
every single signal on held-out missions.

Usage:
    python scripts/train_fusion.py                      # collect, then train
    python scripts/train_fusion.py --reuse-data          # train on the saved ticks
    python scripts/train_fusion.py --detector models/yolo/labelfix/weights/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.decision.dqn_controller import DqnLocalController
from sentry_ai.decision.mlp_fusion import FeatureGroup
from sentry_ai.interfaces.decision import SceneEvidence
from sentry_ai.interfaces.perception import IDenoiser
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor
from sentry_ai.simulation.factory import Mission
from sentry_ai.training.fusion import (
    ActionReport,
    FusionDataset,
    FusionRecorder,
    FusionTrainer,
    FusionTrainingOutcome,
    collect,
)
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)

#: The runs M7 compares: every signal, then each one alone.
VARIANTS: dict[str, tuple[FeatureGroup, ...]] = {
    "fusion": tuple(FeatureGroup),
    "dqn_only": (FeatureGroup.POLICY,),
    "camera_only": (FeatureGroup.CAMERA,),
    "lstm_only": (FeatureGroup.BEHAVIOUR,),
}


def main() -> int:
    """Collect (unless reusing), train every variant, report M7."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)
    config = _apply_overrides(loader.load_fusion_config(args.config), args)

    train_path = config.dataset_dir / "train.npz"
    val_path = config.dataset_dir / "val.npz"
    if not args.reuse_data:
        _collect_all(loader, args, config, train_path, val_path)
    train_set, val_set = FusionDataset.load(train_path), FusionDataset.load(val_path)
    _describe(train_set, val_set)

    trainer = FusionTrainer(config)
    outcomes = {
        name: trainer.train(train_set, val_set, groups, _run_name(args.name, name))
        for name, groups in VARIANTS.items()
    }
    _report(outcomes, val_set, trainer.resolve_device(), config.critical_weight)
    return 0


def _collect_all(
    loader: ConfigLoader,
    args: argparse.Namespace,
    config: FusionTrainingConfig,
    train_path: Path,
    val_path: Path,
) -> None:
    """Run the collection missions and save both splits."""
    app_config = loader.load_app_config(args.app_config)
    factory = MissionFactory.from_app_config(loader, app_config, belief_lag=config.belief_lag)
    source = _evidence_source(loader, args, app_config.sensor_config_path, config.seed)
    policy = DqnLocalController.from_file(config.dqn_weights, device=args.device)
    predictor = LstmMotionPredictor.from_checkpoint(config.lstm_weights, device=args.device)

    def evidence_for(mission: Mission) -> Callable[[], SceneEvidence]:
        return partial(source.evidence, mission.city_map)

    for path, seeds in ((train_path, config.training_seeds), (val_path, config.validation_seeds)):
        recorder = FusionRecorder(
            policy,
            predictor,
            config.oracle_drive_probability,
            random.Random(config.seed + seeds[0]),
        )
        print(f"collecting {len(seeds)} missions (seeds {seeds[0]}-{seeds[-1]}) ...", flush=True)
        dataset = collect(factory.build, evidence_for, seeds, recorder, config.max_ticks)
        dataset.save(path)
        print(f"  {len(dataset):,} ticks, {int(dataset.critical.sum()):,} critical -> {path}")


def _evidence_source(
    loader: ConfigLoader, args: argparse.Namespace, sensor_path: Path | None, seed: int
) -> OnboardEvidenceSource:
    """The onboard camera; YOLO on degraded frames when ``--detector`` is given."""
    if sensor_path is None:
        raise SystemExit("the app config must set 'sensor_config' to collect fusion data")
    sensor_config = loader.load_sensor_config(sensor_path)
    rig = SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, sensor_path))
    if args.detector is None:
        return OnboardEvidenceSource(rig)

    from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415

    return OnboardEvidenceSource(
        rig,
        detector=YoloDetector(loader.resolve(args.detector), device=args.device),
        denoiser=_denoiser(loader, args),
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(seed)),
    )


def _describe(train_set: FusionDataset, val_set: FusionDataset) -> None:
    for name, data in (("train", train_set), ("val", val_set)):
        share = data.critical.mean() if len(data) else 0.0
        print(
            f"{name:<6}{len(data):>8,} ticks  {int(data.critical.sum()):>6,} critical ({share:.1%})"
        )


def _report(
    outcomes: dict[str, FusionTrainingOutcome],
    val_set: FusionDataset,
    device: str,
    critical_weight: float,
) -> None:
    """Every variant next to the DQN alone, and the M7 verdict."""
    rows = {
        "DQN alone (no model)": ActionReport.score(
            val_set.labels, val_set.dqn_predictions, val_set.critical, critical_weight
        )
    }
    rows.update({name: outcome.report for name, outcome in outcomes.items()})
    print(f"\nTrained on {device}. Held-out missions: {len(set(val_set.missions.tolist()))}")
    print(f"  {'':<22}{'accuracy':>9}  {'caught':>7}  {'false-override':>15}  {'macro-F1':>9}")
    for name, report in rows.items():
        print(
            f"  {name:<22}{report.accuracy:>9.4f}  {report.critical_accuracy:>7.1%}  "
            f"{report.false_override_rate:>15.2%}  {report.macro_f1:>9.4f}"
        )
    print("  (caught = share of the stale-map DQN's mistakes corrected;")
    print("   false-override = share of ticks the DQN had right that were changed)")
    fusion = rows.pop("fusion")
    met = all(fusion.weighted_accuracy > other.weighted_accuracy for other in rows.values())
    verdict = "MET" if met else "NOT MET"
    print(f"\nM7 (fusion beats every single signal on weighted accuracy): {verdict}")
    print("  The mission-level test is scripts/evaluate_fusion.py.")
    print(f"  weights     {outcomes['fusion'].weights_path}")


def _run_name(base: str, variant: str) -> str:
    return base if variant == "fusion" else f"{base}_{variant}"


def _apply_overrides(
    config: FusionTrainingConfig, args: argparse.Namespace
) -> FusionTrainingConfig:
    """Let command-line flags win over the config file, for quick experiments."""
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.missions is not None:
        config = replace(
            config, train_missions=args.missions, val_missions=max(1, args.missions // 4)
        )
    if args.lag is not None:
        config = replace(config, belief_lag=args.lag)
    if args.critical_weight is not None:
        config = replace(config, critical_weight=args.critical_weight)
    return replace(config, device=args.device) if args.device else config


def _denoiser(loader: ConfigLoader, args: argparse.Namespace) -> IDenoiser | None:
    """The Phase 4 autoencoder in front of the onboard detector, when ``--denoiser`` names one."""
    if args.denoiser is None:
        return None

    from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder  # noqa: PLC0415

    return ConvDenoisingAutoencoder.from_checkpoint(loader.resolve(args.denoiser), args.device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the decision-fusion MLP.")
    parser.add_argument("--config", default="configs/training/fusion.yaml", help="Fusion config.")
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--reuse-data", action="store_true", help="Skip collection.")
    parser.add_argument("--detector", help="YOLO weights for the onboard camera (default: labels).")
    parser.add_argument("--denoiser", help="Autoencoder applied before --detector.")
    parser.add_argument("--device", default="cpu", help='Model device ("cpu", "cuda", index).')
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--missions", type=int, help="Override training missions (val = 1/4).")
    parser.add_argument("--lag", type=int, help="Override the map lag, in refreshes.")
    parser.add_argument("--critical-weight", type=float, help="Override the critical-tick weight.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
