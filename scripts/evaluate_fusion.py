#!/usr/bin/env python3
"""Mission-level evaluation of decision fusion: crashes avoided (milestone M7).

``train_fusion.py`` scores fusion per decision. This scores what the
decisions are for: whole missions, driven on a command-center map that
lags reality by ``belief_lag`` refreshes, where the vehicle's own camera is
the only thing that sees a fresh collapse in time.

Four drivers, on identical missions — same hazard seed, same start tile,
same lag — none of them used for training or for choosing a checkpoint
(the seeds come after the validation range):

* the waypoint follower (Phase 2);
* the DQN alone (Phase 6);
* the DQN with the hand-written camera-veto rule;
* the DQN with the trained MLP fusion (Phase 7).

Usage:
    python scripts/evaluate_fusion.py
    python scripts/evaluate_fusion.py --missions 60 --detector models/yolo/labelfix/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.decision.fusion import CameraVetoFusion, MlpFusion
from sentry_ai.interfaces.decision import IDecisionFusion
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.perception.onboard import OnboardSensing
from sentry_ai.simulation.factory import Mission
from sentry_ai.simulation.grid_source import GroundTruthGridSource, LaggedGridSource
from sentry_ai.training.dqn import ControllerScore, drive_missions
from sentry_ai.training.fusion import Pipeline, assemble_pipeline
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


class _FusedDrivers:
    """Builds a fused controller per mission and attaches its camera once the city exists."""

    def __init__(self, pipeline: Pipeline, fusion: IDecisionFusion) -> None:
        self._pipeline = pipeline
        self._fusion = fusion
        self._sensing: dict[int, OnboardSensing] = {}

    def controller(self) -> ILocalController:
        sensing = self._pipeline.sensing_factory()
        controller = FusedLocalController(
            self._pipeline.local, self._pipeline.predictor, self._fusion, sensing.sightings
        )
        self._sensing[id(controller)] = sensing
        return controller

    def prepare(self, mission: Mission, controller: ILocalController | None) -> None:
        self._sensing.pop(id(controller)).attach(mission.city_map)


def main() -> int:
    """Drive the held-out missions four ways and compare."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = loader.load_fusion_config(args.config)
    pipeline = assemble_pipeline(
        loader,
        loader.resolve(args.sensors),
        dqn_weights=loader.resolve(args.dqn),
        lstm_weights=loader.resolve(args.lstm),
        detector_weights=loader.resolve(args.detector),
        device=args.device,
        seed=config.seed + 1,
    )
    fusion_weights = (
        loader.resolve(args.fusion)
        if args.fusion
        else config.runs_dir / args.run / "fusion" / "best.pt"
    )
    factory = MissionFactory.from_app_config(loader, app_config)
    first = config.validation_seeds.stop
    seeds = list(range(first, first + args.missions))

    def lagged(seed: int, controller: ILocalController | None) -> Mission:
        return factory.build(
            seed, controller, LaggedGridSource(GroundTruthGridSource(), config.belief_lag)
        )

    rule = _FusedDrivers(pipeline, CameraVetoFusion())
    mlp = _FusedDrivers(pipeline, MlpFusion.from_checkpoint(fusion_weights, device=args.device))
    drivers: dict[str, tuple[Callable[[], ILocalController | None], object]] = {
        "waypoint follower": (lambda: None, None),
        "DQN alone": (lambda: pipeline.local, None),
        "DQN + camera-veto rule": (rule.controller, rule.prepare),
        "DQN + MLP fusion": (mlp.controller, mlp.prepare),
    }

    print(
        f"{len(seeds)} held-out missions (seeds {seeds[0]}-{seeds[-1]}), "
        f"command-center map {config.belief_lag} refreshes behind reality\n"
    )
    print(
        f"  {'':<24}{'rescued':>8}{'lost':>6}{'collisions':>12}{'damage':>8}"
        f"{'completed':>11}{'ticks':>7}"
    )
    scores: dict[str, ControllerScore] = {}
    for name, (make, prepare) in drivers.items():
        score = drive_missions(
            lagged, make, seeds, config.max_ticks, prepare=prepare  # type: ignore[arg-type]
        )
        scores[name] = score
        print(
            f"  {name:<24}{score.rescued:>8}{score.lost:>6}{score.collisions:>12}"
            f"{score.mean_damage:>7.1f}%{score.completion_rate:>10.0%}{score.mean_ticks:>7.0f}",
            flush=True,
        )

    alone, fused = scores["DQN alone"], scores["DQN + MLP fusion"]
    avoided = alone.collisions - fused.collisions
    print(
        f"\nFusion vs the DQN alone: {avoided:+d} collisions avoided "
        f"({fused.collisions} vs {alone.collisions}), "
        f"{fused.rescued - alone.rescued:+d} rescued, {fused.lost - alone.lost:+d} lost, "
        f"vehicle damage {alone.mean_damage:.1f}% -> {fused.mean_damage:.1f}% per mission"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decision fusion on whole missions.")
    parser.add_argument(
        "--config",
        default="configs/training/fusion.yaml",
        help="Fusion config (default: configs/training/fusion.yaml).",
    )
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--sensors", default="configs/sensors.yaml", help="Camera config.")
    parser.add_argument("--missions", type=int, default=40, help="Held-out missions to drive.")
    parser.add_argument("--dqn", default="models/dqn/sentry/best.zip", help="DQN weights.")
    parser.add_argument("--lstm", default="models/lstm/dqn/best.pt", help="LSTM weights.")
    parser.add_argument(
        "--detector",
        default="models/yolo/labelfix/weights/best.pt",
        help="YOLO weights for the onboard camera.",
    )
    parser.add_argument("--fusion", help="Fusion weights (default: the run's fusion/best.pt).")
    parser.add_argument("--run", default="sentry", help="Training run name.")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
