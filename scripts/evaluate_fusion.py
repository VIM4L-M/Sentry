#!/usr/bin/env python3
"""Mission-level test of the fusion MLP (Phase 7): does it prevent the crashes?

Runs the held-out evaluation missions of ``configs/training/fusion.yaml``
with the command-center map lagging the world, three ways:

* **fresh map, DQN** — no lag at all: the ceiling.
* **lagged map, DQN** — the DQN alone, driving on the stale map.
* **lagged map, fusion** — the same DQN, with the LSTM, the onboard camera
  and the fusion network arbitrating every tick.

Per-tick scores say whether fusion picks the right action; this says
whether that matters — collisions, damage, rescues.

Usage:
    python scripts/evaluate_fusion.py
    python scripts/evaluate_fusion.py --missions 10 --weights models/fusion/sentry/best.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from sentry_ai.common.logging_config import setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.decision.dqn_controller import DqnLocalController
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.decision.mlp_fusion import MlpFusion
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.interfaces.perception import IDenoiser
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor
from sentry_ai.simulation.factory import Mission
from sentry_ai.simulation.grid_source import LaggedGridSource
from sentry_ai.simulation.onboard_reports import OnboardHazardReporter
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Builds the controller for one mission; given the mission so a camera can bind to it.
ControllerFor = Callable[[Mission], ILocalController]


@dataclass
class Tally:
    """Mission outcomes summed over a set of missions."""

    missions: int = 0
    rescued: int = 0
    lost: int = 0
    collisions: int = 0
    damage: float = 0.0
    completed: int = 0
    overrides: int = 0


def main() -> int:
    """Run the three configurations on identical missions and print the table."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)
    config = loader.load_fusion_config(args.config)
    seeds = (
        list(config.evaluation_seeds)[: args.missions]
        if args.missions
        else list(config.evaluation_seeds)
    )

    dqn = DqnLocalController.from_file(config.dqn_weights, device=args.device)
    predictor = LstmMotionPredictor.from_checkpoint(config.lstm_weights, device=args.device)
    weights = (
        loader.resolve(args.weights) if args.weights else config.runs_dir / "sentry" / "best.pt"
    )
    threshold = config.override_threshold if args.threshold is None else args.threshold
    report_threshold = (
        config.report_threshold if args.report_threshold is None else args.report_threshold
    )
    fusion = MlpFusion.from_checkpoint(weights, device=args.device, override_threshold=threshold)
    source = _evidence_source(loader, args, app_config.sensor_config_path)

    def fused(mission: Mission, report: bool = True) -> ILocalController:
        lag = mission.controller.grid_source
        reporter = (
            OnboardHazardReporter(lag, mission.controller, report_threshold)
            if report and isinstance(lag, LaggedGridSource)
            else None
        )
        return FusedLocalController(
            dqn,
            partial(source.evidence, mission.city_map),
            fusion,
            predictor,
            on_override=reporter,
        )

    runs = {
        "fresh map, DQN": (0, lambda _mission: dqn),
        "lagged map, DQN": (config.belief_lag, lambda _mission: dqn),
        "lagged, fusion only": (config.belief_lag, lambda mission: fused(mission, report=False)),
        "lagged, fusion+reports": (config.belief_lag, fused),
    }
    print(f"{len(seeds)} held-out missions, map lag {config.belief_lag} refreshes\n")
    rows = {
        name: _run(
            MissionFactory.from_app_config(loader, app_config, belief_lag=lag),
            seeds,
            make,
            config.max_ticks,
        )
        for name, (lag, make) in runs.items()
    }
    _print(rows)
    return 0


def _run(factory: MissionFactory, seeds: list[int], make: ControllerFor, max_ticks: int) -> Tally:
    """Drive every seed with a fresh controller from ``make``."""
    tally = Tally()
    for seed in seeds:
        holder: list[ILocalController] = []
        mission = factory.build(seed, _Deferred(holder))
        holder.append(make(mission))
        stats = mission.engine.run(max_ticks)
        tally.missions += 1
        tally.rescued += stats.victims_rescued
        tally.lost += stats.victims_lost
        tally.collisions += stats.collisions
        tally.damage += 100.0 - mission.city_map.vehicle.health_percent
        tally.completed += int(mission.controller.phase.value == "completed")
        controller = holder[0]
        if isinstance(controller, FusedLocalController):
            tally.overrides += controller.overrides
    return tally


class _Deferred(ILocalController):
    """Forwards to a controller built after the mission — the camera needs the mission's city."""

    def __init__(self, holder: list[ILocalController]) -> None:
        self._holder = holder

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Ask the controller the holder now contains."""
        return self._holder[0].decide(observation)


def _evidence_source(
    loader: ConfigLoader, args: argparse.Namespace, sensor_path: Path | None
) -> OnboardEvidenceSource:
    """The onboard camera; YOLO on degraded frames when ``--detector`` is given."""
    if sensor_path is None:
        raise SystemExit("the app config must set 'sensor_config'")
    sensor_config = loader.load_sensor_config(sensor_path)
    rig = SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, sensor_path))
    if args.detector is None:
        return OnboardEvidenceSource(rig)

    from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415

    return OnboardEvidenceSource(
        rig,
        detector=YoloDetector(loader.resolve(args.detector), device=args.device),
        denoiser=_denoiser(loader, args),
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(0)),
    )


def _print(rows: dict[str, Tally]) -> None:
    columns = ("rescued", 8), ("lost", 6), ("collisions", 12), ("damage", 8), ("completed", 11)
    header = "".join(f"{name:>{width}}" for name, width in columns)
    print(f"  {'':<22}{header}{'overrides':>11}")
    for name, t in rows.items():
        print(
            f"  {name:<22}{t.rescued:>8}{t.lost:>6}{t.collisions:>12}{t.damage:>8.0f}"
            f"{t.completed:>6}/{t.missions:<4}{t.overrides:>11}"
        )
    stale, fused = rows["lagged map, DQN"], rows["lagged, fusion+reports"]
    if stale.collisions:
        cut = 1.0 - fused.collisions / stale.collisions
        print(f"\nFusion removes {cut:.0%} of the collisions the stale map causes.")


def _denoiser(loader: ConfigLoader, args: argparse.Namespace) -> IDenoiser | None:
    """The Phase 4 autoencoder in front of the onboard detector, when ``--denoiser`` names one."""
    if args.denoiser is None:
        return None

    from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder  # noqa: PLC0415

    return ConvDenoisingAutoencoder.from_checkpoint(loader.resolve(args.denoiser), args.device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission-level evaluation of the fusion MLP.")
    parser.add_argument("--config", default="configs/training/fusion.yaml", help="Fusion config.")
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--weights", help="Fusion checkpoint (default: runs_dir/sentry/best.pt).")
    parser.add_argument("--detector", help="YOLO weights for the onboard camera (default: labels).")
    parser.add_argument("--denoiser", help="Autoencoder applied before --detector.")
    parser.add_argument("--missions", type=int, help="Evaluate only the first N missions.")
    parser.add_argument("--threshold", type=float, help="Override the config's override threshold.")
    parser.add_argument("--report-threshold", type=float, help="Override the report threshold.")
    parser.add_argument("--device", default="cpu", help='Model device ("cpu", "cuda", index).')
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
