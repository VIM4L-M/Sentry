#!/usr/bin/env python3
"""Scores the occupancy grid the detector builds against ground truth (Phase 3.3).

``evaluate_yolo.py`` answers "how good is the detector?" in mAP. This answers
the question the mission actually depends on: **would the command center make
the right decisions using the map this detector produces?**

They are not the same question. A detector can post a fine mAP and still
strand someone, because what reaches the planner is not a list of boxes but
an occupancy grid, and the three ways that grid can be wrong cost very
different amounts — see :mod:`sentry_ai.perception.grid_metrics`.

Runs the full chain on one instant of the real city::

    SensorRig -> FrameDegrader -> YoloDetector -> DetectionMerger
              -> OccupancyGridBuilder -> GridComparison

Usage:
    python scripts/evaluate_grid.py
    python scripts/evaluate_grid.py --weights models/yolo/labelfix/weights/best.pt
    python scripts/evaluate_grid.py --device cuda
    python scripts/evaluate_grid.py --perfect   # ground truth instead of the model
    python scripts/evaluate_grid.py --denoiser models/autoencoder/sentry/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.perception import IDenoiser
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.grid_metrics import GridComparison
from sentry_ai.perception.grid_source import (
    DetectedGridSource,
    GroundTruthObserver,
    IFrameObserver,
    ModelObserver,
)
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Build a belief grid from the cameras and score it against the world."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)

    sensor_config = loader.load_sensor_config(args.sensors)
    rig = SensorRig.from_config(
        sensor_config, SensorPalette.from_config(loader, args.sensors)
    )
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))

    source = DetectedGridSource(
        rig=rig,
        observer=_observer(loader, args),
        builder=OccupancyGridBuilder.from_city_map(city_map, min_confidence=args.min_confidence),
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(args.seed)),
        denoiser=_denoiser(loader, args),
    )
    belief = source.grid_for(city_map, city_map.vehicle.position)

    _report(GridComparison.between(OccupancyGrid.from_city_map(city_map), belief),
            city_map, belief, args)
    return 0


def _observer(loader: ConfigLoader, args: argparse.Namespace) -> IFrameObserver:
    """How frames become detections: the model, or the answer key.

    ``--perfect`` substitutes the rasterizer's own annotations, which isolates
    the projection and merge steps from the detector entirely. Any
    disagreement it still reports is a pipeline defect, not a weights problem.
    """
    if args.perfect:
        logger.info("Using ground-truth annotations in place of the detector")
        return GroundTruthObserver()

    from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415

    return ModelObserver(
        YoloDetector(
            weights_path=loader.resolve(args.weights),
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    )


def _denoiser(loader: ConfigLoader, args: argparse.Namespace) -> IDenoiser | None:
    """The Phase 4 autoencoder between degrader and detector, if one was named."""
    if args.denoiser is None:
        return None

    from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder  # noqa: PLC0415

    return ConvDenoisingAutoencoder.from_checkpoint(loader.resolve(args.denoiser), args.device)


def _report(
    result: GridComparison,
    city_map: CityMap,
    belief: OccupancyGrid,
    args: argparse.Namespace,
) -> None:
    """Print the scores, then say whether a mission could actually run on this."""
    print(f"Belief grid from {'ground truth' if args.perfect else args.weights}\n")
    print(result.summary())

    print("\nMission viability")
    print(f"  {'every victim on the map':<28}{_yes(result.finds_every_victim)}")
    print(f"  {'no hazard driven into':<28}{_yes(result.is_drivable)}")
    reachable = _reachable(city_map, belief)
    print(f"  {'victims reachable by A*':<28}{reachable}/{len(city_map.victims)}")

    if result.missed_victims:
        tiles = ", ".join(str(p.as_tuple()) for p in sorted(result.missed_victims,
                                                            key=lambda p: (p.y, p.x)))
        print(f"\nMissed victims at {tiles} — nobody would be dispatched to them.")


def _reachable(city_map: CityMap, belief: OccupancyGrid) -> int:
    """How many victims A* can still route to on the believed map."""
    planner = AStarPlanner()
    return sum(
        not planner.plan(belief, city_map.vehicle.position, victim.position).is_empty
        for victim in city_map.victims
    )


def _yes(value: bool) -> str:
    return "yes" if value else "NO"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a detector-built occupancy grid.")
    parser.add_argument("--config", default="configs/app.yaml", help="Root application config.")
    parser.add_argument("--sensors", default="configs/sensors.yaml", help="Camera network config.")
    parser.add_argument(
        "--weights",
        default="models/yolo/labelfix/weights/best.pt",
        help="Detector checkpoint to build the grid from.",
    )
    parser.add_argument(
        "--perfect",
        action="store_true",
        help="Use ground-truth annotations instead of the model, to isolate the pipeline.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or index.")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Detector confidence threshold."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop merged detections below this before building the grid.",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Inference resolution; must match training."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the frame degrader.")
    parser.add_argument(
        "--denoiser",
        help="Denoising autoencoder checkpoint (Phase 4), applied before the detector.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
