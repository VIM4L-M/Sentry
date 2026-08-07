#!/usr/bin/env python3
"""Composition root for a live rescue mission.

Wires every Phase 2 part together — config, city, occupancy grid, A*
planner, mission controller, local controller, engine, renderer — and runs
it in a window. This is the only place that knows which *concrete* adapters
are in play; everything downstream sees ports.

Usage:
    python scripts/run_simulation.py [--config configs/app.yaml]
    python scripts/run_simulation.py --headless [--max-ticks 5000]

Controls (windowed): Escape quit, Space pause, R restart the mission, Tab
toggle manual driving, G occupancy-grid view, C camera panel, arrow keys /
WASD to drive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.grid_source import DetectedGridSource, ModelObserver
from sentry_ai.rendering.simulation_app import (
    MissionScene,
    SimulationApp,
    build_mode_switch,
)
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.grid_source import GroundTruthGridSource
from sentry_ai.simulation.hazards import build_world_processes
from sentry_ai.simulation.mission import MissionController, MissionStats
from sentry_ai.simulation.waypoint_follower import WaypointFollower

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_TICKS = 5000


def main() -> int:
    """Parse arguments, compose the mission, and run it."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)
    logger = get_logger(__name__)

    scene = _build_scene(loader, app_config, args)
    logger.info(
        "Mission ready: %dx%d city, %d victim(s), %d fire(s)",
        scene.city_map.width,
        scene.city_map.height,
        len(scene.city_map.victims),
        len(scene.city_map.fires),
    )

    if args.headless:
        scene.engine.run(max_ticks=args.max_ticks)
        engine = scene.engine
    else:
        app = SimulationApp(
            city_map=scene.city_map,
            engine=scene.engine,
            render_config=app_config.render,
            theme=Theme.from_config(loader, app_config.render.palette_config_path),
            mode_switch=scene.mode_switch,
            sensor_rig=_build_sensor_rig(loader, app_config),
            restart=lambda: _build_scene(loader, app_config, args),
        )
        app.run()
        # After restarts this is a different engine to the one we started
        # with, and it is the mission that actually just ran.
        engine = app.engine

    _report(engine.stats, engine.mission)
    return 0


def _build_scene(
    loader: ConfigLoader, app_config: AppConfig, args: argparse.Namespace
) -> MissionScene:
    """Compose a complete, unstarted mission.

    Called once at launch and again for every ``R`` press, so restarting is
    exactly "build another one" rather than an attempt to rewind a world
    that a mission has already mutated.
    """
    simulation_config, vehicle_config = _load_mission_configs(loader, app_config)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    grid_source = _build_grid_source(loader, app_config, args, city_map)
    mission = MissionController(
        city_map=city_map,
        grid=grid_source.grid_for(city_map, city_map.vehicle.position),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
        grid_source=grid_source,
    )
    mode_switch = build_mode_switch(WaypointFollower())
    engine = SimulationEngine(
        city_map=city_map,
        mission=mission,
        controller=mode_switch,
        simulation_config=simulation_config,
        vehicle_config=vehicle_config,
        world_processes=build_world_processes(simulation_config.hazards),
    )
    return MissionScene(city_map=city_map, engine=engine, mode_switch=mode_switch)


def _build_grid_source(
    loader: ConfigLoader,
    app_config: AppConfig,
    args: argparse.Namespace,
    city_map: CityMap,
) -> IOccupancyGridSource:
    """Where the command center's map comes from: the world, or the cameras.

    The Phase 3.5 switch. Everything downstream — the planner, the mission
    state machine, the vehicle — is identical either way and never learns
    which one it got. Press ``G`` with ``--perception`` on and the grid view
    shows a *belief*, wrong in the ways docs/architecture/phase3-detection.md
    describes, rather than the world.
    """
    if not args.perception:
        return GroundTruthGridSource()

    path = app_config.sensor_config_path
    if path is None:
        raise ConfigurationError("--perception needs 'sensor_config' set in the app config")

    from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415

    sensor_config = loader.load_sensor_config(path)
    return DetectedGridSource(
        rig=SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, path)),
        observer=ModelObserver(
            YoloDetector(
                weights_path=loader.resolve(args.weights),
                confidence=args.confidence,
                image_size=args.image_size,
                device=args.device,
            )
        ),
        builder=OccupancyGridBuilder.from_city_map(city_map),
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(args.seed)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SENTRY AI rescue mission.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to app.yaml, relative to the project root (default: configs/app.yaml).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the mission with no window and print the result.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_MAX_TICKS,
        help=f"Headless tick budget before giving up (default: {DEFAULT_MAX_TICKS}).",
    )
    perception = parser.add_argument_group("perception (Phase 3.5)")
    perception.add_argument(
        "--perception",
        action="store_true",
        help="Plan on what the cameras see instead of on ground truth.",
    )
    perception.add_argument(
        "--weights",
        default="models/yolo/labelfix/weights/best.pt",
        help="Detector checkpoint used by --perception.",
    )
    perception.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    perception.add_argument(
        "--confidence", type=float, default=0.25, help="Detector confidence threshold."
    )
    perception.add_argument(
        "--image-size", type=int, default=256, help="Inference resolution; must match training."
    )
    perception.add_argument("--seed", type=int, default=0, help="Seed for the frame degrader.")
    return parser.parse_args()


def _load_mission_configs(loader: ConfigLoader, app_config: AppConfig) -> tuple:
    """Load the simulation and vehicle configs the app config points at."""
    if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
        raise ConfigurationError(
            "app config must set 'simulation_config' and 'vehicle_config' to run a mission"
        )
    return (
        loader.load_simulation_config(app_config.simulation_config_path),
        loader.load_vehicle_config(app_config.vehicle_config_path),
    )


def _build_sensor_rig(loader: ConfigLoader, app_config: AppConfig) -> SensorRig | None:
    """The camera network for the window, or ``None`` if none is configured.

    Optional rather than required: a mission is perfectly watchable without
    the camera strip, and an app config that omits ``sensor_config`` should
    still open a window rather than fail.
    """
    path = app_config.sensor_config_path
    if path is None:
        return None
    return SensorRig.from_config(
        loader.load_sensor_config(path), SensorPalette.from_config(loader, path)
    )


def _report(stats: MissionStats, mission: MissionController) -> None:
    """Print the mission outcome as the script's final user-facing output."""
    reason = f" — {stats.failure_reason}" if stats.failure_reason else ""
    print(f"Mission {mission.phase.value}{reason}")
    for label, value in stats.as_display_rows():
        print(f"  {label:<12}{value}")


if __name__ == "__main__":
    raise SystemExit(main())
