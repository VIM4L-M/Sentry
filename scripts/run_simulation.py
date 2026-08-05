#!/usr/bin/env python3
"""Composition root for a live rescue mission.

Wires every Phase 2 part together — config, city, occupancy grid, A*
planner, mission controller, local controller, engine, renderer — and runs
it in a window. This is the only place that knows which *concrete* adapters
are in play; everything downstream sees ports.

Usage:
    python scripts/run_simulation.py [--config configs/app.yaml]
    python scripts/run_simulation.py --headless [--max-ticks 5000]

Controls (windowed): Escape quit, Space pause, Tab toggle manual driving,
arrow keys / WASD to drive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.rendering.simulation_app import SimulationApp, build_mode_switch
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine
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

    simulation_config, vehicle_config = _load_mission_configs(loader, app_config)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
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
    logger.info(
        "Mission ready: %dx%d city, %d victim(s), %d fire(s)",
        city_map.width,
        city_map.height,
        len(city_map.victims),
        len(city_map.fires),
    )

    if args.headless:
        engine.run(max_ticks=args.max_ticks)
    else:
        theme = Theme.from_config(loader, app_config.render.palette_config_path)
        SimulationApp(
            city_map=city_map,
            engine=engine,
            render_config=app_config.render,
            theme=theme,
            mode_switch=mode_switch,
            sensor_rig=_build_sensor_rig(loader, app_config),
        ).run()

    _report(engine.stats, mission)
    return 0


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
