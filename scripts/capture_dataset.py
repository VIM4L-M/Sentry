#!/usr/bin/env python3
"""Composition root for generating the synthetic perception dataset.

Runs a rescue mission and captures the camera network as it goes, writing
what Phases 3 and 4 need to train against:

    <output>/images/<camera>_<tick>.png       degraded frames (detector input)
    <output>/labels/<camera>_<tick>.txt       YOLO labels for those frames
    <output>/clean/<camera>_<tick>.png        the matching clean frames
    <output>/classes.txt                      class id -> name

The clean frames are not a debugging luxury: paired with the degraded ones
they *are* the Unit IV autoencoder's training set, and they are only
perfectly aligned because both come from the same rasterization.

Samples are taken every ``--every`` ticks while the mission runs, so the
dataset spans a whole disaster — fires that grow and burn out, buildings
that come down, victims that disappear as they are rescued — rather than
one static frame repeated.

Usage:
    python scripts/capture_dataset.py --output data/synthetic
    python scripts/capture_dataset.py --output data/synthetic --every 5 --seed 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import YOLO_CLASSES
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.hazards import build_world_processes
from sentry_ai.simulation.mission import MissionController
from sentry_ai.simulation.waypoint_follower import WaypointFollower

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_TICKS = 5000

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, run a mission, and write the captured dataset."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)

    sensor_path = _require_sensor_config(app_config)
    rig = SensorRig.from_config(
        loader.load_sensor_config(sensor_path),
        SensorPalette.from_config(loader, sensor_path),
    )
    degrader = FrameDegrader(
        loader.load_sensor_config(sensor_path).degradation,
        np.random.default_rng(args.seed),
    )

    engine, city_map = _build_mission(loader, app_config)
    logger.info("CCTV coverage of the city: %.1f%%", rig.coverage(city_map) * 100)

    written = _capture_mission(engine, city_map, rig, degrader, args)
    print(f"Wrote {written} sample(s) to {args.output}")
    return 0


def _capture_mission(
    engine: SimulationEngine,
    city_map: CityMap,
    rig: SensorRig,
    degrader: FrameDegrader,
    args: argparse.Namespace,
) -> int:
    """Tick the mission, capturing every ``args.every`` ticks."""
    output = Path(args.output)
    _write_classes(output)
    written = 0

    for tick in range(args.max_ticks):
        if tick % args.every == 0:
            written += _write_sample(output, rig, degrader, city_map, tick)
        if engine.tick() is None:
            break
    return written


def _write_sample(
    output: Path,
    rig: SensorRig,
    degrader: FrameDegrader,
    city_map: CityMap,
    tick: int,
) -> int:
    """Write every camera's clean frame, degraded frame, and labels."""
    for frame in rig.capture_all(city_map):
        stem = f"{frame.camera_id}_{tick:05d}"
        _save_png(output / "clean" / f"{stem}.png", frame.pixels)
        _save_png(output / "images" / f"{stem}.png", degrader.degrade(frame).pixels)
        _write_text(output / "labels" / f"{stem}.txt", "\n".join(frame.to_yolo_lines()))
    return 1


def _build_mission(
    loader: ConfigLoader, app_config: AppConfig
) -> tuple[SimulationEngine, CityMap]:
    """Compose the same mission ``run_simulation.py`` does, hazards included."""
    if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
        raise ConfigurationError(
            "app config must set 'simulation_config' and 'vehicle_config' to capture a mission"
        )
    simulation = loader.load_simulation_config(app_config.simulation_config_path)
    vehicle = loader.load_vehicle_config(app_config.vehicle_config_path)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(simulation.planner),
        config=simulation.mission,
    )
    engine = SimulationEngine(
        city_map=city_map,
        mission=mission,
        controller=WaypointFollower(),
        simulation_config=simulation,
        vehicle_config=vehicle,
        world_processes=build_world_processes(simulation.hazards),
    )
    return engine, city_map


def _require_sensor_config(app_config: AppConfig) -> Path:
    """The sensors config path, or a clear error if the app config omits it."""
    if app_config.sensor_config_path is None:
        raise ConfigurationError("app config must set 'sensor_config' to capture frames")
    return app_config.sensor_config_path


def _write_classes(output: Path) -> None:
    """Write the class-id ordering the label files are numbered against."""
    _write_text(output / "classes.txt", "\n".join(kind.value for kind in YOLO_CLASSES))


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _save_png(path: Path, pixels: np.ndarray) -> None:
    """Save an ``(h, w, 3)`` uint8 array as a PNG.

    Pygame is already a dependency and can encode PNGs, so this avoids
    pulling in Pillow purely to write files. The dummy video driver is
    forced first: encoding must not require a display.
    """
    import pygame  # imported lazily so the sensors package stays UI-free

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    path.parent.mkdir(parents=True, exist_ok=True)
    surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
    pygame.image.save(surface, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a synthetic perception dataset.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to app.yaml, relative to the project root (default: configs/app.yaml).",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic",
        help="Directory to write the dataset into (default: data/synthetic).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Capture one sample every N simulation ticks (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for smoke and sensor noise, so a dataset is reproducible (default: 0).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_MAX_TICKS,
        help=f"Tick budget for the captured mission (default: {DEFAULT_MAX_TICKS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
