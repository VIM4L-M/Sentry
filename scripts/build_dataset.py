#!/usr/bin/env python3
"""Composition root for the synthetic perception dataset.

Runs a series of rescue missions — each a *different* disaster on the same
map, driven by a different hazard seed — and writes what the cameras saw in
the layout Ultralytics expects:

    <output>/data.yaml                  dataset descriptor
    <output>/images/{train,val}/*.png   degraded frames (detector input)
    <output>/labels/{train,val}/*.txt   YOLO labels for those frames
    <output>/clean/{train,val}/*.png    matching clean frames (Unit IV pairs)

Validation missions are held out whole. Splitting individual frames at
random would put near-duplicates on both sides — consecutive ticks of one
mission look almost identical — and the resulting mAP would measure
memorisation rather than detection.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --train-missions 20 --val-missions 5
    python scripts/build_dataset.py --output data/tiny --train-missions 2 --val-missions 1
    python scripts/build_dataset.py --sensors configs/sensors_v75.yaml --output data/v75
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
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.factory import Mission, build_mission
from sentry_ai.training.dataset import (
    CaptureOptions,
    DatasetLayout,
    DatasetStats,
    MissionSource,
    YoloDatasetBuilder,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, run the missions, and write the dataset."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)

    sensor_path = _resolve_sensor_config(app_config, loader, args.sensors)
    sensor_config = loader.load_sensor_config(sensor_path)
    rig = SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, sensor_path))

    builder = YoloDatasetBuilder(
        rig=rig,
        # Seeded so a rebuild produces identical corruption. The variety
        # that matters comes from the hazards, not from the sensor noise.
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(args.seed)),
        layout=DatasetLayout(root=loader.resolve(args.output)),
    )
    stats = builder.build(
        mission_source=_mission_source(loader, app_config),
        options=CaptureOptions(
            train_missions=args.train_missions,
            val_missions=args.val_missions,
            capture_every=args.every,
            max_ticks=args.max_ticks,
        ),
    )
    _report(stats, loader.resolve(args.output))
    return 0


def _mission_source(loader: ConfigLoader, app_config: AppConfig) -> MissionSource:
    """A factory that builds a fresh mission for a given hazard seed.

    A new ``CityMap`` per mission, deliberately: a mission mutates the city
    it runs in, so reusing one would mean every mission after the first
    started from a half-burnt ruin.
    """
    simulation_config, vehicle_config = _load_mission_configs(loader, app_config)
    map_data = loader.load_yaml(app_config.map_config_path)

    def build(seed: int) -> Mission:
        return build_mission(
            city_map=CityMap.from_config(map_data),
            simulation_config=simulation_config,
            vehicle_config=vehicle_config,
            hazard_seed=seed,
        )

    return build


def _load_mission_configs(loader: ConfigLoader, app_config: AppConfig) -> tuple:
    """Load the simulation and vehicle configs the app config points at."""
    if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
        raise ConfigurationError(
            "app config must set 'simulation_config' and 'vehicle_config' to run missions"
        )
    return (
        loader.load_simulation_config(app_config.simulation_config_path),
        loader.load_vehicle_config(app_config.vehicle_config_path),
    )


def _resolve_sensor_config(
    app_config: AppConfig, loader: ConfigLoader, override: str | None
) -> Path:
    """Which sensors file to capture through.

    An override exists so a camera experiment — a different marker size, a
    different ``tile_size_px`` — is one command against one extra YAML file,
    rather than an edit to the shipped config that every other run then
    silently inherits.
    """
    if override is not None:
        return loader.resolve(override)
    if app_config.sensor_config_path is None:
        raise ConfigurationError("app config must set 'sensor_config' to capture frames")
    return app_config.sensor_config_path


def _report(stats: DatasetStats, output: Path) -> None:
    """Print what was written, including the class balance."""
    print(f"Dataset written to {output}")
    for label, value in stats.as_display_rows():
        print(f"  {label:<20}{value}")


def _parse_args() -> argparse.Namespace:
    defaults = CaptureOptions()
    parser = argparse.ArgumentParser(description="Build the synthetic perception dataset.")
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
        "--sensors",
        help=(
            "Override the sensors config the app config points at, for camera "
            "experiments (marker size, tile_size_px). Default: whatever app.yaml names."
        ),
    )
    parser.add_argument(
        "--train-missions",
        type=int,
        default=defaults.train_missions,
        help=f"Disasters captured for training (default: {defaults.train_missions}).",
    )
    parser.add_argument(
        "--val-missions",
        type=int,
        default=defaults.val_missions,
        help=f"Disasters held out for validation (default: {defaults.val_missions}).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=defaults.capture_every,
        help=f"Capture one sample every N ticks (default: {defaults.capture_every}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for smoke and sensor noise, so a rebuild is identical (default: 0).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=defaults.max_ticks,
        help=f"Tick budget per mission (default: {defaults.max_ticks}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
