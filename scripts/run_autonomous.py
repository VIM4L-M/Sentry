#!/usr/bin/env python3
"""Composition root for a fully autonomous mission (Phase 8, milestone M8).

Every trained model, wired together: the four CCTV cameras -> denoiser ->
YOLO -> the command center's map; A* routes on it; the DQN proposes each
move, the LSTM predicts the vehicle's behaviour, the onboard camera looks
ahead, and the fusion network decides. No human input anywhere.

At the end it prints the outcome and where each tick's time went, and writes
the mission record (outcome, timing, path, events) for the dashboard.

Usage:
    python scripts/run_autonomous.py                  # watch it in the window
    python scripts/run_autonomous.py --headless       # no window; prints the result
    python scripts/run_autonomous.py --headless --perception-every 4   # slow machine

Keys (windowed): Escape quit, Space pause, R restart, Tab manual override,
G the command center's camera-built map, C the camera panel.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.autonomy.record import MissionRecord
from sentry_ai.autonomy.stack import AutonomousMission, AutonomyStack
from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig, AutonomyConfig
from sentry_ai.domain.map import CityMap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_TICKS = 3000

logger = get_logger(__name__)


def main() -> int:
    """Load the stack, run one mission, report it."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = loader.load_autonomy_config(args.config)
    if args.perception_every is not None:
        config = replace(config, perception_every=args.perception_every)
    if args.device is not None:
        config = replace(config, device=args.device)
    stack = AutonomyStack.load(loader, app_config, config)

    if args.headless:
        run = _mission(loader, app_config, stack, args.seed)
        record = run.run(max_ticks=args.max_ticks)
    else:
        record = _windowed(loader, app_config, stack, args.seed)

    _report(record, config)
    print(f"\nMission record: {record.save(config.missions_dir)}")
    return 0


def _mission(
    loader: ConfigLoader,
    app_config: AppConfig,
    stack: AutonomyStack,
    seed: int | None,
    wrap: object = None,
) -> AutonomousMission:
    if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
        raise ConfigurationError("app config must set 'simulation_config' and 'vehicle_config'")
    return stack.mission(
        city_map=CityMap.from_config(loader.load_yaml(app_config.map_config_path)),
        simulation_config=loader.load_simulation_config(app_config.simulation_config_path),
        vehicle_config=loader.load_vehicle_config(app_config.vehicle_config_path),
        hazard_seed=seed,
        wrap=wrap,
    )


def _windowed(
    loader: ConfigLoader, app_config: AppConfig, stack: AutonomyStack, seed: int | None
) -> MissionRecord:
    """Run in the mission window; ``R`` builds a fresh full-stack mission."""
    from sentry_ai.rendering.simulation_app import (  # noqa: PLC0415 - Pygame only when windowed
        MissionScene,
        SimulationApp,
        build_mode_switch,
    )
    from sentry_ai.rendering.theme import Theme  # noqa: PLC0415
    from sentry_ai.sensors.palette import SensorPalette  # noqa: PLC0415
    from sentry_ai.sensors.rig import SensorRig  # noqa: PLC0415

    runs: list[AutonomousMission] = []

    def scene() -> MissionScene:
        run = _mission(loader, app_config, stack, seed, wrap=build_mode_switch)
        runs.append(run)
        return MissionScene(
            city_map=run.mission.city_map,
            engine=run.mission.engine,
            mode_switch=run.driver,  # type: ignore[arg-type]
        )

    first = scene()
    sensors = app_config.sensor_config_path
    app = SimulationApp(
        city_map=first.city_map,
        engine=first.engine,
        render_config=app_config.render,
        theme=Theme.from_config(loader, app_config.render.palette_config_path),
        mode_switch=first.mode_switch,
        sensor_rig=(
            None
            if sensors is None
            else SensorRig.from_config(
                loader.load_sensor_config(sensors), SensorPalette.from_config(loader, sensors)
            )
        ),
        restart=scene,
    )
    app.run()
    return runs[-1].record()


def _report(record: MissionRecord, config: AutonomyConfig) -> None:
    stats = record.stats
    reason = f" — {record.failure_reason}" if record.failure_reason else ""
    print(f"\nMission {record.outcome}{reason}")
    print(
        f"  rescued {stats['victims_rescued']}, lost {stats['victims_lost']}, "
        f"unreachable {stats['victims_unreachable']}, collisions {stats['collisions']}, "
        f"ticks {stats['ticks']}, vehicle health {stats['vehicle_health']:.0f}%"
    )
    print(f"\nWhere the time goes (perception every {config.perception_every} tick(s)):")
    print(f"  {'stage':<20}{'calls':>7}{'ms/call':>10}{'ms/tick':>10}")
    for timing in record.timings:
        print(
            f"  {timing['stage']:<20}{timing['calls']:>7}"
            f"{timing['mean_ms']:>10.1f}{timing['per_tick_ms']:>10.1f}"
        )
    if record.mean_tick_ms:
        budget = 100.0  # 10 simulated ticks per second
        speed = budget / record.mean_tick_ms
        print(f"\n  {record.mean_tick_ms:.0f} ms per tick = {speed:.2f}x real time")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fully autonomous SENTRY AI mission.")
    parser.add_argument("--config", default="configs/autonomy.yaml", help="Autonomy config.")
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--headless", action="store_true", help="No window; print the result.")
    parser.add_argument("--seed", type=int, help="Hazard seed (default: the config's).")
    parser.add_argument(
        "--max-ticks", type=int, default=DEFAULT_MAX_TICKS, help="Headless tick budget."
    )
    parser.add_argument("--perception-every", type=int, help="Override perception_every.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
