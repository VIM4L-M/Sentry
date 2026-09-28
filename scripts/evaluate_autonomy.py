#!/usr/bin/env python3
"""Milestone M8: the full autonomous stack, on held-out missions.

Drives ``eval_missions`` missions — fresh hazard seeds, random start tiles,
never used for training — twice each:

* **full stack**: camera-built map (denoiser + YOLO), A*, DQN, LSTM, onboard
  camera, MLP fusion. Zero human input.
* **reference**: the Phase 2 waypoint follower on the *ground-truth* map —
  perfect perception, perfect driving on this map. The ceiling.

M8 is met when the full stack completes at least ``completion_threshold`` of
the missions and rescues at least ``rescue_threshold`` of what the reference
rescues on the same missions. Every full-stack mission is written as a
record, and the per-stage timing across all of them is reported — the
profile against the hardware it ran on.

Usage:
    python scripts/evaluate_autonomy.py
    python scripts/evaluate_autonomy.py --missions 10 --perception-every 4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.autonomy.record import MissionRecord
from sentry_ai.autonomy.stack import AutonomyStack
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AutonomyConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.simulation.mission import MissionPhase
from sentry_ai.training.dqn import ControllerScore, drive_missions
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TICKS = 3000

logger = get_logger(__name__)


def main() -> int:
    """Drive every held-out mission with the full stack and the reference; judge M8."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = loader.load_autonomy_config(args.config)
    if args.perception_every is not None:
        config = replace(config, perception_every=args.perception_every)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.missions is not None:
        config = replace(config, eval_missions=args.missions)
    stack = AutonomyStack.load(loader, app_config, config)
    factory = MissionFactory.from_app_config(loader, app_config)
    if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
        raise SystemExit("app config must set 'simulation_config' and 'vehicle_config'")
    simulation = loader.load_simulation_config(app_config.simulation_config_path)
    vehicle = loader.load_vehicle_config(app_config.vehicle_config_path)
    seeds = list(range(config.eval_seed_base, config.eval_seed_base + config.eval_missions))

    print(
        f"{len(seeds)} held-out missions, random starts, perception every {config.perception_every}"
    )
    records: list[MissionRecord] = []
    for index, seed in enumerate(seeds, start=1):
        run = stack.mission(
            CityMap.from_config(factory.map_data_for(seed)), simulation, vehicle, hazard_seed=seed
        )
        record = run.run(max_ticks=MAX_TICKS)
        record.save(config.missions_dir)
        records.append(record)
        print(
            f"  {index:>2}/{len(seeds)} seed {seed}: {record.outcome:<9} "
            f"rescued {record.stats['victims_rescued']} lost {record.stats['victims_lost']} "
            f"collisions {record.stats['collisions']} ({record.mean_tick_ms:.0f} ms/tick)",
            flush=True,
        )
    reference = drive_missions(factory.build, lambda: None, seeds, MAX_TICKS)

    _report(records, reference, config)
    return 0


def _report(
    records: list[MissionRecord], reference: ControllerScore, config: AutonomyConfig
) -> None:
    completed = sum(record.outcome == MissionPhase.COMPLETED.value for record in records)
    rescued = sum(record.stats["victims_rescued"] for record in records)
    lost = sum(record.stats["victims_lost"] for record in records)
    collisions = sum(record.stats["collisions"] for record in records)
    damage = sum(100.0 - record.stats["vehicle_health"] for record in records) / len(records)
    ticks = sum(record.ticks for record in records) / len(records)
    print(
        f"\n  {'':<32}{'rescued':>8}{'lost':>6}{'collisions':>12}{'damage':>8}"
        f"{'completed':>11}{'ticks':>7}"
    )
    print(
        f"  {'full stack (camera-built map)':<32}{rescued:>8}{lost:>6}{collisions:>12}"
        f"{damage:>7.1f}%{completed / len(records):>10.0%}{ticks:>7.0f}"
    )
    print(
        f"  {'reference (ground-truth map)':<32}{reference.rescued:>8}{reference.lost:>6}"
        f"{reference.collisions:>12}{reference.mean_damage:>7.1f}%"
        f"{reference.completion_rate:>10.0%}{reference.mean_ticks:>7.0f}"
    )

    stages: dict[str, list[float]] = {}
    for record in records:
        for timing in record.timings:
            stages.setdefault(timing["stage"], []).append(timing["per_tick_ms"])
    print("\n  Mean cost per tick, across every mission:")
    for stage, values in stages.items():
        print(f"    {stage:<20}{sum(values) / len(values):>8.1f} ms")

    completion = completed / len(records)
    ratio = rescued / reference.rescued if reference.rescued else 1.0
    met = completion >= config.completion_threshold and ratio >= config.rescue_threshold
    print(
        f"\nM8 (complete >= {config.completion_threshold:.0%} of missions and rescue "
        f">= {config.rescue_threshold:.0%} of the reference): "
        f"{'MET' if met else 'NOT MET'} — {completion:.0%} completed, "
        f"{ratio:.0%} of reference rescues"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full autonomous stack (M8).")
    parser.add_argument("--config", default="configs/autonomy.yaml", help="Autonomy config.")
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--missions", type=int, help="Override eval_missions.")
    parser.add_argument("--perception-every", type=int, help="Override perception_every.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
