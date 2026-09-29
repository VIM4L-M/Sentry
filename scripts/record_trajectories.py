#!/usr/bin/env python3
"""Composition root for the Unit III trajectory dataset.

Runs a series of rescue missions — each a different disaster on the same
map, driven by a different hazard seed — and records the vehicle's state
after every tick:

    <output>/train.json   trajectories for training
    <output>/val.json     trajectories from held-out missions

Split by mission: consecutive windows of one trajectory overlap almost
entirely, so splitting windows at random would measure memorisation.

Each mission starts the vehicle on a different road tile, drawn from its
seed. With the shipped fixed start every mission drives nearly the same
route, and 99.4% of validation windows turned out to be copies of training
windows — the split measured memorisation. ``--fixed-start`` reproduces that
for comparison.

Missions run on the ground-truth grid. The sequence model learns how the
vehicle *drives*, and the driving is the same whichever map producer is in
use; ground truth is simply much faster to simulate than four cameras and a
detector per tick.

Usage:
    python scripts/record_trajectories.py
    python scripts/record_trajectories.py --train-missions 300 --val-missions 80
    python scripts/record_trajectories.py --fixed-start --output data/trajectories_fixed
    python scripts/record_trajectories.py --dqn models/dqn/sentry/best.zip \
        --output data/trajectories_dqn        # the LSTM fusion reads (Phase 7)
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.simulation.factory import Mission
from sentry_ai.training.missions import MissionFactory
from sentry_ai.training.trajectories import TrajectoryRecorder, TrajectorySet

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: First hazard seed. Far from the seeds build_dataset.py uses (0-15), so the
#: two datasets do not share disasters — not that it matters for correctness,
#: but it keeps the two experiments independent.
DEFAULT_SEED_BASE = 1000

logger = get_logger(__name__)


def main() -> int:
    """Record training and validation trajectories."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)

    city_template = loader.load_yaml(app_config.map_config_path)
    probe = CityMap.from_config(city_template)
    factory = MissionFactory.from_app_config(
        loader, app_config, randomise_start=not args.fixed_start
    )
    build = factory.build
    if args.dqn is not None:
        # Record how the trained DQN drives, not the waypoint follower: the
        # LSTM that feeds fusion must predict the driver fusion actually sits
        # on. The DQN reverses and pre-turns; the follower never does.
        from sentry_ai.decision.dqn_controller import DqnLocalController  # noqa: PLC0415

        dqn = DqnLocalController.from_file(loader.resolve(args.dqn))
        build = partial(_build_with, factory, dqn)
    recorder = TrajectoryRecorder(build, max_ticks=args.max_ticks)
    output = loader.resolve(args.output)

    first_val = args.seed_base + args.train_missions
    splits = {
        "train": range(args.seed_base, first_val),
        "val": range(first_val, first_val + args.val_missions),
    }
    print(f"Trajectories written to {output}")
    for split, seeds in splits.items():
        trajectories = recorder.record_set(seeds, probe.width, probe.height)
        trajectories.save(output / f"{split}.json")
        _report(split, trajectories)
    return 0


def _build_with(factory: MissionFactory, driver: ILocalController, seed: int) -> Mission:
    """One mission for ``seed``, driven by ``driver``."""
    return factory.build(seed, driver)


def _report(split: str, trajectories: TrajectorySet) -> None:
    outcomes: dict[str, int] = {}
    for trajectory in trajectories.trajectories:
        outcomes[trajectory.outcome] = outcomes.get(trajectory.outcome, 0) + 1
    summary = ", ".join(f"{count} {outcome}" for outcome, count in sorted(outcomes.items()))
    print(
        f"  {split:<6}{len(trajectories.trajectories)} missions, "
        f"{trajectories.total_states} states ({summary})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record vehicle trajectories for the LSTM.")
    parser.add_argument("--config", default="configs/app.yaml", help="Root application config.")
    parser.add_argument(
        "--output",
        default="data/trajectories",
        help="Directory to write train.json and val.json (default: data/trajectories).",
    )
    parser.add_argument(
        "--train-missions", type=int, default=150, help="Missions for training (default: 150)."
    )
    parser.add_argument(
        "--val-missions", type=int, default=50, help="Held-out missions (default: 50)."
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help=f"First hazard seed (default: {DEFAULT_SEED_BASE}).",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=5000, help="Tick budget per mission (default: 5000)."
    )
    parser.add_argument(
        "--dqn", help="Drive with this trained DQN instead of the waypoint follower."
    )
    parser.add_argument(
        "--fixed-start",
        action="store_true",
        help="Start every mission on the map's own start tile (the memorisation baseline).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
