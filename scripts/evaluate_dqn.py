#!/usr/bin/env python3
"""Scores the DQN against the waypoint follower on held-out missions (milestone M6).

Both controllers drive exactly the same missions — same hazard seed, same
random start tile — through the same ``ILocalController`` slot the live
pipeline uses. Reported per controller: victims rescued and lost,
collisions, completion rate and why missions failed, and mean mission
length.

M6 is met when the DQN rescues at least ``rescue_threshold`` (from
``configs/training/dqn.yaml``) of what the follower rescues.

Usage:
    python scripts/evaluate_dqn.py
    python scripts/evaluate_dqn.py --weights models/dqn/sentry/best.zip --device cuda
    python scripts/evaluate_dqn.py --simulation configs/simulation_stress.yaml   # harsher disaster
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.decision.dqn_controller import DqnLocalController
from sentry_ai.training.dqn import ControllerScore, PolicyComparison, drive_missions
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Drive every held-out mission with both controllers and compare."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = loader.load_dqn_config(args.config)
    weights = (
        loader.resolve(args.weights) if args.weights else config.runs_dir / args.run / "best.zip"
    )
    policy = DqnLocalController.from_file(weights, device=args.device)
    if args.simulation is not None:
        app_config = replace(app_config, simulation_config_path=loader.resolve(args.simulation))
    factory = MissionFactory.from_app_config(loader, app_config)
    seeds = list(config.evaluation_seeds)

    comparison = PolicyComparison(
        baseline=drive_missions(factory.build, lambda: None, seeds, config.max_episode_steps),
        candidate=drive_missions(factory.build, lambda: policy, seeds, config.max_episode_steps),
    )
    print(f"DQN {weights}\n{len(seeds)} held-out missions, random starts")
    print(f"hazards: {app_config.simulation_config_path}\n")
    print(f"  {'':<18}{'rescued':>8}{'lost':>6}{'collisions':>12}{'completed':>11}{'ticks':>7}")
    for name, score in (("waypoint follower", comparison.baseline), ("DQN", comparison.candidate)):
        _print_row(name, score)
    for name, score in (("waypoint follower", comparison.baseline), ("DQN", comparison.candidate)):
        reasons = score.failure_reasons()
        if reasons:
            listed = ", ".join(f"{count} {reason}" for reason, count in reasons.items())
            print(f"  {name} failures: {listed}")

    worse = [
        (b.seed, b.rescued, c.rescued)
        for b, c in zip(comparison.baseline.results, comparison.candidate.results, strict=True)
        if c.rescued < b.rescued
    ]
    if worse:
        print("\n  Missions where the DQN rescued fewer:")
        for seed, follower, dqn in worse:
            print(f"    seed {seed}: follower {follower}, DQN {dqn}")

    verdict = "MET" if comparison.meets(config.rescue_threshold) else "NOT MET"
    print(
        f"\nM6 (rescues >= {config.rescue_threshold:.0%} of the follower's): {verdict} "
        f"— {comparison.rescue_ratio:.1%}"
    )
    return 0


def _print_row(name: str, score: ControllerScore) -> None:
    print(
        f"  {name:<18}{score.rescued:>8}{score.lost:>6}{score.collisions:>12}"
        f"{score.completion_rate:>10.0%}{score.mean_ticks:>7.0f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DQN local controller.")
    parser.add_argument(
        "--config",
        default="configs/training/dqn.yaml",
        help="DQN config (default: configs/training/dqn.yaml).",
    )
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--weights", help="Model to score (default: the run's best.zip).")
    parser.add_argument("--run", default="sentry", help="Training run name to score.")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    parser.add_argument(
        "--simulation",
        help="Evaluate under a different simulation config, e.g. configs/simulation_stress.yaml.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
