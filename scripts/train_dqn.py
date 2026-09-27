#!/usr/bin/env python3
"""Composition root for training the DQN local controller (Unit V).

Reads ``configs/training/dqn.yaml``, trains a Stable-Baselines3 DQN in
``SentryEnv`` — whole missions, random starts, the real physics — and every
``eval_every`` steps drives held-out missions with it next to the waypoint
follower, keeping the checkpoint that rescues the most.

Usage:
    python scripts/train_dqn.py
    python scripts/train_dqn.py --timesteps 50000 --name quick
    python scripts/train_dqn.py --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import DqnTrainingConfig
from sentry_ai.training.dqn import DqnTrainer, DqnTrainingOutcome
from sentry_ai.training.missions import MissionFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)


def main() -> int:
    """Parse arguments, train the DQN, and report it against the follower."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.app_config)
    setup_logging(app_config.logging_config_path)

    config = _apply_overrides(loader.load_dqn_config(args.config), args)
    trainer = DqnTrainer(config, MissionFactory.from_app_config(loader, app_config).build)
    print(
        f"Training DQN for {config.total_timesteps:,} steps on {trainer.resolve_device()}; "
        f"evaluating every {config.eval_every:,} on {config.eval_episodes} held-out missions",
        flush=True,
    )
    outcome = trainer.train(run_name=args.name, on_evaluation=_print_evaluation)
    _report(outcome, config)
    return 0


def _apply_overrides(config: DqnTrainingConfig, args: argparse.Namespace) -> DqnTrainingConfig:
    if args.timesteps is not None:
        config = replace(config, total_timesteps=args.timesteps)
    if args.eval_every is not None:
        config = replace(config, eval_every=args.eval_every)
    if args.device is not None:
        config = replace(config, device=args.device)
    return config


def _print_evaluation(row: dict[str, float]) -> None:
    print(
        f"step {int(row['step']):>8,}  rescued {int(row['rescued']):>3}  "
        f"({row['rescue_ratio']:.0%} of follower)  lost {int(row['lost']):>2}  "
        f"collisions {int(row['collisions']):>4}  completed {row['completion_rate']:.0%}  "
        f"mean ticks {row['mean_ticks']:.0f}",
        flush=True,
    )


def _report(outcome: DqnTrainingOutcome, config: DqnTrainingConfig) -> None:
    comparison = outcome.comparison
    print(f"\nBest checkpoint: step {outcome.best_step:,}")
    print(f"  {'':<18}{'rescued':>8}{'lost':>6}{'collisions':>12}{'completed':>11}{'ticks':>7}")
    for name, score in (("waypoint follower", comparison.baseline), ("DQN", comparison.candidate)):
        print(
            f"  {name:<18}{score.rescued:>8}{score.lost:>6}{score.collisions:>12}"
            f"{score.completion_rate:>10.0%}{score.mean_ticks:>7.0f}"
        )
    verdict = "MET" if comparison.meets(config.rescue_threshold) else "NOT MET"
    print(
        f"\nM6 (rescues >= {config.rescue_threshold:.0%} of the follower's): {verdict} "
        f"— {comparison.rescue_ratio:.0%}"
    )
    print(f"  weights     {outcome.weights_path}")
    print(f"  log         {outcome.run_dir / 'metrics.csv'}")
    print("  Score on the full held-out set with scripts/evaluate_dqn.py.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DQN local controller.")
    parser.add_argument(
        "--config",
        default="configs/training/dqn.yaml",
        help="DQN config (default: configs/training/dqn.yaml).",
    )
    parser.add_argument("--app-config", default="configs/app.yaml", help="Root app config.")
    parser.add_argument("--name", default="sentry", help="Run name under runs_dir.")
    parser.add_argument("--timesteps", type=int, help="Override total training steps.")
    parser.add_argument("--eval-every", type=int, help="Override the evaluation interval.")
    parser.add_argument("--device", help='Override the device ("cpu", "cuda", or an index).')
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
