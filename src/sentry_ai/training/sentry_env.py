"""``SentryEnv``: a whole rescue mission as a Gymnasium environment (Unit V).

One episode is one mission; one step is one simulation tick in which the
agent, instead of the waypoint follower, chooses the vehicle's action.
Everything else is the real simulation, unmodified: the A* command center
still plans the route, hazards still spread and collapse, and
:class:`~sentry_ai.simulation.vehicle_controller.VehicleController` still
applies the same physics the live window uses. A policy cannot learn
against rules the deployed system does not enforce.

**How the agent takes the wheel.** The engine asks its controller for a
decision inside ``tick()``. The environment installs a controller that
simply returns whatever action the agent chose for this step, so the
engine's own ordering — hazards, then the decision, then physics, then the
mission — is untouched.

**What it is paid for** is local (ADR 0002): progress toward the waypoint
the command center set, pickups and deliveries, and penalties for
collisions, damage, fire, time and battery. See
:class:`~sentry_ai.config.schema.RewardConfig`.

Episodes draw a hazard seed from the training range and, through
:class:`~sentry_ai.training.missions.MissionFactory`, start the vehicle on a
random road tile, so the policy cannot learn one route by heart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sentry_ai.config.schema import RewardConfig
from sentry_ai.decision.dqn_controller import POLICY_FEATURES, action_at, policy_features
from sentry_ai.domain.entities import Position
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.simulation.engine import TickResult
from sentry_ai.simulation.factory import Mission
from sentry_ai.simulation.mission import MissionPhase

#: Builds a mission for a seed, driven by the given controller.
EpisodeFactory = Callable[[int, ILocalController | None], Mission]


class AgentDriver(ILocalController):
    """The controller the environment installs: it does what the agent said."""

    def __init__(self) -> None:
        self.action = LocalAction.STOP

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Return the agent's action for this tick, with indicator Q-values."""
        return LocalDecision(
            action=self.action,
            q_values={action: float(action is self.action) for action in LOCAL_ACTION_ORDER},
        )


class SentryEnv(gym.Env[NDArray[np.float32], np.int64]):
    """A rescue mission the agent drives, one tick per step."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        factory: EpisodeFactory,
        seeds: Sequence[int],
        reward: RewardConfig,
        max_episode_steps: int,
    ) -> None:
        """Create the environment.

        Args:
            factory: Builds a fresh mission for a seed, driven by a given
                controller — :meth:`MissionFactory.build`.
            seeds: Hazard seeds episodes are drawn from.
            reward: The reward weights.
            max_episode_steps: Steps before an episode is truncated.

        Raises:
            ValueError: If ``seeds`` is empty or the step limit is not positive.
        """
        super().__init__()
        if not seeds:
            raise ValueError("SentryEnv needs at least one mission seed")
        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")
        self._factory = factory
        self._seeds = list(seeds)
        self._reward = reward
        self._max_steps = max_episode_steps
        self._driver = AgentDriver()
        self._mission: Mission | None = None
        self._steps = 0
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(POLICY_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(LOCAL_ACTION_ORDER))

    @property
    def mission(self) -> Mission:
        """The mission of the current episode.

        Raises:
            RuntimeError: Before the first :meth:`reset`.
        """
        if self._mission is None:
            raise RuntimeError("call reset() before using the environment")
        return self._mission

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a new mission.

        Args:
            seed: Seeds the environment's own generator, which picks the
                mission seed.
            options: ``{"mission_seed": n}`` runs that exact mission instead
                of drawing one — how evaluation pins a mission down.
        """
        super().reset(seed=seed)
        if options and "mission_seed" in options:
            mission_seed = int(options["mission_seed"])
        else:
            mission_seed = int(self._seeds[int(self.np_random.integers(len(self._seeds)))])
        self._driver = AgentDriver()
        self._mission = self._factory(mission_seed, self._driver)
        self._steps = 0
        return policy_features(self._mission.engine.observe()), {"mission_seed": mission_seed}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Run one tick with the agent's action and pay it for the result."""
        mission = self.mission
        engine = mission.engine
        vehicle = mission.city_map.vehicle
        before = engine.observe()
        onboard_before = len(vehicle.onboard_victims)
        rescued_before = engine.stats.victims_rescued

        self._driver.action = action_at(int(action))
        result = engine.tick()
        self._steps += 1

        after = engine.observe()
        delivered = engine.stats.victims_rescued - rescued_before
        picked_up = len(vehicle.onboard_victims) - onboard_before + delivered
        reward = self._pay(before, after, result, picked_up, delivered)

        terminated = result is None or engine.is_done
        truncated = not terminated and self._steps >= self._max_steps
        stats = engine.stats
        info = {
            "phase": engine.mission.phase.value,
            "rescued": stats.victims_rescued,
            "lost": stats.victims_lost,
            "collisions": stats.collisions,
        }
        return policy_features(after), reward, terminated, truncated, info

    def _pay(
        self,
        before: LocalObservation,
        after: LocalObservation,
        result: TickResult | None,
        picked_up: int,
        delivered: int,
    ) -> float:
        """The reward for one tick; each term is one line of RewardConfig."""
        weights = self._reward
        reward = weights.step
        if before.next_waypoint is not None:
            closer = _distance(before.position, before.next_waypoint) - _distance(
                after.position, before.next_waypoint
            )
            reward += weights.progress * closer
        reward += weights.pickup * picked_up + weights.delivery * delivered
        reward += weights.fire_proximity * after.fire_proximity
        if result is not None:
            reward += weights.collision * float(result.outcome.collided)
            reward += weights.damage * result.outcome.damage_taken
            reward += weights.battery * result.outcome.battery_spent
            reward += weights.reverse * float(result.outcome.action is LocalAction.REVERSE)
            if result.phase is MissionPhase.COMPLETED:
                reward += weights.completion
            elif result.phase is MissionPhase.FAILED:
                reward += weights.failure
        return float(reward)


def _distance(a: Position, b: Position) -> int:
    """Tiles apart along the grid — the distance a vehicle actually covers."""
    return abs(a.x - b.x) + abs(a.y - b.y)
