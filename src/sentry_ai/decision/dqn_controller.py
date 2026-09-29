"""DQN implementing :class:`ILocalController` (Unit V).

**What the policy sees** is not :meth:`LocalObservation.as_array` verbatim.
That layout is in world coordinates — absolute position, compass heading,
world-frame offset to the waypoint — and a network given it would have to
learn a rotation before it could learn to drive. The policy instead sees
the same facts *egocentrically*:

=========================  =============================================
``waypoint_forward``       how far ahead the waypoint is, -1..1
``waypoint_right``         how far to the right, -1..1
``has_waypoint``           0 when the route is finished
``blocked_ahead``          the tile in front is impassable
``fire_proximity``         0-1 closeness to the nearest known fire
``battery``                0-1
``at_waypoint``            standing on the waypoint right now
=========================  =============================================

Absolute position is left out on purpose. A policy that knows where it is
can memorise the city — Phase 5 measured exactly that happening — and an
egocentric one has to learn *how to drive*, which is the same everywhere
and the same after a building collapses.

:func:`policy_features` is the single definition, used by the training
environment and by this adapter, so the two cannot drift apart.

Stable-Baselines3 is imported only when a model is loaded; importing this
module costs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)

logger = get_logger(__name__)

#: Length of :func:`policy_features`' output.
POLICY_FEATURES = 7

#: Length with the two traffic inputs appended (``traffic=True``): a road user
#: on the tile ahead, and one two tiles ahead. A traffic-trained DQN is
#: recognised by its observation size, so older 7-input models keep working.
TRAFFIC_POLICY_FEATURES = POLICY_FEATURES + 2

#: Waypoint offsets are clipped to this many tiles and scaled to -1..1. A
#: route advances tile by tile, so the waypoint is normally one tile away;
#: three leaves room to see it after a detour without letting a far
#: waypoint dominate the input.
_WAYPOINT_REACH = 3.0


def policy_features(observation: LocalObservation, traffic: bool = False) -> NDArray[np.float32]:
    """The egocentric vector the policy acts on; see the module docstring.

    ``traffic`` appends ``road_user_ahead`` and ``road_user_ahead_far``, the
    inputs of a DQN trained among cars and pedestrians.
    """
    base = _base_features(observation)
    if not traffic:
        return base
    extra = np.array(
        [float(observation.road_user_ahead), float(observation.road_user_ahead_far)],
        dtype=np.float32,
    )
    return np.concatenate([base, extra])


def _base_features(observation: LocalObservation) -> NDArray[np.float32]:
    heading_x, heading_y = observation.heading.delta
    waypoint = observation.next_waypoint
    if waypoint is None:
        forward = right = 0.0
    else:
        dx = waypoint.x - observation.position.x
        dy = waypoint.y - observation.position.y
        forward = float(dx * heading_x + dy * heading_y)
        # Turning right from north (0, -1) faces east (1, 0): right = (-hy, hx).
        right = float(dx * -heading_y + dy * heading_x)
    return np.array(
        [
            _scaled(forward),
            _scaled(right),
            float(waypoint is not None),
            float(observation.blocked_ahead),
            observation.fire_proximity,
            observation.battery_percent / 100.0,
            float(waypoint is not None and waypoint == observation.position),
        ],
        dtype=np.float32,
    )


def action_at(index: int) -> LocalAction:
    """The action a network output index stands for.

    Raises:
        ValueError: If ``index`` is outside :data:`LOCAL_ACTION_ORDER`.
    """
    if not 0 <= index < len(LOCAL_ACTION_ORDER):
        raise ValueError(f"action index {index} outside 0..{len(LOCAL_ACTION_ORDER) - 1}")
    return LOCAL_ACTION_ORDER[index]


class DqnLocalController(ILocalController):
    """Drives with a trained Stable-Baselines3 DQN, greedily."""

    def __init__(self, model: Any) -> None:
        """Wrap a loaded ``stable_baselines3.DQN``.

        Args:
            model: Anything exposing SB3's ``policy.obs_to_tensor`` and
                ``q_net`` — in practice a loaded DQN. Typed loosely so
                importing this module does not import SB3.
        """
        self._model = model
        shape: tuple[int, ...] = tuple(
            getattr(getattr(model, "observation_space", None), "shape", None) or ()
        )
        self._traffic = shape == (TRAFFIC_POLICY_FEATURES,)

    @property
    def sees_traffic(self) -> bool:
        """Whether this model was trained with the two road-user inputs."""
        return self._traffic

    @classmethod
    def from_file(cls, weights_path: Path, device: str = "cpu") -> DqnLocalController:
        """Load a ``.zip`` written by ``scripts/train_dqn.py``.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"DQN weights not found: {weights_path}")

        from stable_baselines3 import DQN  # noqa: PLC0415 - keeps SB3 off the import path

        model = DQN.load(str(weights_path), device=device)
        logger.info("Loaded DQN local controller from %s on %s", weights_path, device)
        return cls(model)

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Take the action with the highest Q-value; report all of them.

        Greedy — no exploration at inference. The Q-values are the network's
        real estimates, which is what Phase 7's fusion network will read.
        """
        q_values = self.q_values(observation)
        best = max(range(len(q_values)), key=q_values.__getitem__)
        return LocalDecision(
            action=action_at(best),
            q_values={action: q_values[i] for i, action in enumerate(LOCAL_ACTION_ORDER)},
        )

    def q_values(self, observation: LocalObservation) -> list[float]:
        """The network's value estimate for each action, in :data:`LOCAL_ACTION_ORDER`."""
        import torch  # noqa: PLC0415 - SB3 already requires it

        features = policy_features(observation, traffic=self._traffic)
        tensor, _ = self._model.policy.obs_to_tensor(features)
        with torch.no_grad():
            values = self._model.q_net(tensor)
        return [float(value) for value in values.squeeze(0).cpu().tolist()]


def _scaled(tiles: float) -> float:
    """Clip a tile offset to the reach and scale it to -1..1."""
    return max(-_WAYPOINT_REACH, min(_WAYPOINT_REACH, tiles)) / _WAYPOINT_REACH
