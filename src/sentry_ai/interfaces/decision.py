"""Ports for navigation policy (Unit V) and decision fusion (Unit I).

``INavigationPolicy`` is implemented in Phase 6 by a Deep Q-Network trained
with Stable-Baselines3. ``IDecisionFusion`` is implemented in Phase 7 by an
MLP (PyTorch, Adam, ReLU, Dropout) that combines every upstream AI signal
into the vehicle's final action. This file defines the contracts only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from sentry_ai.interfaces.perception import Detection
from sentry_ai.interfaces.sequence import BehaviourSignal


class VehicleAction(Enum):
    """The discrete actions the vehicle's policy can choose between."""

    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    HOLD_POSITION = "hold_position"
    PICKUP_VICTIM = "pickup_victim"
    DROP_OFF_VICTIM = "drop_off_victim"


@dataclass(frozen=True)
class PolicyOutput:
    """An :class:`INavigationPolicy`'s chosen action and its Q-value estimates."""

    action: VehicleAction
    q_values: dict[VehicleAction, float]


@dataclass(frozen=True)
class FinalAction:
    """The single action the vehicle will execute this tick, after fusion."""

    action: VehicleAction
    rationale_score: float


class INavigationPolicy(ABC):
    """Chooses a navigation action from an observation.

    Implemented in Phase 6 by a Deep Q-Network (Unit V — Reinforcement
    Learning) trained inside the Gymnasium ``SentryEnv``.
    """

    @abstractmethod
    def act(self, observation: NDArray[np.float32]) -> PolicyOutput:
        """Return the chosen action and its Q-value estimates for ``observation``."""
        raise NotImplementedError


class IDecisionFusion(ABC):
    """Fuses perception, sequence, and policy signals into one final action.

    Implemented in Phase 7 by an MLP (Unit I — Deep Learning Fundamentals:
    PyTorch, Adam, ReLU, Dropout).
    """

    @abstractmethod
    def fuse(
        self,
        detections: list[Detection],
        behaviour: BehaviourSignal,
        policy_output: PolicyOutput,
    ) -> FinalAction:
        """Combine every upstream signal into the vehicle's final action this tick."""
        raise NotImplementedError
