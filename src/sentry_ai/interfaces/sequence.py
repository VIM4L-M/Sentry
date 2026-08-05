"""Port for the sequence model that predicts near-term vehicle behaviour.

Implemented in Phase 5 by an LSTM (Unit III — Sequence Models). This file
defines the contract and its supporting value types only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from sentry_ai.domain.entities import Position


class BehaviourClass(Enum):
    """Coarse near-term behaviour categories the sequence model predicts."""

    ADVANCE = "advance"
    RETREAT = "retreat"
    HOLD = "hold"
    DIVERT = "divert"


@dataclass(frozen=True)
class VehicleState:
    """One timestep of recorded vehicle state, as fed to the sequence model."""

    position: Position
    battery_percent: float
    heading_degrees: float


@dataclass(frozen=True)
class BehaviourSignal:
    """An :class:`IMotionPredictor`'s prediction for the upcoming timestep."""

    predicted_class: BehaviourClass
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"BehaviourSignal.confidence must be within 0.0-1.0, got {self.confidence}"
            )


class IMotionPredictor(ABC):
    """Predicts near-term vehicle movement/behaviour from a window of past states.

    Implemented in Phase 5 by an LSTM (Unit III — Sequence Models).
    """

    @abstractmethod
    def predict(self, state_history: Sequence[VehicleState]) -> BehaviourSignal:
        """Return the predicted behaviour given ``state_history`` (oldest first)."""
        raise NotImplementedError
