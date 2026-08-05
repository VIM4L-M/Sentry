"""Port for decision fusion (Unit I) — the last stage before the vehicle acts.

Implemented in Phase 7 by an MLP (PyTorch, Adam, ReLU, Dropout) that
combines every upstream AI signal — the detector's findings, the LSTM's
behaviour prediction, and the DQN's Q-values — into the single action the
vehicle executes this tick. This file defines the contract only.

The navigation contracts that used to live here (``VehicleAction``,
``PolicyOutput``, ``INavigationPolicy``) moved to
:mod:`sentry_ai.interfaces.navigation` when routing was split into a global
A* tier and a local DQN tier — see
``docs/adr/0002-two-tier-navigation-and-command-center.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision
from sentry_ai.interfaces.perception import Detection
from sentry_ai.interfaces.sequence import BehaviourSignal


@dataclass(frozen=True)
class FinalAction:
    """The single action the vehicle will execute this tick, after fusion.

    Attributes:
        action: The egocentric move to execute.
        rationale_score: The fusion network's confidence in this action,
            0-1. Surfaced on the dashboard so an operator can see how sure
            the system was, not just what it did.
    """

    action: LocalAction
    rationale_score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rationale_score <= 1.0:
            raise DomainValidationError(
                f"FinalAction.rationale_score must be within 0.0-1.0, got {self.rationale_score}"
            )


class IDecisionFusion(ABC):
    """Fuses perception, sequence, and local-policy signals into one action.

    Implemented in Phase 7 by an MLP (Unit I — Deep Learning Fundamentals:
    PyTorch, Adam, ReLU, Dropout).
    """

    @abstractmethod
    def fuse(
        self,
        detections: list[Detection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """Combine every upstream signal into the vehicle's final action this tick."""
        raise NotImplementedError
