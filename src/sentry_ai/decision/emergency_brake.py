"""Automatic emergency braking for cars and pedestrians (Phase 9, key ``6``).

The trained driver never saw another road user: the DQN, the LSTM and the
fusion MLP were all trained on empty streets. Rather than claim they handle
traffic, this is a separate, rule-based safety layer — the same split a real
car makes between its learned planner and its AEB. It wraps whatever
controller is driving and watches the vehicle's surround sensors: if the
move it is about to make would drive into a car or a person, it stops
instead, and waits for them to clear.

Switched off, the vehicle drives on regardless, and each time it would hit
someone is counted — the number the demo shows next to the brake count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.traffic import AgentKind
from sentry_ai.interfaces.navigation import (
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)

logger = get_logger(__name__)

#: What the surround sensors report: occupied tile -> what occupies it.
TrafficSensor = Callable[[], Mapping[tuple[int, int], AgentKind]]


@dataclass(frozen=True)
class BrakeEvent:
    """One tick the brake intervened: who was in the way, and what it overruled."""

    kind: AgentKind
    overruled: LocalAction


class EmergencyBrake(ILocalController):
    """Stops the vehicle rather than let it move into a road user's tile."""

    def __init__(self, inner: ILocalController, sensor: TrafficSensor) -> None:
        """Wrap ``inner``, reading road users from ``sensor`` every tick."""
        self._inner = inner
        self._sensor = sensor
        self.enabled = True
        self.brakes: Counter[AgentKind] = Counter()
        self.hits: Counter[AgentKind] = Counter()
        self.last: BrakeEvent | None = None

    @property
    def inner(self) -> ILocalController:
        """The controller whose decisions are being checked."""
        return self._inner

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """The inner decision, or STOP if it would drive into someone."""
        decision = self._inner.decide(observation)
        self.last = None
        target = _target(observation, decision.action)
        if target is None:
            return decision
        kind = self._sensor().get(target)
        if kind is None:
            return decision
        if not self.enabled:
            self.hits[kind] += 1
            logger.warning("Drove into a %s at %s (brake off)", kind.value, target)
            return decision
        self.brakes[kind] += 1
        self.last = BrakeEvent(kind, decision.action)
        return LocalDecision(action=LocalAction.STOP, q_values=decision.q_values)


def _target(observation: LocalObservation, action: LocalAction) -> tuple[int, int] | None:
    """The tile ``action`` would move the vehicle into, or ``None`` if it stays put."""
    dx, dy = observation.heading.delta
    if action is LocalAction.MOVE_FORWARD:
        return observation.position.x + dx, observation.position.y + dy
    if action is LocalAction.REVERSE:
        return observation.position.x - dx, observation.position.y - dy
    return None
