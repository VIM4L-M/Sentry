"""What ADVANCE, RETREAT, HOLD and DIVERT mean, as a rule over two states.

The four classes on :class:`~sentry_ai.interfaces.sequence.BehaviourClass`
arrived in Phase 1 with names and no definition. This module is the
definition. It is kinematic and egocentric — every class is judged against
the way the vehicle was facing at the start, not against a compass or a
goal:

* **HOLD** — the vehicle ends where it started.
* **ADVANCE** — it went mostly the way it was facing.
* **RETREAT** — it went mostly the way it was *not* facing: backing out,
  or a U-turn after a replan.
* **DIVERT** — it went sideways: took a corner, swerved.

"Mostly" is a cone: the net displacement is within ``cone_degrees`` of
straight ahead (or straight behind). At the default 30 degrees, three tiles
forward and one sideways is still an advance; two and two is a diversion.

Egocentric on purpose, for the same reason the vehicle's action space is:
"about to turn" means the same thing heading north as heading west, so a
model that learns it once has learned it for every direction.

One rule serves both the labels the LSTM trains on and the persistence
baseline it has to beat. If they were written separately, the baseline and
the model could be scored against subtly different questions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.sequence import BehaviourClass, VehicleState

#: Default half-angle of the ADVANCE and RETREAT cones.
DEFAULT_CONE_DEGREES = 30.0

#: Compass bearing of each heading: clockwise from north, as
#: ``VehicleState.heading_degrees`` is defined.
_BEARINGS: dict[Heading, float] = {
    Heading.NORTH: 0.0,
    Heading.EAST: 90.0,
    Heading.SOUTH: 180.0,
    Heading.WEST: 270.0,
}

#: Stable ordering for encoding classes as network output indices. Appending
#: is safe; reordering invalidates every trained checkpoint.
BEHAVIOUR_ORDER: tuple[BehaviourClass, ...] = (
    BehaviourClass.ADVANCE,
    BehaviourClass.RETREAT,
    BehaviourClass.HOLD,
    BehaviourClass.DIVERT,
)


def heading_to_degrees(heading: Heading) -> float:
    """Compass bearing of ``heading``: 0 north, 90 east, clockwise."""
    return _BEARINGS[heading]


def facing_vector(heading_degrees: float) -> tuple[float, float]:
    """Unit ``(dx, dy)`` for a compass bearing, in tile space (``y`` grows down).

    North is ``(0, -1)`` and east ``(1, 0)``, matching
    :attr:`~sentry_ai.domain.enums.Heading.delta`.
    """
    radians = math.radians(heading_degrees)
    return math.sin(radians), -math.cos(radians)


def classify_motion(
    start: VehicleState, end: VehicleState, cone_degrees: float = DEFAULT_CONE_DEGREES
) -> BehaviourClass:
    """Which behaviour carried the vehicle from ``start`` to ``end``.

    Judged against the way the vehicle faced at ``start``.

    Raises:
        ValueError: If ``cone_degrees`` is not strictly between 0 and 90 —
            at 90 the ADVANCE and RETREAT cones would cover everything and
            DIVERT could never happen.
    """
    if not 0.0 < cone_degrees < 90.0:
        raise ValueError(f"cone_degrees must be within (0, 90), got {cone_degrees}")

    dx = end.position.x - start.position.x
    dy = end.position.y - start.position.y
    if dx == 0 and dy == 0:
        return BehaviourClass.HOLD

    face_x, face_y = facing_vector(start.heading_degrees)
    alignment = (dx * face_x + dy * face_y) / math.hypot(dx, dy)
    threshold = math.cos(math.radians(cone_degrees))
    if alignment >= threshold:
        return BehaviourClass.ADVANCE
    if alignment <= -threshold:
        return BehaviourClass.RETREAT
    return BehaviourClass.DIVERT


def persistence_baseline(
    history: Sequence[VehicleState], horizon: int, cone_degrees: float = DEFAULT_CONE_DEGREES
) -> BehaviourClass:
    """Predict that the vehicle keeps doing what it did over the last ``horizon`` steps.

    The naive baseline milestone M5 is measured against. Strong on this
    problem — a vehicle driving down a street keeps driving down it — so
    beating it means the model has learned *when behaviour changes*, which
    is the only part worth predicting.

    Raises:
        ValueError: If ``history`` is shorter than ``horizon + 1`` states.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if len(history) <= horizon:
        raise ValueError(
            f"persistence needs at least {horizon + 1} states, got {len(history)}"
        )
    return classify_motion(history[-1 - horizon], history[-1], cone_degrees)
