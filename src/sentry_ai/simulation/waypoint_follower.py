"""A deterministic :class:`ILocalController` that drives a planned route.

This is Phase 2's stand-in for the Phase 6 DQN, and it exists for two
reasons beyond "something has to drive the vehicle before the model is
trained":

* It makes the whole mission loop runnable and testable with no ML stack
  installed, so the simulation can be validated independently of training.
* It is the **baseline** the DQN must beat. A learned local controller
  that cannot outperform greedy waypoint-chasing around dynamic obstacles
  has not earned its place in the pipeline.

The Q-values it reports are indicator values (1.0 for the chosen action,
0.0 otherwise), not learned estimates — it satisfies the contract without
pretending to be a value function.
"""

from __future__ import annotations

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)


class WaypointFollower(ILocalController):
    """Turns toward the next waypoint, then drives at it one tile per tick."""

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Choose the action that makes the most progress toward the waypoint.

        Precedence: stop when there is nothing to chase or the way ahead is
        blocked (the mission controller reacts by replanning), turn when
        the vehicle is not yet facing the waypoint, otherwise drive
        forward.
        """
        return _decision(self._choose(observation))

    def _choose(self, observation: LocalObservation) -> LocalAction:
        waypoint = observation.next_waypoint
        if waypoint is None or waypoint == observation.position:
            return LocalAction.STOP

        desired = _heading_toward(observation.position, waypoint)
        if desired is None:
            return LocalAction.STOP
        if desired is not observation.heading:
            return _turn_toward(observation.heading, desired)
        if observation.blocked_ahead:
            return LocalAction.STOP
        return LocalAction.MOVE_FORWARD


def _heading_toward(origin: Position, target: Position) -> Heading | None:
    """The heading that steps from ``origin`` toward ``target``.

    Resolves the larger axis first, so a diagonal offset (which a
    tile-by-tile route never produces, but a replan mid-turn can) still
    yields sensible progress. Returns ``None`` when the positions match.
    """
    delta_x = target.x - origin.x
    delta_y = target.y - origin.y
    if delta_x == 0 and delta_y == 0:
        return None
    if abs(delta_x) >= abs(delta_y):
        return Heading.EAST if delta_x > 0 else Heading.WEST
    return Heading.SOUTH if delta_y > 0 else Heading.NORTH


def _turn_toward(current: Heading, desired: Heading) -> LocalAction:
    """Whichever single turn brings ``current`` closer to ``desired``.

    A 180-degree correction is resolved as a right turn; the next tick
    completes it. Reversing instead would move the vehicle backwards off
    its route.
    """
    return LocalAction.TURN_LEFT if current.turn_left() is desired else LocalAction.TURN_RIGHT


def _decision(action: LocalAction) -> LocalDecision:
    """Wrap ``action`` in a decision with indicator Q-values."""
    return LocalDecision(
        action=action,
        q_values={candidate: float(candidate is action) for candidate in LOCAL_ACTION_ORDER},
    )
