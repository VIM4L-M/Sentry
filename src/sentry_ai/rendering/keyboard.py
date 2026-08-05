"""Keyboard input as a drop-in local controller.

Lets a human drive the rescue vehicle through exactly the same action
space and physics the DQN will use — which makes manual mode a debugging
tool *and* a fair human baseline, rather than a separate code path with
its own rules.

Lives in ``rendering`` rather than ``simulation`` because it is an input
adapter over Pygame: the application layer must not import a UI toolkit.
"""

from __future__ import annotations

import pygame

from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)

#: Which key produces which action. Arrow keys and WASD both work, so the
#: layout suits either hand.
KEY_BINDINGS: dict[int, LocalAction] = {
    pygame.K_UP: LocalAction.MOVE_FORWARD,
    pygame.K_w: LocalAction.MOVE_FORWARD,
    pygame.K_DOWN: LocalAction.REVERSE,
    pygame.K_s: LocalAction.REVERSE,
    pygame.K_LEFT: LocalAction.TURN_LEFT,
    pygame.K_a: LocalAction.TURN_LEFT,
    pygame.K_RIGHT: LocalAction.TURN_RIGHT,
    pygame.K_d: LocalAction.TURN_RIGHT,
}


class KeyboardController(ILocalController):
    """Turns the currently-held keys into a :class:`LocalAction` each tick.

    Reads held keys rather than key-press events so holding a direction
    keeps the vehicle moving, and releasing everything stops it — the
    behaviour a driver expects.
    """

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Return the action bound to the first held key, or ``STOP``."""
        del observation  # a human does not need the observation vector
        pressed = pygame.key.get_pressed()
        action = next(
            (bound for key, bound in KEY_BINDINGS.items() if pressed[key]),
            LocalAction.STOP,
        )
        return LocalDecision(
            action=action,
            q_values={candidate: float(candidate is action) for candidate in LOCAL_ACTION_ORDER},
        )
