"""Unit tests for sentry_ai.rendering.keyboard.KeyboardController."""

from __future__ import annotations

import pygame
import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction, LocalObservation
from sentry_ai.rendering.keyboard import KEY_BINDINGS, KeyboardController

_OBSERVATION = LocalObservation(
    position=Position(1, 1),
    heading=Heading.NORTH,
    battery_percent=100.0,
    next_waypoint=None,
    blocked_ahead=False,
    fire_proximity=0.0,
)


class _HeldKeys:
    """Stands in for ``pygame.key.get_pressed()``'s sequence-like result."""

    def __init__(self, *held: int) -> None:
        self._held = set(held)

    def __getitem__(self, key: int) -> bool:
        return key in self._held


@pytest.fixture
def controller() -> KeyboardController:
    return KeyboardController()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (pygame.K_UP, LocalAction.MOVE_FORWARD),
        (pygame.K_w, LocalAction.MOVE_FORWARD),
        (pygame.K_DOWN, LocalAction.REVERSE),
        (pygame.K_s, LocalAction.REVERSE),
        (pygame.K_LEFT, LocalAction.TURN_LEFT),
        (pygame.K_a, LocalAction.TURN_LEFT),
        (pygame.K_RIGHT, LocalAction.TURN_RIGHT),
        (pygame.K_d, LocalAction.TURN_RIGHT),
    ],
)
def test_each_binding_produces_its_action(
    controller: KeyboardController,
    monkeypatch: pytest.MonkeyPatch,
    key: int,
    expected: LocalAction,
) -> None:
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _HeldKeys(key))
    assert controller.decide(_OBSERVATION).action is expected


def test_no_keys_held_stops_the_vehicle(
    controller: KeyboardController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _HeldKeys())
    assert controller.decide(_OBSERVATION).action is LocalAction.STOP


def test_unbound_key_is_ignored(
    controller: KeyboardController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _HeldKeys(pygame.K_q))
    assert controller.decide(_OBSERVATION).action is LocalAction.STOP


def test_q_values_cover_every_action(
    controller: KeyboardController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _HeldKeys(pygame.K_UP))
    decision = controller.decide(_OBSERVATION)
    assert set(decision.q_values) == set(LOCAL_ACTION_ORDER)
    assert decision.q_values[LocalAction.MOVE_FORWARD] == 1.0


def test_every_binding_maps_to_a_real_action() -> None:
    assert set(KEY_BINDINGS.values()) <= set(LOCAL_ACTION_ORDER)
