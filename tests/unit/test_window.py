"""Unit tests for the screen-fitting window (sentry_ai.rendering.window)."""

from __future__ import annotations

import os

import pygame
import pytest

from sentry_ai.rendering import window


@pytest.fixture(autouse=True)
def _headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    yield
    pygame.display.quit()


def test_a_design_that_fits_draws_straight_on_the_window(monkeypatch) -> None:
    monkeypatch.setattr(window, "_desktop_size", lambda: (2880, 1800))
    fitted = window.open_window((1600, 900), "t")
    assert not fitted.scaled
    assert fitted.canvas is fitted.display


def test_a_design_too_big_is_scaled_down_keeping_its_shape(monkeypatch) -> None:
    monkeypatch.setattr(window, "_desktop_size", lambda: (1366, 768))
    fitted = window.open_window((1600, 900), "t")
    assert fitted.scaled
    assert fitted.canvas.get_size() == (1600, 900)
    w, h = fitted.display.get_size()
    assert w <= 1366 and h <= 768 * 0.9
    assert abs(w / h - 1600 / 900) < 0.01
    fitted.present()
