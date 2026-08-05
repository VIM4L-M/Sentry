"""Shared pytest fixtures.

Sets the SDL video/audio drivers to the headless "dummy" backend before
anything imports Pygame, so the whole suite (including rendering tests)
runs without a display — required for CI and this sandboxed environment.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    """The repository root, for tests that need to load real config files."""
    return PROJECT_ROOT
