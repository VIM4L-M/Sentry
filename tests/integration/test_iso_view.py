"""Integration tests for the isometric 3D view (sentry_ai.rendering.iso_renderer)."""

from __future__ import annotations

from pathlib import Path

import pygame
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.iso_renderer import IsoRenderer
from sentry_ai.rendering.theme import Theme
from sentry_ai.training.missions import MissionFactory


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


def test_draws_a_running_mission_inside_its_area(loader: ConfigLoader) -> None:
    pygame.init()
    mission = MissionFactory.from_app_config(loader, loader.load_app_config()).build(7)
    mission.engine.run(40)
    renderer = IsoRenderer(Theme.from_config(loader, "configs/render.yaml"))
    surface = pygame.Surface((1000, 600))
    surface.fill((255, 0, 255))
    area = pygame.Rect(100, 20, 840, 560)
    renderer.advance_animation(0.3)
    renderer.draw(surface, mission.city_map, area, route=mission.controller.route)

    outside = [surface.get_at((x, y))[:3] for x, y in ((5, 5), (990, 590), (50, 300))]
    assert all(color == (255, 0, 255) for color in outside)
    assert surface.get_at(area.center)[:3] != (255, 0, 255)


def test_draws_every_shipped_map(loader: ConfigLoader, project_root: Path) -> None:
    pygame.init()
    renderer = IsoRenderer(Theme.from_config(loader, "configs/render.yaml"))
    for path in sorted((project_root / "configs" / "maps").glob("*.yaml")):
        city = CityMap.from_config(loader.load_yaml(path))
        renderer.draw(pygame.Surface((840, 560)), city, pygame.Rect(0, 0, 840, 560))
