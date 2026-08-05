"""Integration test for sentry_ai.rendering.app.PreviewApp.

Runs headless (SDL "dummy" driver, set in conftest) and injects a QUIT
event before calling ``run()`` so the loop executes exactly one frame and
returns, instead of blocking forever waiting for a real window close.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.app import PreviewApp
from sentry_ai.rendering.theme import Theme


def test_preview_app_runs_one_frame_and_exits_on_quit(project_root: Path) -> None:
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")
    map_data = loader.load_yaml(app_config.map_config_path)
    city_map = CityMap.from_config(map_data)
    theme = Theme.from_config(loader, app_config.render.palette_config_path)

    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    app = PreviewApp(city_map=city_map, render_config=app_config.render, theme=theme)
    app.run()  # must return promptly instead of hanging
