"""Integration test: config loading, CityMap construction, and rendering
wired together end-to-end, headless (SDL "dummy" driver, set in conftest).

This is Phase 1's walking-skeleton proof: everything that ``scripts/run_preview.py``
does, minus the actual event loop.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.theme import Theme


def test_full_pipeline_renders_without_error(project_root: Path) -> None:
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")

    map_data = loader.load_yaml(app_config.map_config_path)
    city_map = CityMap.from_config(map_data)

    theme = Theme.from_config(loader, app_config.render.palette_config_path)
    renderer = MapRenderer(theme=theme, tile_size_px=app_config.render.tile_size_px)

    pygame.init()
    try:
        surface = pygame.display.set_mode(
            (
                city_map.width * app_config.render.tile_size_px,
                city_map.height * app_config.render.tile_size_px,
            )
        )
        renderer.draw(surface, city_map)  # must not raise

        # Sanity check the surface actually received non-background pixels
        # (i.e. something was drawn, not just a blank fill).
        background = theme.background.as_tuple()
        pixels_sampled = [
            surface.get_at((x, y))[:3]
            for x in range(0, surface.get_width(), 7)
            for y in range(0, surface.get_height(), 7)
        ]
        assert any(pixel != background for pixel in pixels_sampled)
    finally:
        pygame.quit()
