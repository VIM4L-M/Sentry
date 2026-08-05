"""Phase 1 composition root for the visual preview: a window that renders
a static disaster city until the user closes it.

This intentionally contains no game-loop mechanics beyond "draw and pump
events" — no input handling, no tick-based state updates, no HUD. Those
belong to the real simulation loop, introduced in Phase 2.
"""

from __future__ import annotations

import pygame

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import RenderConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.theme import Theme

logger = get_logger(__name__)


class PreviewApp:
    """Opens a window and renders a static ``CityMap`` until closed."""

    def __init__(self, city_map: CityMap, render_config: RenderConfig, theme: Theme) -> None:
        """Create the preview app.

        Args:
            city_map: The (already validated) disaster city to display.
            render_config: Window size/title/frame-rate settings.
            theme: Color palette to render with.
        """
        self._city_map = city_map
        self._render_config = render_config
        self._renderer = MapRenderer(theme=theme, tile_size_px=render_config.tile_size_px)

    def run(self) -> None:
        """Open the window and block until the user closes it or presses Escape."""
        pygame.init()
        try:
            surface = pygame.display.set_mode(
                (
                    self._city_map.width * self._render_config.tile_size_px,
                    self._city_map.height * self._render_config.tile_size_px,
                )
            )
            pygame.display.set_caption(self._render_config.window_title)
            clock = pygame.time.Clock()
            logger.info(
                "Preview window opened: %dx%d tiles at %dpx/tile",
                self._city_map.width,
                self._city_map.height,
                self._render_config.tile_size_px,
            )

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                self._renderer.draw(surface, self._city_map)
                pygame.display.flip()
                clock.tick(self._render_config.target_fps)
        finally:
            pygame.quit()
            logger.info("Preview window closed")
