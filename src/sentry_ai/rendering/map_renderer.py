"""Draws a ``CityMap``'s current state onto a Pygame surface.

Phase 1 draws a static snapshot only: terrain tiles as flat colored
squares, entities as colored circles on top. No animation, no movement,
no HUD — those arrive with the simulation engine in Phase 2.
"""

from __future__ import annotations

import pygame

from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.map import CityMap, MapEntity
from sentry_ai.interfaces.navigation import Route
from sentry_ai.rendering.theme import Theme

#: Entity markers are drawn smaller than a full tile so the terrain
#: underneath them stays visible.
_ENTITY_RADIUS_FRACTION = 0.35

#: Route breadcrumbs are smaller again, so they read as a trail behind the
#: entities rather than competing with them.
_ROUTE_RADIUS_FRACTION = 0.12


class MapRenderer:
    """Renders a :class:`CityMap` onto a Pygame surface using a :class:`Theme`."""

    def __init__(self, theme: Theme, tile_size_px: int) -> None:
        """Create a renderer.

        Args:
            theme: Color palette to draw with.
            tile_size_px: Edge length, in pixels, of one map tile.
        """
        self._theme = theme
        self._tile_size_px = tile_size_px

    def draw(
        self, surface: pygame.Surface, city_map: CityMap, route: Route | None = None
    ) -> None:
        """Draw ``city_map``'s terrain and entities onto ``surface``.

        Args:
            surface: Target surface.
            city_map: World to draw.
            route: Optional planned route, drawn as breadcrumbs beneath the
                entities. ``None`` (the default) keeps the Phase 1 static
                preview exactly as it was.
        """
        surface.fill(self._theme.background.as_tuple())
        self._draw_terrain(surface, city_map)
        if route is not None:
            self._draw_route(surface, route)
        for victim in city_map.victims:
            self._draw_entity(surface, victim)
        for fire in city_map.fires:
            self._draw_entity(surface, fire)
        for obstacle in city_map.obstacles:
            self._draw_entity(surface, obstacle)
        self._draw_entity(surface, city_map.vehicle)
        self._draw_heading(surface, city_map.vehicle)

    def _draw_terrain(self, surface: pygame.Surface, city_map: CityMap) -> None:
        for y in range(city_map.height):
            for x in range(city_map.width):
                terrain_type = city_map.tile_at(Position(x, y))
                color = self._theme.terrain_colors[terrain_type]
                rect = pygame.Rect(
                    x * self._tile_size_px,
                    y * self._tile_size_px,
                    self._tile_size_px,
                    self._tile_size_px,
                )
                pygame.draw.rect(surface, color.as_tuple(), rect)

    def _draw_entity(self, surface: pygame.Surface, entity: MapEntity) -> None:
        color = self._theme.entity_colors[entity.entity_kind]
        radius = max(2, int(self._tile_size_px * _ENTITY_RADIUS_FRACTION))
        pygame.draw.circle(surface, color.as_tuple(), self._center(entity.position), radius)

    def _draw_route(self, surface: pygame.Surface, route: Route) -> None:
        """Mark every tile the vehicle still intends to drive through."""
        radius = max(1, int(self._tile_size_px * _ROUTE_RADIUS_FRACTION))
        color = self._theme.hud.accent.as_tuple()
        for waypoint in route:
            pygame.draw.circle(surface, color, self._center(waypoint), radius)

    def _draw_heading(self, surface: pygame.Surface, vehicle: Vehicle) -> None:
        """Draw a stub from the vehicle's center showing which way it faces."""
        center = self._center(vehicle.position)
        dx, dy = vehicle.heading.delta
        reach = self._tile_size_px // 2
        end = (center[0] + dx * reach, center[1] + dy * reach)
        pygame.draw.line(surface, self._theme.hud.text.as_tuple(), center, end, width=2)

    def _center(self, position: Position) -> tuple[int, int]:
        """Pixel center of ``position``'s tile."""
        half = self._tile_size_px // 2
        return (
            position.x * self._tile_size_px + half,
            position.y * self._tile_size_px + half,
        )
