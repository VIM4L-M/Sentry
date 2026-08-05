"""Draws a ``CityMap``'s current state onto a Pygame surface.

The operator's view of the city: terrain, entities, the planned route, and
optionally the CCTV footprints. Strictly read-only — the renderer never
advances a mission or mutates an entity.

Two things here exist purely so a viewer can follow a mission without
narration. Entities are drawn as :mod:`sentry_ai.rendering.glyphs` rather
than coloured discs, so a victim is distinguishable from debris at a
glance. And the *previous* route is drawn greyed out beneath the active
one, so a replan is visible as a fork rather than as a number ticking up in
the stats panel.
"""

from __future__ import annotations

import pygame

from sentry_ai.common.color import Color
from sentry_ai.domain.entities import Obstacle, Position, Vehicle
from sentry_ai.domain.enums import TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import Route
from sentry_ai.rendering import glyphs
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.camera import CameraView

#: Route breadcrumbs, as a fraction of a tile.
_ROUTE_RADIUS_FRACTION = 0.13

#: How far the abandoned route's colour is pulled toward the panel colour.
_STALE_ROUTE_BLEND = 0.62

#: Terrain types drawn with a road marking down the middle.
_ROAD_TERRAIN = frozenset({TerrainType.ROAD})

#: Terrain types drawn as raised blocks with a highlight edge.
_BLOCK_TERRAIN = frozenset({TerrainType.BUILDING})

#: Point size for the camera-footprint labels drawn on the map.
_CAMERA_LABEL_SIZE_PX = 12


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
        self._flicker = 0.0
        self._font: pygame.font.Font | None = None

    def advance_animation(self, delta_seconds: float) -> None:
        """Advance time-based effects (currently the fire flicker).

        Separate from :meth:`draw` so animation is tied to wall-clock frame
        time rather than to how often something happens to be redrawn, and
        so a headless caller can render a frame without any animation at
        all.
        """
        self._flicker = (self._flicker + delta_seconds) % 1.0

    def draw(
        self,
        surface: pygame.Surface,
        city_map: CityMap,
        route: Route | None = None,
        previous_route: Route | None = None,
        camera_views: tuple[CameraView, ...] = (),
    ) -> None:
        """Draw the city onto ``surface``.

        Args:
            surface: Target surface.
            city_map: World to draw.
            route: The active plan, drawn as breadcrumbs. ``None`` keeps the
                Phase 1 static preview exactly as it was.
            previous_route: The plan this one replaced, drawn greyed out
                beneath it so a replan is legible.
            camera_views: CCTV footprints to outline and label. Empty by
                default — the preview script wants no camera furniture.
        """
        surface.fill(self._theme.background.as_tuple())
        self._draw_terrain(surface, city_map)
        if previous_route is not None:
            self._draw_route(surface, previous_route, self._stale_route_color())
        if route is not None:
            self._draw_route(surface, route, self._theme.hud.accent)
        self._draw_entities(surface, city_map)
        for view in camera_views:
            self._draw_camera_footprint(surface, view)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------

    def _draw_terrain(self, surface: pygame.Surface, city_map: CityMap) -> None:
        """Fill every tile, adding markings that separate roads from blocks."""
        for y in range(city_map.height):
            for x in range(city_map.width):
                position = Position(x, y)
                terrain = city_map.tile_at(position)
                rect = self._tile_rect(position)
                pygame.draw.rect(surface, self._theme.terrain_colors[terrain].as_tuple(), rect)
                if terrain in _BLOCK_TERRAIN:
                    self._draw_block_edge(surface, rect, terrain)
                elif terrain in _ROAD_TERRAIN:
                    self._draw_road_marking(surface, rect, terrain)

    def _draw_block_edge(
        self, surface: pygame.Surface, rect: pygame.Rect, terrain: TerrainType
    ) -> None:
        """Outline a building so a block reads as masonry, not a colour field."""
        edge = _shade(self._theme.terrain_colors[terrain], 0.78)
        pygame.draw.rect(surface, edge.as_tuple(), rect, width=1)

    def _draw_road_marking(
        self, surface: pygame.Surface, rect: pygame.Rect, terrain: TerrainType
    ) -> None:
        """A short dash down the centre of a road tile."""
        marking = _shade(self._theme.terrain_colors[terrain], 1.35)
        length = max(2, rect.height // 3)
        width = max(1, rect.width // 14)
        pygame.draw.rect(
            surface,
            marking.as_tuple(),
            pygame.Rect(rect.centerx - width // 2, rect.centery - length // 2, width, length),
        )

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _draw_entities(self, surface: pygame.Surface, city_map: CityMap) -> None:
        """Draw every entity as its glyph, hazards beneath rescuables."""
        colors = self._theme.entity_colors
        # The cross is lightened rather than drawn in the safe zone's own
        # colour, which would make it invisible against the tile beneath it.
        glyphs.draw_hospital(
            surface,
            self._tile_rect(city_map.safe_zone.position),
            _shade(self._theme.terrain_colors[TerrainType.SAFE_ZONE], 1.9),
        )
        for obstacle in city_map.obstacles:
            self._draw_obstacle(surface, obstacle)
        for fire in city_map.fires:
            glyphs.draw_fire(
                surface,
                self._tile_rect(fire.position),
                colors[fire.entity_kind],
                _shade(colors[fire.entity_kind], 1.5),
                self._flicker,
            )
        for victim in city_map.victims:
            # Only the still-trapped: someone aboard the vehicle or already
            # delivered is not at those coordinates any more, and leaving
            # their glyph behind makes the map disagree with the grid, the
            # camera frames, and the rescued counter all at once.
            if victim.status is VictimStatus.TRAPPED:
                glyphs.draw_victim(
                    surface, self._tile_rect(victim.position), colors[victim.entity_kind]
                )
        self._draw_vehicle(surface, city_map.vehicle)

    def _draw_obstacle(self, surface: pygame.Surface, obstacle: Obstacle) -> None:
        """Draw an obstacle as what it actually is.

        Every obstacle blocks movement identically, but a tree and a
        collapsed building mean different things to someone watching, so
        they are not drawn with the same glyph.
        """
        rect = self._tile_rect(obstacle.position)
        if obstacle.kind is TerrainType.TREE:
            glyphs.draw_tree(
                surface, rect, _shade(self._theme.terrain_colors[TerrainType.TREE], 1.6)
            )
        else:
            glyphs.draw_debris(surface, rect, self._theme.entity_colors[obstacle.entity_kind])

    def _draw_vehicle(self, surface: pygame.Surface, vehicle: Vehicle) -> None:
        """Draw the rescue vehicle as a chevron pointing along its heading."""
        glyphs.draw_vehicle(
            surface,
            self._tile_rect(vehicle.position),
            self._theme.entity_colors[vehicle.entity_kind],
            vehicle.heading,
        )

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_route(self, surface: pygame.Surface, route: Route, color: Color) -> None:
        """Mark every tile a route intends to drive through."""
        radius = max(1, int(self._tile_size_px * _ROUTE_RADIUS_FRACTION))
        for waypoint in route:
            pygame.draw.circle(surface, color.as_tuple(), self._center(waypoint), radius)

    def _draw_camera_footprint(self, surface: pygame.Surface, view: CameraView) -> None:
        """Outline and label one camera's field of view.

        Drawn as an outline rather than a translucent fill: four overlapping
        translucent rectangles wash out the map underneath them, and the
        overlap — the part worth seeing — is exactly where the wash is
        worst. Outlines make the overlapping band read as two borders.
        """
        color = self._theme.hud.text
        rect = pygame.Rect(
            view.origin.x * self._tile_size_px,
            view.origin.y * self._tile_size_px,
            view.width_tiles * self._tile_size_px,
            view.height_tiles * self._tile_size_px,
        )
        pygame.draw.rect(surface, color.as_tuple(), rect, width=1)

        label = self._label_font().render(view.camera_id, True, color.as_tuple())
        backdrop = label.get_rect().inflate(6, 2)
        backdrop.topleft = (rect.left + 3, rect.top + 3)
        pygame.draw.rect(surface, self._theme.hud.panel.as_tuple(), backdrop)
        surface.blit(label, (backdrop.left + 3, backdrop.top + 1))

    def _label_font(self) -> pygame.font.Font:
        """The camera-label font, created on first use.

        Lazy because most renders draw no cameras, and because a
        ``MapRenderer`` must stay constructible before ``pygame.font`` has
        been initialised — the static preview never needs one.
        """
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont(
                "consolas,dejavusansmono,monospace", _CAMERA_LABEL_SIZE_PX
            )
        return self._font

    def _stale_route_color(self) -> Color:
        """The muted colour an abandoned route is drawn in."""
        return self._theme.hud.accent.blended_with(self._theme.hud.panel, _STALE_ROUTE_BLEND)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _tile_rect(self, position: Position) -> pygame.Rect:
        """Pixel bounds of ``position``'s tile."""
        return pygame.Rect(
            position.x * self._tile_size_px,
            position.y * self._tile_size_px,
            self._tile_size_px,
            self._tile_size_px,
        )

    def _center(self, position: Position) -> tuple[int, int]:
        """Pixel center of ``position``'s tile."""
        half = self._tile_size_px // 2
        return (
            position.x * self._tile_size_px + half,
            position.y * self._tile_size_px + half,
        )


def _shade(color: Color, factor: float) -> Color:
    """A darker (``factor`` < 1) or lighter version of ``color``."""
    return Color(
        r=min(255, round(color.r * factor)),
        g=min(255, round(color.g * factor)),
        b=min(255, round(color.b * factor)),
    )
