"""An isometric 3D view of the same city (Phase 9 demo view, key ``V``).

Draws exactly what :class:`~sentry_ai.rendering.map_renderer.MapRenderer`
draws, from an elevated three-quarter angle: buildings rise as shaded
blocks, collapsed ones as broken stumps, fires burn above the roofline, and
the vehicle is a small truck on the street. It reads the same
:class:`CityMap` and :class:`Route` and changes nothing, so switching views
mid-mission is free.

Pure pygame on purpose — no 3D engine dependency, so it runs anywhere the
2D view does. Tiles are drawn back to front (painter's algorithm), which is
exact for a grid of axis-aligned blocks seen from one fixed corner.
"""

from __future__ import annotations

import math

import pygame

from sentry_ai.common.color import Color
from sentry_ai.domain.entities import Position, Vehicle, Victim
from sentry_ai.domain.enums import EntityKind, TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import Route
from sentry_ai.rendering import glyphs
from sentry_ai.rendering.theme import Theme

#: Building heights, in tile half-widths, chosen per tile from its coordinates
#: so the skyline is varied but identical every frame.
_STOREYS = (0.6, 0.85, 1.15, 1.5)

#: Height of a collapsed building's stump, and of a tree's crown.
_STUMP = 0.3
_CROWN = 0.9

#: Face shading: the lit side, the side in shadow.
_LIT, _SHADOW = 0.82, 0.62

#: Pure black and white, for shading faces and lighting highlights.
_BLACK, _WHITE = Color(0, 0, 0), Color(255, 255, 255)


class IsoRenderer:
    """Renders a :class:`CityMap` isometrically into a given rectangle."""

    def __init__(self, theme: Theme) -> None:
        """Create a renderer that draws with ``theme``'s colours."""
        self._theme = theme
        self._flicker = 0.0

    def advance_animation(self, delta_seconds: float) -> None:
        """Advance the fire flicker; tied to frame time, like the 2D view."""
        self._flicker = (self._flicker + delta_seconds) % 1.0

    def draw(
        self,
        surface: pygame.Surface,
        city_map: CityMap,
        area: pygame.Rect,
        route: Route | None = None,
    ) -> None:
        """Draw the whole city, fitted and centred inside ``area``."""
        pygame.draw.rect(surface, self._theme.background.as_tuple(), area)
        half = self._half_width(city_map, area)
        origin = (
            area.centerx - (city_map.width - city_map.height) * half // 2,
            area.top
            + int(max(_STOREYS) * half)
            + (area.height - self._drawn_height(city_map, half)) // 2,
        )
        route_tiles = set(route.waypoints) if route is not None else set()
        fires = {fire.position for fire in city_map.fires}
        victims = {v.position: v for v in city_map.victims if v.status is not VictimStatus.ONBOARD}
        victims = {p: v for p, v in victims.items() if v.status is not VictimStatus.RESCUED}
        for depth in range(city_map.width + city_map.height - 1):
            for x in range(max(0, depth - city_map.height + 1), min(city_map.width, depth + 1)):
                position = Position(x, depth - x)
                self._draw_tile(surface, city_map, position, origin, half, position in route_tiles)
                if position in victims:
                    self._draw_victim(surface, victims[position], origin, half)
                if position in fires:
                    self._draw_fire(surface, city_map, position, origin, half)
                if position == city_map.vehicle.position:
                    self._draw_vehicle(surface, city_map.vehicle, origin, half)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _half_width(city_map: CityMap, area: pygame.Rect) -> int:
        """Half a tile's diamond width, the largest that fits ``area``."""
        span = city_map.width + city_map.height
        by_width = area.width / span
        by_height = area.height / (span / 2 + max(_STOREYS) + 1)
        return max(4, int(min(by_width, by_height)))

    @staticmethod
    def _drawn_height(city_map: CityMap, half: int) -> int:
        return int((city_map.width + city_map.height) * half / 2 + max(_STOREYS) * half)

    @staticmethod
    def _ground(position: Position, origin: tuple[int, int], half: int) -> tuple[float, float]:
        """Screen point of a tile's centre at ground level."""
        return (
            origin[0] + (position.x - position.y) * half,
            origin[1] + (position.x + position.y + 1) * half / 2,
        )

    def _diamond(
        self, position: Position, origin: tuple[int, int], half: int, lift: float = 0.0
    ) -> list[tuple[float, float]]:
        cx, cy = self._ground(position, origin, half)
        cy -= lift * half
        return [(cx, cy - half / 2), (cx + half, cy), (cx, cy + half / 2), (cx - half, cy)]

    # ------------------------------------------------------------------
    # Tiles
    # ------------------------------------------------------------------

    def _draw_tile(
        self,
        surface: pygame.Surface,
        city_map: CityMap,
        position: Position,
        origin: tuple[int, int],
        half: int,
        on_route: bool,
    ) -> None:
        terrain = city_map.tile_at(position)
        color = self._theme.terrain_colors[terrain]
        if terrain is TerrainType.BUILDING:
            self._prism(
                surface,
                position,
                origin,
                half,
                _STOREYS[(position.x * 7 + position.y * 3) % 4],
                color,
            )
        elif terrain is TerrainType.COLLAPSED_BUILDING:
            self._prism(surface, position, origin, half, _STUMP, color)
        elif terrain is TerrainType.TREE:
            self._flat(
                surface, position, origin, half, self._theme.terrain_colors[TerrainType.OPEN_GROUND]
            )
            self._tree(surface, position, origin, half, color)
        else:
            self._flat(surface, position, origin, half, color)
            if position == city_map.safe_zone.position:
                cx, cy = self._ground(position, origin, half)
                rect = pygame.Rect(0, 0, half, half)
                rect.center = (int(cx), int(cy - half * 0.4))
                glyphs.draw_hospital(surface, rect, color.blended_with(_WHITE, 0.7))
        if on_route:
            cx, cy = self._ground(position, origin, half)
            radius = max(2, half // 6)
            pygame.draw.circle(
                surface, self._theme.hud.accent.as_tuple(), (int(cx), int(cy)), radius
            )

    def _flat(
        self,
        surface: pygame.Surface,
        position: Position,
        origin: tuple[int, int],
        half: int,
        color: Color,
    ) -> None:
        points = self._diamond(position, origin, half)
        pygame.draw.polygon(surface, color.as_tuple(), points)
        pygame.draw.polygon(surface, color.blended_with(_BLACK, 0.12).as_tuple(), points, 1)

    def _prism(
        self,
        surface: pygame.Surface,
        position: Position,
        origin: tuple[int, int],
        half: int,
        height: float,
        color: Color,
    ) -> None:
        """A block ``height`` half-widths tall: two visible walls and a roof."""
        base = self._diamond(position, origin, half)
        top = self._diamond(position, origin, half, lift=height)
        dark = _BLACK
        left_wall = [base[3], base[2], top[2], top[3]]
        right_wall = [base[2], base[1], top[1], top[2]]
        pygame.draw.polygon(surface, color.blended_with(dark, 1 - _SHADOW).as_tuple(), left_wall)
        pygame.draw.polygon(surface, color.blended_with(dark, 1 - _LIT).as_tuple(), right_wall)
        pygame.draw.polygon(surface, color.as_tuple(), top)
        pygame.draw.polygon(surface, color.blended_with(dark, 0.45).as_tuple(), top, 1)

    def _tree(
        self,
        surface: pygame.Surface,
        position: Position,
        origin: tuple[int, int],
        half: int,
        color: Color,
    ) -> None:
        cx, cy = self._ground(position, origin, half)
        trunk = color.blended_with(_BLACK, 0.55)
        pygame.draw.line(
            surface, trunk.as_tuple(), (cx, cy), (cx, cy - _CROWN * half * 0.6), max(1, half // 6)
        )
        crown = pygame.Rect(0, 0, int(half * 1.1), int(half * 0.9))
        crown.center = (int(cx), int(cy - _CROWN * half))
        pygame.draw.ellipse(surface, color.as_tuple(), crown)

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _draw_fire(
        self,
        surface: pygame.Surface,
        city_map: CityMap,
        position: Position,
        origin: tuple[int, int],
        half: int,
    ) -> None:
        """A flame standing on the tile, above the roofline when it is a building."""
        terrain = city_map.tile_at(position)
        lift = (
            _STOREYS[(position.x * 7 + position.y * 3) % 4]
            if terrain is TerrainType.BUILDING
            else 0.0
        )
        cx, cy = self._ground(position, origin, half)
        rect = pygame.Rect(0, 0, int(half * 1.3), int(half * 1.6))
        rect.midbottom = (int(cx), int(cy - lift * half + half * 0.2))
        color = self._theme.entity_colors[EntityKind.FIRE]
        glyphs.draw_fire(surface, rect, color, color.blended_with(_WHITE, 0.6), self._flicker)

    def _draw_victim(
        self, surface: pygame.Surface, victim: Victim, origin: tuple[int, int], half: int
    ) -> None:
        cx, cy = self._ground(victim.position, origin, half)
        color = self._theme.entity_colors[EntityKind.VICTIM]
        if victim.status is VictimStatus.LOST:
            color = color.blended_with(self._theme.hud.panel, 0.7)
        body = pygame.Rect(0, 0, max(3, half // 2), int(half * 0.8))
        body.midbottom = (int(cx), int(cy))
        pygame.draw.ellipse(surface, color.as_tuple(), body)
        pygame.draw.circle(
            surface, color.as_tuple(), (int(cx), int(cy - half * 0.95)), max(2, half // 4)
        )
        if victim.status is VictimStatus.TRAPPED:
            pulse = 1.0 + 0.25 * math.sin(self._flicker * math.tau)
            ring = pygame.Rect(0, 0, int(half * 1.4 * pulse), int(half * 0.7 * pulse))
            ring.center = (int(cx), int(cy))
            pygame.draw.ellipse(surface, self._theme.hud.warning.as_tuple(), ring, 1)

    def _draw_vehicle(
        self, surface: pygame.Surface, vehicle: Vehicle, origin: tuple[int, int], half: int
    ) -> None:
        """A small truck: a coloured block with a windscreen facing the heading."""
        color = self._theme.entity_colors[EntityKind.VEHICLE]
        cx, cy = self._ground(vehicle.position, origin, half)
        dx, dy = vehicle.heading.delta
        fx, fy = (dx - dy) * half * 0.35, (dx + dy) * half * 0.18
        body = [
            (cx - fx - fy * 1.6, cy - fy + fx * 0.3),
            (cx + fx - fy * 1.6, cy + fy + fx * 0.3),
            (cx + fx + fy * 1.6, cy + fy - fx * 0.3),
            (cx - fx + fy * 1.6, cy - fy - fx * 0.3),
        ]
        roof = [(px, py - half * 0.45) for px, py in body]
        dark = _BLACK
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            wall = [body[a], body[b], roof[b], roof[a]]
            pygame.draw.polygon(surface, color.blended_with(dark, 0.3).as_tuple(), wall)
        pygame.draw.polygon(surface, color.as_tuple(), roof)
        nose = (cx + fx, cy + fy - half * 0.45)
        pygame.draw.circle(
            surface, _WHITE.as_tuple(), (int(nose[0]), int(nose[1])), max(2, half // 6)
        )
