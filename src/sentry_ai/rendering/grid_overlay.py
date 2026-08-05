"""Renders the occupancy grid as the planner actually sees it.

The map view shows the *world*. This shows the command center's **belief**
about the world — the integer-coded grid A* plans on. They are the same
thing today, and from Phase 3 they will not be: the grid will be built from
YOLO detections and will be wrong in interesting ways.

Being able to flip between the two with one key is the difference between
"the vehicle drove somewhere strange" and "the grid thought that tile was
debris". It is a debugging tool first and a demo aid second.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from sentry_ai.common.color import Color
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.rendering.theme import Theme

#: Colour per occupancy code. Impassable codes are deliberately hot and
#: traversable ones cold, so a glance answers "where can it drive?".
CODE_COLORS: dict[OccupancyCode, Color] = {
    OccupancyCode.ROAD: Color(38, 42, 48),
    OccupancyCode.BUILDING: Color(96, 100, 110),
    OccupancyCode.FIRE: Color(206, 78, 32),
    OccupancyCode.DEBRIS: Color(132, 96, 54),
    OccupancyCode.VICTIM: Color(232, 92, 148),
    OccupancyCode.HOSPITAL: Color(58, 168, 140),
    OccupancyCode.VEHICLE: Color(72, 156, 236),
}

#: Point size of the digit drawn in each cell.
_CODE_LABEL_SIZE_PX = 11

#: Below this tile size the digits are unreadable and are dropped.
_MIN_TILE_FOR_LABELS_PX = 18


@dataclass(frozen=True)
class GridOverlayLegend:
    """One legend entry: the code, its name, and the colour it is drawn in."""

    code: OccupancyCode
    label: str
    color: Color

    @property
    def text(self) -> str:
        """``"2 fire"`` — the digit and its meaning."""
        return f"{int(self.code)} {self.label}"


def legend() -> list[GridOverlayLegend]:
    """Every occupancy code in numeric order, for a HUD legend."""
    return [
        GridOverlayLegend(code=code, label=code.name.lower(), color=CODE_COLORS[code])
        for code in sorted(CODE_COLORS, key=int)
    ]


class GridOverlayRenderer:
    """Draws an :class:`OccupancyGrid` as coloured, numbered cells."""

    def __init__(self, theme: Theme, tile_size_px: int) -> None:
        """Create a renderer.

        Args:
            theme: Used only for the cell-outline colour, so the debug view
                stays visually related to the rest of the window.
            tile_size_px: Edge length of one cell, matching the map view so
                toggling between them does not move anything.
        """
        self._theme = theme
        self._tile_size_px = tile_size_px
        self._font: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface, grid: OccupancyGrid) -> None:
        """Fill ``surface``'s map area with the grid's colour-coded cells."""
        show_labels = self._tile_size_px >= _MIN_TILE_FOR_LABELS_PX
        for y in range(grid.height):
            for x in range(grid.width):
                code = grid.code_at(Position(x, y))
                rect = pygame.Rect(
                    x * self._tile_size_px,
                    y * self._tile_size_px,
                    self._tile_size_px,
                    self._tile_size_px,
                )
                pygame.draw.rect(surface, CODE_COLORS[code].as_tuple(), rect)
                pygame.draw.rect(surface, self._theme.background.as_tuple(), rect, width=1)
                if show_labels:
                    self._draw_code(surface, rect, code)

    def _draw_code(
        self, surface: pygame.Surface, rect: pygame.Rect, code: OccupancyCode
    ) -> None:
        """Write the cell's integer code, which is the real contract."""
        glyph = self._label_font().render(
            str(int(code)), True, self._theme.hud.text.as_tuple()
        )
        surface.blit(glyph, glyph.get_rect(center=rect.center))

    def _label_font(self) -> pygame.font.Font:
        """The digit font, created on first use."""
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont(
                "consolas,dejavusansmono,monospace", _CODE_LABEL_SIZE_PX
            )
        return self._font
