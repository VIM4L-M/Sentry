"""Draws the mission HUD: the operator's instrument panel.

Reads live state off the :class:`~sentry_ai.simulation.mission.MissionController`
and the vehicle, and draws it into a strip below the city. Strictly
read-only, like every renderer here — the HUD never advances a mission or
touches vehicle state.

The layout is a fixed grid of label/value rows plus two gauges, sized from
:class:`HudLayout` rather than pixel literals scattered through the drawing
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from sentry_ai.domain.entities import Vehicle
from sentry_ai.rendering.theme import Color, Theme
from sentry_ai.simulation.mission import MissionController, MissionPhase


@dataclass(frozen=True)
class HudLayout:
    """Pixel geometry of the HUD strip.

    Attributes:
        height_px: Total strip height below the map.
        padding_px: Inset from the strip's edges.
        row_height_px: Vertical pitch between text rows.
        gauge_width_px: Width of the battery and health bars.
        font_size_px: Point size for all HUD text.
    """

    height_px: int = 116
    padding_px: int = 12
    row_height_px: int = 18
    gauge_width_px: int = 180
    font_size_px: int = 15


class HudRenderer:
    """Renders mission telemetry into a strip below the map."""

    def __init__(self, theme: Theme, layout: HudLayout | None = None) -> None:
        """Create a HUD renderer.

        Args:
            theme: Palette to draw with; only ``theme.hud`` is used.
            layout: Pixel geometry. Defaults to :class:`HudLayout`.

        Note:
            ``pygame.font`` must already be initialised — the composition
            root owns ``pygame.init()``, not this class.
        """
        self._theme = theme
        self._layout = layout or HudLayout()
        self._font = pygame.font.SysFont(
            "consolas,dejavusansmono,monospace", self._layout.font_size_px
        )

    @property
    def height_px(self) -> int:
        """How much vertical space this HUD needs."""
        return self._layout.height_px

    def draw(self, surface: pygame.Surface, mission: MissionController, vehicle: Vehicle) -> None:
        """Draw the whole HUD strip along the bottom of ``surface``."""
        layout = self._layout
        top = surface.get_height() - layout.height_px
        pygame.draw.rect(
            surface,
            self._theme.hud.panel.as_tuple(),
            pygame.Rect(0, top, surface.get_width(), layout.height_px),
        )

        self._draw_status(surface, mission, top + layout.padding_px)
        self._draw_stats(surface, mission, top + layout.padding_px)
        self._draw_gauges(surface, vehicle, top + layout.padding_px)

    def _draw_status(self, surface: pygame.Surface, mission: MissionController, top: int) -> None:
        """Phase, failure reason, and the waypoint currently being chased."""
        layout = self._layout
        phase_color = (
            self._theme.hud.warning
            if mission.phase is MissionPhase.FAILED
            else self._theme.hud.accent
        )
        self._text(surface, mission.phase.value.replace("_", " ").upper(), layout.padding_px, top,
                   phase_color)

        waypoint = mission.next_waypoint()
        waypoint_text = waypoint.as_tuple() if waypoint is not None else "—"
        detail = mission.stats.failure_reason or f"waypoint {waypoint_text}"
        self._text(surface, detail, layout.padding_px, top + layout.row_height_px)
        self._text(
            surface,
            f"route {len(mission.route)} tiles",
            layout.padding_px,
            top + layout.row_height_px * 2,
        )

    def _draw_stats(self, surface: pygame.Surface, mission: MissionController, top: int) -> None:
        """The mission's running counters, in two columns."""
        layout = self._layout
        column_x = layout.padding_px + 220
        for index, (label, value) in enumerate(mission.stats.as_display_rows()):
            x = column_x + (index // 3) * 190
            y = top + (index % 3) * layout.row_height_px
            self._text(surface, f"{label:<12}{value}", x, y)

    def _draw_gauges(self, surface: pygame.Surface, vehicle: Vehicle, top: int) -> None:
        """Battery and health bars, right-aligned."""
        layout = self._layout
        x = surface.get_width() - layout.gauge_width_px - layout.padding_px
        self._gauge(surface, "BATTERY", vehicle.battery_percent, x, top)
        self._gauge(surface, "HEALTH", vehicle.health_percent, x, top + layout.row_height_px * 2)

    def _gauge(
        self, surface: pygame.Surface, label: str, percent: float, x: int, y: int
    ) -> None:
        """One labelled horizontal bar, drawn empty-then-filled."""
        layout = self._layout
        self._text(surface, f"{label} {percent:5.1f}%", x, y)
        track = pygame.Rect(x, y + layout.row_height_px, layout.gauge_width_px, 8)
        pygame.draw.rect(surface, self._theme.hud.panel.as_tuple(), track)
        pygame.draw.rect(surface, self._theme.hud.text.as_tuple(), track, width=1)

        filled = int(track.width * max(0.0, min(percent, 100.0)) / 100.0)
        if filled <= 0:
            return
        color = self._theme.hud.warning if percent <= 25.0 else self._theme.hud.accent
        pygame.draw.rect(
            surface, color.as_tuple(), pygame.Rect(track.x, track.y, filled, track.height)
        )

    def _text(
        self, surface: pygame.Surface, message: str, x: int, y: int, color: Color | None = None
    ) -> None:
        """Blit one line of HUD text at ``(x, y)``."""
        rendered = self._font.render(
            str(message), True, (color or self._theme.hud.text).as_tuple()
        )
        surface.blit(rendered, (x, y))
