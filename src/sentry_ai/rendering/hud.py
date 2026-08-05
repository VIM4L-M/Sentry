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
from sentry_ai.simulation.events import EventKind
from sentry_ai.simulation.mission import MissionController, MissionPhase

#: Colour role per event kind. Only three roles exist in the HUD palette, so
#: events map onto "routine", "good news", and "bad news" rather than
#: getting a colour each.
_EVENT_TONE: dict[EventKind, str] = {
    EventKind.MISSION: "accent",
    EventKind.ROUTE: "text",
    EventKind.RESCUE: "accent",
    EventKind.HAZARD: "warning",
    EventKind.FAILURE: "warning",
}


@dataclass(frozen=True)
class HudLayout:
    """Pixel geometry of the HUD strip.

    Attributes:
        height_px: Total strip height below the map.
        padding_px: Inset from the strip's edges.
        row_height_px: Vertical pitch between text rows.
        gauge_width_px: Width of the battery and health bars.
        font_size_px: Point size for all HUD text.
        stats_left_px: Where the stats block starts, clear of the status text.
        stats_column_width_px: Horizontal pitch between stats columns.
        events_left_px: Where the event log starts, clear of the stats.
        events_rows: How many recent events the log shows.
        stats_rows_per_column: How many stats rows stack before wrapping into
            the next column. Set so the block stays clear of the gauges as
            :meth:`~sentry_ai.simulation.mission.MissionStats.as_display_rows`
            grows — adding a metric should not silently overlap them.
    """

    height_px: int = 116
    padding_px: int = 12
    row_height_px: int = 18
    gauge_width_px: int = 180
    font_size_px: int = 15
    stats_left_px: int = 220
    stats_column_width_px: int = 190
    stats_rows_per_column: int = 5
    events_left_px: int = 610
    events_rows: int = 5


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
        self._draw_events(surface, mission, top + layout.padding_px)
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
        """The mission's running counters, wrapped into columns."""
        layout = self._layout
        column_x = layout.padding_px + layout.stats_left_px
        per_column = layout.stats_rows_per_column
        for index, (label, value) in enumerate(mission.stats.as_display_rows()):
            x = column_x + (index // per_column) * layout.stats_column_width_px
            y = top + (index % per_column) * layout.row_height_px
            self._text(surface, f"{label:<12}{value}", x, y)

    def _draw_events(self, surface: pygame.Surface, mission: MissionController, top: int) -> None:
        """The most recent mission events, oldest at the top.

        This is the panel that lets someone watch a demo and understand
        *why* the vehicle did what it did — "world changed under the route"
        followed by "routing to the hospital" tells the whole replanning
        story without a word of narration.
        """
        layout = self._layout
        x = layout.padding_px + layout.events_left_px
        available = self._events_width(surface)
        for index, event in enumerate(mission.events.recent(layout.events_rows)):
            timestamp, message = event.as_row()
            color = getattr(self._theme.hud, _EVENT_TONE[event.kind])
            self._text(
                surface,
                self._ellipsize(f"{timestamp} {message}", available),
                x,
                top + index * layout.row_height_px,
                color,
            )

    def _events_width(self, surface: pygame.Surface) -> int:
        """Horizontal room the event log has before the gauges begin."""
        layout = self._layout
        gauges_left = surface.get_width() - layout.gauge_width_px - layout.padding_px
        return gauges_left - (layout.padding_px + layout.events_left_px) - layout.padding_px

    def _ellipsize(self, text: str, max_width_px: int) -> str:
        """Trim ``text`` with a trailing ellipsis until it fits.

        Event messages are written for humans and vary in length, so the
        panel has to cope rather than assume. Measured with the real font
        rather than an assumed character width — the HUD font is monospace
        today, but nothing enforces that.
        """
        if max_width_px <= 0 or self._font.size(text)[0] <= max_width_px:
            return text
        trimmed = text
        while trimmed and self._font.size(f"{trimmed}…")[0] > max_width_px:
            trimmed = trimmed[:-1]
        return f"{trimmed}…"

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
