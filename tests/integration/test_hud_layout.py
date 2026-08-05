"""Integration tests for the HUD's layout arithmetic.

These exist because the stats block wraps into columns and the gauges are
right-aligned, so the two can collide silently as
:meth:`~sentry_ai.simulation.mission.MissionStats.as_display_rows` grows.
Nothing raises when that happens — the text simply draws on top of the
bars — so it needs a test rather than a code review.

Text is measured with the real font (SDL's "dummy" driver, set in
conftest), not estimated from a character-width guess.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import pygame
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import MissionConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.rendering.camera_panel import CameraPanelLayout
from sentry_ai.rendering.hud import HudLayout, HudRenderer
from sentry_ai.rendering.theme import Theme
from sentry_ai.simulation.mission import MissionController, MissionStats

#: The shipped city at the shipped tile size, plus the camera strip — the
#: window the HUD actually has to fit inside. Keep this in step with
#: SimulationApp._create_surface; the whole point of these tests is that the
#: HUD stays inside the window it is given.
_WINDOW_WIDTH_PX = 30 * 28 + CameraPanelLayout().width_px


@pytest.fixture
def theme(project_root: Path) -> Theme:
    loader = ConfigLoader(project_root=project_root)
    return Theme.from_config(loader, "configs/render.yaml")


@pytest.fixture
def renderer(theme: Theme) -> HudRenderer:
    pygame.init()
    return HudRenderer(theme)


def _rendered_width(renderer: HudRenderer, text: str) -> int:
    """Width of ``text`` in the HUD's own font."""
    font: pygame.font.Font = renderer._font  # noqa: SLF001 — measuring real metrics
    return int(font.size(text)[0])


def _stats_right_edge(renderer: HudRenderer, layout: HudLayout) -> int:
    """The rightmost pixel the stats block can reach."""
    rows = MissionStats().as_display_rows()
    columns = ceil(len(rows) / layout.stats_rows_per_column)
    last_column_x = (
        layout.padding_px + layout.stats_left_px + (columns - 1) * layout.stats_column_width_px
    )
    widest = max(_rendered_width(renderer, f"{label:<12}{value}") for label, value in rows)
    return last_column_x + widest


class TestStatsFitBesideTheGauges:
    def test_the_stats_block_does_not_reach_the_gauges(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        gauges_left = _WINDOW_WIDTH_PX - layout.gauge_width_px - layout.padding_px
        assert _stats_right_edge(renderer, layout) <= gauges_left

    def test_the_status_block_does_not_reach_the_stats(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        widest_status = _rendered_width(renderer, "returning to hospital")
        assert layout.padding_px + widest_status <= layout.padding_px + layout.stats_left_px

    def test_every_stats_row_fits_inside_the_panel(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        used = layout.padding_px + layout.stats_rows_per_column * layout.row_height_px
        assert used + layout.padding_px <= layout.height_px

    def test_the_gauges_fit_inside_the_panel(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        # Health sits two rows below battery, and its bar is 8px tall.
        used = layout.padding_px + layout.row_height_px * 3 + 8
        assert used + layout.padding_px <= layout.height_px


class TestEventsFitTheirColumn:
    def test_the_event_log_starts_clear_of_the_stats(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        assert _stats_right_edge(renderer, layout) <= layout.padding_px + layout.events_left_px

    def test_the_event_log_ends_before_the_gauges(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        events_left = layout.padding_px + layout.events_left_px
        gauges_left = _WINDOW_WIDTH_PX - layout.gauge_width_px - layout.padding_px
        assert events_left < gauges_left

    def test_a_long_message_is_ellipsized_rather_than_overrunning(
        self, renderer: HudRenderer
    ) -> None:
        """Event text is written for humans and varies; the panel must cope."""
        surface = pygame.Surface((_WINDOW_WIDTH_PX, 400))
        available = renderer._events_width(surface)  # noqa: SLF001 — real geometry
        long_line = " 12.3s " + "a very long hazard description " * 4

        fitted = renderer._ellipsize(long_line, available)  # noqa: SLF001
        assert fitted.endswith("…")
        assert _rendered_width(renderer, fitted) <= available

    def test_a_short_message_is_left_alone(self, renderer: HudRenderer) -> None:
        surface = pygame.Surface((_WINDOW_WIDTH_PX, 400))
        available = renderer._events_width(surface)  # noqa: SLF001
        assert renderer._ellipsize("12.3s ok", available) == "12.3s ok"  # noqa: SLF001

    def test_every_event_row_fits_inside_the_panel(self, renderer: HudRenderer) -> None:
        layout = HudLayout()
        used = layout.padding_px + layout.events_rows * layout.row_height_px
        assert used + layout.padding_px <= layout.height_px


class TestDrawingAFullyPopulatedHud:
    def _mission(self, project_root: Path) -> MissionController:
        loader = ConfigLoader(project_root=project_root)
        city_map = CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))
        return MissionController(
            city_map=city_map,
            grid=OccupancyGrid.from_city_map(city_map),
            planner=AStarPlanner(),
            config=MissionConfig(),
            stats=MissionStats(
                ticks=9999,
                elapsed_seconds=999.9,
                victims_rescued=99,
                victims_unreachable=99,
                replans=99,
                collisions=99,
                tiles_travelled=9999,
                hazard_events=99,
                routes_cut_by_hazards=99,
            ),
        )

    def test_the_real_draw_path_survives_worst_case_values(
        self, renderer: HudRenderer, project_root: Path
    ) -> None:
        mission = self._mission(project_root)
        surface = pygame.Surface((_WINDOW_WIDTH_PX, 560 + renderer.height_px))
        renderer.draw(surface, mission, mission.city_map.vehicle)

        # The panel must have been painted: the strip is no longer blank.
        strip_top = surface.get_height() - renderer.height_px
        assert surface.get_at((4, strip_top + 4))[:3] != (0, 0, 0)

    def test_worst_case_stats_still_clear_the_gauges(
        self, renderer: HudRenderer, project_root: Path
    ) -> None:
        """Widths are measured against real values, not the zeroed defaults."""
        layout = HudLayout()
        rows = self._mission(project_root).stats.as_display_rows()
        columns = ceil(len(rows) / layout.stats_rows_per_column)
        last_column_x = (
            layout.padding_px
            + layout.stats_left_px
            + (columns - 1) * layout.stats_column_width_px
        )
        widest = max(_rendered_width(renderer, f"{label:<12}{value}") for label, value in rows)
        gauges_left = _WINDOW_WIDTH_PX - layout.gauge_width_px - layout.padding_px
        assert last_column_x + widest <= gauges_left
