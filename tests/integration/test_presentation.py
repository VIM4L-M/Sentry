"""Integration tests for the presentation layer: glyphs, grid view, cameras.

Rendering tests can only assert so much — "it looks right" is not
executable. What they *can* pin down is that each glyph marks the tile it
was given and stays inside it, that the debug view colours cells by their
occupancy code, and that the camera panel survives the frames the rig
actually produces. Those are the failures that would otherwise reach a
demo.

Headless throughout (SDL "dummy" driver, set in conftest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import pytest

from sentry_ai.common.color import Color
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading, TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.rendering import glyphs
from sentry_ai.rendering.camera_panel import CameraPanelRenderer
from sentry_ai.rendering.grid_overlay import CODE_COLORS, GridOverlayRenderer, legend
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig

_INK = Color(255, 0, 255)
_BLANK = (0, 0, 0)


@pytest.fixture(scope="module", autouse=True)
def _pygame_ready() -> None:
    pygame.init()


@pytest.fixture
def theme(project_root: Path) -> Theme:
    loader = ConfigLoader(project_root=project_root)
    return Theme.from_config(loader, "configs/render.yaml")


@pytest.fixture
def city_map(project_root: Path) -> CityMap:
    loader = ConfigLoader(project_root=project_root)
    return CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))


@pytest.fixture
def rig(project_root: Path) -> SensorRig:
    loader = ConfigLoader(project_root=project_root)
    return SensorRig.from_config(
        loader.load_sensor_config("configs/sensors.yaml"),
        SensorPalette.from_config(loader, "configs/sensors.yaml"),
    )


def _painted(surface: pygame.Surface, rect: pygame.Rect) -> int:
    """How many pixels inside ``rect`` are no longer blank."""
    return sum(
        1
        for x in range(rect.left, rect.right)
        for y in range(rect.top, rect.bottom)
        if surface.get_at((x, y))[:3] != _BLANK
    )


class TestGlyphsStayInsideTheirTile:
    """A glyph that bleeds into a neighbouring tile misreports the world."""

    _SIZE = 32

    def _canvas(self) -> tuple[pygame.Surface, pygame.Rect]:
        surface = pygame.Surface((self._SIZE * 3, self._SIZE * 3))
        surface.fill(_BLANK)
        return surface, pygame.Rect(self._SIZE, self._SIZE, self._SIZE, self._SIZE)

    @pytest.mark.parametrize(
        "draw",
        [
            glyphs.draw_victim,
            glyphs.draw_debris,
            glyphs.draw_tree,
            glyphs.draw_hospital,
        ],
    )
    def test_simple_glyphs_mark_their_tile_and_only_their_tile(self, draw: object) -> None:
        surface, rect = self._canvas()
        draw(surface, rect, _INK)  # type: ignore[operator]
        assert _painted(surface, rect) > 0
        assert _painted(surface, surface.get_rect()) == _painted(surface, rect)

    def test_the_vehicle_chevron_stays_inside_its_tile(self) -> None:
        surface, rect = self._canvas()
        glyphs.draw_vehicle(surface, rect, _INK, Heading.NORTH)
        assert _painted(surface, rect) > 0
        assert _painted(surface, surface.get_rect()) == _painted(surface, rect)

    def test_the_chevron_points_a_different_way_per_heading(self) -> None:
        renders = []
        for heading in (Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST):
            surface, rect = self._canvas()
            glyphs.draw_vehicle(surface, rect, _INK, heading)
            renders.append(pygame.image.tobytes(surface, "RGB"))
        assert len(set(renders)) == 4

    def test_fire_flickers_over_time(self) -> None:
        """Static fire made the city look like a screenshot."""
        renders = []
        for phase in (0.0, 0.25, 0.75):
            surface, rect = self._canvas()
            glyphs.draw_fire(surface, rect, _INK, Color(255, 255, 0), flicker=phase)
            renders.append(pygame.image.tobytes(surface, "RGB"))
        assert len(set(renders)) == 3

    def test_a_flickering_flame_never_escapes_its_tile(self) -> None:
        for phase in (0.0, 0.2, 0.4, 0.6, 0.8):
            surface, rect = self._canvas()
            glyphs.draw_fire(surface, rect, _INK, Color(255, 255, 0), flicker=phase)
            assert _painted(surface, surface.get_rect()) == _painted(surface, rect)


class TestMapRenderer:
    def test_a_rescued_victim_leaves_the_map(self, theme: Theme, city_map: CityMap) -> None:
        """Otherwise the map disagrees with the grid and the rescued counter."""
        renderer = MapRenderer(theme, tile_size_px=16)
        surface = pygame.Surface((city_map.width * 16, city_map.height * 16))
        target = city_map.victims[0]
        tile = pygame.Rect(target.position.x * 16, target.position.y * 16, 16, 16)

        renderer.draw(surface, city_map)
        before = pygame.image.tobytes(surface.subsurface(tile), "RGB")
        target.status = VictimStatus.RESCUED
        renderer.draw(surface, city_map)
        assert pygame.image.tobytes(surface.subsurface(tile), "RGB") != before

    def test_trees_and_rubble_do_not_look_the_same(
        self, theme: Theme, city_map: CityMap
    ) -> None:
        renderer = MapRenderer(theme, tile_size_px=16)
        surface = pygame.Surface((city_map.width * 16, city_map.height * 16))
        renderer.draw(surface, city_map)

        kinds = {obstacle.kind: obstacle.position for obstacle in city_map.obstacles}
        tree, rubble = kinds[TerrainType.TREE], kinds[TerrainType.RUBBLE]
        tree_pixels = pygame.image.tobytes(
            surface.subsurface(pygame.Rect(tree.x * 16, tree.y * 16, 16, 16)), "RGB"
        )
        rubble_pixels = pygame.image.tobytes(
            surface.subsurface(pygame.Rect(rubble.x * 16, rubble.y * 16, 16, 16)), "RGB"
        )
        assert tree_pixels != rubble_pixels

    def test_the_previous_route_is_drawn_differently_from_the_active_one(
        self, theme: Theme, city_map: CityMap
    ) -> None:
        """A replan has to be visible, not just counted."""
        from sentry_ai.interfaces.navigation import Route

        renderer = MapRenderer(theme, tile_size_px=16)
        surface = pygame.Surface((city_map.width * 16, city_map.height * 16))
        tile = Position(5, 5)
        route = Route(waypoints=(tile,), cost=1.0)

        renderer.draw(surface, city_map, route=route)
        active = surface.get_at((tile.x * 16 + 8, tile.y * 16 + 8))[:3]
        renderer.draw(surface, city_map, previous_route=route)
        stale = surface.get_at((tile.x * 16 + 8, tile.y * 16 + 8))[:3]
        assert active != stale

    def test_camera_footprints_are_drawn_only_when_asked(
        self, theme: Theme, city_map: CityMap, rig: SensorRig
    ) -> None:
        renderer = MapRenderer(theme, tile_size_px=16)
        surface = pygame.Surface((city_map.width * 16, city_map.height * 16))

        renderer.draw(surface, city_map)
        without = pygame.image.tobytes(surface, "RGB")
        renderer.draw(surface, city_map, camera_views=rig.cctv_views)
        assert pygame.image.tobytes(surface, "RGB") != without

    def test_the_preview_call_still_works_unchanged(
        self, theme: Theme, city_map: CityMap
    ) -> None:
        """Phase 1's two-argument call must keep working."""
        renderer = MapRenderer(theme, tile_size_px=8)
        surface = pygame.Surface((city_map.width * 8, city_map.height * 8))
        renderer.draw(surface, city_map)


class TestGridOverlay:
    def test_every_occupancy_code_has_a_colour(self) -> None:
        assert set(CODE_COLORS) == set(OccupancyCode)

    def test_the_legend_is_ordered_by_code(self) -> None:
        assert [int(entry.code) for entry in legend()] == list(range(7))
        assert legend()[2].text == "2 fire"

    def test_cells_are_painted_by_their_code(self, theme: Theme) -> None:
        grid = OccupancyGrid.empty(width=3, height=1)
        grid.mark(Position(1, 0), OccupancyCode.FIRE)
        renderer = GridOverlayRenderer(theme, tile_size_px=24)
        surface = pygame.Surface((72, 24))

        renderer.draw(surface, grid)
        assert surface.get_at((36, 12))[:3] != surface.get_at((12, 12))[:3]

    def test_the_whole_shipped_grid_renders(self, theme: Theme, city_map: CityMap) -> None:
        grid = OccupancyGrid.from_city_map(city_map)
        renderer = GridOverlayRenderer(theme, tile_size_px=28)
        surface = pygame.Surface((city_map.width * 28, city_map.height * 28))
        renderer.draw(surface, grid)

    def test_tiny_cells_drop_their_digits_rather_than_smearing(
        self, theme: Theme
    ) -> None:
        grid = OccupancyGrid.empty(width=4, height=4)
        renderer = GridOverlayRenderer(theme, tile_size_px=6)
        renderer.draw(pygame.Surface((24, 24)), grid)


class TestCameraPanel:
    def test_the_panel_renders_every_frame_the_rig_produces(
        self, theme: Theme, city_map: CityMap, rig: SensorRig
    ) -> None:
        panel = CameraPanelRenderer(theme)
        surface = pygame.Surface((panel.width_px, 560))
        surface.fill(_BLANK)
        panel.draw(surface, rig.capture_all(city_map), left=0, top=0, height=560)
        assert _painted(surface, surface.get_rect()) > 0

    def test_an_empty_frame_list_is_not_an_error(self, theme: Theme) -> None:
        panel = CameraPanelRenderer(theme)
        panel.draw(pygame.Surface((240, 400)), [], left=0, top=0, height=400)

    def test_a_cramped_panel_does_not_raise(
        self, theme: Theme, city_map: CityMap, rig: SensorRig
    ) -> None:
        """Thumbnails share the available height; five into 40px is degenerate."""
        panel = CameraPanelRenderer(theme)
        panel.draw(pygame.Surface((240, 40)), rig.capture_all(city_map), 0, 0, 40)

    def test_thumbnails_shrink_rather_than_overflow(
        self, theme: Theme, city_map: CityMap, rig: SensorRig
    ) -> None:
        panel = CameraPanelRenderer(theme)
        surface = pygame.Surface((panel.width_px + 40, 560))
        surface.fill(_BLANK)
        panel.draw(surface, rig.capture_all(city_map), left=0, top=0, height=560)

        # Nothing may be drawn to the right of the panel's own width.
        overflow = pygame.Rect(panel.width_px, 0, 40, 560)
        assert _painted(surface, overflow) == 0

    def test_frames_carry_their_detections_into_the_panel(
        self, theme: Theme, city_map: CityMap, rig: SensorRig
    ) -> None:
        """The boxes drawn here are ground truth today and YOLO output in Phase 3."""
        frames = rig.capture_cctv(city_map)
        assert any(frame.annotations for frame in frames)

        panel = CameraPanelRenderer(theme)
        surface = pygame.Surface((panel.width_px, 560))
        panel.draw(surface, frames, left=0, top=0, height=560)
        assert np.any(pygame.surfarray.array3d(surface) > 0)
