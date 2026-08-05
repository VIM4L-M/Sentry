"""Unit tests for sentry_ai.sensors.rasterizer.FrameRasterizer.

The rasterizer is the source of every training label in the project, so
these tests care most about one thing: that what is painted and what is
annotated describe the same world.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sentry_ai.common.color import Color
from sentry_ai.domain.entities import Obstacle, Position
from sentry_ai.domain.enums import EntityKind, TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.perception import Detection
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rasterizer import FrameRasterizer

_ROAD = Color(10, 20, 30)
_BUILDING = Color(200, 200, 200)
_VICTIM = Color(240, 0, 120)
_DEBRIS = Color(60, 50, 40)
_VEHICLE = Color(0, 128, 255)
_VOID = Color(1, 2, 3)

# 6x4 city. Road across row 1, a building at (2, 2), a tree at (5, 3).
_MAP_DATA: dict[str, Any] = {
    "width": 6,
    "height": 4,
    "grid": ["......", "======", "..#...", ".....t"],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
    "vehicle_start": [0, 3],
    "victims": [{"id": "v1", "position": [4, 2]}],
    "fires": [{"id": "f1", "position": [1, 3], "intensity": 1.0, "radius": 1}],
}


def _palette(jitter: int = 0) -> SensorPalette:
    return SensorPalette(
        terrain={terrain: Color(90, 90, 90) for terrain in TerrainType}
        | {TerrainType.ROAD: _ROAD, TerrainType.BUILDING: _BUILDING},
        void=_VOID,
        fire_core=Color(255, 200, 0),
        fire_edge=Color(180, 60, 0),
        victim=_VICTIM,
        debris=_DEBRIS,
        vehicle=_VEHICLE,
        texture_jitter=jitter,
    )


def _full_view(tile_size_px: int = 8) -> CameraView:
    return CameraView(
        camera_id="cam_full",
        origin=Position(0, 0),
        width_tiles=6,
        height_tiles=4,
        tile_size_px=tile_size_px,
    )


@pytest.fixture
def city_map() -> CityMap:
    return CityMap.from_config(_MAP_DATA)


def _render(city_map: CityMap, view: CameraView | None = None, jitter: int = 0) -> CameraFrame:
    return FrameRasterizer(_palette(jitter)).render(city_map, view or _full_view())


def _labels(frame: CameraFrame, kind: EntityKind) -> list[Detection]:
    return [detection for detection in frame.annotations if detection.label is kind]


class TestFrameShape:
    def test_the_frame_matches_the_camera_geometry(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        assert frame.pixels.shape == (32, 48, 3)
        assert frame.pixels.dtype == np.uint8

    def test_every_annotation_is_ground_truth(self, city_map: CityMap) -> None:
        assert all(detection.confidence == 1.0 for detection in _render(city_map).annotations)


class TestTerrainPainting:
    def test_a_road_tile_is_painted_the_road_colour(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        assert tuple(frame.pixels[12, 20]) == _ROAD.as_tuple()

    def test_a_building_tile_is_painted_the_building_colour(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        # World tile (2, 2) -> pixels x 16..24, y 16..24.
        assert tuple(frame.pixels[20, 20]) == _BUILDING.as_tuple()

    def test_tiles_outside_the_map_are_painted_void(self, city_map: CityMap) -> None:
        view = CameraView(
            camera_id="overhang",
            origin=Position(4, 2),
            width_tiles=4,
            height_tiles=4,
            tile_size_px=8,
        )
        frame = _render(city_map, view)
        # World (6, 2) is off the east edge -> frame pixels x 16..24, y 0..8.
        assert tuple(frame.pixels[4, 20]) == _VOID.as_tuple()


class TestVictimLabels:
    def test_a_trapped_victim_is_painted_and_labelled(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        victims = _labels(frame, EntityKind.VICTIM)
        assert len(victims) == 1
        assert frame.world_position_of(victims[0]) == Position(4, 2)

    def test_the_victim_marker_is_smaller_than_its_tile(self, city_map: CityMap) -> None:
        """Full-tile targets would make the detection problem trivial."""
        box = _labels(_render(city_map), EntityKind.VICTIM)[0].bbox
        assert (box.x_max - box.x_min) < 8

    def test_a_victim_aboard_the_vehicle_is_not_labelled(self, city_map: CityMap) -> None:
        city_map.victims[0].status = VictimStatus.ONBOARD
        assert _labels(_render(city_map), EntityKind.VICTIM) == []

    def test_a_rescued_victim_is_not_labelled(self, city_map: CityMap) -> None:
        city_map.victims[0].status = VictimStatus.RESCUED
        assert _labels(_render(city_map), EntityKind.VICTIM) == []

    def test_a_victim_outside_the_footprint_is_not_labelled(self, city_map: CityMap) -> None:
        view = CameraView(
            camera_id="corner",
            origin=Position(0, 0),
            width_tiles=2,
            height_tiles=2,
            tile_size_px=8,
        )
        assert _labels(_render(city_map, view), EntityKind.VICTIM) == []


class TestFireLabels:
    def test_a_fire_is_labelled_once_over_its_whole_footprint(
        self, city_map: CityMap
    ) -> None:
        fires = _labels(_render(city_map), EntityKind.FIRE)
        assert len(fires) == 1
        box = fires[0].bbox
        # Radius-1 disc at (1, 3) spans world x 0..2 -> pixels 0..24.
        assert (box.x_min, box.x_max) == (0, 24)

    def test_a_dimmer_fire_paints_differently(self, city_map: CityMap) -> None:
        """Intensity has to be visible, or the detector cannot learn it."""
        bright = _render(city_map).pixels.copy()
        city_map.fires[0].intensity = 0.1
        assert not np.array_equal(bright, _render(city_map).pixels)

    def test_a_fire_off_camera_is_not_labelled(self, city_map: CityMap) -> None:
        view = CameraView(
            camera_id="ne",
            origin=Position(4, 0),
            width_tiles=2,
            height_tiles=2,
            tile_size_px=8,
        )
        assert _labels(_render(city_map, view), EntityKind.FIRE) == []


class TestObstacleLabels:
    def test_map_obstacles_are_labelled(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        obstacles = _labels(frame, EntityKind.OBSTACLE)
        assert [frame.world_position_of(o) for o in obstacles] == [Position(5, 3)]

    def test_a_collapse_appears_in_the_next_capture(self, city_map: CityMap) -> None:
        """The camera has to see the world change, or replanning is blind."""
        before = len(_labels(_render(city_map), EntityKind.OBSTACLE))
        city_map.terrain[Position(3, 0)] = TerrainType.COLLAPSED_BUILDING
        city_map.obstacles.append(
            Obstacle(
                obstacle_id="new",
                position=Position(3, 0),
                kind=TerrainType.COLLAPSED_BUILDING,
            )
        )
        assert len(_labels(_render(city_map), EntityKind.OBSTACLE)) == before + 1


class TestVehiclePainting:
    def test_the_vehicle_is_painted(self, city_map: CityMap) -> None:
        frame = _render(city_map)
        # Vehicle at (0, 3) -> tile pixels x 0..8, y 24..32; marker is inset.
        assert tuple(frame.pixels[28, 4]) == _VEHICLE.as_tuple()

    def test_the_vehicle_is_not_a_detection_target(self, city_map: CityMap) -> None:
        """It is the ego agent, not something the detector should hunt for."""
        assert _labels(_render(city_map), EntityKind.VEHICLE) == []


class TestTextureAndDeterminism:
    def test_zero_jitter_gives_flat_colour(self, city_map: CityMap) -> None:
        frame = _render(city_map, jitter=0)
        road_row = frame.pixels[12, 16:24]
        assert len(np.unique(road_row.reshape(-1, 3), axis=0)) == 1

    def test_jitter_adds_per_pixel_grain(self, city_map: CityMap) -> None:
        frame = _render(city_map, jitter=20)
        road_row = frame.pixels[12, 16:24]
        assert len(np.unique(road_row.reshape(-1, 3), axis=0)) > 1

    def test_the_same_scene_renders_identically_twice(self, city_map: CityMap) -> None:
        rasterizer = FrameRasterizer(_palette(jitter=15))
        first = rasterizer.render(city_map, _full_view())
        second = rasterizer.render(city_map, _full_view())
        assert np.array_equal(first.pixels, second.pixels)

    def test_a_fresh_rasterizer_reproduces_the_same_grain(self, city_map: CityMap) -> None:
        """Texture is seeded by camera id, not by process state."""
        first = FrameRasterizer(_palette(jitter=15)).render(city_map, _full_view())
        second = FrameRasterizer(_palette(jitter=15)).render(city_map, _full_view())
        assert np.array_equal(first.pixels, second.pixels)

    def test_different_cameras_get_different_grain(self, city_map: CityMap) -> None:
        rasterizer = FrameRasterizer(_palette(jitter=20))
        left = rasterizer.render(city_map, _full_view())
        right_view = CameraView(
            camera_id="cam_other",
            origin=Position(0, 0),
            width_tiles=6,
            height_tiles=4,
            tile_size_px=8,
        )
        right = rasterizer.render(city_map, right_view)
        assert not np.array_equal(left.pixels, right.pixels)

    def test_a_changed_world_produces_a_changed_frame(self, city_map: CityMap) -> None:
        before = _render(city_map, jitter=10).pixels.copy()
        city_map.victims[0].status = VictimStatus.RESCUED
        assert not np.array_equal(before, _render(city_map, jitter=10).pixels)
