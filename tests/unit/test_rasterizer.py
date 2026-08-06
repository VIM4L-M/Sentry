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
from sentry_ai.config.schema import MarkerScaleConfig
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


class TestVictimInRubble:
    """A victim pinned under debris is one tile with one answer: VICTIM.

    Annotating both left the detector an unwinnable choice — a 13 px
    obstacle box containing an 8 px victim box, twelve obstacles to every
    victim — and it always answered OBSTACLE, which is an impassable code.
    A trapped victim became a wall the planner routed around. Measured
    victim recall was precisely the share of victims *not* pinned in
    rubble, so these tests guard a real regression rather than a theory.
    """

    def _pinned(self, city_map: CityMap) -> CityMap:
        """Bury the victim at (4, 2) under debris."""
        city_map.terrain[Position(4, 2)] = TerrainType.RUBBLE
        city_map.obstacles.append(
            Obstacle(obstacle_id="rubble", position=Position(4, 2), kind=TerrainType.RUBBLE)
        )
        return city_map

    def test_the_victim_is_still_labelled(self, city_map: CityMap) -> None:
        frame = _render(self._pinned(city_map))
        victims = _labels(frame, EntityKind.VICTIM)
        assert len(victims) == 1
        assert frame.world_position_of(victims[0]) == Position(4, 2)

    def test_the_debris_under_them_is_not_labelled(self, city_map: CityMap) -> None:
        frame = _render(self._pinned(city_map))
        positions = [frame.world_position_of(o) for o in _labels(frame, EntityKind.OBSTACLE)]
        assert Position(4, 2) not in positions

    def test_debris_elsewhere_is_still_labelled(self, city_map: CityMap) -> None:
        """Only the victim's own tile is affected."""
        frame = _render(self._pinned(city_map))
        positions = [frame.world_position_of(o) for o in _labels(frame, EntityKind.OBSTACLE)]
        assert Position(5, 3) in positions

    def test_the_debris_is_still_painted(self, city_map: CityMap) -> None:
        """The rubble is physically there; a brown ring around a pink core
        is what 'trapped in rubble' looks like from above."""
        frame = _render(self._pinned(city_map))
        # Tile (4, 2) -> pixels x 32..40, y 16..24. Debris fills 0.8 of it,
        # the victim only 0.5, so the ring between them stays debris.
        assert tuple(frame.pixels[17, 33]) == _DEBRIS.as_tuple()
        assert tuple(frame.pixels[20, 36]) == _VICTIM.as_tuple()

    def test_no_victim_box_is_nested_inside_an_obstacle_box(self, city_map: CityMap) -> None:
        """The exact condition that cost 45.4% of victim recall."""
        frame = _render(self._pinned(city_map))
        obstacles = [o.bbox for o in _labels(frame, EntityKind.OBSTACLE)]
        for victim in _labels(frame, EntityKind.VICTIM):
            centre_x = (victim.bbox.x_min + victim.bbox.x_max) / 2
            centre_y = (victim.bbox.y_min + victim.bbox.y_max) / 2
            assert not any(
                box.x_min <= centre_x <= box.x_max and box.y_min <= centre_y <= box.y_max
                for box in obstacles
            )

    def test_a_rescued_victim_frees_the_debris_label(self, city_map: CityMap) -> None:
        """Precedence lasts exactly as long as someone is trapped there."""
        pinned = self._pinned(city_map)
        pinned.victims[0].status = VictimStatus.RESCUED
        frame = _render(pinned)
        positions = [frame.world_position_of(o) for o in _labels(frame, EntityKind.OBSTACLE)]
        assert Position(4, 2) in positions


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


class TestMarkerScales:
    """Marker size is the project's main control over small-object recall.

    A victim is the smallest annotated class and sits near the resolution
    floor of YOLOv8's finest detection head, so these pin down that the
    configured fraction really does reach the pixels — an experiment that
    silently ignored its own setting would be worse than no experiment.
    """

    def _victim_size(self, city_map: CityMap, scale: float) -> tuple[int, int]:
        rasterizer = FrameRasterizer(_palette(), markers=MarkerScaleConfig(victim=scale))
        frame = rasterizer.render(city_map, _full_view(tile_size_px=16))
        box = _labels(frame, EntityKind.VICTIM)[0].bbox
        return (box.x_max - box.x_min, box.y_max - box.y_min)

    def test_the_default_victim_fills_half_its_tile(self, city_map: CityMap) -> None:
        assert self._victim_size(city_map, MarkerScaleConfig.victim) == (8, 8)

    def test_a_larger_scale_paints_a_larger_victim(self, city_map: CityMap) -> None:
        assert self._victim_size(city_map, 0.75) == (12, 12)

    def test_the_label_matches_the_painted_pixels(self, city_map: CityMap) -> None:
        """Ground truth must describe what was actually drawn, at any scale."""
        rasterizer = FrameRasterizer(_palette(), markers=MarkerScaleConfig(victim=0.75))
        frame = rasterizer.render(city_map, _full_view(tile_size_px=16))
        box = _labels(frame, EntityKind.VICTIM)[0].bbox

        painted = np.all(frame.pixels == np.array(_VICTIM.as_tuple()), axis=-1)
        rows, cols = np.nonzero(painted)
        assert (cols.min(), cols.max() + 1) == (box.x_min, box.x_max)
        assert (rows.min(), rows.max() + 1) == (box.y_min, box.y_max)
