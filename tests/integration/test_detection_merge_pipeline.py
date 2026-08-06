"""The merger against the real camera rig and the real city (Phase 3.2).

The unit tests drive :class:`DetectionMerger` with hand-built boxes. These
drive it with the actual four-camera network from ``configs/sensors.yaml``
and the actual city, using the rasterizer's *ground-truth* annotations in
place of a model's predictions.

That substitution is the point. It isolates the merge step completely: if a
victim standing in the two-tile overlap between the west and east cameras
comes back as two victims here, the fault is in the merging, not in the
detector — because a perfect detector is exactly what is being fed in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import SensorConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.perception.merger import CameraObservation, DetectionMerger, WorldDetection
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig

_SENSOR_CONFIG = "configs/sensors.yaml"


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


@pytest.fixture
def sensor_config(loader: ConfigLoader) -> SensorConfig:
    return loader.load_sensor_config(_SENSOR_CONFIG)


@pytest.fixture
def city_map(loader: ConfigLoader) -> CityMap:
    return CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))


@pytest.fixture
def rig(loader: ConfigLoader, sensor_config: SensorConfig) -> SensorRig:
    return SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, _SENSOR_CONFIG))


def _as_observations(frames: list[CameraFrame]) -> list[CameraObservation]:
    """Feed each frame's ground truth in as if a perfect detector produced it."""
    return [CameraObservation(frame.view, frame.annotations) for frame in frames]


def _of(found: list[WorldDetection], kind: EntityKind) -> list[WorldDetection]:
    return [one for one in found if one.label is kind]


class TestVictimsAcrossTheRealNetwork:
    def test_every_trapped_victim_is_reported_exactly_once(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        trapped = {
            victim.position
            for victim in city_map.victims
            if victim.status is VictimStatus.TRAPPED
        }
        found = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))

        victims = _of(found, EntityKind.VICTIM)
        assert {one.position for one in victims} == trapped
        assert len(victims) == len(trapped)

    def test_a_victim_in_the_camera_overlap_is_not_double_counted(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """The strongest case for merging existing at all.

        The two camera columns overlap by two tiles precisely so this
        happens. Without merging the command center would dispatch twice.
        """
        overlap = [
            tile
            for tile in (
                Position(x, y)
                for x in range(city_map.width)
                for y in range(city_map.height)
            )
            if sum(view.covers(tile) for view in rig.cctv_views) > 1
        ]
        assert overlap, "the shipped camera layout no longer overlaps"

        target = overlap[len(overlap) // 2]
        city_map.victims[0].position = target
        city_map.victims[0].status = VictimStatus.TRAPPED

        found = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))
        at_target = [one for one in _of(found, EntityKind.VICTIM) if target in one.tiles]

        assert len(at_target) == 1
        assert at_target[0].corroborated
        assert len(at_target[0].camera_ids) > 1

    def test_a_rescued_victim_disappears_from_the_merged_view(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        before = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))
        for victim in city_map.victims:
            victim.status = VictimStatus.RESCUED
        after = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))

        assert _of(before, EntityKind.VICTIM)
        assert _of(after, EntityKind.VICTIM) == []


class TestHazardsAcrossTheRealNetwork:
    def test_every_merged_tile_is_on_the_map(self, rig: SensorRig, city_map: CityMap) -> None:
        """Projection must never invent a tile outside the city."""
        found = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))
        assert found
        for one in found:
            for tile in one.tiles:
                assert city_map.in_bounds(tile)

    def test_a_fire_straddling_two_cameras_is_one_fire(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        seam_x = max(view.origin.x for view in rig.cctv_views)
        city_map.fires[0].position = Position(seam_x, 5)
        city_map.fires[0].radius = 3
        city_map.fires[0].intensity = 1.0

        found = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))
        touching = [
            one for one in _of(found, EntityKind.FIRE) if Position(seam_x, 5) in one.tiles
        ]

        assert len(touching) == 1
        assert touching[0].corroborated

    def test_merging_reduces_the_detection_count(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """If nothing ever merged, the whole step would be dead weight."""
        observations = _as_observations(rig.capture_cctv(city_map))
        raw = sum(len(observation.detections) for observation in observations)
        assert len(DetectionMerger().merge(observations)) < raw


class TestGroundTruthAgreement:
    def test_no_two_merged_victims_share_a_tile(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        found = _of(
            DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map))),
            EntityKind.VICTIM,
        )
        seen: set[Position] = set()
        for one in found:
            assert seen.isdisjoint(one.tiles)
            seen.update(one.tiles)

    def test_the_onboard_camera_can_join_the_merge(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """The vehicle's own view is another sensor, not a special case."""
        cctv_only = DetectionMerger().merge(_as_observations(rig.capture_cctv(city_map)))
        with_onboard = DetectionMerger().merge(_as_observations(rig.capture_all(city_map)))

        victims = {one.position for one in _of(with_onboard, EntityKind.VICTIM)}
        assert {one.position for one in _of(cctv_only, EntityKind.VICTIM)} == victims
