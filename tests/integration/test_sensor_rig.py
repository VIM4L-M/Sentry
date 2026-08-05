"""Integration tests for the sensor rig against the repo's real configs.

These pin down the properties later phases depend on and that a careless
edit to ``configs/sensors.yaml`` would silently break: that the CCTV
network sees the whole city, that every victim is captured by someone, and
that the projection from a detection back to a map tile is exact.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import SensorConfig
from sentry_ai.domain.enums import EntityKind, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.sensors.degradation import FrameDegrader
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
    return SensorRig.from_config(
        sensor_config, SensorPalette.from_config(loader, _SENSOR_CONFIG)
    )


def _detections(frames: list[CameraFrame], kind: EntityKind) -> list[tuple[int, int]]:
    """World tiles where ``kind`` was seen, across every frame given."""
    return [
        frame.world_position_of(detection).as_tuple()
        for frame in frames
        for detection in frame.annotations
        if detection.label is kind
    ]


class TestRigValidation:
    def test_duplicate_camera_ids_are_rejected(
        self, loader: ConfigLoader, sensor_config: SensorConfig
    ) -> None:
        """Two cameras with one id would make a frame's provenance ambiguous."""
        palette = SensorPalette.from_config(loader, _SENSOR_CONFIG)
        clashing = replace(
            sensor_config,
            cameras=(sensor_config.cameras[0], sensor_config.cameras[0]),
        )
        with pytest.raises(ConfigValidationError, match="must be unique"):
            SensorRig.from_config(clashing, palette)

    def test_the_onboard_camera_may_not_reuse_a_cctv_id(
        self, loader: ConfigLoader, sensor_config: SensorConfig
    ) -> None:
        palette = SensorPalette.from_config(loader, _SENSOR_CONFIG)
        clashing = replace(
            sensor_config,
            onboard=replace(sensor_config.onboard, camera_id=sensor_config.cameras[0].camera_id),
        )
        with pytest.raises(ConfigValidationError, match="must be unique"):
            SensorRig.from_config(clashing, palette)


class TestCoverage:
    def test_the_cctv_network_sees_the_whole_city(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """A blind spot is invisible downstream — uncovered tiles just go stale."""
        assert rig.coverage(city_map) == pytest.approx(1.0)

    def test_the_cameras_overlap_rather_than_butting_up_exactly(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """Overlap is what gives the command center duplicates to merge."""
        total = sum(len(view.world_tiles()) for view in rig.cctv_views)
        assert total > city_map.width * city_map.height

    def test_every_victim_is_seen_by_at_least_one_camera(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        seen = set(_detections(rig.capture_cctv(city_map), EntityKind.VICTIM))
        assert {victim.position.as_tuple() for victim in city_map.victims} <= seen

    def test_both_starting_fires_are_seen(self, rig: SensorRig, city_map: CityMap) -> None:
        assert len(_detections(rig.capture_cctv(city_map), EntityKind.FIRE)) >= 2


class TestFrameGeometry:
    def test_cctv_frames_are_stride_friendly(self, rig: SensorRig, city_map: CityMap) -> None:
        """YOLO wants both dimensions to be multiples of 32."""
        for frame in rig.capture_cctv(city_map):
            assert frame.width % 32 == 0
            assert frame.height % 32 == 0

    def test_every_cctv_frame_has_the_same_shape(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        shapes = {frame.pixels.shape for frame in rig.capture_cctv(city_map)}
        assert len(shapes) == 1

    def test_capture_all_appends_the_onboard_view_last(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        frames = rig.capture_all(city_map)
        assert len(frames) == len(rig.cctv_views) + 1
        assert frames[-1].camera_id == "onboard"

    def test_the_onboard_camera_always_contains_the_vehicle(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        for position in (city_map.vehicle.position, *[v.position for v in city_map.victims]):
            city_map.vehicle.position = position
            frame = rig.capture_onboard(city_map)
            assert frame.view.covers(position)


class TestProjectionAgainstTheRealMap:
    def test_every_detection_lands_on_the_tile_it_was_painted_from(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        frames = rig.capture_cctv(city_map)
        for tile in _detections(frames, EntityKind.VICTIM):
            assert any(victim.position.as_tuple() == tile for victim in city_map.victims)

    def test_a_rescued_victim_stops_being_detected(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        target = city_map.victims[0]
        before = _detections(rig.capture_cctv(city_map), EntityKind.VICTIM)
        target.status = VictimStatus.RESCUED
        after = _detections(rig.capture_cctv(city_map), EntityKind.VICTIM)
        assert target.position.as_tuple() in before
        assert target.position.as_tuple() not in after


class TestDeterminism:
    def test_capturing_an_unchanged_city_twice_is_byte_identical(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        """Otherwise 'did the world change?' becomes a question about the renderer."""
        first = rig.capture_all(city_map)
        second = rig.capture_all(city_map)
        assert all(np.array_equal(a.pixels, b.pixels) for a, b in zip(first, second, strict=True))

    def test_a_rebuilt_rig_produces_the_same_imagery(
        self, loader: ConfigLoader, sensor_config: SensorConfig, city_map: CityMap
    ) -> None:
        palette = SensorPalette.from_config(loader, _SENSOR_CONFIG)
        first = SensorRig.from_config(sensor_config, palette).capture_cctv(city_map)
        second = SensorRig.from_config(sensor_config, palette).capture_cctv(city_map)
        assert all(np.array_equal(a.pixels, b.pixels) for a, b in zip(first, second, strict=True))


class TestDegradedCapture:
    def test_a_degraded_capture_keeps_every_label(
        self, rig: SensorRig, sensor_config: SensorConfig, city_map: CityMap
    ) -> None:
        degrader = FrameDegrader(sensor_config.degradation, np.random.default_rng(0))
        for frame in rig.capture_cctv(city_map):
            assert degrader.degrade(frame).annotations == frame.annotations

    def test_yolo_labels_export_for_every_camera(
        self, rig: SensorRig, city_map: CityMap
    ) -> None:
        exported = [line for frame in rig.capture_cctv(city_map) for line in frame.to_yolo_lines()]
        assert exported
        for line in exported:
            class_id, *values = line.split()
            assert class_id in {"0", "1", "2"}
            assert all(0.0 <= float(value) <= 1.0 for value in values)
