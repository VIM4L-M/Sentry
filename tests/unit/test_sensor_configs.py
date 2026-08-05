"""Unit tests for the sensor config dataclasses, loader, and palette."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sentry_ai.common.color import Color
from sentry_ai.common.exceptions import (
    ConfigurationError,
    ConfigValidationError,
)
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import (
    CameraSpec,
    DegradationConfig,
    OnboardCameraConfig,
    SensorConfig,
)
from sentry_ai.domain.enums import TerrainType
from sentry_ai.sensors.palette import SensorPalette

_MINIMAL: dict[str, Any] = {
    "tile_size_px": 8,
    "cameras": [{"id": "a", "origin": [0, 0], "size": [4, 4]}],
    "palette": {
        "void": [0, 0, 0],
        "fire_core": [255, 200, 0],
        "fire_edge": [180, 60, 0],
        "victim": [240, 0, 120],
        "debris": [60, 50, 40],
        "vehicle": [0, 128, 255],
        "terrain": {terrain.value: [90, 90, 90] for terrain in TerrainType},
    },
}


def _write(tmp_path: Path, data: dict[str, Any]) -> ConfigLoader:
    (tmp_path / "sensors.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return ConfigLoader(project_root=tmp_path)


class TestCameraSpecValidation:
    def test_an_empty_id_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="id must not be empty"):
            CameraSpec(camera_id=" ", origin_x=0, origin_y=0, width_tiles=1, height_tiles=1)

    def test_a_negative_origin_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="origin"):
            CameraSpec(camera_id="a", origin_x=-1, origin_y=0, width_tiles=1, height_tiles=1)

    @pytest.mark.parametrize("field", ["width_tiles", "height_tiles"])
    def test_a_non_positive_size_is_rejected(self, field: str) -> None:
        kwargs: dict[str, Any] = {
            "camera_id": "a",
            "origin_x": 0,
            "origin_y": 0,
            "width_tiles": 1,
            "height_tiles": 1,
            field: 0,
        }
        with pytest.raises(ConfigValidationError):
            CameraSpec(**kwargs)


class TestSensorConfigValidation:
    def test_a_rig_with_no_cameras_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="at least one camera"):
            SensorConfig(cameras=())

    def test_a_non_positive_tile_size_is_rejected(self) -> None:
        spec = CameraSpec(camera_id="a", origin_x=0, origin_y=0, width_tiles=1, height_tiles=1)
        with pytest.raises(ConfigValidationError, match="tile_size_px"):
            SensorConfig(tile_size_px=0, cameras=(spec,))

    def test_onboard_rejects_a_non_positive_span(self) -> None:
        with pytest.raises(ConfigValidationError, match="span_tiles"):
            OnboardCameraConfig(span_tiles=0)

    @pytest.mark.parametrize("density", [-0.1, 1.1])
    def test_degradation_rejects_out_of_range_smoke(self, density: float) -> None:
        with pytest.raises(ConfigValidationError, match="smoke_density"):
            DegradationConfig(smoke_density=density)

    def test_degradation_rejects_an_out_of_range_grey(self) -> None:
        with pytest.raises(ConfigValidationError, match="smoke_grey"):
            DegradationConfig(smoke_grey=256)

    def test_degradation_rejects_negative_blur(self) -> None:
        with pytest.raises(ConfigValidationError, match="blur_radius"):
            DegradationConfig(blur_radius=-1)

    def test_degradation_rejects_negative_noise(self) -> None:
        with pytest.raises(ConfigValidationError, match="noise_std"):
            DegradationConfig(noise_std=-1.0)


class TestLoadingSensorConfig:
    def test_a_minimal_file_uses_dataclass_defaults(self, tmp_path: Path) -> None:
        config = _write(tmp_path, _MINIMAL).load_sensor_config("sensors.yaml")
        assert config.onboard == OnboardCameraConfig()
        assert config.degradation == DegradationConfig()

    def test_cameras_are_parsed_in_order(self, tmp_path: Path) -> None:
        data = dict(_MINIMAL)
        data["cameras"] = [
            {"id": "first", "origin": [0, 0], "size": [4, 4]},
            {"id": "second", "origin": [4, 0], "size": [6, 2]},
        ]
        config = _write(tmp_path, data).load_sensor_config("sensors.yaml")
        assert [camera.camera_id for camera in config.cameras] == ["first", "second"]
        assert config.cameras[1].origin_x == 4
        assert (config.cameras[1].width_tiles, config.cameras[1].height_tiles) == (6, 2)

    def test_a_camera_without_an_id_gets_a_positional_one(self, tmp_path: Path) -> None:
        data = dict(_MINIMAL)
        data["cameras"] = [{"origin": [0, 0], "size": [2, 2]}]
        config = _write(tmp_path, data).load_sensor_config("sensors.yaml")
        assert config.cameras[0].camera_id == "cctv_0"

    def test_a_malformed_camera_list_is_rejected(self, tmp_path: Path) -> None:
        data = dict(_MINIMAL)
        data["cameras"] = {"not": "a list"}
        with pytest.raises(ConfigurationError, match="must be a list"):
            _write(tmp_path, data).load_sensor_config("sensors.yaml")

    def test_a_malformed_origin_is_rejected(self, tmp_path: Path) -> None:
        data = dict(_MINIMAL)
        data["cameras"] = [{"id": "a", "origin": [1, 2, 3], "size": [2, 2]}]
        with pytest.raises(ConfigurationError, match="2-element"):
            _write(tmp_path, data).load_sensor_config("sensors.yaml")

    def test_a_wrongly_typed_value_is_rejected(self, tmp_path: Path) -> None:
        data = dict(_MINIMAL)
        data["tile_size_px"] = "eight"
        with pytest.raises(ConfigurationError, match="must be an integer"):
            _write(tmp_path, data).load_sensor_config("sensors.yaml")


class TestLoadingPalette:
    def test_a_complete_palette_loads(self, tmp_path: Path) -> None:
        palette = SensorPalette.from_config(_write(tmp_path, _MINIMAL), "sensors.yaml")
        assert set(palette.terrain) == set(TerrainType)
        assert palette.victim.as_tuple() == (240, 0, 120)

    def test_a_missing_palette_section_is_rejected(self, tmp_path: Path) -> None:
        data = {key: value for key, value in _MINIMAL.items() if key != "palette"}
        with pytest.raises(ConfigValidationError, match="'palette' mapping"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")

    def test_a_missing_terrain_colour_is_rejected(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {**_MINIMAL, "palette": dict(_MINIMAL["palette"])}
        data["palette"]["terrain"] = {
            terrain.value: [1, 2, 3]
            for terrain in TerrainType
            if terrain is not TerrainType.ROAD
        }
        with pytest.raises(ConfigValidationError, match="terrain.road"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")

    def test_a_missing_entity_colour_is_rejected(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {**_MINIMAL, "palette": dict(_MINIMAL["palette"])}
        del data["palette"]["victim"]
        with pytest.raises(ConfigValidationError, match="victim"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")

    def test_a_malformed_colour_is_rejected(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {**_MINIMAL, "palette": dict(_MINIMAL["palette"])}
        data["palette"]["victim"] = [1, 2]
        with pytest.raises(ConfigValidationError, match="3-element"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")

    def test_an_out_of_range_channel_is_rejected(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {**_MINIMAL, "palette": dict(_MINIMAL["palette"])}
        data["palette"]["victim"] = [300, 0, 0]
        with pytest.raises(ConfigValidationError, match="0-255"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")

    def test_a_palette_built_without_every_terrain_is_rejected(self) -> None:
        """Guards direct construction, which the YAML path cannot reach."""
        colour = Color(1, 2, 3)
        with pytest.raises(ConfigValidationError, match="terrain is missing"):
            SensorPalette(
                terrain={TerrainType.ROAD: colour},
                void=colour,
                fire_core=colour,
                fire_edge=colour,
                victim=colour,
                debris=colour,
                vehicle=colour,
                texture_jitter=0,
            )

    def test_excessive_jitter_is_rejected(self, tmp_path: Path) -> None:
        """Grain that swamps the class colours would make the labels unlearnable."""
        data: dict[str, Any] = {**_MINIMAL, "palette": dict(_MINIMAL["palette"])}
        data["palette"]["texture_jitter"] = 200
        with pytest.raises(ConfigValidationError, match="texture_jitter"):
            SensorPalette.from_config(_write(tmp_path, data), "sensors.yaml")


class TestShippedSensorConfig:
    def test_the_app_config_points_at_the_sensors_file(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        app_config = loader.load_app_config("configs/app.yaml")
        assert app_config.sensor_config_path is not None
        assert app_config.sensor_config_path.is_file()

    def test_the_shipped_config_loads_and_validates(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        config = loader.load_sensor_config("configs/sensors.yaml")
        assert len(config.cameras) == 4
        assert {camera.camera_id for camera in config.cameras} == {
            "cctv_nw",
            "cctv_ne",
            "cctv_sw",
            "cctv_se",
        }
