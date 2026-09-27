"""Unit tests for the Phase 4 config: AutoencoderTrainingConfig and DegradationConfig.scaled.

``scaled`` is small, but the denoiser's whole training distribution and its
stress test are built from it, so its edges — saturation, rounding, zero —
are pinned down here rather than discovered in a training curve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigurationError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AutoencoderTrainingConfig, DegradationConfig


class TestShippedConfig:
    def test_the_shipped_file_loads(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        config = loader.load_autoencoder_config("configs/training/autoencoder.yaml")

        assert config.dataset_dir == project_root / "data/synthetic"
        assert config.runs_dir == project_root / "models/autoencoder"
        assert config.skip_connections is True
        assert config.loss == "l1"

    def test_the_crop_fits_the_smallest_camera(self, project_root: Path) -> None:
        """The 144 px onboard view is the smallest frame the dataset holds."""
        loader = ConfigLoader(project_root=project_root)
        config = loader.load_autoencoder_config("configs/training/autoencoder.yaml")
        sensors = loader.load_sensor_config("configs/sensors.yaml")

        onboard_px = sensors.onboard.span_tiles * sensors.onboard.tile_size_px
        assert config.crop_size <= onboard_px

    def test_omitted_keys_fall_back_to_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "ae.yaml").write_text("epochs: 3\n", encoding="utf-8")
        config = ConfigLoader(project_root=tmp_path).load_autoencoder_config("ae.yaml")

        assert config.epochs == 3
        assert config.base_channels == AutoencoderTrainingConfig().base_channels

    def test_a_wrongly_typed_value_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "ae.yaml").write_text("skip_connections: 'yes'\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader(project_root=tmp_path).load_autoencoder_config("ae.yaml")


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"base_channels": 0},
            {"depth": 0},
            {"epochs": 0},
            {"batch_size": -1},
            {"patience": -1},
            {"num_workers": -1},
            {"learning_rate": 0.0},
            {"weight_decay": -0.1},
            {"loss": "huber"},
            {"severity_min": 2.0, "severity_max": 1.0},
            {"severity_min": -0.5},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigValidationError):
            AutoencoderTrainingConfig(**overrides)  # type: ignore[arg-type]

    def test_the_crop_must_survive_every_downsampling(self) -> None:
        with pytest.raises(ConfigValidationError, match="multiple of 2"):
            AutoencoderTrainingConfig(crop_size=100, depth=3)

    def test_a_crop_that_divides_is_accepted(self) -> None:
        assert AutoencoderTrainingConfig(crop_size=96, depth=5).crop_size == 96


class TestScaledDegradation:
    def test_severity_one_is_unchanged(self) -> None:
        config = DegradationConfig()
        assert config.scaled(1.0) == config

    def test_severity_zero_is_a_clean_frame(self) -> None:
        scaled = DegradationConfig().scaled(0.0)
        assert scaled.smoke_density == 0.0
        assert scaled.blur_radius == 0
        assert scaled.noise_std == 0.0

    def test_strengths_scale_linearly(self) -> None:
        scaled = DegradationConfig(smoke_density=0.2, noise_std=8.0).scaled(2.0)
        assert scaled.smoke_density == pytest.approx(0.4)
        assert scaled.noise_std == pytest.approx(16.0)

    def test_smoke_saturates_at_fully_opaque(self) -> None:
        assert DegradationConfig(smoke_density=0.45).scaled(3.0).smoke_density == 1.0

    def test_blur_rounds_half_up(self) -> None:
        """Python's round() would send 1.5 and 2.5 both to 2; this must not."""
        config = DegradationConfig(blur_radius=1)
        assert config.scaled(1.5).blur_radius == 2
        assert config.scaled(2.5).blur_radius == 3
        assert config.scaled(0.4).blur_radius == 0

    def test_smoke_colour_never_scales(self) -> None:
        assert DegradationConfig(smoke_grey=150).scaled(2.0).smoke_grey == 150

    def test_negative_severity_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            DegradationConfig().scaled(-1.0)
