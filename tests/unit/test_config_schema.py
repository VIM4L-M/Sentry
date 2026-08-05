"""Unit tests for sentry_ai.config.schema dataclass validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.schema import AppConfig, RenderConfig


def _render_config(**overrides: object) -> RenderConfig:
    defaults: dict[str, object] = {
        "window_title": "Test",
        "tile_size_px": 16,
        "target_fps": 30,
        "palette_config_path": Path("configs/render.yaml"),
    }
    defaults.update(overrides)
    return RenderConfig(**defaults)  # type: ignore[arg-type]


def test_render_config_accepts_valid_values() -> None:
    config = _render_config()
    assert config.tile_size_px == 16


@pytest.mark.parametrize("tile_size_px", [0, -1])
def test_render_config_rejects_non_positive_tile_size(tile_size_px: int) -> None:
    with pytest.raises(ConfigValidationError):
        _render_config(tile_size_px=tile_size_px)


@pytest.mark.parametrize("target_fps", [0, -30])
def test_render_config_rejects_non_positive_fps(target_fps: int) -> None:
    with pytest.raises(ConfigValidationError):
        _render_config(target_fps=target_fps)


def test_render_config_rejects_blank_title() -> None:
    with pytest.raises(ConfigValidationError):
        _render_config(window_title="   ")


def test_app_config_rejects_extensionless_map_path() -> None:
    with pytest.raises(ConfigValidationError):
        AppConfig(
            logging_config_path=Path("configs/logging.yaml"),
            map_config_path=Path("configs/maps/city_default"),  # no suffix
            render=_render_config(),
        )
