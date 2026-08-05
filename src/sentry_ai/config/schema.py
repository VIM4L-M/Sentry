"""Typed dataclasses describing the shape of ``configs/app.yaml``.

Each dataclass validates itself in ``__post_init__`` so an invalid config
fails fast, at load time, with a clear message — never deep inside a
render loop or training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentry_ai.common.exceptions import ConfigValidationError


@dataclass(frozen=True)
class RenderConfig:
    """Window and drawing tunables for the Pygame renderer.

    Attributes:
        window_title: Text shown in the OS window title bar.
        tile_size_px: Edge length, in pixels, of one map tile when drawn.
        target_fps: Frame-rate cap passed to the Pygame clock.
        palette_config_path: Absolute path to the color palette YAML file.
    """

    window_title: str
    tile_size_px: int
    target_fps: int
    palette_config_path: Path

    def __post_init__(self) -> None:
        if not self.window_title.strip():
            raise ConfigValidationError("render.window_title must not be empty")
        if self.tile_size_px <= 0:
            raise ConfigValidationError(
                f"render.tile_size_px must be positive, got {self.tile_size_px}"
            )
        if self.target_fps <= 0:
            raise ConfigValidationError(
                f"render.target_fps must be positive, got {self.target_fps}"
            )
        if not self.palette_config_path.suffix:
            raise ConfigValidationError(
                f"render.palette_config must point at a file, got {self.palette_config_path}"
            )


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration, composing the other config files.

    Attributes:
        logging_config_path: Absolute path to the ``dictConfig`` YAML file.
        map_config_path: Absolute path to the disaster-city map YAML file.
        render: Window/drawing tunables.
    """

    logging_config_path: Path
    map_config_path: Path
    render: RenderConfig

    def __post_init__(self) -> None:
        if not self.logging_config_path.suffix:
            raise ConfigValidationError(
                f"app.logging_config must point at a file, got {self.logging_config_path}"
            )
        if not self.map_config_path.suffix:
            raise ConfigValidationError(
                f"app.map_config must point at a file, got {self.map_config_path}"
            )
