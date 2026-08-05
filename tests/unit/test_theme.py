"""Unit tests for sentry_ai.rendering.theme."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.enums import EntityKind, TerrainType
from sentry_ai.rendering.theme import Color, Theme

_FULL_PALETTE = """
background: [10, 10, 10]
terrain:
  open_ground: [1, 1, 1]
  road: [2, 2, 2]
  building: [3, 3, 3]
  collapsed_building: [4, 4, 4]
  rubble: [5, 5, 5]
  tree: [6, 6, 6]
  safe_zone: [7, 7, 7]
  blocked_road: [8, 8, 8]
entities:
  vehicle: [9, 9, 9]
  victim: [10, 10, 10]
  fire: [11, 11, 11]
  smoke: [12, 12, 12]
  obstacle: [13, 13, 13]
"""


class TestColor:
    @pytest.mark.parametrize("channel", [-1, 256])
    def test_rejects_out_of_range_channel(self, channel: int) -> None:
        with pytest.raises(ConfigValidationError):
            Color(r=channel, g=0, b=0)

    def test_as_tuple(self) -> None:
        assert Color(1, 2, 3).as_tuple() == (1, 2, 3)


class TestThemeFromConfig:
    def test_loads_full_palette(self, tmp_path: Path) -> None:
        palette_file = tmp_path / "render.yaml"
        palette_file.write_text(_FULL_PALETTE, encoding="utf-8")
        loader = ConfigLoader(project_root=tmp_path)

        theme = Theme.from_config(loader, "render.yaml")

        assert theme.background.as_tuple() == (10, 10, 10)
        assert theme.terrain_colors[TerrainType.BUILDING].as_tuple() == (3, 3, 3)
        assert theme.entity_colors[EntityKind.FIRE].as_tuple() == (11, 11, 11)

    def test_missing_terrain_color_raises(self, tmp_path: Path) -> None:
        incomplete = _FULL_PALETTE.replace("  building: [3, 3, 3]\n", "")
        palette_file = tmp_path / "render.yaml"
        palette_file.write_text(incomplete, encoding="utf-8")
        loader = ConfigLoader(project_root=tmp_path)

        with pytest.raises(ConfigValidationError, match="terrain.building"):
            Theme.from_config(loader, "render.yaml")

    def test_missing_entity_color_raises(self, tmp_path: Path) -> None:
        incomplete = _FULL_PALETTE.replace("  fire: [11, 11, 11]\n", "")
        palette_file = tmp_path / "render.yaml"
        palette_file.write_text(incomplete, encoding="utf-8")
        loader = ConfigLoader(project_root=tmp_path)

        with pytest.raises(ConfigValidationError, match="entities.fire"):
            Theme.from_config(loader, "render.yaml")

    def test_missing_background_raises(self, tmp_path: Path) -> None:
        incomplete = _FULL_PALETTE.replace("background: [10, 10, 10]\n", "")
        palette_file = tmp_path / "render.yaml"
        palette_file.write_text(incomplete, encoding="utf-8")
        loader = ConfigLoader(project_root=tmp_path)

        with pytest.raises(ConfigValidationError, match="background"):
            Theme.from_config(loader, "render.yaml")


def test_real_render_config_loads(project_root: Path) -> None:
    """The actual configs/render.yaml shipped in the repo must be valid."""
    loader = ConfigLoader(project_root=project_root)
    theme = Theme.from_config(loader, "configs/render.yaml")
    assert len(theme.terrain_colors) == len(TerrainType)
    assert len(theme.entity_colors) == len(EntityKind)
