"""Unit tests for sentry_ai.config.loader.ConfigLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import AssetNotFoundError, ConfigurationError
from sentry_ai.config.loader import ConfigLoader


def test_resolve_relative_path_joins_project_root(tmp_path: Path) -> None:
    loader = ConfigLoader(project_root=tmp_path)
    assert loader.resolve("configs/app.yaml") == tmp_path / "configs" / "app.yaml"


def test_resolve_absolute_path_is_unchanged(tmp_path: Path) -> None:
    loader = ConfigLoader(project_root=tmp_path)
    absolute = tmp_path / "elsewhere" / "app.yaml"
    assert loader.resolve(absolute) == absolute


def test_load_yaml_missing_file_raises_asset_not_found(tmp_path: Path) -> None:
    loader = ConfigLoader(project_root=tmp_path)
    with pytest.raises(AssetNotFoundError):
        loader.load_yaml("does_not_exist.yaml")


def test_load_yaml_invalid_syntax_raises_configuration_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("key: [unclosed", encoding="utf-8")
    loader = ConfigLoader(project_root=tmp_path)
    with pytest.raises(ConfigurationError):
        loader.load_yaml(bad_file)


def test_load_yaml_non_mapping_raises_configuration_error(tmp_path: Path) -> None:
    list_file = tmp_path / "list.yaml"
    list_file.write_text("- one\n- two\n", encoding="utf-8")
    loader = ConfigLoader(project_root=tmp_path)
    with pytest.raises(ConfigurationError):
        loader.load_yaml(list_file)


def test_load_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    loader = ConfigLoader(project_root=tmp_path)
    assert loader.load_yaml(empty_file) == {}


def test_load_app_config_success(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    app_yaml = tmp_path / "configs" / "app.yaml"
    app_yaml.write_text(
        """
logging_config: "configs/logging.yaml"
map_config: "configs/maps/city_default.yaml"
render:
  window_title: "Test Window"
  tile_size_px: 16
  target_fps: 30
  palette_config: "configs/render.yaml"
""",
        encoding="utf-8",
    )

    loader = ConfigLoader(project_root=tmp_path)
    app_config = loader.load_app_config("configs/app.yaml")

    assert app_config.logging_config_path == tmp_path / "configs" / "logging.yaml"
    assert app_config.map_config_path == tmp_path / "configs" / "maps" / "city_default.yaml"
    assert app_config.render.window_title == "Test Window"
    assert app_config.render.tile_size_px == 16
    assert app_config.render.target_fps == 30
    assert app_config.render.palette_config_path == tmp_path / "configs" / "render.yaml"


def test_load_app_config_missing_top_level_key_raises(tmp_path: Path) -> None:
    app_yaml = tmp_path / "app.yaml"
    app_yaml.write_text('logging_config: "configs/logging.yaml"\n', encoding="utf-8")
    loader = ConfigLoader(project_root=tmp_path)
    with pytest.raises(ConfigurationError):
        loader.load_app_config(app_yaml)


def test_load_app_config_missing_render_key_raises(tmp_path: Path) -> None:
    app_yaml = tmp_path / "app.yaml"
    app_yaml.write_text(
        """
logging_config: "configs/logging.yaml"
map_config: "configs/maps/city_default.yaml"
render:
  window_title: "Test"
  tile_size_px: 16
""",
        encoding="utf-8",
    )
    loader = ConfigLoader(project_root=tmp_path)
    with pytest.raises(ConfigurationError):
        loader.load_app_config(app_yaml)


def test_real_app_config_loads(project_root: Path) -> None:
    """The actual configs/app.yaml shipped in the repo must load cleanly."""
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")
    assert app_config.map_config_path.is_file()
    assert app_config.logging_config_path.is_file()
    assert app_config.render.palette_config_path.is_file()
