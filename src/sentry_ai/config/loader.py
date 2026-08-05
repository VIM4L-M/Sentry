"""Loads YAML configuration files into validated dataclasses.

``ConfigLoader`` is constructed with an explicit project root (dependency
injection — never ``Path.cwd()`` or a hardcoded absolute path) so it works
identically whether invoked from a script, a test, or a Streamlit app
launched from an arbitrary working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sentry_ai.common.exceptions import AssetNotFoundError, ConfigurationError
from sentry_ai.common.types import PathLike
from sentry_ai.config.schema import AppConfig, RenderConfig

_DEFAULT_APP_CONFIG = "configs/app.yaml"


class ConfigLoader:
    """Resolves and parses YAML config files relative to a project root."""

    def __init__(self, project_root: PathLike) -> None:
        """Create a loader rooted at ``project_root``.

        Args:
            project_root: Directory that relative config paths (in YAML
                files and in method calls) are resolved against. Typically
                the repository root.
        """
        self._project_root = Path(project_root).resolve()

    @property
    def project_root(self) -> Path:
        """The resolved, absolute project root this loader is rooted at."""
        return self._project_root

    def resolve(self, relative_path: PathLike) -> Path:
        """Resolve ``relative_path`` against the project root.

        An already-absolute path is returned unchanged, so callers can pass
        either project-relative or fully-qualified paths uniformly.
        """
        path = Path(relative_path)
        return path if path.is_absolute() else (self._project_root / path)

    def load_yaml(self, relative_path: PathLike) -> dict[str, Any]:
        """Read and parse a YAML file, returning its top-level mapping.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If the file is not valid YAML, or its
                top-level document is not a mapping.
        """
        path = self.resolve(relative_path)
        if not path.is_file():
            raise AssetNotFoundError(f"Config file not found: {path}")

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Config file is not valid YAML: {path}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file must contain a YAML mapping: {path}")
        return data

    def load_app_config(self, relative_path: PathLike = _DEFAULT_APP_CONFIG) -> AppConfig:
        """Load and validate the root application config.

        Args:
            relative_path: Path to ``app.yaml``, relative to the project
                root unless already absolute. Defaults to the conventional
                ``configs/app.yaml``.

        Raises:
            AssetNotFoundError: If the app config or a file it references
                does not exist.
            ConfigurationError: If a required key is missing or malformed.
        """
        data = self.load_yaml(relative_path)

        for key in ("logging_config", "map_config", "render"):
            if key not in data:
                raise ConfigurationError(f"app config missing required key: '{key}'")

        render_data = data["render"]
        if not isinstance(render_data, dict):
            raise ConfigurationError("app config key 'render' must be a mapping")

        try:
            render = RenderConfig(
                window_title=render_data["window_title"],
                tile_size_px=render_data["tile_size_px"],
                target_fps=render_data["target_fps"],
                palette_config_path=self.resolve(render_data["palette_config"]),
            )
        except KeyError as exc:
            raise ConfigurationError(f"app config 'render' section missing key: {exc}") from exc

        return AppConfig(
            logging_config_path=self.resolve(data["logging_config"]),
            map_config_path=self.resolve(data["map_config"]),
            render=render,
        )
