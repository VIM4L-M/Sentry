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
from sentry_ai.config.schema import (
    AppConfig,
    BatteryConfig,
    MissionConfig,
    PlannerConfig,
    RenderConfig,
    SimulationConfig,
    VehicleConfig,
)

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
            simulation_config_path=self._optional_path(data, "simulation_config"),
            vehicle_config_path=self._optional_path(data, "vehicle_config"),
        )

    def load_simulation_config(self, relative_path: PathLike) -> SimulationConfig:
        """Load ``simulation.yaml`` into a :class:`SimulationConfig`.

        Every key is optional: an omitted key falls back to the dataclass
        default, so a minimal config file stays legal and adding a tunable
        never breaks an existing one.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
        """
        data = self.load_yaml(relative_path)
        mission_data = _require_mapping(data.get("mission", {}), "simulation.mission")
        planner_data = _require_mapping(data.get("planner", {}), "simulation.planner")

        mission = MissionConfig(
            time_limit_seconds=_as_float(
                mission_data, "time_limit_seconds", MissionConfig.time_limit_seconds
            ),
            replan_on_blocked_route=_as_bool(
                mission_data, "replan_on_blocked_route", MissionConfig.replan_on_blocked_route
            ),
            min_battery_to_continue=_as_float(
                mission_data, "min_battery_to_continue", MissionConfig.min_battery_to_continue
            ),
        )
        planner = PlannerConfig(
            fire_risk_penalty=_as_float(
                planner_data, "fire_risk_penalty", PlannerConfig.fire_risk_penalty
            ),
            fire_risk_radius=_as_int(
                planner_data, "fire_risk_radius", PlannerConfig.fire_risk_radius
            ),
            turn_penalty=_as_float(planner_data, "turn_penalty", PlannerConfig.turn_penalty),
        )
        return SimulationConfig(
            tick_rate_hz=_as_float(data, "tick_rate_hz", SimulationConfig.tick_rate_hz),
            mission=mission,
            planner=planner,
        )

    def load_vehicle_config(self, relative_path: PathLike) -> VehicleConfig:
        """Load ``vehicle.yaml`` into a :class:`VehicleConfig`.

        Same optional-key policy as :meth:`load_simulation_config`.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
        """
        data = self.load_yaml(relative_path)
        battery_data = _require_mapping(data.get("battery", {}), "vehicle.battery")

        battery = BatteryConfig(
            initial_percent=_as_float(
                battery_data, "initial_percent", BatteryConfig.initial_percent
            ),
            drain_per_move=_as_float(battery_data, "drain_per_move", BatteryConfig.drain_per_move),
            drain_per_turn=_as_float(battery_data, "drain_per_turn", BatteryConfig.drain_per_turn),
            drain_per_idle_tick=_as_float(
                battery_data, "drain_per_idle_tick", BatteryConfig.drain_per_idle_tick
            ),
        )
        return VehicleConfig(
            capacity=_as_int(data, "capacity", VehicleConfig.capacity),
            battery=battery,
            collision_damage_percent=_as_float(
                data, "collision_damage_percent", VehicleConfig.collision_damage_percent
            ),
            fire_damage_per_tick=_as_float(
                data, "fire_damage_per_tick", VehicleConfig.fire_damage_per_tick
            ),
            sensor_range_tiles=_as_float(
                data, "sensor_range_tiles", VehicleConfig.sensor_range_tiles
            ),
        )

    def _optional_path(self, data: dict[str, Any], key: str) -> Path | None:
        """Resolve ``data[key]`` as a path, or ``None`` when the key is absent."""
        value = data.get(key)
        return self.resolve(value) if value is not None else None


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"config section '{label}' must be a mapping, got {type(value)}")
    return value


def _as_float(data: dict[str, Any], key: str, default: float) -> float:
    """Read ``data[key]`` as a float, falling back to ``default`` when absent."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"config key '{key}' must be a number, got {value!r}")
    return float(value)


def _as_int(data: dict[str, Any], key: str, default: int) -> int:
    """Read ``data[key]`` as an int, falling back to ``default`` when absent."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"config key '{key}' must be an integer, got {value!r}")
    return value


def _as_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    """Read ``data[key]`` as a bool, falling back to ``default`` when absent."""
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"config key '{key}' must be true or false, got {value!r}")
    return value
