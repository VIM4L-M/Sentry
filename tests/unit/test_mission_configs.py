"""Unit tests for the Phase 2 config dataclasses and their loader methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigurationError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import (
    BatteryConfig,
    MissionConfig,
    PlannerConfig,
    SimulationConfig,
    VehicleConfig,
)


class TestValidation:
    def test_planner_rejects_negative_fire_penalty(self) -> None:
        with pytest.raises(ConfigValidationError):
            PlannerConfig(fire_risk_penalty=-1.0)

    def test_planner_rejects_negative_turn_penalty(self) -> None:
        with pytest.raises(ConfigValidationError):
            PlannerConfig(turn_penalty=-0.1)

    def test_mission_rejects_non_positive_time_limit(self) -> None:
        with pytest.raises(ConfigValidationError):
            MissionConfig(time_limit_seconds=0.0)

    @pytest.mark.parametrize("reserve", [-1.0, 101.0])
    def test_mission_rejects_out_of_range_battery_reserve(self, reserve: float) -> None:
        with pytest.raises(ConfigValidationError):
            MissionConfig(min_battery_to_continue=reserve)

    def test_simulation_rejects_non_positive_tick_rate(self) -> None:
        with pytest.raises(ConfigValidationError):
            SimulationConfig(tick_rate_hz=0.0)

    def test_seconds_per_tick_is_the_inverse_of_the_rate(self) -> None:
        assert SimulationConfig(tick_rate_hz=20.0).seconds_per_tick == pytest.approx(0.05)

    def test_battery_rejects_negative_drain(self) -> None:
        with pytest.raises(ConfigValidationError):
            BatteryConfig(drain_per_move=-0.1)

    def test_battery_rejects_zero_initial_charge(self) -> None:
        with pytest.raises(ConfigValidationError):
            BatteryConfig(initial_percent=0.0)

    def test_vehicle_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ConfigValidationError):
            VehicleConfig(capacity=0)

    def test_vehicle_rejects_non_positive_sensor_range(self) -> None:
        with pytest.raises(ConfigValidationError):
            VehicleConfig(sensor_range_tiles=0.0)


class TestSimulationLoader:
    def test_loads_every_section(self, tmp_path: Path) -> None:
        (tmp_path / "simulation.yaml").write_text(
            """
tick_rate_hz: 20.0
mission:
  time_limit_seconds: 120.0
  replan_on_blocked_route: false
  min_battery_to_continue: 25.0
planner:
  fire_risk_penalty: 3.0
  fire_risk_radius: 4
  turn_penalty: 0.9
""",
            encoding="utf-8",
        )
        config = ConfigLoader(project_root=tmp_path).load_simulation_config("simulation.yaml")
        assert config.tick_rate_hz == 20.0
        assert config.mission.replan_on_blocked_route is False
        assert config.mission.min_battery_to_continue == 25.0
        assert config.planner.fire_risk_radius == 4

    def test_empty_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "simulation.yaml").write_text("", encoding="utf-8")
        config = ConfigLoader(project_root=tmp_path).load_simulation_config("simulation.yaml")
        assert config == SimulationConfig()

    def test_string_where_a_number_belongs_raises(self, tmp_path: Path) -> None:
        (tmp_path / "simulation.yaml").write_text('tick_rate_hz: "fast"\n', encoding="utf-8")
        with pytest.raises(ConfigurationError, match="tick_rate_hz"):
            ConfigLoader(project_root=tmp_path).load_simulation_config("simulation.yaml")

    def test_number_where_a_bool_belongs_raises(self, tmp_path: Path) -> None:
        (tmp_path / "simulation.yaml").write_text(
            "mission:\n  replan_on_blocked_route: 1\n", encoding="utf-8"
        )
        with pytest.raises(ConfigurationError, match="replan_on_blocked_route"):
            ConfigLoader(project_root=tmp_path).load_simulation_config("simulation.yaml")

    def test_non_mapping_section_raises(self, tmp_path: Path) -> None:
        (tmp_path / "simulation.yaml").write_text("planner: [1, 2]\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="simulation.planner"):
            ConfigLoader(project_root=tmp_path).load_simulation_config("simulation.yaml")


class TestVehicleLoader:
    def test_loads_every_section(self, tmp_path: Path) -> None:
        (tmp_path / "vehicle.yaml").write_text(
            """
capacity: 3
battery:
  initial_percent: 80.0
  drain_per_move: 0.5
collision_damage_percent: 7.5
sensor_range_tiles: 8.0
""",
            encoding="utf-8",
        )
        config = ConfigLoader(project_root=tmp_path).load_vehicle_config("vehicle.yaml")
        assert config.capacity == 3
        assert config.battery.initial_percent == 80.0
        assert config.battery.drain_per_move == 0.5
        # Unspecified keys keep their defaults.
        assert config.battery.drain_per_turn == BatteryConfig.drain_per_turn
        assert config.collision_damage_percent == 7.5
        assert config.sensor_range_tiles == 8.0

    def test_float_where_an_int_belongs_raises(self, tmp_path: Path) -> None:
        (tmp_path / "vehicle.yaml").write_text("capacity: 2.5\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="capacity"):
            ConfigLoader(project_root=tmp_path).load_vehicle_config("vehicle.yaml")


class TestShippedConfigs:
    def test_real_simulation_config_loads(self, project_root: Path) -> None:
        config = ConfigLoader(project_root=project_root).load_simulation_config(
            "configs/simulation.yaml"
        )
        assert config.tick_rate_hz > 0.0

    def test_real_vehicle_config_loads(self, project_root: Path) -> None:
        config = ConfigLoader(project_root=project_root).load_vehicle_config("configs/vehicle.yaml")
        assert config.capacity >= 1

    def test_app_config_points_at_both(self, project_root: Path) -> None:
        app_config = ConfigLoader(project_root=project_root).load_app_config("configs/app.yaml")
        assert app_config.simulation_config_path is not None
        assert app_config.simulation_config_path.is_file()
        assert app_config.vehicle_config_path is not None
        assert app_config.vehicle_config_path.is_file()

    def test_paths_are_optional_for_preview_only_configs(self, tmp_path: Path) -> None:
        (tmp_path / "app.yaml").write_text(
            """
logging_config: "configs/logging.yaml"
map_config: "configs/maps/city_default.yaml"
render:
  window_title: "Preview"
  tile_size_px: 16
  target_fps: 30
  palette_config: "configs/render.yaml"
""",
            encoding="utf-8",
        )
        app_config = ConfigLoader(project_root=tmp_path).load_app_config("app.yaml")
        assert app_config.simulation_config_path is None
        assert app_config.vehicle_config_path is None
