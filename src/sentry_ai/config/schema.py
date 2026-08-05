"""Typed dataclasses describing the shape of ``configs/app.yaml``.

Each dataclass validates itself in ``__post_init__`` so an invalid config
fails fast, at load time, with a clear message — never deep inside a
render loop or training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class PlannerConfig:
    """Cost model for the global A* route planner.

    Attributes:
        fire_risk_penalty: Extra traversal cost added to a tile within
            ``fire_risk_radius`` of a known fire. Higher values make the
            planner detour further to keep clear of heat; ``0.0`` makes it
            purely shortest-path.
        fire_risk_radius: How many tiles a fire's risk penalty reaches.
        turn_penalty: Extra cost per change of direction, which biases the
            planner toward straighter routes that need fewer waypoints.
    """

    fire_risk_penalty: float = 6.0
    fire_risk_radius: int = 2
    turn_penalty: float = 0.4

    def __post_init__(self) -> None:
        if self.fire_risk_penalty < 0.0:
            raise ConfigValidationError(
                f"planner.fire_risk_penalty must be non-negative, got {self.fire_risk_penalty}"
            )
        if self.fire_risk_radius < 0:
            raise ConfigValidationError(
                f"planner.fire_risk_radius must be non-negative, got {self.fire_risk_radius}"
            )
        if self.turn_penalty < 0.0:
            raise ConfigValidationError(
                f"planner.turn_penalty must be non-negative, got {self.turn_penalty}"
            )


@dataclass(frozen=True)
class MissionConfig:
    """Rules that decide when a mission succeeds, fails, or replans.

    Attributes:
        time_limit_seconds: Wall-clock mission budget. The mission fails
            when it elapses with victims still unrescued.
        replan_on_blocked_route: Whether the command center re-runs the
            planner when the vehicle reports its route is obstructed.
        min_battery_to_continue: Battery percentage below which the mission
            controller stops seeking victims and heads for the hospital.
    """

    time_limit_seconds: float = 300.0
    replan_on_blocked_route: bool = True
    min_battery_to_continue: float = 15.0

    def __post_init__(self) -> None:
        if self.time_limit_seconds <= 0.0:
            raise ConfigValidationError(
                f"mission.time_limit_seconds must be positive, got {self.time_limit_seconds}"
            )
        if not 0.0 <= self.min_battery_to_continue <= 100.0:
            raise ConfigValidationError(
                f"mission.min_battery_to_continue must be within 0.0-100.0, "
                f"got {self.min_battery_to_continue}"
            )


@dataclass(frozen=True)
class SimulationConfig:
    """Timing and rules for the tick-based simulation engine.

    Attributes:
        tick_rate_hz: Fixed simulation steps per second. Decoupled from
            ``RenderConfig.target_fps`` on purpose — physics must not
            change speed when the frame rate does.
        mission: Mission success/failure/replanning rules.
        planner: Global route-planner cost model.
    """

    tick_rate_hz: float = 10.0
    mission: MissionConfig = field(default_factory=MissionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    def __post_init__(self) -> None:
        if self.tick_rate_hz <= 0.0:
            raise ConfigValidationError(
                f"simulation.tick_rate_hz must be positive, got {self.tick_rate_hz}"
            )

    @property
    def seconds_per_tick(self) -> float:
        """Simulated time advanced by one tick."""
        return 1.0 / self.tick_rate_hz


@dataclass(frozen=True)
class BatteryConfig:
    """How the rescue vehicle spends its battery.

    Attributes:
        initial_percent: Charge the vehicle starts a mission with.
        drain_per_move: Cost of one forward or reverse step, in percent.
        drain_per_turn: Cost of one 90-degree turn, in percent.
        drain_per_idle_tick: Cost of a tick spent stopped, in percent.
    """

    initial_percent: float = 100.0
    drain_per_move: float = 0.35
    drain_per_turn: float = 0.10
    drain_per_idle_tick: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.initial_percent <= 100.0:
            raise ConfigValidationError(
                f"battery.initial_percent must be within 0.0-100.0 (exclusive of 0), "
                f"got {self.initial_percent}"
            )
        for name, value in (
            ("drain_per_move", self.drain_per_move),
            ("drain_per_turn", self.drain_per_turn),
            ("drain_per_idle_tick", self.drain_per_idle_tick),
        ):
            if value < 0.0:
                raise ConfigValidationError(f"battery.{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class VehicleConfig:
    """Physical limits of the rescue vehicle.

    Attributes:
        capacity: How many victims can ride at once.
        battery: Battery drain model.
        collision_damage_percent: Health lost when the vehicle drives into
            something impassable.
        fire_damage_per_tick: Health lost per tick spent inside a fire tile.
        sensor_range_tiles: How far the onboard sensors reach. Sets the
            horizon for the fire-proximity term in the local observation,
            and from Phase 3 the footprint of the onboard camera.
    """

    capacity: int = 2
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    collision_damage_percent: float = 5.0
    fire_damage_per_tick: float = 2.0
    sensor_range_tiles: float = 5.0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigValidationError(f"vehicle.capacity must be at least 1, got {self.capacity}")
        if self.sensor_range_tiles <= 0.0:
            raise ConfigValidationError(
                f"vehicle.sensor_range_tiles must be positive, got {self.sensor_range_tiles}"
            )
        for name, value in (
            ("collision_damage_percent", self.collision_damage_percent),
            ("fire_damage_per_tick", self.fire_damage_per_tick),
        ):
            if value < 0.0:
                raise ConfigValidationError(f"vehicle.{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration, composing the other config files.

    Attributes:
        logging_config_path: Absolute path to the ``dictConfig`` YAML file.
        map_config_path: Absolute path to the disaster-city map YAML file.
        render: Window/drawing tunables.
        simulation_config_path: Absolute path to ``simulation.yaml``, or
            ``None`` for a preview-only config that never runs a mission.
        vehicle_config_path: Absolute path to ``vehicle.yaml``, or ``None``
            for a preview-only config that never runs a mission.
    """

    logging_config_path: Path
    map_config_path: Path
    render: RenderConfig
    simulation_config_path: Path | None = None
    vehicle_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.logging_config_path.suffix:
            raise ConfigValidationError(
                f"app.logging_config must point at a file, got {self.logging_config_path}"
            )
        if not self.map_config_path.suffix:
            raise ConfigValidationError(
                f"app.map_config must point at a file, got {self.map_config_path}"
            )
