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
        urgency_weight: How many tiles of detour a dying victim is worth.
            Objectives are ranked by planned route cost minus
            ``urgency_weight * (1 - health/100)``, so at ``0.0`` the
            controller always takes the cheapest victim to reach and at high
            values it will cross the city for someone critical.
    """

    time_limit_seconds: float = 300.0
    replan_on_blocked_route: bool = True
    min_battery_to_continue: float = 15.0
    urgency_weight: float = 14.0

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
        if self.urgency_weight < 0.0:
            raise ConfigValidationError(
                f"mission.urgency_weight must be non-negative, got {self.urgency_weight}"
            )


@dataclass(frozen=True)
class FireSpreadConfig:
    """Cellular-automaton parameters for how fire grows, spreads, and dies.

    Attributes:
        enabled: Whether fire evolves at all. ``False`` freezes every fire
            at its starting intensity and radius.
        interval_seconds: Simulated time between automaton steps. Fire is
            deliberately slower than the tick rate — a fire that mutated
            ten times a second would make routing meaningless.
        growth_per_step: Intensity gained per step while a fire is growing.
        burnout_per_step: Intensity lost per step once a fire has peaked
            and started to burn itself out.
        ignition_chance: Probability, per flammable neighbour per step and
            scaled by the parent fire's intensity, that the fire spreads.
        max_radius: Ceiling on a single fire's footprint, in tiles.
        max_active_fires: Ceiling on simultaneous fires, so a mission
            cannot degenerate into a fully-engulfed city.
    """

    enabled: bool = True
    interval_seconds: float = 3.0
    growth_per_step: float = 0.12
    burnout_per_step: float = 0.18
    ignition_chance: float = 0.16
    max_radius: int = 3
    max_active_fires: int = 12

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ConfigValidationError(
                f"hazards.fire.interval_seconds must be positive, got {self.interval_seconds}"
            )
        for name, value in (
            ("growth_per_step", self.growth_per_step),
            ("burnout_per_step", self.burnout_per_step),
            ("ignition_chance", self.ignition_chance),
        ):
            if not 0.0 <= value <= 1.0:
                raise ConfigValidationError(
                    f"hazards.fire.{name} must be within 0.0-1.0, got {value}"
                )
        if self.max_radius < 1:
            raise ConfigValidationError(
                f"hazards.fire.max_radius must be at least 1, got {self.max_radius}"
            )
        if self.max_active_fires < 1:
            raise ConfigValidationError(
                f"hazards.fire.max_active_fires must be at least 1, got {self.max_active_fires}"
            )


@dataclass(frozen=True)
class DebrisCollapseConfig:
    """How often a weakened building drops debris into the street.

    This is the specification's "dynamic obstacle": something that appears
    mid-mission on a tile the planner already believed was clear.

    Attributes:
        enabled: Whether collapses happen at all.
        interval_seconds: Simulated time between collapse attempts.
        collapse_chance: Probability that an attempt actually produces a
            collapse.
        max_collapses: Ceiling on collapses per mission, so the city stays
            navigable.
    """

    enabled: bool = True
    interval_seconds: float = 6.0
    collapse_chance: float = 0.35
    max_collapses: int = 6

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ConfigValidationError(
                f"hazards.debris.interval_seconds must be positive, got {self.interval_seconds}"
            )
        if not 0.0 <= self.collapse_chance <= 1.0:
            raise ConfigValidationError(
                f"hazards.debris.collapse_chance must be within 0.0-1.0, "
                f"got {self.collapse_chance}"
            )
        if self.max_collapses < 0:
            raise ConfigValidationError(
                f"hazards.debris.max_collapses must be non-negative, got {self.max_collapses}"
            )


@dataclass(frozen=True)
class VictimRiskConfig:
    """How fast trapped victims deteriorate while they wait.

    Without this a victim beside a fire is in no more danger than one in an
    empty street, so "grab the nearest one" is always right and the command
    center never has to make a real choice.

    Attributes:
        enabled: Whether victims deteriorate at all. ``False`` restores the
            earlier behaviour where waiting costs a victim nothing.
        interval_seconds: Simulated time between health steps.
        base_drain: Health lost per step by any trapped victim.
        fire_drain: Additional health lost per step at the seat of a fire,
            scaled linearly down to zero at ``fire_radius``.
        fire_radius: How far a fire's effect on a victim reaches, in tiles.
            Deliberately larger than a fire's drawn footprint — smoke and
            heat hurt well beyond the flames.
    """

    enabled: bool = True
    interval_seconds: float = 1.0
    base_drain: int = 1
    fire_drain: int = 6
    fire_radius: float = 4.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ConfigValidationError(
                f"hazards.victims.interval_seconds must be positive, "
                f"got {self.interval_seconds}"
            )
        if self.fire_radius <= 0.0:
            raise ConfigValidationError(
                f"hazards.victims.fire_radius must be positive, got {self.fire_radius}"
            )
        for name, value in (("base_drain", self.base_drain), ("fire_drain", self.fire_drain)):
            if value < 0:
                raise ConfigValidationError(
                    f"hazards.victims.{name} must be non-negative, got {value}"
                )


@dataclass(frozen=True)
class HazardConfig:
    """Everything that changes the city without the vehicle touching it.

    Attributes:
        seed: Seed for the hazard random number generator. Fixed by default
            so a mission replays identically; change it to sample a
            different disaster from the same starting map.
        fire: Fire spread model.
        debris: Building-collapse model.
        victims: How fast trapped victims deteriorate.
    """

    seed: int = 20250805
    fire: FireSpreadConfig = field(default_factory=FireSpreadConfig)
    debris: DebrisCollapseConfig = field(default_factory=DebrisCollapseConfig)
    victims: VictimRiskConfig = field(default_factory=VictimRiskConfig)


@dataclass(frozen=True)
class SimulationConfig:
    """Timing and rules for the tick-based simulation engine.

    Attributes:
        tick_rate_hz: Fixed simulation steps per second. Decoupled from
            ``RenderConfig.target_fps`` on purpose — physics must not
            change speed when the frame rate does.
        mission: Mission success/failure/replanning rules.
        planner: Global route-planner cost model.
        hazards: Processes that evolve the city on their own.
    """

    tick_rate_hz: float = 10.0
    mission: MissionConfig = field(default_factory=MissionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)

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
class CameraSpec:
    """One fixed CCTV camera's footprint, in world tiles.

    Attributes:
        camera_id: Unique identifier, also the seed for this camera's
            sensor grain — renaming a camera changes its imagery.
        origin_x: Left edge of the covered region.
        origin_y: Top edge of the covered region.
        width_tiles: Width of the covered region.
        height_tiles: Height of the covered region.
    """

    camera_id: str
    origin_x: int
    origin_y: int
    width_tiles: int
    height_tiles: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ConfigValidationError("sensors.cameras[].id must not be empty")
        if self.origin_x < 0 or self.origin_y < 0:
            raise ConfigValidationError(
                f"sensors camera '{self.camera_id}' origin must be non-negative, "
                f"got ({self.origin_x}, {self.origin_y})"
            )
        for name, value in (("width", self.width_tiles), ("height", self.height_tiles)):
            if value <= 0:
                raise ConfigValidationError(
                    f"sensors camera '{self.camera_id}' {name} must be positive, got {value}"
                )


@dataclass(frozen=True)
class OnboardCameraConfig:
    """The vehicle-mounted camera's geometry.

    Attributes:
        camera_id: Identifier stamped onto its frames.
        span_tiles: Side length of the square window that follows the
            vehicle. Odd values keep the vehicle centred.
        tile_size_px: Pixels per tile. Finer than the CCTV cameras by
            default: this is the view local obstacle detection runs on.
    """

    camera_id: str = "onboard"
    span_tiles: int = 9
    tile_size_px: int = 16

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ConfigValidationError("sensors.onboard.id must not be empty")
        for name, value in (
            ("span_tiles", self.span_tiles),
            ("tile_size_px", self.tile_size_px),
        ):
            if value <= 0:
                raise ConfigValidationError(f"sensors.onboard.{name} must be positive, got {value}")


@dataclass(frozen=True)
class DegradationConfig:
    """How badly a camera frame is corrupted before the denoiser sees it.

    Attributes:
        smoke_density: Peak opacity of the smoke field, 0-1. At ``0.0`` no
            smoke is applied.
        smoke_grey: Grey level smoke blends toward, 0-255.
        blur_radius: Number of 3-tap smoothing passes. ``0`` leaves the
            frame sharp.
        noise_std: Standard deviation of additive Gaussian sensor noise, in
            8-bit levels.
    """

    smoke_density: float = 0.45
    smoke_grey: int = 150
    blur_radius: int = 1
    noise_std: float = 8.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.smoke_density <= 1.0:
            raise ConfigValidationError(
                f"sensors.degradation.smoke_density must be within 0.0-1.0, "
                f"got {self.smoke_density}"
            )
        if not 0 <= self.smoke_grey <= 255:
            raise ConfigValidationError(
                f"sensors.degradation.smoke_grey must be within 0-255, got {self.smoke_grey}"
            )
        if self.blur_radius < 0:
            raise ConfigValidationError(
                f"sensors.degradation.blur_radius must be non-negative, got {self.blur_radius}"
            )
        if self.noise_std < 0.0:
            raise ConfigValidationError(
                f"sensors.degradation.noise_std must be non-negative, got {self.noise_std}"
            )


@dataclass(frozen=True)
class SensorConfig:
    """The whole camera network: CCTV layout, onboard camera, corruption.

    Attributes:
        tile_size_px: Pixels per tile for the fixed CCTV cameras.
        cameras: Fixed camera footprints, in capture order.
        onboard: The vehicle-mounted camera.
        degradation: Smoke/blur/noise model applied to captured frames.
    """

    tile_size_px: int = 16
    cameras: tuple[CameraSpec, ...] = ()
    onboard: OnboardCameraConfig = field(default_factory=OnboardCameraConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    def __post_init__(self) -> None:
        if self.tile_size_px <= 0:
            raise ConfigValidationError(
                f"sensors.tile_size_px must be positive, got {self.tile_size_px}"
            )
        if not self.cameras:
            raise ConfigValidationError("sensors.cameras must define at least one camera")


@dataclass(frozen=True)
class YoloTrainingConfig:
    """Hyperparameters for fine-tuning YOLOv8n (Unit II).

    Attributes:
        dataset_dir: Directory holding ``data.yaml`` and the split folders.
        runs_dir: Where Ultralytics writes weights and training curves.
        pretrained_weights: Local checkpoint to transfer from, resolved
            against the project root. A local path rather than a bare name
            so a training run has no hidden network dependency —
            ``scripts/fetch_pretrained.py`` puts it there.
        image_size: Training and inference resolution. The same value must
            reach ``YoloDetector`` — a mismatch silently misplaces boxes.
        epochs: Maximum passes over the training split.
        batch_size: Images per step.
        patience: Epochs without validation improvement before stopping.
        horizontal_flip: Probability of mirroring a frame left-to-right.
        vertical_flip: Probability of mirroring top-to-bottom. Legal here
            only because the city is viewed from directly overhead, which
            makes a flipped frame a plausible city rather than an impossible
            one.
        mosaic: Probability of stitching four frames into one.
        scale: Random zoom fraction.
        translate: Random shift fraction.
        hsv_value: Brightness jitter fraction.
        confidence: Minimum detection score at inference.
        iou: Non-maximum-suppression threshold.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
        seed: Fixed so a run is reproducible.
    """

    dataset_dir: Path = Path("data/synthetic")
    runs_dir: Path = Path("models/yolo")
    pretrained_weights: Path = Path("models/pretrained/yolov8n.pt")
    image_size: int = 256
    epochs: int = 60
    batch_size: int = 16
    patience: int = 15
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    mosaic: float = 0.4
    scale: float = 0.3
    translate: float = 0.1
    hsv_value: float = 0.3
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "auto"
    seed: int = 20250806

    def __post_init__(self) -> None:
        for name, value in (
            ("image_size", self.image_size),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ConfigValidationError(f"yolo.{name} must be positive, got {value}")
        if self.patience < 0:
            raise ConfigValidationError(f"yolo.patience must be non-negative, got {self.patience}")
        if self.image_size % 32 != 0:
            raise ConfigValidationError(
                f"yolo.image_size must be a multiple of 32 (YOLO's stride), "
                f"got {self.image_size}"
            )
        for name, fraction in (
            ("horizontal_flip", self.horizontal_flip),
            ("vertical_flip", self.vertical_flip),
            ("mosaic", self.mosaic),
            ("scale", self.scale),
            ("translate", self.translate),
            ("hsv_value", self.hsv_value),
            ("confidence", self.confidence),
            ("iou", self.iou),
        ):
            if not 0.0 <= fraction <= 1.0:
                raise ConfigValidationError(
                    f"yolo.{name} must be within 0.0-1.0, got {fraction}"
                )


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
        sensor_config_path: Absolute path to ``sensors.yaml``, or ``None``
            for a config that never captures camera frames.
    """

    logging_config_path: Path
    map_config_path: Path
    render: RenderConfig
    simulation_config_path: Path | None = None
    vehicle_config_path: Path | None = None
    sensor_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.logging_config_path.suffix:
            raise ConfigValidationError(
                f"app.logging_config must point at a file, got {self.logging_config_path}"
            )
        if not self.map_config_path.suffix:
            raise ConfigValidationError(
                f"app.map_config must point at a file, got {self.map_config_path}"
            )
