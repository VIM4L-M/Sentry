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
        block_confirm_refreshes: How many consecutive map refreshes must show
            the rest of the route blocked before the command center drops it
            and replans. A detector flickers; a collapse persists. ``1``
            reacts to a single frame, which can abandon a victim over a
            one-tick phantom; ``3`` (0.3 s) ignores flicker and still reacts
            to a real collapse. Adopted from the teammate's Phase 8 build.
    """

    time_limit_seconds: float = 300.0
    replan_on_blocked_route: bool = True
    min_battery_to_continue: float = 15.0
    urgency_weight: float = 14.0
    block_confirm_refreshes: int = 3

    def __post_init__(self) -> None:
        if self.block_confirm_refreshes < 1:
            raise ConfigValidationError(
                f"mission.block_confirm_refreshes must be at least 1, "
                f"got {self.block_confirm_refreshes}"
            )
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
class TrafficConfig:
    """Other road users: cars and pedestrians sharing the streets (Phase 9).

    Off by default, so every earlier map, test and trained model is
    unchanged. The command center's map never contains them — they move too
    fast to plan around — so only the vehicle's own sensors and its
    emergency brake stand between it and a collision.

    Attributes:
        enabled: Whether any traffic is spawned.
        cars: Cars driving the road tiles.
        pedestrians: People walking the pavements, sometimes crossing.
        car_step_ticks: Ticks between a car's moves (the rescue vehicle
            moves every tick, so cars are slower and it can catch up).
        pedestrian_step_ticks: Ticks between a pedestrian's steps.
        autos: Autorickshaws, slower than cars, on the road tiles.
        two_wheelers: Motorbikes and scooters, as quick as cars.
        cows: Cattle wandering the road and the verge, slowest of all.
        auto_step_ticks: Ticks between an autorickshaw's moves.
        two_wheeler_step_ticks: Ticks between a two-wheeler's moves.
        cow_step_ticks: Ticks between a cow's steps.
        crossing_chance: Chance a pedestrian at the kerb steps into the road.
        seed: Seed for spawning and every choice an agent makes.
    """

    enabled: bool = False
    cars: int = 0
    pedestrians: int = 0
    car_step_ticks: int = 2
    pedestrian_step_ticks: int = 4
    crossing_chance: float = 0.15
    seed: int = 20250928
    autos: int = 0
    two_wheelers: int = 0
    cows: int = 0
    auto_step_ticks: int = 3
    two_wheeler_step_ticks: int = 2
    cow_step_ticks: int = 6

    def __post_init__(self) -> None:
        for name, value in (
            ("cars", self.cars),
            ("pedestrians", self.pedestrians),
            ("autos", self.autos),
            ("two_wheelers", self.two_wheelers),
            ("cows", self.cows),
        ):
            if value < 0:
                raise ConfigValidationError(f"traffic.{name} must be non-negative, got {value}")
        for name, value in (
            ("car_step_ticks", self.car_step_ticks),
            ("pedestrian_step_ticks", self.pedestrian_step_ticks),
            ("auto_step_ticks", self.auto_step_ticks),
            ("two_wheeler_step_ticks", self.two_wheeler_step_ticks),
            ("cow_step_ticks", self.cow_step_ticks),
        ):
            if value < 1:
                raise ConfigValidationError(f"traffic.{name} must be at least 1, got {value}")
        if not 0.0 <= self.crossing_chance <= 1.0:
            raise ConfigValidationError(
                f"traffic.crossing_chance must be within 0.0-1.0, got {self.crossing_chance}"
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
        hazards: Processes that evolve the city on their own.
        traffic: Cars and pedestrians sharing the streets; off by default.
    """

    tick_rate_hz: float = 10.0
    mission: MissionConfig = field(default_factory=MissionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)

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
class MarkerScaleConfig:
    """How much of its tile each annotated entity's marker fills.

    These decide the pixel size of every training target, so they are the
    single most important control over small-object recall. At the default
    ``tile_size_px`` of 16 a victim is 8 px across, which lands on roughly
    one cell of YOLOv8's finest (stride-8) detection head — the resolution
    floor of the architecture. Debris, at 0.8, is half again as wide and is
    detected almost perfectly. That gap is why these live in config: it
    makes marker size an experiment rather than a source edit.

    Attributes:
        victim: Fraction of a tile a victim's marker occupies.
        debris: Fraction of a tile a piece of debris occupies.
        vehicle: Fraction of a tile the ego vehicle occupies. Not annotated,
            but drawn, so the detector has seen it.
    """

    victim: float = 0.5
    debris: float = 0.8
    vehicle: float = 0.7

    def __post_init__(self) -> None:
        for name, value in (
            ("victim", self.victim),
            ("debris", self.debris),
            ("vehicle", self.vehicle),
        ):
            if not 0.0 < value <= 1.0:
                raise ConfigValidationError(
                    f"sensors.markers.{name} must be within 0.0 (exclusive) to 1.0, got {value}"
                )


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

    def scaled(self, severity: float) -> DegradationConfig:
        """The same corruption, made ``severity`` times as strong.

        ``1.0`` is this config unchanged and ``0.0`` is a clean frame. Smoke
        opacity saturates at fully opaque; blur passes round half up, so a
        radius-1 blur becomes radius 2 at severity 1.5. The smoke colour is a
        property of the smoke, not of how much there is, so it never scales.

        This is how the denoiser is trained on more than the one corruption
        level the dataset was captured at, and how it is stress-tested beyond
        it.
        """
        if severity < 0.0:
            raise ConfigValidationError(
                f"degradation severity must be non-negative, got {severity}"
            )
        return DegradationConfig(
            smoke_density=min(1.0, self.smoke_density * severity),
            smoke_grey=self.smoke_grey,
            blur_radius=int(self.blur_radius * severity + 0.5),
            noise_std=self.noise_std * severity,
        )


@dataclass(frozen=True)
class SensorConfig:
    """The whole camera network: CCTV layout, onboard camera, corruption.

    Attributes:
        tile_size_px: Pixels per tile for the fixed CCTV cameras.
        cameras: Fixed camera footprints, in capture order.
        onboard: The vehicle-mounted camera.
        markers: How much of a tile each annotated entity fills.
        degradation: Smoke/blur/noise model applied to captured frames.
    """

    tile_size_px: int = 16
    cameras: tuple[CameraSpec, ...] = ()
    onboard: OnboardCameraConfig = field(default_factory=OnboardCameraConfig)
    markers: MarkerScaleConfig = field(default_factory=MarkerScaleConfig)
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


#: Reconstruction losses the autoencoder trainer knows how to build.
AUTOENCODER_LOSSES: tuple[str, ...] = ("l1", "mse")


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    """Architecture and hyperparameters for the denoising autoencoder (Unit IV).

    Attributes:
        dataset_dir: The dataset ``build_dataset.py`` wrote. Training reads
            its ``clean/train`` frames and corrupts them on the fly;
            validation reads the stored ``images/val``/``clean/val`` pairs,
            which are exactly what the detector is shown.
        runs_dir: Where checkpoints and the metric log are written.
        base_channels: Feature maps at full resolution. Each level down
            doubles it.
        depth: Number of stride-2 downsamplings. Every level halves the
            resolution the bottleneck works at.
        skip_connections: Carry each encoder level's features across to the
            matching decoder level. See docs/architecture/phase4-denoising.md
            for why this defaults to on, and why it is still a switch.
        crop_size: Side of the square training crops, in pixels. Must fit
            inside the smallest frame (the 144 px onboard view) and divide
            by ``2 ** depth``.
        severity_min: Weakest corruption sampled for a training crop, as a
            multiple of ``sensors.degradation``.
        severity_max: Strongest corruption sampled. Training above 1.0 is
            what lets the denoiser cope with more smoke than the dataset was
            captured in.
        epochs: Maximum passes over the training frames.
        batch_size: Crops per optimisation step.
        learning_rate: Adam step size.
        weight_decay: Adam L2 penalty. ``0.0`` disables it.
        patience: Epochs without a validation improvement before stopping.
            ``0`` disables early stopping.
        loss: ``"l1"`` or ``"mse"``.
        num_workers: DataLoader worker processes. ``0`` loads in the
            training process, which is the safe choice on Windows.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
        seed: Fixed so a run is reproducible, including every crop and
            every corruption.
    """

    dataset_dir: Path = Path("data/synthetic")
    runs_dir: Path = Path("models/autoencoder")
    base_channels: int = 32
    depth: int = 3
    skip_connections: bool = True
    crop_size: int = 128
    severity_min: float = 0.5
    severity_max: float = 2.0
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    loss: str = "l1"
    num_workers: int = 2
    device: str = "auto"
    seed: int = 20250807

    def __post_init__(self) -> None:
        for name, value in (
            ("base_channels", self.base_channels),
            ("depth", self.depth),
            ("crop_size", self.crop_size),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ConfigValidationError(f"autoencoder.{name} must be positive, got {value}")
        for name, value in (("patience", self.patience), ("num_workers", self.num_workers)):
            if value < 0:
                raise ConfigValidationError(
                    f"autoencoder.{name} must be non-negative, got {value}"
                )
        if self.crop_size % (2**self.depth) != 0:
            raise ConfigValidationError(
                f"autoencoder.crop_size must be a multiple of 2 ** depth "
                f"({2**self.depth}), got {self.crop_size}"
            )
        if not 0.0 <= self.severity_min <= self.severity_max:
            raise ConfigValidationError(
                f"autoencoder.severity_min/max must satisfy 0 <= min <= max, "
                f"got {self.severity_min}/{self.severity_max}"
            )
        if self.learning_rate <= 0.0:
            raise ConfigValidationError(
                f"autoencoder.learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ConfigValidationError(
                f"autoencoder.weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.loss not in AUTOENCODER_LOSSES:
            raise ConfigValidationError(
                f"autoencoder.loss must be one of {AUTOENCODER_LOSSES}, got {self.loss!r}"
            )


@dataclass(frozen=True)
class LstmTrainingConfig:
    """Architecture and hyperparameters for the behaviour LSTM (Unit III).

    Attributes:
        trajectories_dir: Where ``record_trajectories.py`` wrote the
            ``train.json`` and ``val.json`` trajectory files.
        runs_dir: Where checkpoints and the metric log are written.
        window: States the model sees per prediction, oldest first.
        horizon: How many ticks ahead the predicted behaviour spans. Must be
            shorter than ``window`` so the persistence baseline — which
            looks back ``horizon`` ticks — can be scored on the same input.
        cone_degrees: Half-angle of the ADVANCE/RETREAT cones; see
            :mod:`sentry_ai.sequence.behaviour`.
        hidden_size: LSTM hidden units per layer.
        num_layers: Stacked LSTM layers.
        dropout: Dropout between LSTM layers and before the classifier.
        epochs: Maximum passes over the training windows.
        batch_size: Windows per optimisation step.
        learning_rate: Adam step size.
        weight_decay: Adam L2 penalty. ``0.0`` disables it.
        patience: Epochs without a validation improvement before stopping.
            ``0`` disables early stopping.
        class_weighting: Weight the loss by square-root inverse class
            frequency. About seven in ten windows are ADVANCE; unweighted,
            the cheapest way to a low loss is to say ADVANCE every time.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
        seed: Fixed so a run is reproducible.
    """

    trajectories_dir: Path = Path("data/trajectories")
    runs_dir: Path = Path("models/lstm")
    window: int = 8
    horizon: int = 4
    cone_degrees: float = 30.0
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    class_weighting: bool = True
    device: str = "auto"
    seed: int = 20250808

    def __post_init__(self) -> None:
        for name, value in (
            ("window", self.window),
            ("horizon", self.horizon),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ConfigValidationError(f"lstm.{name} must be positive, got {value}")
        if self.horizon >= self.window:
            raise ConfigValidationError(
                f"lstm.horizon ({self.horizon}) must be shorter than lstm.window "
                f"({self.window}), or the persistence baseline cannot see far enough back"
            )
        if not 0.0 < self.cone_degrees < 90.0:
            raise ConfigValidationError(
                f"lstm.cone_degrees must be within (0, 90), got {self.cone_degrees}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigValidationError(f"lstm.dropout must be within [0, 1), got {self.dropout}")
        if self.learning_rate <= 0.0:
            raise ConfigValidationError(
                f"lstm.learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ConfigValidationError(
                f"lstm.weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.patience < 0:
            raise ConfigValidationError(f"lstm.patience must be non-negative, got {self.patience}")


@dataclass(frozen=True)
class RewardConfig:
    """What the DQN is paid for, per tick (Unit V).

    Local by design (ADR 0002): the policy is rewarded for driving the route
    the command center planned, not for choosing it. Every term is a weight
    on a fact the physics or the mission already reports.

    Attributes:
        progress: Per tile closer to the waypoint that was active before the
            tick. Moving away costs the same amount, so there is nothing to
            gain by wandering off and back.
        pickup: Per victim taken aboard.
        delivery: Per victim delivered to the hospital.
        collision: Per refused move into something impassable.
        damage: Per health percentage lost (collision or fire).
        fire_proximity: Scaled by the 0-1 closeness to the nearest fire.
        step: Every tick. Makes dawdling cost something, including STOP.
        battery: Per battery percentage spent.
        reverse: Per REVERSE action. The physics charge reversing exactly
            what driving forward costs, and an early policy exploited that,
            driving 14% of its tiles backwards in stretches of up to ten —
            cheaper than turning round. A rescue vehicle reverses to back
            out of a dead end, not to cross town; this makes a U-turn the
            cheaper way to change direction. ``0.0`` restores the old
            behaviour.
        completion: Once, when the mission completes.
        failure: Once, when the mission fails.
        road_user_hit: Per move into a car or a pedestrian (Phase 9
            traffic), on top of ``collision``. An ambulance that injures
            people on its way has failed at its job, so this is set far
            above any other penalty; ``0.0`` where there is no traffic.
    """

    progress: float = 1.0
    pickup: float = 5.0
    delivery: float = 10.0
    collision: float = -2.0
    damage: float = -0.2
    fire_proximity: float = -0.5
    step: float = -0.05
    battery: float = -0.1
    reverse: float = -0.2
    completion: float = 10.0
    failure: float = -10.0
    road_user_hit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("progress", "pickup", "delivery", "completion"):
            if getattr(self, name) < 0.0:
                raise ConfigValidationError(f"dqn.reward.{name} must be non-negative")
        for name in (
            "collision",
            "damage",
            "fire_proximity",
            "step",
            "battery",
            "reverse",
            "failure",
            "road_user_hit",
        ):
            if getattr(self, name) > 0.0:
                raise ConfigValidationError(
                    f"dqn.reward.{name} is a penalty and must be zero or negative"
                )


@dataclass(frozen=True)
class DqnTrainingConfig:
    """The environment, the DQN, and the bar it has to clear (Unit V).

    Attributes:
        runs_dir: Where the model, its metadata, and evaluations are written.
        train_seeds: Hazard seeds training episodes are drawn from. Each
            episode also starts the vehicle on a random road tile.
        eval_seeds: Held-out seeds, never trained on.
        seed_base: First training seed; evaluation seeds follow training.
        max_episode_steps: Episode cut-off, well under the mission timer.
        total_timesteps: Environment steps to train for.
        learning_rate: Adam step size.
        buffer_size: Replay buffer capacity, in transitions.
        learning_starts: Random steps collected before learning begins.
        batch_size: Transitions per gradient step.
        gamma: Discount factor.
        train_freq: Environment steps between gradient steps.
        target_update_interval: Steps between target-network syncs.
        exploration_fraction: Share of training over which epsilon decays.
        exploration_final_eps: Epsilon after the decay.
        hidden_sizes: Q-network hidden layers.
        eval_every: Steps between held-out evaluations during training.
        eval_episodes: Held-out missions per evaluation.
        rescue_threshold: M6 bar — the DQN must rescue at least this share
            of what the waypoint follower rescues on the same missions.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
        seed: Fixed so a run is reproducible.
        reward: The reward weights.
        traffic_features: Give the policy the two road-user inputs
            (``TRAFFIC_POLICY_FEATURES``). For training among cars and
            pedestrians; the app config must then enable traffic.
    """

    runs_dir: Path = Path("models/dqn")
    train_seeds: int = 400
    eval_seeds: int = 40
    seed_base: int = 3000
    max_episode_steps: int = 1500
    total_timesteps: int = 300_000
    learning_rate: float = 5e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 2_000
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    hidden_sizes: tuple[int, ...] = (64, 64)
    eval_every: int = 25_000
    eval_episodes: int = 20
    rescue_threshold: float = 0.9
    device: str = "auto"
    seed: int = 20250809
    reward: RewardConfig = field(default_factory=RewardConfig)
    traffic_features: bool = False

    def __post_init__(self) -> None:
        for name in (
            "train_seeds",
            "eval_seeds",
            "max_episode_steps",
            "total_timesteps",
            "buffer_size",
            "batch_size",
            "train_freq",
            "target_update_interval",
            "eval_every",
            "eval_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ConfigValidationError(f"dqn.{name} must be positive")
        if self.learning_starts < 0:
            raise ConfigValidationError("dqn.learning_starts must be non-negative")
        if self.learning_rate <= 0.0:
            raise ConfigValidationError("dqn.learning_rate must be positive")
        for name in ("gamma", "exploration_fraction", "exploration_final_eps"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigValidationError(f"dqn.{name} must be within 0.0-1.0")
        if not 0.0 < self.rescue_threshold <= 1.0:
            raise ConfigValidationError("dqn.rescue_threshold must be within (0, 1]")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ConfigValidationError("dqn.hidden_sizes must be a non-empty list of positives")

    @property
    def training_seeds(self) -> range:
        """The hazard seeds training episodes draw from."""
        return range(self.seed_base, self.seed_base + self.train_seeds)

    @property
    def evaluation_seeds(self) -> range:
        """Held-out seeds, directly after the training ones."""
        first = self.seed_base + self.train_seeds
        return range(first, first + self.eval_seeds)


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


#: The two fusion labels; see ``FusionTrainingConfig.label_mode``.
LABEL_MODES = ("truth", "veto")


@dataclass(frozen=True)
class FusionTrainingConfig:
    """Scenario, data and hyperparameters for the fusion MLP (Unit I).

    Attributes:
        dataset_dir: Where ``train_fusion.py`` writes the collected samples.
        runs_dir: Where checkpoints and the metric log are written.
        dqn_weights: The DQN whose decisions fusion arbitrates.
        lstm_weights: The motion predictor fusion listens to.
        belief_lag: Refreshes the command center's map trails the world by
            — the Phase 7 scenario. With no lag the planner already routes
            around every hazard and fusion has nothing to correct.
        train_missions: Missions collected for training.
        val_missions: Held-out missions for validation, never trained on.
        eval_missions: Further held-out missions for the mission-level
            comparison in ``evaluate_fusion.py``.
        seed_base: First training seed; validation and evaluation seeds
            follow in order, clear of the DQN's own seed range.
        oracle_drive_probability: Share of collection ticks driven by the
            ground-truth decision instead of the DQN's. Pure DQN driving
            only visits the states the DQN reaches; mixing in the corrected
            action shows the network what happens after a correction too.
        max_ticks: Hard stop per mission.
        hidden_sizes: Width of each hidden layer.
        dropout: Dropout after every hidden activation.
        epochs: Maximum passes over the training samples.
        batch_size: Samples per optimisation step.
        learning_rate: Adam step size.
        weight_decay: Adam L2 penalty.
        patience: Epochs without a validation improvement before stopping.
        override_threshold: At inference, the probability fusion's choice
            needs before it may overrule the DQN.
        report_hazards: Whether a fusion override also reports what the
            camera saw to the command center's map, which then replans.
            Off by default: measured on 40 missions it removed the last few
            collisions but cost a third of the completed missions.
        report_threshold: Camera confidence needed before a hazard seen
            ahead is reported to the command center's map.
        critical_weight: Loss weight on ticks where the DQN, reading the
            stale map, disagrees with the ground-truth decision. They are a
            few percent of all ticks and the only ones where fusion earns
            its place.
        label_mode: What fusion is taught to output. ``"truth"``: the
            action the DQN would take on the true map (any action — the
            original Phase 7 label). ``"veto"``: the DQN's own action,
            unless it would move into a tile that is really impassable, in
            which case STOP — the label the teammate's build used, which
            only asks the network to learn when to veto.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
        seed: Fixed so a run is reproducible.
    """

    dataset_dir: Path = Path("data/fusion")
    runs_dir: Path = Path("models/fusion")
    dqn_weights: Path = Path("models/dqn/sentry/best.zip")
    lstm_weights: Path = Path("models/lstm/sentry/best.pt")
    belief_lag: int = 30
    train_missions: int = 120
    val_missions: int = 30
    eval_missions: int = 40
    seed_base: int = 5000
    oracle_drive_probability: float = 0.3
    max_ticks: int = 3000
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    critical_weight: float = 2.0
    label_mode: str = "truth"
    override_threshold: float = 0.8
    report_hazards: bool = False
    report_threshold: float = 0.5
    device: str = "auto"
    seed: int = 20250810

    def __post_init__(self) -> None:
        for name in (
            "train_missions",
            "val_missions",
            "eval_missions",
            "max_ticks",
            "epochs",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ConfigValidationError(f"fusion.{name} must be positive")
        if self.belief_lag < 0 or self.patience < 0:
            raise ConfigValidationError("fusion.belief_lag and fusion.patience must be >= 0")
        if not 0.0 <= self.oracle_drive_probability <= 1.0:
            raise ConfigValidationError("fusion.oracle_drive_probability must be within 0.0-1.0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigValidationError("fusion.dropout must be within [0, 1)")
        if self.learning_rate <= 0.0 or self.critical_weight <= 0.0:
            raise ConfigValidationError("fusion.learning_rate and critical_weight must be positive")
        if self.weight_decay < 0.0:
            raise ConfigValidationError("fusion.weight_decay must be non-negative")
        if not 0.0 <= self.report_threshold <= 1.0:
            raise ConfigValidationError("fusion.report_threshold must be within 0.0-1.0")
        if not 0.0 <= self.override_threshold <= 1.0:
            raise ConfigValidationError("fusion.override_threshold must be within 0.0-1.0")
        if self.label_mode not in LABEL_MODES:
            raise ConfigValidationError(
                f"fusion.label_mode must be one of {LABEL_MODES}, got {self.label_mode!r}"
            )
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ConfigValidationError("fusion.hidden_sizes must be a non-empty list of positives")

    @property
    def training_seeds(self) -> range:
        """Hazard seeds of the training missions."""
        return range(self.seed_base, self.seed_base + self.train_missions)

    @property
    def validation_seeds(self) -> range:
        """Hazard seeds of the validation missions."""
        start = self.seed_base + self.train_missions
        return range(start, start + self.val_missions)

    @property
    def evaluation_seeds(self) -> range:
        """Hazard seeds of the mission-level evaluation."""
        start = self.seed_base + self.train_missions + self.val_missions
        return range(start, start + self.eval_missions)
