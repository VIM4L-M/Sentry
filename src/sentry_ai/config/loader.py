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
    AutoencoderTrainingConfig,
    BatteryConfig,
    CameraSpec,
    DebrisCollapseConfig,
    DegradationConfig,
    DqnTrainingConfig,
    FireSpreadConfig,
    FusionTrainingConfig,
    HazardConfig,
    LstmTrainingConfig,
    MarkerScaleConfig,
    MissionConfig,
    OnboardCameraConfig,
    PlannerConfig,
    RenderConfig,
    RewardConfig,
    SensorConfig,
    SimulationConfig,
    TrafficConfig,
    VehicleConfig,
    VictimRiskConfig,
    YoloTrainingConfig,
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
            sensor_config_path=self._optional_path(data, "sensor_config"),
        )

    def load_sensor_config(self, relative_path: PathLike) -> SensorConfig:
        """Load the camera network from ``sensors.yaml``.

        Only the ``cameras`` list is required — every other key falls back
        to its dataclass default. The ``palette`` section of the same file
        is read separately by
        :meth:`~sentry_ai.sensors.palette.SensorPalette.from_config`, which
        owns what the cameras look like.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If no cameras are defined or a camera's
                geometry is invalid.
        """
        data = self.load_yaml(relative_path)
        onboard_data = _require_mapping(data.get("onboard", {}), "sensors.onboard")
        marker_data = _require_mapping(data.get("markers", {}), "sensors.markers")
        degradation_data = _require_mapping(data.get("degradation", {}), "sensors.degradation")

        return SensorConfig(
            tile_size_px=_as_int(data, "tile_size_px", SensorConfig.tile_size_px),
            cameras=_camera_specs(data.get("cameras", [])),
            onboard=OnboardCameraConfig(
                camera_id=str(onboard_data.get("id", OnboardCameraConfig.camera_id)),
                span_tiles=_as_int(onboard_data, "span_tiles", OnboardCameraConfig.span_tiles),
                tile_size_px=_as_int(
                    onboard_data, "tile_size_px", OnboardCameraConfig.tile_size_px
                ),
            ),
            markers=MarkerScaleConfig(
                victim=_as_float(marker_data, "victim", MarkerScaleConfig.victim),
                debris=_as_float(marker_data, "debris", MarkerScaleConfig.debris),
                vehicle=_as_float(marker_data, "vehicle", MarkerScaleConfig.vehicle),
            ),
            degradation=DegradationConfig(
                smoke_density=_as_float(
                    degradation_data, "smoke_density", DegradationConfig.smoke_density
                ),
                smoke_grey=_as_int(degradation_data, "smoke_grey", DegradationConfig.smoke_grey),
                blur_radius=_as_int(
                    degradation_data, "blur_radius", DegradationConfig.blur_radius
                ),
                noise_std=_as_float(degradation_data, "noise_std", DegradationConfig.noise_std),
            ),
        )

    def load_yolo_config(self, relative_path: PathLike) -> YoloTrainingConfig:
        """Load ``configs/training/yolo.yaml`` into a :class:`YoloTrainingConfig`.

        Same optional-key policy as every other config here. ``dataset_dir``
        and ``runs_dir`` are resolved against the project root so a training
        run works from any working directory.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If a value is out of range.
        """
        data = self.load_yaml(relative_path)
        defaults = YoloTrainingConfig()
        return YoloTrainingConfig(
            dataset_dir=self.resolve(str(data.get("dataset_dir", defaults.dataset_dir))),
            runs_dir=self.resolve(str(data.get("runs_dir", defaults.runs_dir))),
            pretrained_weights=self.resolve(
                str(data.get("pretrained_weights", defaults.pretrained_weights))
            ),
            image_size=_as_int(data, "image_size", defaults.image_size),
            epochs=_as_int(data, "epochs", defaults.epochs),
            batch_size=_as_int(data, "batch_size", defaults.batch_size),
            patience=_as_int(data, "patience", defaults.patience),
            horizontal_flip=_as_float(data, "horizontal_flip", defaults.horizontal_flip),
            vertical_flip=_as_float(data, "vertical_flip", defaults.vertical_flip),
            mosaic=_as_float(data, "mosaic", defaults.mosaic),
            scale=_as_float(data, "scale", defaults.scale),
            translate=_as_float(data, "translate", defaults.translate),
            hsv_value=_as_float(data, "hsv_value", defaults.hsv_value),
            confidence=_as_float(data, "confidence", defaults.confidence),
            iou=_as_float(data, "iou", defaults.iou),
            device=str(data.get("device", defaults.device)),
            seed=_as_int(data, "seed", defaults.seed),
        )

    def load_autoencoder_config(self, relative_path: PathLike) -> AutoencoderTrainingConfig:
        """Load ``configs/training/autoencoder.yaml`` into a :class:`AutoencoderTrainingConfig`.

        Same optional-key policy and path resolution as :meth:`load_yolo_config`.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If a value is out of range.
        """
        data = self.load_yaml(relative_path)
        defaults = AutoencoderTrainingConfig()
        return AutoencoderTrainingConfig(
            dataset_dir=self.resolve(str(data.get("dataset_dir", defaults.dataset_dir))),
            runs_dir=self.resolve(str(data.get("runs_dir", defaults.runs_dir))),
            base_channels=_as_int(data, "base_channels", defaults.base_channels),
            depth=_as_int(data, "depth", defaults.depth),
            skip_connections=_as_bool(data, "skip_connections", defaults.skip_connections),
            crop_size=_as_int(data, "crop_size", defaults.crop_size),
            severity_min=_as_float(data, "severity_min", defaults.severity_min),
            severity_max=_as_float(data, "severity_max", defaults.severity_max),
            epochs=_as_int(data, "epochs", defaults.epochs),
            batch_size=_as_int(data, "batch_size", defaults.batch_size),
            learning_rate=_as_float(data, "learning_rate", defaults.learning_rate),
            weight_decay=_as_float(data, "weight_decay", defaults.weight_decay),
            patience=_as_int(data, "patience", defaults.patience),
            loss=str(data.get("loss", defaults.loss)),
            num_workers=_as_int(data, "num_workers", defaults.num_workers),
            device=str(data.get("device", defaults.device)),
            seed=_as_int(data, "seed", defaults.seed),
        )

    def load_lstm_config(self, relative_path: PathLike) -> LstmTrainingConfig:
        """Load ``configs/training/lstm.yaml`` into a :class:`LstmTrainingConfig`.

        Same optional-key policy and path resolution as :meth:`load_yolo_config`.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If a value is out of range.
        """
        data = self.load_yaml(relative_path)
        defaults = LstmTrainingConfig()
        return LstmTrainingConfig(
            trajectories_dir=self.resolve(
                str(data.get("trajectories_dir", defaults.trajectories_dir))
            ),
            runs_dir=self.resolve(str(data.get("runs_dir", defaults.runs_dir))),
            window=_as_int(data, "window", defaults.window),
            horizon=_as_int(data, "horizon", defaults.horizon),
            cone_degrees=_as_float(data, "cone_degrees", defaults.cone_degrees),
            hidden_size=_as_int(data, "hidden_size", defaults.hidden_size),
            num_layers=_as_int(data, "num_layers", defaults.num_layers),
            dropout=_as_float(data, "dropout", defaults.dropout),
            epochs=_as_int(data, "epochs", defaults.epochs),
            batch_size=_as_int(data, "batch_size", defaults.batch_size),
            learning_rate=_as_float(data, "learning_rate", defaults.learning_rate),
            weight_decay=_as_float(data, "weight_decay", defaults.weight_decay),
            patience=_as_int(data, "patience", defaults.patience),
            class_weighting=_as_bool(data, "class_weighting", defaults.class_weighting),
            device=str(data.get("device", defaults.device)),
            seed=_as_int(data, "seed", defaults.seed),
        )

    def load_dqn_config(self, relative_path: PathLike) -> DqnTrainingConfig:
        """Load ``configs/training/dqn.yaml`` into a :class:`DqnTrainingConfig`.

        Same optional-key policy as every other config; ``reward`` is an
        optional nested section.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If a value is out of range.
        """
        data = self.load_yaml(relative_path)
        defaults = DqnTrainingConfig()
        reward_data = _require_mapping(data.get("reward", {}), "dqn.reward")
        reward_defaults = RewardConfig()
        hidden = data.get("hidden_sizes", list(defaults.hidden_sizes))
        if not isinstance(hidden, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) for size in hidden
        ):
            raise ConfigurationError(
                f"config key 'hidden_sizes' must be a list of ints, got {hidden!r}"
            )
        return DqnTrainingConfig(
            runs_dir=self.resolve(str(data.get("runs_dir", defaults.runs_dir))),
            train_seeds=_as_int(data, "train_seeds", defaults.train_seeds),
            eval_seeds=_as_int(data, "eval_seeds", defaults.eval_seeds),
            seed_base=_as_int(data, "seed_base", defaults.seed_base),
            max_episode_steps=_as_int(data, "max_episode_steps", defaults.max_episode_steps),
            total_timesteps=_as_int(data, "total_timesteps", defaults.total_timesteps),
            learning_rate=_as_float(data, "learning_rate", defaults.learning_rate),
            buffer_size=_as_int(data, "buffer_size", defaults.buffer_size),
            learning_starts=_as_int(data, "learning_starts", defaults.learning_starts),
            batch_size=_as_int(data, "batch_size", defaults.batch_size),
            gamma=_as_float(data, "gamma", defaults.gamma),
            train_freq=_as_int(data, "train_freq", defaults.train_freq),
            target_update_interval=_as_int(
                data, "target_update_interval", defaults.target_update_interval
            ),
            exploration_fraction=_as_float(
                data, "exploration_fraction", defaults.exploration_fraction
            ),
            exploration_final_eps=_as_float(
                data, "exploration_final_eps", defaults.exploration_final_eps
            ),
            hidden_sizes=tuple(hidden),
            eval_every=_as_int(data, "eval_every", defaults.eval_every),
            eval_episodes=_as_int(data, "eval_episodes", defaults.eval_episodes),
            rescue_threshold=_as_float(data, "rescue_threshold", defaults.rescue_threshold),
            device=str(data.get("device", defaults.device)),
            seed=_as_int(data, "seed", defaults.seed),
            reward=RewardConfig(
                **{
                    name: _as_float(reward_data, name, getattr(reward_defaults, name))
                    for name in (
                        "progress",
                        "pickup",
                        "delivery",
                        "collision",
                        "damage",
                        "fire_proximity",
                        "step",
                        "battery",
                        "reverse",
                        "completion",
                        "failure",
                        "road_user_hit",
                    )
                }
            ),
            traffic_features=_as_bool(data, "traffic_features", defaults.traffic_features),
        )

    def load_fusion_config(self, relative_path: PathLike) -> FusionTrainingConfig:
        """Load ``configs/training/fusion.yaml`` into a :class:`FusionTrainingConfig`.

        Same optional-key policy and path resolution as every other config.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigurationError: If a value has the wrong type.
            ConfigValidationError: If a value is out of range.
        """
        data = self.load_yaml(relative_path)
        d = FusionTrainingConfig()

        def path(key: str, default: Path) -> Path:
            return self.resolve(str(data.get(key, default)))

        return FusionTrainingConfig(
            dataset_dir=path("dataset_dir", d.dataset_dir),
            runs_dir=path("runs_dir", d.runs_dir),
            dqn_weights=path("dqn_weights", d.dqn_weights),
            lstm_weights=path("lstm_weights", d.lstm_weights),
            belief_lag=_as_int(data, "belief_lag", d.belief_lag),
            train_missions=_as_int(data, "train_missions", d.train_missions),
            val_missions=_as_int(data, "val_missions", d.val_missions),
            eval_missions=_as_int(data, "eval_missions", d.eval_missions),
            seed_base=_as_int(data, "seed_base", d.seed_base),
            oracle_drive_probability=_as_float(
                data, "oracle_drive_probability", d.oracle_drive_probability
            ),
            max_ticks=_as_int(data, "max_ticks", d.max_ticks),
            hidden_sizes=_as_int_tuple(data, "hidden_sizes", d.hidden_sizes),
            dropout=_as_float(data, "dropout", d.dropout),
            epochs=_as_int(data, "epochs", d.epochs),
            batch_size=_as_int(data, "batch_size", d.batch_size),
            learning_rate=_as_float(data, "learning_rate", d.learning_rate),
            weight_decay=_as_float(data, "weight_decay", d.weight_decay),
            patience=_as_int(data, "patience", d.patience),
            critical_weight=_as_float(data, "critical_weight", d.critical_weight),
            label_mode=str(data.get("label_mode", d.label_mode)),
            override_threshold=_as_float(data, "override_threshold", d.override_threshold),
            report_hazards=_as_bool(data, "report_hazards", d.report_hazards),
            report_threshold=_as_float(data, "report_threshold", d.report_threshold),
            device=str(data.get("device", d.device)),
            seed=_as_int(data, "seed", d.seed),
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
            urgency_weight=_as_float(
                mission_data, "urgency_weight", MissionConfig.urgency_weight
            ),
            block_confirm_refreshes=_as_int(
                mission_data, "block_confirm_refreshes", MissionConfig.block_confirm_refreshes
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
            hazards=_hazard_config(_require_mapping(data.get("hazards", {}), "simulation.hazards")),
            traffic=_traffic_config(
                _require_mapping(data.get("traffic", {}), "simulation.traffic")
            ),
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


def _camera_specs(entries: Any) -> tuple[CameraSpec, ...]:
    """Parse the ``cameras`` list of a sensors config file."""
    if not isinstance(entries, list):
        raise ConfigurationError(
            f"config section 'sensors.cameras' must be a list, got {type(entries)}"
        )
    specs: list[CameraSpec] = []
    for index, entry in enumerate(entries):
        camera = _require_mapping(entry, f"sensors.cameras[{index}]")
        origin = camera.get("origin", [0, 0])
        size = camera.get("size", [1, 1])
        try:
            origin_x, origin_y = origin
            width, height = size
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"sensors.cameras[{index}] needs 2-element 'origin' and 'size' lists"
            ) from exc
        specs.append(
            CameraSpec(
                camera_id=str(camera.get("id", f"cctv_{index}")),
                origin_x=int(origin_x),
                origin_y=int(origin_y),
                width_tiles=int(width),
                height_tiles=int(height),
            )
        )
    return tuple(specs)


def _hazard_config(data: dict[str, Any]) -> HazardConfig:
    """Build the hazard config from the ``hazards`` section of ``simulation.yaml``."""
    fire_data = _require_mapping(data.get("fire", {}), "simulation.hazards.fire")
    debris_data = _require_mapping(data.get("debris", {}), "simulation.hazards.debris")
    victim_data = _require_mapping(data.get("victims", {}), "simulation.hazards.victims")

    fire = FireSpreadConfig(
        enabled=_as_bool(fire_data, "enabled", FireSpreadConfig.enabled),
        interval_seconds=_as_float(
            fire_data, "interval_seconds", FireSpreadConfig.interval_seconds
        ),
        growth_per_step=_as_float(fire_data, "growth_per_step", FireSpreadConfig.growth_per_step),
        burnout_per_step=_as_float(
            fire_data, "burnout_per_step", FireSpreadConfig.burnout_per_step
        ),
        ignition_chance=_as_float(fire_data, "ignition_chance", FireSpreadConfig.ignition_chance),
        max_radius=_as_int(fire_data, "max_radius", FireSpreadConfig.max_radius),
        max_active_fires=_as_int(
            fire_data, "max_active_fires", FireSpreadConfig.max_active_fires
        ),
    )
    debris = DebrisCollapseConfig(
        enabled=_as_bool(debris_data, "enabled", DebrisCollapseConfig.enabled),
        interval_seconds=_as_float(
            debris_data, "interval_seconds", DebrisCollapseConfig.interval_seconds
        ),
        collapse_chance=_as_float(
            debris_data, "collapse_chance", DebrisCollapseConfig.collapse_chance
        ),
        max_collapses=_as_int(debris_data, "max_collapses", DebrisCollapseConfig.max_collapses),
    )
    victims = VictimRiskConfig(
        enabled=_as_bool(victim_data, "enabled", VictimRiskConfig.enabled),
        interval_seconds=_as_float(
            victim_data, "interval_seconds", VictimRiskConfig.interval_seconds
        ),
        base_drain=_as_int(victim_data, "base_drain", VictimRiskConfig.base_drain),
        fire_drain=_as_int(victim_data, "fire_drain", VictimRiskConfig.fire_drain),
        fire_radius=_as_float(victim_data, "fire_radius", VictimRiskConfig.fire_radius),
    )
    return HazardConfig(
        seed=_as_int(data, "seed", HazardConfig.seed),
        fire=fire,
        debris=debris,
        victims=victims,
    )


def _traffic_config(data: dict[str, Any]) -> TrafficConfig:
    """Build the traffic config from the ``traffic`` section of ``simulation.yaml``."""
    d = TrafficConfig
    return TrafficConfig(
        enabled=_as_bool(data, "enabled", d.enabled),
        cars=_as_int(data, "cars", d.cars),
        pedestrians=_as_int(data, "pedestrians", d.pedestrians),
        car_step_ticks=_as_int(data, "car_step_ticks", d.car_step_ticks),
        pedestrian_step_ticks=_as_int(data, "pedestrian_step_ticks", d.pedestrian_step_ticks),
        crossing_chance=_as_float(data, "crossing_chance", d.crossing_chance),
        seed=_as_int(data, "seed", d.seed),
        autos=_as_int(data, "autos", d.autos),
        two_wheelers=_as_int(data, "two_wheelers", d.two_wheelers),
        cows=_as_int(data, "cows", d.cows),
        auto_step_ticks=_as_int(data, "auto_step_ticks", d.auto_step_ticks),
        two_wheeler_step_ticks=_as_int(data, "two_wheeler_step_ticks", d.two_wheeler_step_ticks),
        cow_step_ticks=_as_int(data, "cow_step_ticks", d.cow_step_ticks),
    )


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


def _as_int_tuple(data: dict[str, Any], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Read ``data[key]`` as a list of ints, falling back to ``default`` when absent."""
    value = data.get(key, list(default))
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ConfigurationError(f"config key '{key}' must be a list of ints, got {value!r}")
    return tuple(value)
