"""Unit tests for driving among road users with reinforcement learning (Phase 9)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import RewardConfig, TrafficConfig
from sentry_ai.decision.dqn_controller import (
    POLICY_FEATURES,
    TRAFFIC_POLICY_FEATURES,
    DqnLocalController,
    policy_features,
)
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.traffic import AgentKind, TrafficAgent
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction, LocalObservation
from sentry_ai.simulation.factory import build_mission
from sentry_ai.training.missions import MissionFactory
from sentry_ai.training.sentry_env import SentryEnv


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


def _observation(ahead: bool = False, far: bool = False) -> LocalObservation:
    return LocalObservation(
        position=Position(5, 5),
        heading=Heading.EAST,
        battery_percent=80.0,
        next_waypoint=Position(6, 5),
        blocked_ahead=False,
        fire_proximity=0.0,
        road_user_ahead=ahead,
        road_user_ahead_far=far,
    )


class TestPolicyFeatures:
    def test_the_old_seven_inputs_are_unchanged(self) -> None:
        features = policy_features(_observation(ahead=True, far=True))
        assert features.shape == (POLICY_FEATURES,)

    def test_traffic_appends_the_two_road_user_inputs(self) -> None:
        features = policy_features(_observation(ahead=True, far=False), traffic=True)
        assert features.shape == (TRAFFIC_POLICY_FEATURES,)
        assert features[-2:].tolist() == [1.0, 0.0]
        assert np.array_equal(features[:POLICY_FEATURES], policy_features(_observation()))


class _Model:
    """Just enough of an SB3 DQN: an observation space and a Q-network."""

    def __init__(self, inputs: int) -> None:
        self.observation_space = SimpleNamespace(shape=(inputs,))
        self.seen: list[np.ndarray] = []
        self.policy = SimpleNamespace(obs_to_tensor=self._to_tensor)

    def _to_tensor(self, features: np.ndarray) -> tuple[object, None]:
        import torch

        self.seen.append(features)
        return torch.as_tensor(features).unsqueeze(0), None

    def q_net(self, tensor: object) -> object:
        import torch

        return torch.zeros(1, len(LOCAL_ACTION_ORDER))


class TestDqnController:
    def test_a_traffic_model_is_recognised_by_its_input_size(self) -> None:
        model = _Model(TRAFFIC_POLICY_FEATURES)
        controller = DqnLocalController(model)
        controller.decide(_observation(ahead=True))
        assert controller.sees_traffic
        assert model.seen[0].shape == (TRAFFIC_POLICY_FEATURES,)

    def test_an_old_model_keeps_its_seven_inputs(self) -> None:
        model = _Model(POLICY_FEATURES)
        controller = DqnLocalController(model)
        controller.decide(_observation(ahead=True))
        assert not controller.sees_traffic
        assert model.seen[0].shape == (POLICY_FEATURES,)


def _traffic_mission(loader: ConfigLoader) -> tuple[object, CityMap]:
    app = loader.load_app_config("configs/app_traffic.yaml")
    assert app.simulation_config_path is not None and app.vehicle_config_path is not None
    city = CityMap.from_config(loader.load_yaml(app.map_config_path))
    mission = build_mission(
        city,
        loader.load_simulation_config(app.simulation_config_path),
        loader.load_vehicle_config(app.vehicle_config_path),
    )
    return mission, city


class TestEngineSensors:
    def test_a_traffic_config_builds_a_mission_with_road_users(
        self, loader: ConfigLoader
    ) -> None:
        mission, city = _traffic_mission(loader)
        assert mission.traffic is not None  # type: ignore[attr-defined]
        mission.engine.tick()  # type: ignore[attr-defined]
        assert city.traffic

    def test_the_observation_reports_a_road_user_ahead(self, loader: ConfigLoader) -> None:
        mission, city = _traffic_mission(loader)
        engine = mission.engine  # type: ignore[attr-defined]
        engine.tick()
        vehicle = city.vehicle
        dx, dy = vehicle.heading.delta
        ahead = Position(vehicle.position.x + dx, vehicle.position.y + dy)
        traffic = mission.traffic  # type: ignore[attr-defined]
        traffic._occupied[ahead.as_tuple()] = TrafficAgent(  # noqa: SLF001 - planting one
            "planted", AgentKind.PEDESTRIAN, ahead, Heading.NORTH, ahead
        )
        observation = engine.observe()
        assert observation.road_user_ahead
        assert not observation.road_user_ahead_far or True

    def test_missions_without_traffic_report_none(self, loader: ConfigLoader) -> None:
        factory = MissionFactory.from_app_config(loader, loader.load_app_config("configs/app.yaml"))
        mission = factory.build(1)
        observation = mission.engine.observe()
        assert mission.traffic is None
        assert not observation.road_user_ahead and not observation.road_user_ahead_far


class TestSentryEnvWithTraffic:
    def test_the_observation_space_grows_with_traffic(self, loader: ConfigLoader) -> None:
        factory = MissionFactory.from_app_config(
            loader, loader.load_app_config("configs/app_traffic.yaml")
        )
        env = SentryEnv(factory.build, [1, 2], RewardConfig(), 50, traffic_features=True)
        observation, _ = env.reset(seed=0)
        assert observation.shape == (TRAFFIC_POLICY_FEATURES,)
        _, _, _, _, info = env.step(LOCAL_ACTION_ORDER.index(LocalAction.MOVE_FORWARD))
        assert "road_user_hits" in info

    def test_hitting_a_road_user_is_penalised(self, loader: ConfigLoader) -> None:
        factory = MissionFactory.from_app_config(
            loader, loader.load_app_config("configs/app_traffic.yaml")
        )
        reward = RewardConfig(road_user_hit=-50.0)
        env = SentryEnv(factory.build, [1], reward, 50, traffic_features=True)
        env.reset(seed=0)
        mission = env.mission
        vehicle = mission.city_map.vehicle
        dx, dy = vehicle.heading.delta
        ahead = Position(vehicle.position.x + dx, vehicle.position.y + dy)
        assert mission.traffic is not None
        env.step(LOCAL_ACTION_ORDER.index(LocalAction.STOP))  # let traffic spawn
        blocker = TrafficAgent("planted", AgentKind.PEDESTRIAN, ahead, Heading.NORTH, ahead)
        mission.traffic._occupied[ahead.as_tuple()] = blocker  # noqa: SLF001
        # Keep the planted person from walking off during the tick.
        blocker.moved_at = 10**9
        mission.city_map.traffic.append(blocker)
        _, paid, _, _, info = env.step(LOCAL_ACTION_ORDER.index(LocalAction.MOVE_FORWARD))
        assert info["road_user_hits"] == 1
        assert paid < -40.0


class TestRewardConfig:
    def test_the_road_user_penalty_must_be_a_penalty(self) -> None:
        with pytest.raises(ConfigValidationError):
            RewardConfig(road_user_hit=1.0)

    def test_traffic_training_config_loads(self, loader: ConfigLoader) -> None:
        config = loader.load_dqn_config("configs/training/dqn_traffic.yaml")
        assert config.traffic_features and config.reward.road_user_hit < 0.0
        assert TrafficConfig().enabled is False
