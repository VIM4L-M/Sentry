"""Unit tests for SentryEnv and MissionFactory (Phase 6).

The environment runs real missions on the shipped map — what is being
tested is the join between Gymnasium and the simulation: that the agent's
action is the one the vehicle takes, that each reward term fires on the
fact it names, and that an episode ends when the mission does.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import RewardConfig
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction
from sentry_ai.simulation.waypoint_follower import WaypointFollower
from sentry_ai.training.missions import MissionFactory
from sentry_ai.training.sentry_env import SentryEnv

gymnasium = pytest.importorskip("gymnasium", reason="SentryEnv needs gymnasium")

_ZERO = RewardConfig(
    progress=0.0,
    pickup=0.0,
    delivery=0.0,
    collision=0.0,
    damage=0.0,
    fire_proximity=0.0,
    step=0.0,
    battery=0.0,
    reverse=0.0,
    completion=0.0,
    failure=0.0,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def factory() -> MissionFactory:
    loader = ConfigLoader(project_root=_PROJECT_ROOT)
    return MissionFactory.from_app_config(loader, loader.load_app_config("configs/app.yaml"))


def _env(factory: MissionFactory, reward: RewardConfig = _ZERO, steps: int = 1500) -> SentryEnv:
    return SentryEnv(factory.build, range(100, 110), reward, steps)


def _index(action: LocalAction) -> int:
    return LOCAL_ACTION_ORDER.index(action)


def _follow(env: SentryEnv) -> tuple[float, int, dict[str, object]]:
    """Drive an episode with the waypoint follower's choices; return (return, steps, info)."""
    follower = WaypointFollower()
    total, steps, info = 0.0, 0, {}
    while True:
        action = follower.decide(env.mission.engine.observe()).action
        _, reward, terminated, truncated, info = env.step(_index(action))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, info


class TestMissionFactory:
    def test_the_same_seed_is_the_same_mission(self, factory: MissionFactory) -> None:
        assert factory.map_data_for(7)["vehicle_start"] == factory.map_data_for(7)["vehicle_start"]

    def test_seeds_start_in_different_places(self, factory: MissionFactory) -> None:
        starts = {tuple(factory.map_data_for(seed)["vehicle_start"]) for seed in range(30)}
        assert len(starts) > 10

    def test_a_fixed_start_keeps_the_maps_own(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        app = loader.load_app_config("configs/app.yaml")
        fixed = MissionFactory.from_app_config(loader, app, randomise_start=False)
        original = loader.load_yaml(app.map_config_path)["vehicle_start"]
        assert fixed.map_data_for(3)["vehicle_start"] == original

    def test_the_controller_is_the_one_given(self, factory: MissionFactory) -> None:
        follower = WaypointFollower()
        mission = factory.build(1, follower)
        assert mission.engine._controller is follower  # noqa: SLF001 - the wiring under test


class TestGymContract:
    def test_it_passes_gymnasiums_own_checker(self, factory: MissionFactory) -> None:
        from gymnasium.utils.env_checker import check_env

        check_env(_env(factory), skip_render_check=True)

    def test_observations_are_the_policy_features(self, factory: MissionFactory) -> None:
        env = _env(factory)
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert "mission_seed" in info

    def test_a_mission_seed_can_be_pinned(self, factory: MissionFactory) -> None:
        _, info = _env(factory).reset(options={"mission_seed": 4242})
        assert info["mission_seed"] == 4242

    def test_the_env_seed_makes_episodes_reproducible(self, factory: MissionFactory) -> None:
        first = _env(factory).reset(seed=11)[1]["mission_seed"]
        assert _env(factory).reset(seed=11)[1]["mission_seed"] == first

    def test_using_it_before_reset_is_an_error(self, factory: MissionFactory) -> None:
        with pytest.raises(RuntimeError):
            _env(factory).step(0)

    @pytest.mark.parametrize("kwargs", [{"seeds": []}, {"max_episode_steps": 0}])
    def test_degenerate_setups_are_rejected(
        self, factory: MissionFactory, kwargs: dict[str, object]
    ) -> None:
        options: dict[str, object] = {
            "factory": factory.build,
            "seeds": [1],
            "reward": _ZERO,
            "max_episode_steps": 10,
        }
        options.update(kwargs)
        with pytest.raises(ValueError):
            SentryEnv(**options)  # type: ignore[arg-type]


class TestTheAgentDrives:
    def test_the_vehicle_takes_the_agents_action(self, factory: MissionFactory) -> None:
        env = _env(factory)
        env.reset(options={"mission_seed": 1})
        heading = env.mission.city_map.vehicle.heading
        env.step(_index(LocalAction.TURN_LEFT))
        assert env.mission.city_map.vehicle.heading is heading.turn_left()

    def test_driving_like_the_follower_completes_the_mission(
        self, factory: MissionFactory
    ) -> None:
        env = _env(factory)
        env.reset(options={"mission_seed": 1})
        _, _, info = _follow(env)
        assert info["phase"] == "completed" and info["rescued"] == 4

    def test_episodes_are_truncated_at_the_step_limit(self, factory: MissionFactory) -> None:
        env = _env(factory, steps=5)
        env.reset(options={"mission_seed": 1})
        for _ in range(4):
            assert not env.step(_index(LocalAction.STOP))[3]
        _, _, terminated, truncated, _ = env.step(_index(LocalAction.STOP))
        assert truncated and not terminated


class TestReward:
    def _one_step(
        self, factory: MissionFactory, reward: RewardConfig, action: LocalAction
    ) -> float:
        env = _env(factory, reward)
        env.reset(options={"mission_seed": 1})
        return env.step(_index(action))[1]

    def test_all_zero_weights_pay_nothing(self, factory: MissionFactory) -> None:
        assert self._one_step(factory, _ZERO, LocalAction.MOVE_FORWARD) == 0.0

    def test_the_step_cost_is_charged_every_tick(self, factory: MissionFactory) -> None:
        reward = replace(_ZERO, step=-0.5)
        assert self._one_step(factory, reward, LocalAction.STOP) == pytest.approx(-0.5)

    def test_reversing_is_charged(self, factory: MissionFactory) -> None:
        reward = replace(_ZERO, reverse=-0.3)
        assert self._one_step(factory, reward, LocalAction.REVERSE) == pytest.approx(-0.3)
        assert self._one_step(factory, reward, LocalAction.STOP) == 0.0

    def test_battery_spent_is_charged(self, factory: MissionFactory) -> None:
        reward = replace(_ZERO, battery=-1.0)
        assert self._one_step(factory, reward, LocalAction.TURN_LEFT) < 0.0

    def test_progress_pays_for_closing_on_the_waypoint(self, factory: MissionFactory) -> None:
        env = _env(factory, replace(_ZERO, progress=1.0))
        env.reset(options={"mission_seed": 1})
        follower = WaypointFollower()
        rewards = []
        for _ in range(40):
            action = follower.decide(env.mission.engine.observe()).action
            rewards.append(env.step(_index(action))[1])
        assert max(rewards) == pytest.approx(1.0)  # one tile closer, once per move
        assert min(rewards) >= 0.0  # the follower never drives away

    def test_the_follower_earns_far_more_than_random_driving(
        self, factory: MissionFactory
    ) -> None:
        """The reward must rank good driving above bad, or nothing can be learned."""
        env = _env(factory, RewardConfig(), steps=600)
        env.reset(options={"mission_seed": 1})
        follower_return, _, _ = _follow(env)

        env.reset(options={"mission_seed": 1})
        rng = np.random.default_rng(0)
        random_return = 0.0
        for _ in range(600):
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(5)))
            random_return += reward
            if terminated or truncated:
                break
        assert follower_return > 100.0 > 0.0 > random_return
