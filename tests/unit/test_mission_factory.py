"""Unit tests for sentry_ai.simulation.factory."""

from __future__ import annotations

from typing import Any

import pytest

from sentry_ai.config.schema import SimulationConfig, VehicleConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision, LocalObservation
from sentry_ai.simulation.factory import build_mission
from sentry_ai.simulation.mission import MissionPhase

_MAP_DATA: dict[str, Any] = {
    "width": 7,
    "height": 5,
    "grid": [".......", "..##...", "..##...", "=======", "...t..."],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
    "vehicle_start": [6, 0],
    "victims": [{"id": "victim_a", "position": [5, 4]}],
    "fires": [{"id": "seed_fire", "position": [2, 1], "intensity": 0.5, "radius": 1}],
}


def _city_map() -> CityMap:
    return CityMap.from_config(_MAP_DATA)


class _Stationary:
    """A controller that never moves, so a mission can be observed at rest."""

    def decide(self, observation: LocalObservation) -> LocalDecision:
        del observation
        return LocalDecision(action=LocalAction.STOP, q_values={})


class TestBuildMission:
    def test_a_fresh_mission_starts_in_planning(self) -> None:
        mission = build_mission(_city_map(), SimulationConfig(), VehicleConfig())
        assert mission.controller.phase is MissionPhase.PLANNING
        assert mission.engine.stats.ticks == 0

    def test_the_engine_applies_the_vehicle_config(self) -> None:
        vehicle_config = VehicleConfig(capacity=3)
        mission = build_mission(_city_map(), SimulationConfig(), vehicle_config)
        assert mission.city_map.vehicle.capacity == 3

    def test_the_controller_can_be_injected(self) -> None:
        mission = build_mission(
            _city_map(), SimulationConfig(), VehicleConfig(), controller=_Stationary()
        )
        start = mission.city_map.vehicle.position
        for _ in range(10):
            mission.engine.tick()
        assert mission.city_map.vehicle.position == start

    def test_the_default_controller_actually_drives(self) -> None:
        mission = build_mission(_city_map(), SimulationConfig(), VehicleConfig())
        start = mission.city_map.vehicle.position
        for _ in range(30):
            mission.engine.tick()
        assert mission.city_map.vehicle.position != start

    def test_controller_exposes_the_command_center(self) -> None:
        mission = build_mission(_city_map(), SimulationConfig(), VehicleConfig())
        assert mission.controller is mission.engine.mission


class TestHazardSeedOverride:
    """Re-seeding is the only source of variety the training set has."""

    def _hazards_after(self, seed: int | None, ticks: int = 1200) -> list[tuple[int, int]]:
        """Where fires and collapses ended up — this disaster's fingerprint.

        Driven by a controller that never moves, so the mission runs to its
        time limit rather than finishing in a few seconds. Hazards need time
        to diverge, and a mission that ends promptly barely gives them any.
        """
        mission = build_mission(
            _city_map(),
            SimulationConfig(),
            VehicleConfig(),
            controller=_Stationary(),
            hazard_seed=seed,
        )
        for _ in range(ticks):
            if mission.engine.tick() is None:
                break
        return sorted(
            entity.position.as_tuple()
            for entity in (*mission.city_map.fires, *mission.city_map.obstacles)
        )

    def test_the_same_seed_produces_the_same_disaster(self) -> None:
        assert self._hazards_after(7) == self._hazards_after(7)

    def test_different_seeds_produce_different_disasters(self) -> None:
        outcomes = {tuple(self._hazards_after(seed)) for seed in range(8)}
        assert len(outcomes) > 1

    def test_omitting_the_seed_uses_the_configured_one(self) -> None:
        configured = SimulationConfig().hazards.seed
        assert self._hazards_after(None) == self._hazards_after(configured)

    def test_overriding_the_seed_does_not_mutate_the_config(self) -> None:
        """The config is frozen and shared; the override must be a copy."""
        config = SimulationConfig()
        build_mission(_city_map(), config, VehicleConfig(), hazard_seed=999)
        assert config.hazards.seed != 999


class TestIsolationBetweenMissions:
    def test_two_missions_over_separate_maps_do_not_interfere(self) -> None:
        """The dataset builder runs many missions; they must not bleed."""
        first = build_mission(_city_map(), SimulationConfig(), VehicleConfig(), hazard_seed=1)
        for _ in range(200):
            if first.engine.tick() is None:
                break

        second = build_mission(_city_map(), SimulationConfig(), VehicleConfig(), hazard_seed=1)
        assert second.city_map.vehicle.position == Position(6, 0)
        assert second.engine.stats.ticks == 0
        assert all(victim.health == 100 for victim in second.city_map.victims)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_seed_produces_a_runnable_mission(seed: int) -> None:
    """A seed that crashes a mission would silently corrupt the dataset."""
    mission = build_mission(_city_map(), SimulationConfig(), VehicleConfig(), hazard_seed=seed)
    for _ in range(500):
        if mission.engine.tick() is None:
            break
    assert mission.controller.phase.is_terminal
