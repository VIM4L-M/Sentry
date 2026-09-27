"""Unit tests for trajectory recording in sentry_ai.training.trajectories.

Recording runs real missions on the shipped map, because what is being
tested is exactly the join between the simulation and the dataset: that a
trajectory is the vehicle's actual path, tick by tick, and that randomised
starts are legal tiles to start from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.enums import Heading, TerrainType
from sentry_ai.domain.map import CityMap
from sentry_ai.sequence.behaviour import heading_to_degrees
from sentry_ai.simulation.factory import Mission, build_mission
from sentry_ai.training.missions import start_positions
from sentry_ai.training.trajectories import TrajectoryRecorder, vehicle_state


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


@pytest.fixture
def city(loader: ConfigLoader) -> CityMap:
    app_config = loader.load_app_config("configs/app.yaml")
    return CityMap.from_config(loader.load_yaml(app_config.map_config_path))


def _source(loader: ConfigLoader):  # type: ignore[no-untyped-def]
    app_config = loader.load_app_config("configs/app.yaml")
    simulation = loader.load_simulation_config(app_config.simulation_config_path)
    vehicle = loader.load_vehicle_config(app_config.vehicle_config_path)
    map_data = loader.load_yaml(app_config.map_config_path)

    def build(seed: int) -> Mission:
        return build_mission(
            city_map=CityMap.from_config(map_data),
            simulation_config=simulation,
            vehicle_config=vehicle,
            hazard_seed=seed,
        )

    return build


class TestStartPositions:
    def test_every_start_is_open_road(self, city: CityMap) -> None:
        starts = start_positions(city)
        assert starts
        assert all(city.tile_at(position) is TerrainType.ROAD for position in starts)

    def test_no_start_is_on_top_of_a_victim(self, city: CityMap) -> None:
        victims = {victim.position for victim in city.victims}
        assert not victims & set(start_positions(city))

    def test_every_start_builds_a_valid_map(self, loader: ConfigLoader, city: CityMap) -> None:
        """CityMap validates its start tile; every candidate must pass it."""
        app_config = loader.load_app_config("configs/app.yaml")
        data = loader.load_yaml(app_config.map_config_path)
        for start in start_positions(city):
            CityMap.from_config({**data, "vehicle_start": [start.x, start.y]})

    def test_the_order_is_stable(self, city: CityMap) -> None:
        assert start_positions(city) == start_positions(city)


class TestRecorder:
    def test_a_trajectory_is_the_mission_tick_by_tick(self, loader: ConfigLoader) -> None:
        trajectory = TrajectoryRecorder(_source(loader), max_ticks=40).record(seed=3)
        assert len(trajectory.states) == 41  # the start, then one per tick
        assert trajectory.seed == 3

    def test_the_vehicle_never_teleports(self, loader: ConfigLoader) -> None:
        states = TrajectoryRecorder(_source(loader), max_ticks=200).record(seed=5).states
        for before, after in zip(states, states[1:], strict=False):
            step = abs(after.position.x - before.position.x)
            step += abs(after.position.y - before.position.y)
            assert step <= 1

    def test_a_finished_mission_records_its_outcome(self, loader: ConfigLoader) -> None:
        trajectory = TrajectoryRecorder(_source(loader)).record(seed=1)
        assert trajectory.outcome == "completed"

    def test_the_state_is_what_the_vehicle_holds(self, city: CityMap) -> None:
        city.vehicle.heading = Heading.WEST
        state = vehicle_state(city.vehicle)
        assert state.position == city.vehicle.position
        assert state.heading_degrees == heading_to_degrees(Heading.WEST)
