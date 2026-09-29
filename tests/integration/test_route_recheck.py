"""The command center re-checks its route against every fresh map (adopted from Phase 8).

Ported with the rule from the teammate's build. A route that the latest map
shows blocked on ``block_confirm_refreshes`` consecutive refreshes is dropped
and replanned; a single-refresh phantom is ignored; the vehicle's own tile
never counts; and on a perfect map the rule never fires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import MissionConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.simulation.factory import Mission
from sentry_ai.simulation.grid_source import GroundTruthGridSource
from sentry_ai.training.missions import MissionFactory

_MESSAGE = "route blocked on the latest map"


@pytest.fixture(scope="module")
def factory() -> MissionFactory:
    loader = ConfigLoader(project_root=Path(__file__).resolve().parents[2])
    return MissionFactory.from_app_config(
        loader, loader.load_app_config("configs/app.yaml"), randomise_start=False
    )


class _Always(ILocalController):
    def __init__(self, action: LocalAction) -> None:
        self._action = action

    def decide(self, observation: LocalObservation) -> LocalDecision:
        return LocalDecision(self._action, {a: 0.0 for a in LOCAL_ACTION_ORDER})


class _Phantom(IOccupancyGridSource):
    """Ground truth, plus debris on one tile for the next few refreshes."""

    def __init__(self) -> None:
        self._truth = GroundTruthGridSource()
        self.tile: Position | None = None
        self._left = 0

    def show(self, refreshes: int) -> None:
        self._left = refreshes

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        grid = self._truth.grid_for(city_map, vehicle_position)
        if self._left > 0 and self.tile is not None:
            grid.mark(self.tile, OccupancyCode.DEBRIS)
            self._left -= 1
        return grid


def _waiting_with_phantom(factory: MissionFactory) -> tuple[Mission, _Phantom]:
    """A mission whose vehicle holds still on a planned route, a phantom on its last tile."""
    mission = factory.build(1, _Always(LocalAction.STOP))
    phantom = _Phantom()
    mission.controller.grid_source = phantom
    mission.engine.tick()
    phantom.tile = mission.controller.route.waypoints[-1]
    return mission, phantom


def _rechecks(mission: Mission) -> int:
    log = mission.controller.events
    return sum(_MESSAGE in event.message for event in log.recent(log.capacity))


def test_ground_truth_missions_never_trigger_it(factory: MissionFactory) -> None:
    mission = factory.build(1)
    mission.engine.run(max_ticks=1500)
    assert mission.controller.phase.value == "completed"
    assert _rechecks(mission) == 0


def test_a_blockage_that_persists_replans(factory: MissionFactory) -> None:
    mission, phantom = _waiting_with_phantom(factory)
    confirm = mission.controller.config.block_confirm_refreshes
    phantom.show(refreshes=confirm)
    for _ in range(confirm):
        mission.engine.tick()
    assert _rechecks(mission) == 1


def test_a_one_refresh_phantom_is_ignored(factory: MissionFactory) -> None:
    mission, phantom = _waiting_with_phantom(factory)
    route = mission.controller.route
    phantom.show(refreshes=1)
    for _ in range(mission.controller.config.block_confirm_refreshes + 2):
        mission.engine.tick()
    assert _rechecks(mission) == 0
    assert mission.controller.route is route


def test_the_vehicles_own_tile_never_blocks_its_route(factory: MissionFactory) -> None:
    mission = factory.build(1)
    mission.engine.tick()
    position = mission.city_map.vehicle.position
    mission.controller.grid.mark(position, OccupancyCode.FIRE)
    for _ in range(mission.controller.config.block_confirm_refreshes):
        mission.controller._drop_route_if_blocked(position)  # noqa: SLF001 - the rule under test
    assert mission.controller.next_waypoint() is not None


def test_confirm_refreshes_must_be_positive() -> None:
    with pytest.raises(ConfigValidationError):
        MissionConfig(block_confirm_refreshes=0)
