"""The command center's belief versus the real city (ADR 0003, Phase 7).

Three rules this phase introduced, each tested through whole missions on the
shipped map:

* the vehicle moves through the **real** city — a map that misses debris
  does not let the vehicle drive through it;
* ``LaggedGridSource`` hands back the map as it was ``delay`` refreshes ago;
* the command center re-checks its route against every fresh map, so a
  route through an obstacle the map has only just learned about is dropped
  instead of being driven into.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
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
from sentry_ai.simulation.grid_source import GroundTruthGridSource, LaggedGridSource
from sentry_ai.training.missions import MissionFactory


@pytest.fixture(scope="module")
def factory() -> MissionFactory:
    loader = ConfigLoader(project_root=Path(__file__).resolve().parents[2])
    return MissionFactory.from_app_config(loader, loader.load_app_config("configs/app.yaml"))


class _Always(ILocalController):
    def __init__(self, action: LocalAction) -> None:
        self._action = action

    def decide(self, observation: LocalObservation) -> LocalDecision:
        return LocalDecision(self._action, {a: 0.0 for a in LOCAL_ACTION_ORDER})


class _Blind(IOccupancyGridSource):
    """A map that shows only terrain — no debris, no fire, anywhere."""

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        return OccupancyGrid.terrain_only(city_map)


class _Counting(IOccupancyGridSource):
    """Returns a grid whose (0, 0) cell holds how many times it has been asked."""

    def __init__(self) -> None:
        self.calls = 0

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        self.calls += 1
        grid = OccupancyGrid.empty(3, 3)
        grid.cells[0, 0] = self.calls
        return grid


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
    """A mission whose vehicle holds still on a planned route, and a phantom on its end."""
    phantom = _Phantom()
    mission = factory.build(1, _Always(LocalAction.STOP), phantom)
    mission.engine.tick()
    phantom.tile = mission.controller.route.waypoints[-1]
    return mission, phantom


def _rechecks(mission: Mission) -> int:
    log = mission.controller.events
    return sum("route blocked on the latest map" in e.message for e in log.recent(log.capacity))


class TestTheVehicleMovesInTheRealWorld:
    def test_a_map_without_debris_does_not_let_the_vehicle_through_it(
        self, factory: MissionFactory
    ) -> None:
        """Drive straight ahead forever on a blind map: walls still stop it."""
        mission = factory.build(1, _Always(LocalAction.MOVE_FORWARD), _Blind())
        stats = mission.engine.run(max_ticks=60)
        assert stats.collisions > 0
        assert mission.city_map.is_walkable(mission.city_map.vehicle.position)


class TestLaggedGridSource:
    def test_zero_delay_is_the_inner_source(self) -> None:
        inner = _Counting()
        lagged = LaggedGridSource(inner, delay=0)
        assert [int(lagged.grid_for(None, None).cells[0, 0]) for _ in range(3)] == [1, 2, 3]  # type: ignore[arg-type]

    def test_it_trails_by_the_delay(self) -> None:
        lagged = LaggedGridSource(_Counting(), delay=2)
        seen = [int(lagged.grid_for(None, None).cells[0, 0]) for _ in range(5)]  # type: ignore[arg-type]
        assert seen == [1, 1, 1, 2, 3]

    def test_a_negative_delay_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LaggedGridSource(GroundTruthGridSource(), delay=-1)


class TestRoutesAreRecheckedAgainstFreshMaps:
    def test_ground_truth_missions_are_unchanged(self, factory: MissionFactory) -> None:
        """With a perfect map the re-check never fires, so missions still complete."""
        mission = factory.build(1)
        mission.engine.run(max_ticks=1500)
        assert mission.controller.phase.value == "completed"
        assert not any(
            "route blocked on the latest map" in event.message
            for event in mission.controller.events.recent(mission.controller.events.capacity)
        )

    def test_a_blockage_that_persists_replans(self, factory: MissionFactory) -> None:
        mission, phantom = _waiting_with_phantom(factory)
        confirm = mission.controller.config.block_confirm_refreshes
        phantom.show(refreshes=confirm)
        for _ in range(confirm):
            mission.engine.tick()
        assert _rechecks(mission) == 1

    def test_a_one_refresh_phantom_is_ignored(self, factory: MissionFactory) -> None:
        """A detector flickers; a single false obstacle must not cost the route."""
        mission, phantom = _waiting_with_phantom(factory)
        route = mission.controller.route
        phantom.show(refreshes=1)
        for _ in range(mission.controller.config.block_confirm_refreshes + 2):
            mission.engine.tick()
        assert _rechecks(mission) == 0
        assert mission.controller.route is route

    def test_the_vehicles_own_tile_never_blocks_its_route(self, factory: MissionFactory) -> None:
        """A perceived fire can outrank the vehicle marker; that must not loop replanning."""
        mission = factory.build(1)
        mission.engine.tick()
        grid = mission.controller.grid
        position = mission.city_map.vehicle.position
        grid.mark(position, OccupancyCode.FIRE)
        mission.controller._drop_route_if_blocked(position)  # noqa: SLF001 - the rule under test
        assert mission.controller.next_waypoint() is not None
