"""Unit tests for sentry_ai.simulation.onboard_reports (Phase 7-8)."""

from __future__ import annotations

import numpy as np

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.decision import Region, SceneEvidence
from sentry_ai.interfaces.navigation import LocalObservation
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.simulation.grid_source import LaggedGridSource
from sentry_ai.simulation.onboard_reports import OnboardHazardReporter


class _Empty(IOccupancyGridSource):
    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        return OccupancyGrid(np.zeros((6, 6), dtype=np.uint8))


class _Mission:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate_route(self) -> None:
        self.invalidated += 1


def _observation(heading: Heading = Heading.EAST) -> LocalObservation:
    return LocalObservation(Position(2, 2), heading, 90.0, Position(3, 2), False, 0.0)


def _grid(source: LaggedGridSource) -> OccupancyGrid:
    return source.grid_for(None, Position(0, 0))  # type: ignore[arg-type]


class TestOnboardHazardReporter:
    def test_debris_ahead_is_reported_and_the_route_dropped(self) -> None:
        lag, mission = LaggedGridSource(_Empty(), 3), _Mission()
        reporter = OnboardHazardReporter(lag, mission)  # type: ignore[arg-type]
        reporter(SceneEvidence({(EntityKind.OBSTACLE, Region.AHEAD): 0.9}), _observation())
        assert _grid(lag).code_at(Position(3, 2)) is OccupancyCode.DEBRIS
        assert mission.invalidated == 1 and reporter.reports == 1

    def test_fire_ahead_is_reported_as_fire(self) -> None:
        lag, mission = LaggedGridSource(_Empty(), 3), _Mission()
        OnboardHazardReporter(lag, mission)(  # type: ignore[arg-type]
            SceneEvidence({(EntityKind.FIRE, Region.AHEAD): 0.9}), _observation(Heading.SOUTH)
        )
        assert _grid(lag).code_at(Position(2, 3)) is OccupancyCode.FIRE

    def test_weak_or_sideways_evidence_is_not_reported(self) -> None:
        lag, mission = LaggedGridSource(_Empty(), 3), _Mission()
        reporter = OnboardHazardReporter(lag, mission, threshold=0.5)  # type: ignore[arg-type]
        reporter(SceneEvidence({(EntityKind.OBSTACLE, Region.AHEAD): 0.2}), _observation())
        reporter(SceneEvidence({(EntityKind.OBSTACLE, Region.LEFT): 0.9}), _observation())
        assert reporter.reports == 0 and mission.invalidated == 0

    def test_nothing_is_reported_off_the_map_edge(self) -> None:
        lag, mission = LaggedGridSource(_Empty(), 3), _Mission()
        reporter = OnboardHazardReporter(lag, mission)  # type: ignore[arg-type]
        edge = LocalObservation(Position(0, 0), Heading.NORTH, 90.0, None, False, 0.0)
        reporter(SceneEvidence({(EntityKind.OBSTACLE, Region.AHEAD): 0.9}), edge)
        assert reporter.reports == 0
