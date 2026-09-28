"""Unit tests for LaggedGridSource and the engine's physics seam (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.simulation.grid_source import LaggedGridSource


class _Counting(IOccupancyGridSource):
    """Each call returns a grid whose (0, 0) cell holds the call number's code."""

    def __init__(self) -> None:
        self.calls = 0

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        grid = OccupancyGrid(np.zeros((4, 4), dtype=np.uint8))
        grid.mark(Position(0, 0), OccupancyCode(self.calls % 4))
        self.calls += 1
        return grid


def _code(source: IOccupancyGridSource) -> OccupancyCode:
    return source.grid_for(None, Position(3, 3)).code_at(Position(0, 0))  # type: ignore[arg-type]


class TestLaggedGridSource:
    def test_zero_lag_is_the_wrapped_source(self) -> None:
        source = LaggedGridSource(_Counting(), 0)
        assert [_code(source) for _ in range(3)] == [
            OccupancyCode.ROAD,
            OccupancyCode.BUILDING,
            OccupancyCode.FIRE,
        ]

    def test_answers_lag_calls_late(self) -> None:
        source = LaggedGridSource(_Counting(), 2)
        codes = [_code(source) for _ in range(4)]
        # Calls 0 and 1 have too little history and return the oldest grid.
        assert codes == [
            OccupancyCode.ROAD,
            OccupancyCode.ROAD,
            OccupancyCode.ROAD,
            OccupancyCode.BUILDING,
        ]

    def test_vehicle_is_stamped_where_it_is_now(self) -> None:
        source = LaggedGridSource(_Counting(), 3)
        grid = source.grid_for(None, Position(2, 1))  # type: ignore[arg-type]
        assert grid.code_at(Position(2, 1)) is OccupancyCode.VEHICLE

    def test_old_vehicle_marks_are_cleared(self) -> None:
        class _WithVehicle(IOccupancyGridSource):
            def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
                grid = OccupancyGrid(np.zeros((4, 4), dtype=np.uint8))
                grid.mark_vehicle(vehicle_position)
                return grid

        source = LaggedGridSource(_WithVehicle(), 5)
        source.grid_for(None, Position(0, 0))  # type: ignore[arg-type]
        grid = source.grid_for(None, Position(3, 3))  # type: ignore[arg-type]
        assert grid.code_at(Position(0, 0)) is OccupancyCode.ROAD
        assert grid.positions_with(OccupancyCode.VEHICLE) == [Position(3, 3)]

    def test_rejects_negative_lag(self) -> None:
        with pytest.raises(ValueError):
            LaggedGridSource(_Counting(), -1)

    def test_lag_can_change_while_running(self) -> None:
        source = LaggedGridSource(_Counting(), 0, max_lag=2)
        [_code(source) for _ in range(3)]
        source.lag = 2
        assert _code(source) is OccupancyCode.BUILDING
        source.lag = 0
        # The fifth call's code wraps round to ROAD (codes cycle mod 4).
        assert _code(source) is OccupancyCode.ROAD

    def test_lag_cannot_exceed_its_maximum(self) -> None:
        source = LaggedGridSource(_Counting(), 1, max_lag=3)
        with pytest.raises(ValueError):
            source.lag = 4


class TestLocalReports:
    def test_a_report_overrides_the_stale_map(self) -> None:
        source = LaggedGridSource(_Counting(), 2)
        source.report(Position(1, 1), OccupancyCode.DEBRIS)
        assert source.grid_for(None, Position(3, 3)).code_at(Position(1, 1)) is OccupancyCode.DEBRIS  # type: ignore[arg-type]

    def test_a_report_expires_once_the_map_has_caught_up(self) -> None:
        source = LaggedGridSource(_Counting(), 2)
        source.report(Position(1, 1), OccupancyCode.DEBRIS)
        codes = [
            source.grid_for(None, Position(3, 3)).code_at(Position(1, 1))  # type: ignore[arg-type]
            for _ in range(4)
        ]
        assert codes[:3] == [OccupancyCode.DEBRIS] * 3
        assert codes[3] is OccupancyCode.ROAD
