"""Unit tests for sentry_ai.domain.occupancy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid

# 4x3 city:
#   row0: ....
#   row1: .##.
#   row2: .r.t
_MAP_DATA: dict[str, Any] = {
    "width": 4,
    "height": 3,
    "grid": ["....", ".##.", ".r.t"],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 2},
    "vehicle_start": [0, 1],
    "victims": [{"id": "v1", "position": [2, 2]}],
    "fires": [{"id": "f1", "position": [3, 0], "intensity": 0.8, "radius": 0}],
}


class TestOccupancyCode:
    @pytest.mark.parametrize(
        "code", [OccupancyCode.BUILDING, OccupancyCode.FIRE, OccupancyCode.DEBRIS]
    )
    def test_impassable_codes(self, code: OccupancyCode) -> None:
        assert code.is_traversable is False

    @pytest.mark.parametrize(
        "code",
        [
            OccupancyCode.ROAD,
            OccupancyCode.VICTIM,
            OccupancyCode.HOSPITAL,
            OccupancyCode.VEHICLE,
        ],
    )
    def test_traversable_codes(self, code: OccupancyCode) -> None:
        assert code.is_traversable is True

    def test_numeric_values_match_the_documented_contract(self) -> None:
        """These integers are written into model inputs — they must not drift."""
        assert [int(code) for code in OccupancyCode] == [0, 1, 2, 3, 4, 5, 6]


class TestConstruction:
    def test_empty_is_all_road(self) -> None:
        grid = OccupancyGrid.empty(width=3, height=2)
        assert (grid.width, grid.height) == (3, 2)
        assert np.all(grid.cells == OccupancyCode.ROAD)

    @pytest.mark.parametrize(("width", "height"), [(0, 2), (2, 0), (-1, 1)])
    def test_rejects_non_positive_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(DomainValidationError):
            OccupancyGrid.empty(width=width, height=height)

    def test_rejects_non_2d_cells(self) -> None:
        with pytest.raises(DomainValidationError):
            OccupancyGrid(cells=np.zeros((3,), dtype=np.uint8))


class TestFromCityMap:
    def test_projects_terrain_to_codes(self) -> None:
        grid = OccupancyGrid.from_city_map(CityMap.from_config(_MAP_DATA))
        assert grid.code_at(Position(1, 1)) is OccupancyCode.BUILDING
        assert grid.code_at(Position(1, 2)) is OccupancyCode.DEBRIS  # rubble
        assert grid.code_at(Position(3, 2)) is OccupancyCode.DEBRIS  # tree
        assert grid.code_at(Position(2, 0)) is OccupancyCode.ROAD  # open ground

    def test_entities_override_terrain(self) -> None:
        grid = OccupancyGrid.from_city_map(CityMap.from_config(_MAP_DATA))
        assert grid.code_at(Position(2, 2)) is OccupancyCode.VICTIM
        assert grid.code_at(Position(3, 0)) is OccupancyCode.FIRE
        assert grid.code_at(Position(0, 0)) is OccupancyCode.HOSPITAL
        assert grid.code_at(Position(0, 1)) is OccupancyCode.VEHICLE

    def test_rescued_victims_are_not_stamped(self) -> None:
        city_map = CityMap.from_config(_MAP_DATA)
        city_map.victims[0].status = VictimStatus.RESCUED
        grid = OccupancyGrid.from_city_map(city_map)
        assert grid.code_at(Position(2, 2)) is not OccupancyCode.VICTIM


class TestQueriesAndMutation:
    def test_out_of_bounds_is_not_traversable(self) -> None:
        grid = OccupancyGrid.empty(width=2, height=2)
        assert grid.is_traversable(Position(9, 9)) is False

    def test_code_at_out_of_bounds_raises(self) -> None:
        grid = OccupancyGrid.empty(width=2, height=2)
        with pytest.raises(DomainValidationError):
            grid.code_at(Position(5, 5))

    def test_mark_out_of_bounds_raises(self) -> None:
        grid = OccupancyGrid.empty(width=2, height=2)
        with pytest.raises(DomainValidationError):
            grid.mark(Position(5, 5), OccupancyCode.FIRE)

    def test_mark_radius_covers_a_disc_and_clips_at_the_edge(self) -> None:
        grid = OccupancyGrid.empty(width=5, height=5)
        grid.mark_radius(Position(0, 0), radius=1, code=OccupancyCode.FIRE)
        assert grid.code_at(Position(0, 0)) is OccupancyCode.FIRE
        assert grid.code_at(Position(1, 0)) is OccupancyCode.FIRE
        assert grid.code_at(Position(1, 1)) is OccupancyCode.ROAD  # distance sqrt(2) > 1
        assert grid.code_at(Position(2, 0)) is OccupancyCode.ROAD

    def test_positions_with_finds_every_match(self) -> None:
        grid = OccupancyGrid.empty(width=3, height=3)
        grid.mark(Position(1, 2), OccupancyCode.VICTIM)
        grid.mark(Position(2, 0), OccupancyCode.VICTIM)
        found = {p.as_tuple() for p in grid.positions_with(OccupancyCode.VICTIM)}
        assert found == {(1, 2), (2, 0)}

    def test_as_array_returns_a_defensive_copy(self) -> None:
        grid = OccupancyGrid.empty(width=2, height=2)
        snapshot = grid.as_array()
        grid.mark(Position(0, 0), OccupancyCode.FIRE)
        assert snapshot[0, 0] == OccupancyCode.ROAD
