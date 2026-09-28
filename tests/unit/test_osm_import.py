"""Unit tests for sentry_ai.mapping.osm — OpenStreetMap ways to a disaster city.

Built on synthetic OSM elements, so no network is touched.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from sentry_ai.domain.map import CityMap
from sentry_ai.mapping.osm import (
    BUILDING,
    OPEN,
    ROAD,
    DisasterSpec,
    box_around,
    connect,
    hospital_tile,
    infer_blocks,
    rasterise,
    stage_disaster,
)
from sentry_ai.mapping.overpass import BoundingBox

BOX = BoundingBox(south=0.0, west=0.0, north=1.0, east=1.0)
LEGEND = {".": "open_ground", "#": "building", "=": "road", "x": "collapsed_building",
          "r": "rubble", "t": "tree", "!": "blocked_road"}  # fmt: skip


def _point(x: float, y: float, width: int = 10, height: int = 10) -> dict[str, float]:
    """The lat/lon of grid coordinate (x, y) in BOX."""
    return {"lon": x / width, "lat": 1.0 - y / height}


def _road(points: list[tuple[float, float]]) -> dict[str, Any]:
    return {"tags": {"highway": "residential"}, "geometry": [_point(x, y) for x, y in points]}


def _building(x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return {"tags": {"building": "yes"}, "geometry": [_point(x, y) for x, y in corners]}


class TestRasterise:
    def test_a_road_becomes_a_line_of_road_tiles(self) -> None:
        grid = rasterise([_road([(0.5, 5.5), (9.5, 5.5)])], BOX, 10, 10)
        assert "".join(grid[5]) == ROAD * 10

    def test_a_building_fills_the_tiles_whose_centres_it_covers(self) -> None:
        grid = rasterise([_building(2, 2, 5, 4)], BOX, 10, 10)
        assert grid[2][2:5] == [BUILDING] * 3
        assert grid[3][2:5] == [BUILDING] * 3
        assert grid[4][2] == OPEN

    def test_roads_win_over_buildings(self) -> None:
        grid = rasterise([_building(0, 0, 10, 10), _road([(0.5, 3.5), (9.5, 3.5)])], BOX, 10, 10)
        assert set(grid[3]) == {ROAD}

    def test_diagonal_roads_stay_four_connected(self) -> None:
        grid = connect(
            rasterise([_building(0, 0, 10, 10), _road([(0.5, 0.5), (9.5, 9.5)])], BOX, 10, 10)
        )
        roads = sum(row.count(ROAD) for row in grid)
        assert roads >= 19  # a staircase, not isolated diagonal tiles

    def test_unknown_ways_are_ignored(self) -> None:
        grid = rasterise(
            [{"tags": {"highway": "footway"}, "geometry": [_point(0, 5), _point(9, 5)]}],
            BOX,
            10,
            10,
        )
        assert all(cell == OPEN for row in grid for cell in row)


class TestConnect:
    def test_stranded_pockets_become_building(self) -> None:
        grid = [list(row) for row in ["....#.", "....#.", "#####.", "......"]]
        connected = connect(grid)
        # The 8-tile pocket top-left is smaller than the L-shaped region; it is walled off.
        assert connected[0][0] == BUILDING
        assert connected[3][0] == OPEN


class TestStageDisaster:
    @pytest.fixture
    def grid(self) -> list[list[str]]:
        rows = []
        for y in range(20):
            row = ""
            for x in range(30):
                row += ROAD if x % 6 == 0 or y % 5 == 0 else BUILDING
            rows.append(list(row))
        return rows

    def test_produces_a_loadable_city_with_every_victim_reachable(
        self, grid: list[list[str]]
    ) -> None:
        staged = stage_disaster(grid, random.Random(0), DisasterSpec())
        city = CityMap.from_config({"width": 30, "height": 20, "terrain_legend": LEGEND, **staged})
        assert len(city.victims) >= 3
        assert len(city.fires) >= 1

    def test_is_deterministic_for_a_seed(self, grid: list[list[str]]) -> None:
        a = stage_disaster([r[:] for r in grid], random.Random(5), DisasterSpec())
        b = stage_disaster([r[:] for r in grid], random.Random(5), DisasterSpec())
        assert a == b

    def test_uses_the_given_hospital_position(self, grid: list[list[str]]) -> None:
        staged = stage_disaster(grid, random.Random(0), DisasterSpec(), hospital=(12, 10))
        assert staged["safe_zone"]["position"] == [12, 10]

    def test_refuses_an_area_that_is_all_building(self) -> None:
        with pytest.raises(ValueError):
            stage_disaster([[BUILDING] * 30 for _ in range(20)], random.Random(0), DisasterSpec())


class TestGeography:
    def test_box_has_the_grid_aspect_ratio_on_the_ground(self) -> None:
        box = box_around(13.0, 80.0, 30, 20, 15.0)
        import math

        width_m = (box.east - box.west) * 111_320 * math.cos(math.radians(13.0))
        height_m = (box.north - box.south) * 111_320
        assert width_m == pytest.approx(450, rel=0.01)
        assert height_m == pytest.approx(300, rel=0.01)

    def test_finds_a_hospital_node(self) -> None:
        elements = [{"tags": {"amenity": "hospital"}, "lat": 0.75, "lon": 0.25}]
        assert hospital_tile(elements, BOX, 10, 10) == (2, 2)

    def test_no_hospital_gives_none(self) -> None:
        assert hospital_tile([_road([(0, 0), (1, 1)])], BOX, 10, 10) is None


class TestInferBlocks:
    def test_land_two_tiles_from_a_road_becomes_building(self) -> None:
        grid = [list(row) for row in ["=.....", "=.....", "=....."]]
        filled = infer_blocks(grid)
        assert [row[1] for row in filled] == [OPEN] * 3
        assert all(cell == BUILDING for row in filled for cell in row[2:])

    def test_roads_are_never_filled(self) -> None:
        grid = [list(row) for row in ["......", "======", "......"]]
        assert infer_blocks(grid)[1] == [ROAD] * 6
