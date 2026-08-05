"""Unit tests for sentry_ai.simulation.hazards and the WorldChange port."""

from __future__ import annotations

import random
from typing import Any

import pytest

from sentry_ai.config.schema import DebrisCollapseConfig, FireSpreadConfig, HazardConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.world import WorldChange
from sentry_ai.simulation.hazards import (
    DebrisCollapseProcess,
    FireSpreadProcess,
    build_world_processes,
    orthogonal_neighbours,
)

_STEP = 1.0

# 7x5 city. The building block at x=2..3, y=1..2 is the only fuel and the
# only thing debris can fall off; everything else is open ground or road.
#   row0: .......
#   row1: ..##...
#   row2: ..##...
#   row3: =======   (road — deliberately not flammable)
#   row4: ...t...
_MAP_DATA: dict[str, Any] = {
    "width": 7,
    "height": 5,
    "grid": [".......", "..##...", "..##...", "=======", "...t..."],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
    "vehicle_start": [6, 0],
    "victims": [{"id": "victim_a", "position": [5, 4]}],
    "fires": [{"id": "seed_fire", "position": [2, 1], "intensity": 0.5, "radius": 1}],
}


@pytest.fixture
def city_map() -> CityMap:
    return CityMap.from_config(_MAP_DATA)


def _fire_process(rng_seed: int = 7, **overrides: Any) -> FireSpreadProcess:
    defaults: dict[str, Any] = {"interval_seconds": _STEP}
    defaults.update(overrides)
    return FireSpreadProcess(FireSpreadConfig(**defaults), random.Random(rng_seed))


def _debris_process(rng_seed: int = 7, **overrides: Any) -> DebrisCollapseProcess:
    defaults: dict[str, Any] = {"interval_seconds": _STEP, "collapse_chance": 1.0}
    defaults.update(overrides)
    return DebrisCollapseProcess(DebrisCollapseConfig(**defaults), random.Random(rng_seed))


class TestWorldChange:
    def test_a_change_with_no_tiles_is_empty(self) -> None:
        assert WorldChange.none().is_empty is True

    def test_a_change_with_tiles_is_not_empty(self) -> None:
        assert WorldChange(frozenset({Position(1, 1)}), "x").is_empty is False

    def test_merging_unions_tiles_and_joins_descriptions(self) -> None:
        left = WorldChange(frozenset({Position(0, 0)}), "fire")
        right = WorldChange(frozenset({Position(1, 1)}), "collapse")
        merged = left.merged_with(right)
        assert merged.changed_tiles == {Position(0, 0), Position(1, 1)}
        assert merged.description == "fire; collapse"

    def test_merging_skips_empty_descriptions(self) -> None:
        merged = WorldChange.none().merged_with(WorldChange(frozenset({Position(0, 0)}), "fire"))
        assert merged.description == "fire"


class TestOrthogonalNeighbours:
    def test_yields_four_neighbours_away_from_the_edge(self) -> None:
        assert len(list(orthogonal_neighbours(Position(5, 5)))) == 4

    def test_skips_negative_coordinates_at_the_origin(self) -> None:
        neighbours = {tile.as_tuple() for tile in orthogonal_neighbours(Position(0, 0))}
        assert neighbours == {(1, 0), (0, 1)}


class TestFireCadence:
    def test_a_disabled_process_never_changes_anything(self, city_map: CityMap) -> None:
        assert _fire_process(enabled=False).advance(city_map, 100.0).is_empty

    def test_nothing_happens_before_the_interval_elapses(self, city_map: CityMap) -> None:
        process = _fire_process(interval_seconds=10.0)
        assert process.advance(city_map, 1.0).is_empty
        assert city_map.fires[0].intensity == pytest.approx(0.5)

    def test_the_interval_is_measured_in_simulated_time(self, city_map: CityMap) -> None:
        # 0.25 is exactly representable, so four of them sum to exactly 1.0
        # and the test cannot flake on floating-point accumulation.
        process = _fire_process(interval_seconds=1.0, growth_per_step=0.2, ignition_chance=0.0)
        for _ in range(3):
            process.advance(city_map, 0.25)
        assert city_map.fires[0].intensity == pytest.approx(0.5)
        process.advance(city_map, 0.25)
        assert city_map.fires[0].intensity == pytest.approx(0.7)


class TestFireLifecycle:
    def test_intensity_grows_by_the_configured_step(self, city_map: CityMap) -> None:
        _fire_process(growth_per_step=0.2, ignition_chance=0.0).advance(city_map, _STEP)
        assert city_map.fires[0].intensity == pytest.approx(0.7)

    def test_intensity_is_capped_at_one(self, city_map: CityMap) -> None:
        _fire_process(growth_per_step=0.9, ignition_chance=0.0).advance(city_map, _STEP)
        assert city_map.fires[0].intensity == pytest.approx(1.0)

    def test_radius_scales_with_intensity(self, city_map: CityMap) -> None:
        process = _fire_process(growth_per_step=0.5, ignition_chance=0.0, max_radius=3)
        process.advance(city_map, _STEP)
        assert city_map.fires[0].radius == 3

    def test_a_peaked_fire_burns_out_and_disappears(self, city_map: CityMap) -> None:
        process = _fire_process(growth_per_step=1.0, burnout_per_step=0.5, ignition_chance=0.0)
        for _ in range(4):
            process.advance(city_map, _STEP)
        assert city_map.fires == []

    def test_burning_out_reports_the_tiles_it_gave_back(self, city_map: CityMap) -> None:
        process = _fire_process(growth_per_step=1.0, burnout_per_step=1.0, ignition_chance=0.0)
        process.advance(city_map, _STEP)
        change = process.advance(city_map, _STEP)
        assert city_map.fires == []
        assert Position(2, 1) in change.changed_tiles


class TestFireSpread:
    def _spread_hard(self, city_map: CityMap, steps: int = 6) -> FireSpreadProcess:
        """Run a process whose fires reach full intensity and always ignite."""
        process = _fire_process(
            ignition_chance=1.0, growth_per_step=0.25, burnout_per_step=0.0, max_radius=1
        )
        for _ in range(steps):
            process.advance(city_map, _STEP)
        return process

    def test_fire_never_takes_hold_on_a_road(self, city_map: CityMap) -> None:
        self._spread_hard(city_map)
        for fire in city_map.fires:
            assert city_map.tile_at(fire.position) is not TerrainType.ROAD

    def test_fire_never_takes_hold_on_open_ground(self, city_map: CityMap) -> None:
        self._spread_hard(city_map)
        for fire in city_map.fires:
            assert city_map.tile_at(fire.position) is not TerrainType.OPEN_GROUND

    def test_fire_does_reach_the_rest_of_the_building_block(self, city_map: CityMap) -> None:
        self._spread_hard(city_map)
        assert len(city_map.fires) > 1

    def test_spread_is_capped_by_max_active_fires(self, city_map: CityMap) -> None:
        process = _fire_process(
            ignition_chance=1.0, growth_per_step=0.25, burnout_per_step=0.0, max_active_fires=2
        )
        for _ in range(10):
            process.advance(city_map, _STEP)
        assert len(city_map.fires) <= 2

    def test_a_zero_ignition_chance_never_spreads(self, city_map: CityMap) -> None:
        process = _fire_process(ignition_chance=0.0, growth_per_step=0.05)
        for _ in range(10):
            process.advance(city_map, _STEP)
        assert len(city_map.fires) == 1


class TestFireSparesTheHospital:
    # Fire seated at (2, 0) with a fixed radius of 1. The only flammable tile
    # on its frontier is the tree at (4, 0), which lies inside the safe zone
    # centred on (5, 0) — so a correct implementation never spreads at all.
    _MAP: dict[str, Any] = {
        "width": 7,
        "height": 5,
        "grid": ["..ttt..", ".......", ".......", "=======", "......."],
        "safe_zone": {"position": [5, 0], "radius": 2, "capacity": 4},
        "vehicle_start": [0, 4],
        "victims": [],
        "fires": [{"id": "seed", "position": [2, 0], "intensity": 0.5, "radius": 1}],
    }

    def test_the_only_flammable_neighbour_is_inside_the_safe_zone(self) -> None:
        city_map = CityMap.from_config(self._MAP)
        assert city_map.tile_at(Position(4, 0)) is TerrainType.TREE
        assert city_map.safe_zone.contains(Position(4, 0))

    def test_fire_refuses_to_spread_into_the_safe_zone(self) -> None:
        city_map = CityMap.from_config(self._MAP)
        process = _fire_process(
            ignition_chance=1.0, growth_per_step=0.25, burnout_per_step=0.0, max_radius=1
        )
        for _ in range(8):
            process.advance(city_map, _STEP)
        assert len(city_map.fires) == 1
        assert all(not city_map.safe_zone.contains(fire.position) for fire in city_map.fires)


class TestFireDeterminism:
    def test_the_same_seed_produces_the_same_fires(self) -> None:
        results = []
        for _ in range(2):
            city_map = CityMap.from_config(_MAP_DATA)
            process = _fire_process(rng_seed=99, ignition_chance=0.5, burnout_per_step=0.0)
            for _ in range(8):
                process.advance(city_map, _STEP)
            results.append(sorted(fire.position.as_tuple() for fire in city_map.fires))
        assert results[0] == results[1]

    def test_a_different_seed_can_produce_a_different_disaster(self) -> None:
        outcomes = set()
        for seed in range(12):
            city_map = CityMap.from_config(_MAP_DATA)
            process = _fire_process(rng_seed=seed, ignition_chance=0.3, burnout_per_step=0.0)
            for _ in range(6):
                process.advance(city_map, _STEP)
            outcomes.add(tuple(sorted(f.position.as_tuple() for f in city_map.fires)))
        assert len(outcomes) > 1


class TestDebrisCollapse:
    def test_a_disabled_process_never_collapses(self, city_map: CityMap) -> None:
        assert _debris_process(enabled=False).advance(city_map, 100.0).is_empty

    def test_a_collapse_blocks_the_tile_it_lands_on(self, city_map: CityMap) -> None:
        change = _debris_process().advance(city_map, _STEP)
        assert not change.is_empty
        landed = next(iter(change.changed_tiles))
        assert city_map.tile_at(landed) is TerrainType.COLLAPSED_BUILDING
        assert city_map.is_walkable(landed) is False

    def test_a_collapse_creates_a_trackable_obstacle(self, city_map: CityMap) -> None:
        before = len(city_map.obstacles)
        change = _debris_process().advance(city_map, _STEP)
        assert len(city_map.obstacles) == before + 1
        assert city_map.obstacles[-1].position in change.changed_tiles

    def test_debris_only_lands_beside_a_standing_building(self, city_map: CityMap) -> None:
        process = _debris_process(max_collapses=20)
        landed: list[Position] = []
        for _ in range(20):
            landed.extend(process.advance(city_map, _STEP).changed_tiles)

        assert landed
        for position in landed:
            assert any(
                city_map.in_bounds(neighbour)
                and city_map.tile_at(neighbour) is TerrainType.BUILDING
                for neighbour in orthogonal_neighbours(position)
            )

    def test_debris_never_buries_the_vehicle(self, city_map: CityMap) -> None:
        city_map.vehicle.position = Position(2, 0)
        process = _debris_process(max_collapses=30)
        for _ in range(30):
            assert Position(2, 0) not in process.advance(city_map, _STEP).changed_tiles

    def test_debris_never_buries_a_trapped_victim(self, city_map: CityMap) -> None:
        city_map.victims[0].position = Position(4, 1)
        process = _debris_process(max_collapses=30)
        for _ in range(30):
            assert Position(4, 1) not in process.advance(city_map, _STEP).changed_tiles

    def test_a_rescued_victims_tile_is_fair_game_again(self, city_map: CityMap) -> None:
        city_map.victims[0].position = Position(4, 1)
        city_map.victims[0].status = VictimStatus.RESCUED
        process = _debris_process(max_collapses=40)
        landed: set[Position] = set()
        for _ in range(40):
            landed |= process.advance(city_map, _STEP).changed_tiles
        assert Position(4, 1) in landed

    def test_collapses_stop_at_the_configured_budget(self, city_map: CityMap) -> None:
        before = len(city_map.obstacles)
        process = _debris_process(max_collapses=2)
        for _ in range(20):
            process.advance(city_map, _STEP)
        assert len(city_map.obstacles) == before + 2

    def test_a_zero_chance_never_collapses(self, city_map: CityMap) -> None:
        process = _debris_process(collapse_chance=0.0)
        for _ in range(20):
            assert process.advance(city_map, _STEP).is_empty

    def test_collapse_sites_run_out_rather_than_erroring(self, city_map: CityMap) -> None:
        """Every eligible tile can be consumed; the process must then idle."""
        process = _debris_process(max_collapses=1000)
        for _ in range(60):
            process.advance(city_map, _STEP)
        assert process.advance(city_map, _STEP).is_empty


class TestBuildWorldProcesses:
    def test_builds_a_fire_and_a_debris_process_in_order(self) -> None:
        processes = build_world_processes(HazardConfig())
        assert isinstance(processes[0], FireSpreadProcess)
        assert isinstance(processes[1], DebrisCollapseProcess)

    def test_disabling_fire_leaves_the_debris_stream_untouched(self) -> None:
        """Each process owns its generator, so one cannot perturb the other."""

        def collapse_sites(*, fire_enabled: bool) -> list[tuple[int, int]]:
            city_map = CityMap.from_config(_MAP_DATA)
            processes = build_world_processes(
                HazardConfig(
                    fire=FireSpreadConfig(enabled=fire_enabled, interval_seconds=_STEP),
                    debris=DebrisCollapseConfig(interval_seconds=_STEP, collapse_chance=1.0),
                )
            )
            for _ in range(6):
                for process in processes:
                    process.advance(city_map, _STEP)
            return [obstacle.position.as_tuple() for obstacle in city_map.obstacles]

        assert collapse_sites(fire_enabled=True) == collapse_sites(fire_enabled=False)
