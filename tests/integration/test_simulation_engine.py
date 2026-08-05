"""Integration tests for the Phase 2 mission loop.

Wires the real config files, the real disaster city, A*, the waypoint
follower, and the engine together and runs whole missions headlessly — the
closest thing to ``scripts/run_simulation.py`` that can assert on outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import SimulationConfig, VehicleConfig
from sentry_ai.domain.enums import VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision, LocalObservation
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.hazards import build_world_processes
from sentry_ai.simulation.mission import MissionController, MissionPhase
from sentry_ai.simulation.waypoint_follower import WaypointFollower

_MAX_TICKS = 5000


def _build(
    project_root: Path,
    simulation_config: SimulationConfig | None = None,
    vehicle_config: VehicleConfig | None = None,
    *,
    with_hazards: bool = False,
) -> tuple[SimulationEngine, CityMap]:
    """Compose an engine over the repo's real configs, as the CLI script does.

    Hazards are opt-in so the bulk of these tests pin down the mission loop
    against a static city, and the dynamic-world tests are explicit about
    running a moving target.
    """
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")
    assert app_config.simulation_config_path is not None
    assert app_config.vehicle_config_path is not None

    sim = simulation_config or loader.load_simulation_config(app_config.simulation_config_path)
    vehicle = vehicle_config or loader.load_vehicle_config(app_config.vehicle_config_path)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(sim.planner),
        config=sim.mission,
    )
    processes = build_world_processes(sim.hazards) if with_hazards else []
    engine = SimulationEngine(
        city_map, mission, WaypointFollower(), sim, vehicle, world_processes=processes
    )
    return engine, city_map


class TestAutonomousMission:
    def test_completes_and_rescues_everyone(self, project_root: Path) -> None:
        engine, city_map = _build(project_root)
        stats = engine.run(max_ticks=_MAX_TICKS)

        assert engine.mission.phase is MissionPhase.COMPLETED
        assert stats.victims_rescued == len(city_map.victims)
        assert stats.victims_lost == 0
        assert stats.victims_unreachable == 0
        assert all(victim.status is VictimStatus.RESCUED for victim in city_map.victims)

    def test_ends_with_the_vehicle_home_and_empty(self, project_root: Path) -> None:
        engine, city_map = _build(project_root)
        engine.run(max_ticks=_MAX_TICKS)
        assert city_map.safe_zone.contains(city_map.vehicle.position)
        assert city_map.vehicle.onboard_victims == []

    def test_drives_without_collisions_or_damage(self, project_root: Path) -> None:
        """A* only routes over traversable tiles, so nothing should be hit."""
        engine, city_map = _build(project_root)
        stats = engine.run(max_ticks=_MAX_TICKS)
        assert stats.collisions == 0
        assert city_map.vehicle.health_percent == pytest.approx(100.0)

    def test_finishes_on_one_charge(self, project_root: Path) -> None:
        engine, city_map = _build(project_root)
        engine.run(max_ticks=_MAX_TICKS)
        assert city_map.vehicle.battery_percent > 0.0

    def test_is_deterministic_across_runs(self, project_root: Path) -> None:
        first = _build(project_root)[0].run(max_ticks=_MAX_TICKS)
        second = _build(project_root)[0].run(max_ticks=_MAX_TICKS)
        assert (first.ticks, first.victims_rescued, first.tiles_travelled) == (
            second.ticks,
            second.victims_rescued,
            second.tiles_travelled,
        )


class TestEngineContract:
    def test_tick_returns_none_once_the_mission_ends(self, project_root: Path) -> None:
        engine, _ = _build(project_root)
        engine.run(max_ticks=_MAX_TICKS)
        assert engine.is_done is True
        assert engine.tick() is None

    def test_max_ticks_is_a_hard_stop(self, project_root: Path) -> None:
        engine, _ = _build(project_root)
        stats = engine.run(max_ticks=3)
        assert stats.ticks == 3
        assert engine.is_done is False

    def test_engine_applies_configured_starting_battery(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        vehicle_config = loader.load_vehicle_config("configs/vehicle.yaml")
        engine, city_map = _build(project_root, vehicle_config=vehicle_config)
        assert city_map.vehicle.battery_percent == vehicle_config.battery.initial_percent
        assert city_map.vehicle.capacity == vehicle_config.capacity
        del engine

    def test_tick_reports_what_happened(self, project_root: Path) -> None:
        engine, _ = _build(project_root)
        result = engine.tick()
        assert result is not None
        assert result.tick == 1
        assert isinstance(result.decision, LocalDecision)
        assert result.outcome.action is result.decision.action


class _AlwaysForward:
    """A controller that ignores the route and drives straight into things."""

    def decide(self, observation: LocalObservation) -> LocalDecision:
        del observation
        return LocalDecision(action=LocalAction.MOVE_FORWARD, q_values={})


class TestCollisionHandling:
    def test_a_reckless_controller_collides_and_triggers_replans(
        self, project_root: Path
    ) -> None:
        loader = ConfigLoader(project_root=project_root)
        sim = loader.load_simulation_config("configs/simulation.yaml")
        vehicle = loader.load_vehicle_config("configs/vehicle.yaml")
        city_map = CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))
        mission = MissionController(
            city_map=city_map,
            grid=OccupancyGrid.from_city_map(city_map),
            planner=AStarPlanner(sim.planner),
            config=sim.mission,
        )
        engine = SimulationEngine(city_map, mission, _AlwaysForward(), sim, vehicle)
        stats = engine.run(max_ticks=60)

        assert stats.collisions > 0
        assert city_map.vehicle.health_percent < 100.0


class TestMissionUnderHazards:
    """The same mission, but with fire spreading and buildings coming down."""

    def test_the_mission_still_completes_when_the_city_moves(
        self, project_root: Path
    ) -> None:
        engine, city_map = _build(project_root, with_hazards=True)
        stats = engine.run(max_ticks=_MAX_TICKS)

        assert engine.mission.phase is MissionPhase.COMPLETED
        assert stats.victims_rescued > 0
        assert stats.victims_rescued + stats.victims_unreachable == len(city_map.victims)

    def test_nobody_is_left_behind_when_the_city_is_burning(
        self, project_root: Path
    ) -> None:
        """The headline outcome: every victim out, none lost, none stranded."""
        engine, city_map = _build(project_root, with_hazards=True)
        stats = engine.run(max_ticks=_MAX_TICKS)

        assert stats.victims_rescued == len(city_map.victims)
        assert stats.victims_lost == 0
        assert stats.victims_unreachable == 0

    def test_victims_beside_a_fire_arrive_in_worse_shape(
        self, project_root: Path
    ) -> None:
        """Proximity to fire has to actually cost a victim something."""
        engine, city_map = _build(project_root, with_hazards=True)
        engine.run(max_ticks=_MAX_TICKS)

        by_id = {victim.victim_id: victim for victim in city_map.victims}
        # victim_03 is trapped in the block with fire_02; victim_04 is
        # stranded in the open with nothing burning near them.
        assert by_id["victim_03"].health < by_id["victim_04"].health

    def test_hazards_actually_fire_during_a_mission(self, project_root: Path) -> None:
        engine, _ = _build(project_root, with_hazards=True)
        stats = engine.run(max_ticks=_MAX_TICKS)
        assert stats.hazard_events > 0

    def test_a_collapse_lands_somewhere_the_map_did_not_start_blocked(
        self, project_root: Path
    ) -> None:
        engine, city_map = _build(project_root, with_hazards=True)
        before = len(city_map.obstacles)
        engine.run(max_ticks=_MAX_TICKS)
        assert len(city_map.obstacles) > before

    def test_fire_spreads_beyond_the_two_it_started_with(self, project_root: Path) -> None:
        engine, city_map = _build(project_root, with_hazards=True)
        engine.run(max_ticks=_MAX_TICKS)
        assert any(fire.fire_id.startswith("fire_spread_") for fire in city_map.fires)

    def test_a_hazardous_mission_is_still_deterministic(self, project_root: Path) -> None:
        """The seeded RNG is what makes Phase 6 training runs comparable."""
        first = _build(project_root, with_hazards=True)[0].run(max_ticks=_MAX_TICKS)
        second = _build(project_root, with_hazards=True)[0].run(max_ticks=_MAX_TICKS)
        assert (first.ticks, first.victims_rescued, first.hazard_events) == (
            second.ticks,
            second.victims_rescued,
            second.hazard_events,
        )

    def test_the_tick_result_carries_the_world_change(self, project_root: Path) -> None:
        engine, _ = _build(project_root, with_hazards=True)
        changes = []
        for _ in range(_MAX_TICKS):
            result = engine.tick()
            if result is None:
                break
            if not result.world_change.is_empty:
                changes.append(result.world_change)
        assert changes
        assert all(change.description for change in changes)
