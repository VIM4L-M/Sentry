"""Integration tests for the live mission window.

Runs headless (SDL "dummy" driver, set in conftest) and injects events
before calling ``run()``, so the loop executes a bounded number of frames
and returns instead of blocking on a real window.
"""

from __future__ import annotations

from pathlib import Path

import pygame
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision, LocalObservation
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.rendering.simulation_app import (
    ModeSwitchController,
    SimulationApp,
    build_mode_switch,
)
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.mission import MissionController
from sentry_ai.simulation.waypoint_follower import WaypointFollower


@pytest.fixture
def app(project_root: Path) -> SimulationApp:
    """A fully composed mission window, exactly as run_simulation.py builds one."""
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")
    assert app_config.simulation_config_path is not None
    assert app_config.vehicle_config_path is not None

    simulation_config = loader.load_simulation_config(app_config.simulation_config_path)
    vehicle_config = loader.load_vehicle_config(app_config.vehicle_config_path)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
    )
    mode_switch = build_mode_switch(WaypointFollower())
    engine = SimulationEngine(
        city_map, mission, mode_switch, simulation_config, vehicle_config
    )
    return SimulationApp(
        city_map=city_map,
        engine=engine,
        render_config=app_config.render,
        theme=Theme.from_config(loader, app_config.render.palette_config_path),
        mode_switch=mode_switch,
        sensor_rig=SensorRig.from_config(
            loader.load_sensor_config("configs/sensors.yaml"),
            SensorPalette.from_config(loader, "configs/sensors.yaml"),
        ),
    )


@pytest.fixture
def app_without_cameras(project_root: Path) -> SimulationApp:
    """The same window with no rig, which must still open and run."""
    loader = ConfigLoader(project_root=project_root)
    app_config = loader.load_app_config("configs/app.yaml")
    assert app_config.simulation_config_path is not None
    assert app_config.vehicle_config_path is not None

    simulation_config = loader.load_simulation_config(app_config.simulation_config_path)
    vehicle_config = loader.load_vehicle_config(app_config.vehicle_config_path)
    city_map = CityMap.from_config(loader.load_yaml(app_config.map_config_path))
    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
    )
    engine = SimulationEngine(
        city_map, mission, WaypointFollower(), simulation_config, vehicle_config
    )
    return SimulationApp(
        city_map=city_map,
        engine=engine,
        render_config=app_config.render,
        theme=Theme.from_config(loader, app_config.render.palette_config_path),
    )


def _press(key: int) -> None:
    """Queue one key press followed by a quit, so ``run`` returns."""
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    pygame.event.post(pygame.event.Event(pygame.QUIT))


class TestViewToggles:
    def test_g_switches_to_the_occupancy_grid_view(self, app: SimulationApp) -> None:
        pygame.init()
        _press(pygame.K_g)
        app.run()
        assert app._show_grid is True  # noqa: SLF001 — the toggle is the behaviour

    def test_c_hides_the_camera_panel(self, app: SimulationApp) -> None:
        pygame.init()
        _press(pygame.K_c)
        app.run()
        assert app._show_cameras is False  # noqa: SLF001

    def test_c_does_nothing_without_a_rig(self, app_without_cameras: SimulationApp) -> None:
        pygame.init()
        _press(pygame.K_c)
        app_without_cameras.run()
        assert app_without_cameras._show_cameras is False  # noqa: SLF001

    def test_a_window_without_a_rig_still_runs(
        self, app_without_cameras: SimulationApp
    ) -> None:
        pygame.init()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        app_without_cameras.run()


def test_window_runs_and_exits_on_quit(app: SimulationApp) -> None:
    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()  # must return promptly instead of hanging


def test_window_exits_on_escape(app: SimulationApp) -> None:
    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    app.run()


class _Recorder:
    """A controller that records every observation it is handed."""

    def __init__(self) -> None:
        self.seen: list[LocalObservation] = []

    def decide(self, observation: LocalObservation) -> LocalDecision:
        self.seen.append(observation)
        return LocalDecision(action=LocalAction.STOP, q_values={})


class TestModeSwitch:
    def test_starts_autonomous_and_toggles(self) -> None:
        autonomous, manual = _Recorder(), _Recorder()
        switch = ModeSwitchController(autonomous=autonomous, manual=manual)
        assert switch.manual_mode is False

        observation = LocalObservation(
            position=Position(0, 0),
            heading=Heading.NORTH,
            battery_percent=100.0,
            next_waypoint=None,
            blocked_ahead=False,
            fire_proximity=0.0,
        )
        switch.decide(observation)
        assert len(autonomous.seen) == 1 and not manual.seen

        switch.toggle()
        assert switch.manual_mode is True
        switch.decide(observation)
        assert len(manual.seen) == 1
