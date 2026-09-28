"""Integration tests for the Phase 7-8 stack: map lag, fused controller, mission control.

Run on the shipped city and configs, with no trained models: the onboard
camera reads ground-truth labels and fusion is a rule, so these pin down the
wiring — that a lagging map really does put the vehicle into hazards, that a
controller listening to the camera really does keep it out, and that the
mission-control strip draws every state — without depending on weights.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pygame
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, Heading
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion, Region, SceneEvidence
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision, LocalObservation
from sentry_ai.interfaces.sequence import BehaviourSignal
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.rendering.ai_panel import AiPanelRenderer, MissionControlState
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.grid_source import LaggedGridSource
from sentry_ai.simulation.waypoint_follower import WaypointFollower
from sentry_ai.training.missions import MissionFactory

#: Seeds of missions where a 3 s lag drives the follower into hazards.
SEEDS = range(1000, 1006)
LAG = 30
OBSERVATION = LocalObservation(Position(1, 1), Heading.EAST, 80.0, Position(2, 1), False, 0.0)


class _AvoidWhatTheCameraSees(IDecisionFusion):
    """A hand-written fusion rule: never drive into something the camera sees."""

    def fuse(
        self, evidence: SceneEvidence, behaviour: BehaviourSignal, local_decision: LocalDecision
    ) -> FinalAction:
        blocked = max(
            evidence.at(EntityKind.OBSTACLE, Region.AHEAD),
            evidence.at(EntityKind.FIRE, Region.AHEAD),
        )
        if local_decision.action is LocalAction.MOVE_FORWARD and blocked > 0.5:
            return FinalAction(LocalAction.STOP, blocked)
        return FinalAction(local_decision.action, 1.0)


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


def _rig(loader: ConfigLoader) -> SensorRig:
    app = loader.load_app_config("configs/app.yaml")
    assert app.sensor_config_path is not None
    config = loader.load_sensor_config(app.sensor_config_path)
    return SensorRig.from_config(config, SensorPalette.from_config(loader, app.sensor_config_path))


def _collisions(loader: ConfigLoader, lag: int, guarded: bool) -> int:
    factory = MissionFactory.from_app_config(loader, loader.load_app_config(), belief_lag=lag)
    camera = OnboardEvidenceSource(_rig(loader))
    total = 0
    for seed in SEEDS:
        holder: list[FusedLocalController] = []
        mission = factory.build(seed, _Forward(holder))
        fusion = _AvoidWhatTheCameraSees() if guarded else None
        holder.append(
            FusedLocalController(
                WaypointFollower(), partial(camera.evidence, mission.city_map), fusion
            )
        )
        total += mission.engine.run(3000).collisions
    return total


class _Forward(WaypointFollower):
    """Defers to the controller built once the mission (and its city) exists."""

    def __init__(self, holder: list[FusedLocalController]) -> None:
        self._holder = holder

    def decide(self, observation: LocalObservation) -> LocalDecision:
        return self._holder[0].decide(observation)


class TestMapLagScenario:
    def test_a_fresh_map_causes_no_collisions(self, loader: ConfigLoader) -> None:
        assert _collisions(loader, lag=0, guarded=False) == 0

    def test_a_lagging_map_drives_the_vehicle_into_hazards(self, loader: ConfigLoader) -> None:
        assert _collisions(loader, lag=LAG, guarded=False) > 0

    def test_listening_to_the_camera_prevents_most_of_them(self, loader: ConfigLoader) -> None:
        unguarded = _collisions(loader, lag=LAG, guarded=False)
        guarded = _collisions(loader, lag=LAG, guarded=True)
        assert guarded < unguarded / 2


class TestMissionControlPanel:
    @pytest.fixture
    def state(self, loader: ConfigLoader) -> MissionControlState:
        factory = MissionFactory.from_app_config(loader, loader.load_app_config(), belief_lag=LAG)
        mission = factory.build(1000)
        camera = OnboardEvidenceSource(_rig(loader))
        brain = FusedLocalController(
            WaypointFollower(),
            partial(camera.evidence, mission.city_map),
            _AvoidWhatTheCameraSees(),
            fallback=WaypointFollower(),
        )
        lag = LaggedGridSource(mission.controller.grid_source, 0, max_lag=LAG)
        return MissionControlState(brain, camera, lag, 0.1, collisions=0)

    def _draw(self, loader: ConfigLoader, state: MissionControlState) -> pygame.Surface:
        pygame.init()
        theme = Theme.from_config(loader, "configs/render.yaml")
        panel = AiPanelRenderer(theme)
        surface = pygame.Surface((panel.width_px, 600))
        panel.draw(surface, state, 0, 600)
        return surface

    def test_draws_before_the_first_decision(
        self, loader: ConfigLoader, state: MissionControlState
    ) -> None:
        self._draw(loader, state)

    def test_draws_a_decision_with_every_switch_off(
        self, loader: ConfigLoader, state: MissionControlState
    ) -> None:
        state.brain.decide(OBSERVATION)
        switches = state.brain.switches
        switches.camera = switches.behaviour = switches.fusion = switches.policy = False
        state.brain.decide(OBSERVATION)
        self._draw(loader, state)
        assert state.brain.last is not None
