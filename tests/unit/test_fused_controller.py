"""Unit tests for sentry_ai.decision.fused_controller (Phase 7-8)."""

from __future__ import annotations

from collections.abc import Sequence

from sentry_ai.decision.fused_controller import NO_BEHAVIOUR, FusedLocalController
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, Heading
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion, Region, SceneEvidence
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.sequence import (
    BehaviourClass,
    BehaviourSignal,
    IMotionPredictor,
    VehicleState,
)

DEBRIS_AHEAD = SceneEvidence({(EntityKind.OBSTACLE, Region.AHEAD): 0.9})


class _Forward(ILocalController):
    def decide(self, observation: LocalObservation) -> LocalDecision:
        return LocalDecision(
            LocalAction.MOVE_FORWARD,
            {action: float(action is LocalAction.MOVE_FORWARD) for action in LOCAL_ACTION_ORDER},
        )


class _StopOnDebris(IDecisionFusion):
    """Stops when the camera sees debris ahead; otherwise follows the DQN."""

    def __init__(self) -> None:
        self.seen: list[tuple[SceneEvidence, BehaviourSignal]] = []

    def fuse(
        self, evidence: SceneEvidence, behaviour: BehaviourSignal, local_decision: LocalDecision
    ) -> FinalAction:
        self.seen.append((evidence, behaviour))
        if evidence.at(EntityKind.OBSTACLE, Region.AHEAD) > 0.5:
            return FinalAction(LocalAction.STOP, 0.9)
        return FinalAction(local_decision.action, 0.8)


class _Recorder(IMotionPredictor):
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def predict(self, state_history: Sequence[VehicleState]) -> BehaviourSignal:
        self.lengths.append(len(state_history))
        return BehaviourSignal(BehaviourClass.ADVANCE, 0.7)


def _observation() -> LocalObservation:
    return LocalObservation(
        position=Position(3, 3),
        heading=Heading.NORTH,
        battery_percent=90.0,
        next_waypoint=Position(3, 2),
        blocked_ahead=False,
        fire_proximity=0.0,
    )


def _controller(
    fusion: IDecisionFusion | None = None, predictor: IMotionPredictor | None = None
) -> FusedLocalController:
    return FusedLocalController(_Forward(), lambda: DEBRIS_AHEAD, fusion, predictor)


class TestFusedLocalController:
    def test_fusion_overrides_the_dqn_when_the_camera_sees_debris(self) -> None:
        controller = _controller(_StopOnDebris())
        decision = controller.decide(_observation())
        assert decision.action is LocalAction.STOP
        assert controller.last is not None and controller.last.overridden

    def test_without_a_fusion_model_the_dqn_drives(self) -> None:
        controller = _controller()
        assert controller.decide(_observation()).action is LocalAction.MOVE_FORWARD
        assert controller.last is not None and not controller.last.overridden

    def test_switching_fusion_off_restores_the_dqn(self) -> None:
        controller = _controller(_StopOnDebris())
        controller.switches.fusion = False
        assert controller.decide(_observation()).action is LocalAction.MOVE_FORWARD

    def test_switching_the_camera_off_blinds_fusion(self) -> None:
        fusion = _StopOnDebris()
        controller = _controller(fusion)
        controller.switches.camera = False
        assert controller.decide(_observation()).action is LocalAction.MOVE_FORWARD
        assert fusion.seen[-1][0] == SceneEvidence.empty()

    def test_switching_the_lstm_off_sends_no_behaviour(self) -> None:
        fusion = _StopOnDebris()
        controller = _controller(fusion, _Recorder())
        controller.switches.behaviour = False
        controller.decide(_observation())
        assert fusion.seen[-1][1] == NO_BEHAVIOUR

    def test_the_predictor_sees_a_growing_history(self) -> None:
        predictor = _Recorder()
        controller = _controller(_StopOnDebris(), predictor)
        for _ in range(3):
            controller.decide(_observation())
        assert predictor.lengths == [1, 2, 3]

    def test_reset_forgets_history_and_trace(self) -> None:
        predictor = _Recorder()
        controller = _controller(_StopOnDebris(), predictor)
        controller.decide(_observation())
        controller.reset()
        assert controller.last is None
        controller.decide(_observation())
        assert predictor.lengths[-1] == 1


class _AlwaysStop(IDecisionFusion):
    def fuse(
        self, evidence: SceneEvidence, behaviour: BehaviourSignal, local_decision: LocalDecision
    ) -> FinalAction:
        return FinalAction(LocalAction.STOP, 0.95)


class TestEvidenceGate:
    def test_no_override_when_the_camera_sees_nothing(self) -> None:
        controller = FusedLocalController(_Forward(), SceneEvidence.empty, _AlwaysStop())
        assert controller.decide(_observation()).action is LocalAction.MOVE_FORWARD
        assert controller.overrides == 0

    def test_override_allowed_when_the_camera_sees_a_hazard(self) -> None:
        controller = FusedLocalController(_Forward(), lambda: DEBRIS_AHEAD, _AlwaysStop())
        assert controller.decide(_observation()).action is LocalAction.STOP
