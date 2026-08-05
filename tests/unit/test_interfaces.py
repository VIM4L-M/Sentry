"""Unit tests for the AI module ports (sentry_ai.interfaces).

Phase 1 defines contracts only, so these tests check exactly that: the
ABCs cannot be instantiated directly, a conforming subclass can be, and
each supporting value type enforces its own invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.decision import (
    FinalAction,
    IDecisionFusion,
    INavigationPolicy,
    PolicyOutput,
    VehicleAction,
)
from sentry_ai.interfaces.perception import BoundingBox, Detection, IDenoiser, IVisionDetector
from sentry_ai.interfaces.sequence import BehaviourClass, BehaviourSignal, IMotionPredictor


class TestPortsAreAbstract:
    def test_idenoiser_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IDenoiser()  # type: ignore[abstract]

    def test_ivisiondetector_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IVisionDetector()  # type: ignore[abstract]

    def test_imotionpredictor_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IMotionPredictor()  # type: ignore[abstract]

    def test_inavigationpolicy_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            INavigationPolicy()  # type: ignore[abstract]

    def test_idecisionfusion_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            IDecisionFusion()  # type: ignore[abstract]


class _StubDetector(IVisionDetector):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


def test_conforming_subclass_can_be_instantiated_and_called() -> None:
    detector = _StubDetector()
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


class TestBoundingBox:
    def test_rejects_zero_area(self) -> None:
        with pytest.raises(ValueError, match="positive width/height"):
            BoundingBox(x_min=0, y_min=0, x_max=0, y_max=5)

    def test_accepts_valid_box(self) -> None:
        box = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
        assert box.x_max == 10


class TestDetection:
    def test_rejects_out_of_range_confidence(self) -> None:
        box = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
        with pytest.raises(ValueError, match="confidence"):
            Detection(label=EntityKind.VICTIM, confidence=1.5, bbox=box)


class TestBehaviourSignal:
    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            BehaviourSignal(predicted_class=BehaviourClass.ADVANCE, confidence=-0.1)


class TestPolicyOutputAndFinalAction:
    def test_policy_output_holds_q_values(self) -> None:
        output = PolicyOutput(
            action=VehicleAction.MOVE_NORTH,
            q_values={VehicleAction.MOVE_NORTH: 0.9, VehicleAction.HOLD_POSITION: 0.1},
        )
        assert output.q_values[VehicleAction.MOVE_NORTH] == 0.9

    def test_final_action_holds_rationale_score(self) -> None:
        action = FinalAction(action=VehicleAction.PICKUP_VICTIM, rationale_score=0.87)
        assert action.action is VehicleAction.PICKUP_VICTIM
