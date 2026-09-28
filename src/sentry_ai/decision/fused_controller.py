"""The local controller the vehicle drives with once fusion exists (Phase 7-8).

:class:`FusedLocalController` is an :class:`ILocalController`, so the engine
cannot tell it from the plain DQN — Phase 8's "all five models in the loop"
is a composition-root change, not an engine change. Each tick it:

1. asks the inner controller (the DQN) for its decision on the command
   center's map;
2. asks the motion predictor (the LSTM) what the vehicle is doing, from the
   states this controller has recorded itself;
3. asks the onboard camera what it sees around the vehicle;
4. hands all three to the fusion network and drives what it returns.

Every stage can be switched off at runtime — :class:`SignalSwitches` — which
is what the mission-control screen's ablation keys flip. A switched-off
signal is replaced by its *empty* value (no evidence, no behaviour), not
removed, so the fusion network keeps running on what is left; switching
fusion itself off drives the inner controller's choice unchanged.

What happened on the last tick is kept in :attr:`last` for the display:
every signal, the final action, and whether fusion overrode the DQN;
:attr:`overrides` counts how often it did.

No Torch import here: the models are injected already loaded.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion, SceneEvidence
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.sequence import (
    BehaviourClass,
    BehaviourSignal,
    IMotionPredictor,
    VehicleState,
)
from sentry_ai.sequence.behaviour import heading_to_degrees

#: What a switched-off or absent motion predictor reports: zero confidence,
#: which :func:`~sentry_ai.decision.mlp_fusion.fusion_features` encodes as
#: all zeros — exactly the input an ablated network was trained on.
NO_BEHAVIOUR = BehaviourSignal(predicted_class=BehaviourClass.HOLD, confidence=0.0)

#: States kept for the motion predictor. Enough for any window it uses.
HISTORY_LENGTH = 32


@dataclass
class SignalSwitches:
    """Which upstream signals are live. Flipped by the ablation keys."""

    camera: bool = True
    behaviour: bool = True
    fusion: bool = True
    policy: bool = True


@dataclass(frozen=True)
class FusionTrace:
    """Everything that went into, and came out of, one fused decision.

    Attributes:
        local_decision: The inner controller's (DQN's) choice and Q-values.
        behaviour: The motion predictor's output, or :data:`NO_BEHAVIOUR`.
        evidence: The onboard camera's report, or empty.
        final: The action the vehicle executes.
        overridden: Whether ``final`` differs from the inner controller's
            action — fusion stepped in.
    """

    local_decision: LocalDecision
    behaviour: BehaviourSignal
    evidence: SceneEvidence
    final: FinalAction

    @property
    def overridden(self) -> bool:
        """Whether fusion chose differently from the DQN."""
        return self.final.action is not self.local_decision.action


class FusedLocalController(ILocalController):
    """DQN + LSTM + camera, fused by an MLP, behind the local-controller port."""

    def __init__(
        self,
        inner: ILocalController,
        evidence_source: Callable[[], SceneEvidence],
        fusion: IDecisionFusion | None = None,
        predictor: IMotionPredictor | None = None,
        fallback: ILocalController | None = None,
        on_override: Callable[[SceneEvidence, LocalObservation], None] | None = None,
    ) -> None:
        """Wire the controller.

        Args:
            inner: The policy fusion arbitrates — in practice the DQN.
            evidence_source: Called once per tick for the onboard camera's
                evidence, usually ``OnboardEvidenceSource(...).evidence``
                bound to the live city.
            fusion: The fusion network. ``None`` drives ``inner`` unchanged
                while still recording every signal, so the display works
                before a fusion model is trained.
            predictor: The motion predictor. ``None`` reports
                :data:`NO_BEHAVIOUR`.
            fallback: Drives instead of ``inner`` while
                ``switches.policy`` is off — the waypoint follower, so the
                learned driver can be switched out live.
            on_override: Called whenever fusion overrules the inner
                controller, with the camera's evidence and the observation — how the
                composition root lets the command center hear what the
                vehicle's camera saw.
        """
        self._inner = inner
        self._evidence_source = evidence_source
        self._fusion = fusion
        self._predictor = predictor
        self._fallback = fallback
        self._on_override = on_override
        self._history: deque[VehicleState] = deque(maxlen=HISTORY_LENGTH)
        self.switches = SignalSwitches()
        self.last: FusionTrace | None = None
        self.overrides = 0

    @property
    def has_fusion(self) -> bool:
        """Whether a fusion network is installed at all."""
        return self._fusion is not None

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Fuse this tick's signals into the action the vehicle executes."""
        self._history.append(
            VehicleState(
                position=observation.position,
                battery_percent=observation.battery_percent,
                heading_degrees=heading_to_degrees(observation.heading),
            )
        )
        local = self._driver().decide(observation)
        behaviour = self._behaviour()
        evidence = self._evidence_source() if self.switches.camera else SceneEvidence.empty()
        final = self._final(evidence, behaviour, local)
        self.last = FusionTrace(local, behaviour, evidence, final)
        if final.action is local.action:
            return local
        self.overrides += 1
        if self._on_override is not None:
            self._on_override(evidence, observation)
        return LocalDecision(
            action=final.action,
            q_values={action: float(action is final.action) for action in LOCAL_ACTION_ORDER},
        )

    def reset(self) -> None:
        """Forget the recorded history, for a new mission."""
        self._history.clear()
        self.last = None
        self.overrides = 0

    @property
    def has_fallback(self) -> bool:
        """Whether ``switches.policy`` can switch the learned driver out."""
        return self._fallback is not None

    def _driver(self) -> ILocalController:
        if self.switches.policy or self._fallback is None:
            return self._inner
        return self._fallback

    def _behaviour(self) -> BehaviourSignal:
        if self._predictor is None or not self.switches.behaviour:
            return NO_BEHAVIOUR
        return self._predictor.predict(list(self._history))

    def _final(
        self, evidence: SceneEvidence, behaviour: BehaviourSignal, local: LocalDecision
    ) -> FinalAction:
        if self._fusion is None or not self.switches.fusion:
            return FinalAction(action=local.action, rationale_score=1.0)
        final = self._fusion.fuse(evidence, behaviour, local)
        if final.action is not local.action and not _sees_something(evidence):
            # Correcting the DQN is the camera's job: with nothing in view
            # there is no fresh evidence the DQN lacked, and an override then
            # is the network guessing. Measured on a map it was not trained
            # on, one such guess steered the vehicle into a pocket a fire
            # later closed.
            return FinalAction(action=local.action, rationale_score=final.rationale_score)
        return final


#: Camera confidence above which something counts as seen.
_SEEN = 0.5


def _sees_something(evidence: SceneEvidence) -> bool:
    """Whether the camera reports any hazard or victim with confidence."""
    return any(value >= _SEEN for value in evidence.confidence.values())
