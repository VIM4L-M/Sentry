"""The full local decision pipeline, as one :class:`ILocalController` (Phases 5-7).

Each tick::

    LocalObservation ─┬─► ILocalController (DQN) ────► LocalDecision ─┐
                      ├─► state history ─► IMotionPredictor (LSTM) ──►├─► IDecisionFusion ─► action
                      └─► onboard camera ─► sightings ───────────────►┘

It sits in the engine's ``controller`` slot, so the simulation is unchanged:
the engine hands it an observation and receives a decision, exactly as it
does from the waypoint follower. The decision it returns carries the fused
action and the DQN's Q-values — the network's value estimates are still the
most informative thing to show an operator.

Every tick's inputs and output are kept on :attr:`last_step`, so a data
recorder or a dashboard can read what fusion saw without re-running it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.interfaces.perception import WorldDetection
from sentry_ai.interfaces.sequence import BehaviourSignal, IMotionPredictor, VehicleState
from sentry_ai.sequence.behaviour import heading_to_degrees

#: Returns what the onboard camera sees right now.
Sensing = Callable[[], Sequence[WorldDetection]]

#: States kept for the sequence model. More than any window it uses.
_HISTORY = 64


@dataclass(frozen=True)
class FusionStep:
    """Everything fusion saw and decided on one tick."""

    observation: LocalObservation
    sightings: tuple[WorldDetection, ...]
    behaviour: BehaviourSignal
    local_decision: LocalDecision
    final: FinalAction


class FusedLocalController(ILocalController):
    """DQN + LSTM + onboard camera, fused into the action the vehicle takes."""

    def __init__(
        self,
        local: ILocalController,
        predictor: IMotionPredictor,
        fusion: IDecisionFusion,
        sensing: Sensing,
    ) -> None:
        """Compose the pipeline.

        Args:
            local: The local controller whose proposal fusion weighs — the DQN.
            predictor: The sequence model, fed this controller's own history.
            fusion: Decides the final action.
            sensing: Returns the onboard camera's sightings for this tick.
        """
        self._local = local
        self._predictor = predictor
        self._fusion = fusion
        self._sensing = sensing
        self._history: deque[VehicleState] = deque(maxlen=_HISTORY)
        self.last_step: FusionStep | None = None

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Run every stage once and return the fused action."""
        self._history.append(
            VehicleState(
                position=observation.position,
                battery_percent=observation.battery_percent,
                heading_degrees=heading_to_degrees(observation.heading),
            )
        )
        local_decision = self._local.decide(observation)
        behaviour = self._predictor.predict(list(self._history))
        sightings = tuple(self._sensing())
        final = self._fusion.fuse(observation, sightings, behaviour, local_decision)
        self.last_step = FusionStep(observation, sightings, behaviour, local_decision, final)
        return LocalDecision(action=final.action, q_values=local_decision.q_values)
