"""Port for decision fusion (Unit I) — the last stage before the vehicle acts.

Implemented in Phase 7 by an MLP (PyTorch, Adam, ReLU, Dropout) that
combines every upstream AI signal — what the vehicle's own camera sees, the
LSTM's behaviour prediction, and the DQN's Q-values — into the single
action the vehicle executes this tick. This file defines the contract only.

**Why the camera arrives as** :class:`SceneEvidence` **and not raw
detections.** A detection is a pixel box. Whether that box is "the tile I
am about to drive into" depends on where the camera was pointed and which
way the vehicle faces, and a fusion network given raw boxes would have to
learn that geometry from scratch — and relearn it for every camera. The
projection is exact and already written (``CameraView.tiles_of_box``), so
it runs before fusion and the network sees the answer: *how sure is the
camera that there is fire directly ahead*. See
``docs/adr/0003-egocentric-scene-evidence.md``.

The navigation contracts that used to live here (``VehicleAction``,
``PolicyOutput``, ``INavigationPolicy``) moved to
:mod:`sentry_ai.interfaces.navigation` when routing was split into a global
A* tier and a local DQN tier — see
``docs/adr/0002-two-tier-navigation-and-command-center.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.navigation import LocalAction, LocalDecision
from sentry_ai.interfaces.sequence import BehaviourSignal


class Region(Enum):
    """A place relative to the vehicle, judged against the way it faces.

    ``AHEAD`` is the tile a forward move would enter — the one that decides
    whether driving on is safe. ``AHEAD_FAR`` is the tile beyond it, early
    warning. ``LEFT``/``RIGHT``/``BEHIND`` are the tiles a turn or a reverse
    would lead toward. ``NEARBY`` is anything within two tiles, whatever
    the direction.
    """

    AHEAD = "ahead"
    AHEAD_FAR = "ahead_far"
    LEFT = "left"
    RIGHT = "right"
    BEHIND = "behind"
    NEARBY = "nearby"


#: The detector classes evidence is reported for, in feature order.
EVIDENCE_KINDS: tuple[EntityKind, ...] = (EntityKind.VICTIM, EntityKind.FIRE, EntityKind.OBSTACLE)

#: The regions evidence is reported for, in feature order.
EVIDENCE_REGIONS: tuple[Region, ...] = tuple(Region)


@dataclass(frozen=True)
class SceneEvidence:
    """What the vehicle's own camera reports around it, egocentrically.

    Attributes:
        confidence: The highest detector confidence for each
            ``(kind, region)`` pair that anything was seen in. Pairs with
            nothing seen are absent and read as ``0.0``.
    """

    confidence: dict[tuple[EntityKind, Region], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for (kind, region), value in self.confidence.items():
            if kind not in EVIDENCE_KINDS:
                raise DomainValidationError(f"SceneEvidence has no slot for {kind}")
            if not 0.0 <= value <= 1.0:
                raise DomainValidationError(
                    f"SceneEvidence confidence for {kind.value}/{region.value} must be "
                    f"within 0.0-1.0, got {value}"
                )

    @classmethod
    def empty(cls) -> SceneEvidence:
        """Nothing seen anywhere — also what a blinded camera reports."""
        return cls()

    def at(self, kind: EntityKind, region: Region) -> float:
        """Confidence that ``kind`` is in ``region``, ``0.0`` when unseen."""
        return self.confidence.get((kind, region), 0.0)

    def as_list(self) -> list[float]:
        """Every slot in ``EVIDENCE_KINDS`` x ``EVIDENCE_REGIONS`` order."""
        return [self.at(kind, region) for kind in EVIDENCE_KINDS for region in EVIDENCE_REGIONS]


@dataclass(frozen=True)
class FinalAction:
    """The single action the vehicle will execute this tick, after fusion.

    Attributes:
        action: The egocentric move to execute.
        rationale_score: The fusion network's confidence in this action,
            0-1. Surfaced on the dashboard so an operator can see how sure
            the system was, not just what it did.
    """

    action: LocalAction
    rationale_score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rationale_score <= 1.0:
            raise DomainValidationError(
                f"FinalAction.rationale_score must be within 0.0-1.0, got {self.rationale_score}"
            )


class IDecisionFusion(ABC):
    """Fuses perception, sequence, and local-policy signals into one action.

    Implemented in Phase 7 by an MLP (Unit I — Deep Learning Fundamentals:
    PyTorch, Adam, ReLU, Dropout).
    """

    @abstractmethod
    def fuse(
        self,
        evidence: SceneEvidence,
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """Combine every upstream signal into the vehicle's final action this tick."""
        raise NotImplementedError
