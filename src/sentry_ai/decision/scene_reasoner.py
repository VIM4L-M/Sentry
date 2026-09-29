"""What the vehicle would do about what its camera sees, and why (review demo).

The live-camera demo shows real objects detected by a COCO YOLOv8. This
module turns those detections into the decision an emergency vehicle would
take, in words: "Heavy traffic ahead: I will wait and keep my gap", "Person
in front: stop and give way". It is a small, explainable rule layer that
mirrors the priorities the trained driving policy learned in simulation —
never hit a road user, then reach the victim — so the audience can see the
reasoning, not only the boxes.

It is deliberately rule-based, not a learned model: the DQN drives from the
simulator's map, not from a webcam, so this layer states the same priorities
in a form that works on any photo. Pure Python, no ultralytics import.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """What the vehicle does, most cautious last (higher value wins)."""

    PROCEED = 0
    CAUTION = 1
    SLOW = 2
    WAIT = 3
    STOP = 4


#: What each action is called on screen, and its colour (RGB).
ACTION_TEXT: dict[Action, str] = {
    Action.PROCEED: "PROCEED",
    Action.CAUTION: "GO CAREFULLY",
    Action.SLOW: "SLOW DOWN",
    Action.WAIT: "WAIT",
    Action.STOP: "STOP",
}
ACTION_COLOUR: dict[Action, tuple[int, int, int]] = {
    Action.PROCEED: (40, 214, 140),
    Action.CAUTION: (120, 200, 255),
    Action.SLOW: (255, 200, 60),
    Action.WAIT: (255, 150, 40),
    Action.STOP: (255, 70, 70),
}

VEHICLES = frozenset({"car", "truck", "bus", "auto", "motorcycle", "bicycle"})
ANIMALS = frozenset({"cow", "dog", "horse", "sheep"})

#: A box this tall (share of the frame) is close to the vehicle.
NEAR_HEIGHT = 0.45
#: This many vehicles in view is a traffic queue.
TRAFFIC_COUNT = 3


@dataclass(frozen=True)
class SeenObject:
    """One detection: COCO class name, confidence, and box as frame shares (x0, y0, x1, y1)."""

    name: str
    confidence: float
    box: tuple[float, float, float, float]
    colour: str | None = None  # "red" / "green" / "amber" for a traffic light

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def near(self) -> bool:
        return self.height >= NEAR_HEIGHT

    @property
    def ahead(self) -> bool:
        """Whether the box centre is in the middle third, the vehicle's own lane."""
        centre = (self.box[0] + self.box[2]) / 2
        return 1 / 3 <= centre <= 2 / 3


@dataclass(frozen=True)
class Decision:
    """The action, a one-line headline, and the reasons behind it."""

    action: Action
    headline: str
    reasons: tuple[str, ...]

    @property
    def text(self) -> str:
        return ACTION_TEXT[self.action]


def decide(seen: list[SeenObject]) -> Decision:
    """The most cautious decision any object in view calls for."""
    rules: list[tuple[Action, str]] = []
    people = [o for o in seen if o.name == "person"]
    vehicles = [o for o in seen if o.name in VEHICLES]
    animals = [o for o in seen if o.name in ANIMALS]

    for person in people:
        if person.near or person.ahead:
            rules.append((Action.STOP, "Person in front: I stop and give way"))
        else:
            rules.append((Action.SLOW, "Pedestrian beside the road: I slow down"))
    for animal in animals:
        rules.append((Action.STOP, f"{animal.name.title()} on the road: I wait until it crosses"))
    if any(o.name == "stop sign" for o in seen):
        rules.append((Action.STOP, "Stop sign: I stop and check both ways"))
    rules.extend(_signal_and_traffic(signal_colour(seen), vehicles))
    for vehicle in vehicles:
        kind = _vehicle_name(vehicle.name)
        if vehicle.near and vehicle.ahead:
            rules.append((Action.STOP, f"{kind} close ahead: I brake to keep a safe gap"))
        elif vehicle.near:
            rules.append((Action.SLOW, f"{kind} alongside: I hold my lane and slow"))
        elif vehicle.name in ("bus", "truck"):
            rules.append((Action.CAUTION, f"{kind} ahead: I leave extra room"))
        elif vehicle.name in ("motorcycle", "bicycle", "auto"):
            rules.append((Action.CAUTION, f"{kind} ahead: it may swerve, I stay behind"))

    if not rules:
        return Decision(
            Action.PROCEED,
            "Road clear: I drive on to the victim",
            ("No road users in my path", "Cruise at the safe speed for this street"),
        )
    action = max(a for a, _ in rules)
    reasons = tuple(dict.fromkeys(r for a, r in sorted(rules, key=lambda x: -x[0])))
    return Decision(action, reasons[0], reasons[1:4])


def signal_colour(seen: list[SeenObject]) -> str | None:
    """The signal's colour, weighted by confidence; ``"unknown"`` if unreadable, None if none.

    A far, faint lamp must not overrule the near, confident one, so each
    readable light votes with its detection confidence.
    """
    lights = [o for o in seen if o.name == "traffic light"]
    if not lights:
        return None
    votes: dict[str, float] = {}
    for light in lights:
        if light.colour:
            votes[light.colour] = votes.get(light.colour, 0.0) + light.confidence
    return max(votes, key=lambda c: votes[c]) if votes else "unknown"


def _signal_and_traffic(signal: str | None, vehicles: list[SeenObject]) -> list[tuple[Action, str]]:
    """What the signal and the traffic in front call for, taken together.

    Traffic alone is not a reason to stop: on a green signal the vehicle
    moves with the flow, keeping its gap. It waits only at a red signal or
    behind a queue that is close and not moving.
    """
    count = len(vehicles)
    queue = count >= TRAFFIC_COUNT
    close_queue = queue and any(v.ahead and v.height >= NEAR_HEIGHT / 2 for v in vehicles)
    if signal == "red":
        rules = [(Action.STOP, "Red signal: I stop at the line; siren on, cross only when clear")]
        if queue:
            rules.append((Action.WAIT, f"{count} vehicles queued at the signal: I wait in line"))
        return rules
    if signal == "amber":
        return [(Action.SLOW, "Amber signal: I slow and prepare to stop")]
    if signal == "green":
        if queue:
            return [
                (Action.CAUTION, f"Green signal: I move with the traffic ({count} vehicles)"),
                (Action.CAUTION, "Keeping a safe gap to the vehicle in front"),
            ]
        return [(Action.CAUTION, "Green signal: I go, watching the junction")]
    rules = []
    if signal == "unknown":
        rules.append((Action.CAUTION, "Signal ahead, colour unclear: I slow and check"))
    if close_queue:
        rules.append(
            (Action.WAIT, f"Traffic queue ahead ({count} vehicles): I wait, keeping my gap")
        )
        rules.append((Action.WAIT, "If it does not move, I re-plan to the safer route B"))
    elif queue:
        rules.append((Action.SLOW, f"Traffic ahead ({count} vehicles): I follow at a safe gap"))
    return rules


def _vehicle_name(name: str) -> str:
    return {
        "car": "Car",
        "truck": "Auto/Truck",
        "auto": "Auto-rickshaw",
        "bus": "Bus",
        "motorcycle": "Two-wheeler",
        "bicycle": "Cycle",
    }[name]


class DecisionHold:
    """Keeps a decision on screen long enough to read.

    Detections flicker frame to frame; a new decision replaces the shown one
    at once only if it is more cautious, otherwise after ``hold_s`` seconds.
    """

    def __init__(self, hold_s: float = 1.2) -> None:
        self._hold_s = hold_s
        self._shown: Decision | None = None
        self._since = 0.0

    def update(self, decision: Decision, now: float) -> Decision:
        """The decision to show at time ``now`` (seconds), given the latest one."""
        shown = self._shown
        if (
            shown is None
            or decision == shown
            or decision.action > shown.action
            or now - self._since >= self._hold_s
        ):
            self._shown, self._since = decision, now
            return decision
        return shown
