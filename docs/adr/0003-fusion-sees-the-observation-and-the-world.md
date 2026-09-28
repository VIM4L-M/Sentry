# ADR 0003: Decision fusion sees the vehicle's situation, in world space — and the vehicle moves through the real world

**Status:** Accepted
**Phase:** 7
**Amends:** `IDecisionFusion.fuse` as defined in Phase 1's `interfaces/decision.py`;
`SimulationEngine.tick`'s physics as built in Phase 2

## Context

Phase 1 defined decision fusion as

```python
def fuse(detections: list[Detection], behaviour: BehaviourSignal,
         local_decision: LocalDecision) -> FinalAction
```

Building it exposed two problems, one in the port and one in the engine.

**The port cannot say what is ahead.** A `Detection` is a box in *pixels*.
Whether a piece of debris is on the tile the vehicle is about to drive into
depends on which camera took the frame, where that camera's window sat — the
onboard window is clamped at map edges, so the vehicle is not always at its
centre — and which way the vehicle is facing. The signature carries none of
that. The fusion network would have had to learn camera geometry and heading
from nothing, or be handed them through a side door.

**A wrong map could never cause a crash.** `SimulationEngine.tick` applied the
vehicle's move against `MissionController.grid` — the command center's
*belief*. If the belief missed a collapse, the vehicle drove straight through
the rubble. Nothing a camera saw could ever prevent a collision, because
collisions with unseen obstacles did not exist. Phase 3's "missions on
perception match ground truth" and Phase 6's "zero collisions" were both
measured in that world.

## Decision

**1. The port takes the observation and world-space sightings.**

```python
def fuse(
    self,
    observation: LocalObservation,
    sightings: Sequence[WorldDetection],
    behaviour: BehaviourSignal,
    local_decision: LocalDecision,
) -> FinalAction
```

* `observation` is the same `LocalObservation` the local controller received:
  position, heading, next waypoint, what the belief map says is ahead.
* `sightings` are what the vehicle's own camera reports *now*, already
  projected to map tiles — the `WorldDetection` type Phase 3's merger produces.
  Projection stays with the code that owns camera geometry; fusion never
  touches a pixel.

`WorldDetection` moves from `perception/merger.py` into
`interfaces/perception.py`, because a port may not import an adapter. The
merger re-exports it, so existing imports are unchanged.

**2. The vehicle moves through the real city.** `tick` applies the move against
an occupancy grid built from the true world. The command center still
*plans* on its belief; the vehicle *drives* in reality. This is how a real
vehicle works, and it is what makes a belief error cost something.

**3. A belief can trail reality.** `LaggedGridSource` hands the command center
the grid from N refreshes ago — the communications lag of a real disaster,
where reports arrive late but the vehicle's camera sees the present. It is the
condition under which fusion is evaluated.

## Consequences

* Measured with a 30-tick lag over 30 missions, both the waypoint follower and
  the DQN crash about a hundred times and rescue fewer people. The collisions
  are real, and the onboard camera can see what caused them — which is what
  fusion is for.
* Phase 3–6 results stand. On the shipped map the camera-built grid equals
  ground truth, and with no lag belief and reality coincide, so moving the
  physics onto the true world changes no reported number; all 977 tests
  passed unchanged after the switch.
* The fusion adapter depends on `LocalObservation`, so a future change to that
  value type touches fusion too. That coupling is deliberate: "what is ahead"
  is defined in exactly one place.
