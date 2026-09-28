# ADR 0003: Fusion reads egocentric scene evidence, and is tested against a lagging map

**Status:** Accepted
**Phase:** 7

## Context

Phase 7's fusion network (Unit I) combines three upstream signals into the
action the vehicle executes: the DQN's Q-values, the LSTM's behaviour
prediction, and what a camera sees. Two questions had to be settled before
any network was trained.

**1. What form does the camera's signal take?** The original port was
`fuse(detections: list[Detection], ...)`, with detections as pixel boxes. A
box only means something relative to the camera that produced it and the
way the vehicle faces. A network fed raw boxes would have to learn the
projection from scratch, and learn it again for every camera geometry.

**2. What is there to fuse?** Measured before any fusion code existed:
with the command center's map kept current, the A* planner routes around
every known hazard. The DQN therefore never faces a hazard its own inputs
cannot see, and over 20 missions it had **0 collisions**. A fusion network
has nothing to correct in that setting. Any M7 comparison would come out
at parity by construction, and parity is not evidence of anything.

## Decision

**1. Egocentric evidence.** The port becomes
`fuse(evidence: SceneEvidence, behaviour, local_decision)`. `SceneEvidence`
holds the highest detector confidence for each (class, region) pair. The
regions are named relative to the vehicle: `AHEAD`, `AHEAD_FAR`, `LEFT`,
`RIGHT`, `BEHIND` and `NEARBY`. The projection reuses
`CameraView.tiles_of_box`, the same exact projection the CCTV pipeline
already uses, and lives in `perception/scene_evidence.py`. The network sees
the answer ("0.93 that there is debris directly ahead"), not the geometry.

**2. The map-lag scenario.** `LaggedGridSource` delays the command center's
map by a configurable number of refreshes, modelling CCTV processing, radio
and operator delay. When the map can lag, the engine's physics switches to
the true world (`SimulationEngine(physics_source=...)`). Debris the map has
not heard of still stops the vehicle. Measured, DQN driving, 20 missions:

| map lag | collisions | vehicle damage |
|---|---|---|
| none | 0 | 12 |
| 1 s | 5 | 55 |
| 3 s | 44 | 246 |
| 6 s | 91 | 511 |

The onboard camera sees the present, not the map's past. That gap is what
fusion is trained and judged on. The default lag is 3 s (30 refreshes),
set in `configs/training/fusion.yaml`.

## Consequences

- Nothing called the old port, so changing it cost nothing. The
  `IDecisionFusion` docstring and PROJECT.md §12 record the new contract.
- The default behaviour is unchanged. Lag 0 and no physics source is exactly
  the Phase 2-6 simulation, and the existing tests pass untouched.
- Labels are exact and free. On every tick the DQN is asked twice: once on
  the stale map it really drives on, and once on the true world. The second
  answer is the label; a tick where the two differ is *critical*.
- The scenario is honest about what fusion is for. It does not make the
  DQN better at driving. It makes the whole vehicle safe when one of its
  inputs, the map, is wrong, which is the reason real autonomous vehicles
  fuse onboard sensing with a map at all.
- The lag can be changed while a mission runs (`LaggedGridSource.lag`), so
  the demo can show the failure and the recovery live.
