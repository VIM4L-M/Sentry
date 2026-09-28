# Phase 7 — decision fusion: design notes

**Status:** core complete. An MLP (PyTorch; ReLU, Dropout, Adam) implements
`IDecisionFusion`, combining the DQN, the LSTM and the vehicle's onboard
camera. M7 is met per decision — fusion's macro-F1 beats every single
signal — and, more to the point, on whole missions: with the command
center's map three seconds behind reality, fusion cuts collisions from 150
to 13 and vehicle damage from 20% to under 3% per mission, at no cost in
rescues. Everything trained on CPU.

Getting there required two corrections to earlier phases, recorded in
[ADR 0003](../adr/0003-fusion-sees-the-observation-and-the-world.md).

## What it adds

| piece | where |
|---|---|
| `MlpFusion` (the `IDecisionFusion`), features, `CameraVetoFusion` baseline | `decision/fusion.py` |
| `FusedLocalController` — DQN + LSTM + camera + fusion in the controller slot | `decision/fused_controller.py` |
| `OnboardSensing` — the vehicle's camera as map-space sightings | `perception/onboard.py` |
| `LaggedGridSource` — a command-center map that trails reality | `simulation/grid_source.py` |
| recording, the safe label, training, ablations, decision metrics | `training/fusion.py` |
| config | `configs/training/fusion.yaml` |
| entry points | `scripts/train_fusion.py`, `scripts/evaluate_fusion.py` |

## What runs today

```bash
python scripts/record_trajectories.py --dqn models/dqn/sentry/best.zip --output data/trajectories_dqn
python scripts/train_lstm.py --trajectories data/trajectories_dqn --name dqn   # LSTM on the DQN's driving
python scripts/train_fusion.py --record      # ~15 min: record 200 missions, train 4 models
python scripts/evaluate_fusion.py            # ~6 min: 40 new missions, 4 drivers
```

## Decisions

### Fusion needed a job, and the simulation had to be able to give it one

Phase 1 planned fusion as "combine detector confidences, the LSTM signal
and the DQN's Q-values". Built literally, on a ground-truth map, it has
nothing to do: the DQN already matches a perfect driver (Phase 6), so the
best fusion can do is copy it.

Fusion earns its place when the signals **disagree about the world** — when
the command center's map is wrong and the vehicle's own camera is right.
Two things stood in the way.

**The vehicle drove through its map's mistakes.** `SimulationEngine.tick`
applied moves against the command center's *belief*. Debris the map had
missed was debris the vehicle simply drove through; a wrong map could
never cause a crash. The vehicle now moves through the real city while the
command center keeps planning on its belief (ADR 0003). All 977 tests
passed unchanged across that switch, because until now belief and reality
coincided on the shipped map.

**A map that is always right leaves nothing to catch.** `LaggedGridSource`
gives the command center the map as it was N refreshes ago — the
communications lag of a real disaster, where reports arrive late but the
vehicle's camera sees the present. Measured over 30 missions:

| map lag | collisions (follower / DQN) | rescued (follower / DQN) |
|---|---|---|
| none | 0 / 0 | 107 / 113 |
| 10 refreshes | 10 / 19 | 101 / 107 |
| **30 (3 s)** | **105 / 102** | 94 / 95 |
| 60 | 265 / 226 | 72 / 78 |

Phase 7 trains and evaluates at 30.

### The route is re-checked against every fresh map

Designing the veto exposed a latent gap. With a lagging map, a route
planned just after a collapse can run straight through it; until now the
only thing that ever corrected such a route was the vehicle crashing into
the rubble. Take the crash away and nothing would: the vehicle would wait
at the debris forever. `MissionController` now drops the route whenever a
refreshed map shows a tile on its unvisited remainder impassable — what a
command center does anyway. With a perfect map it never fires (a test
asserts this).

A first version counted the vehicle's own tile, which on a perceived map
can read as fire (fire outranks the vehicle marker). It then replanned the
same route every tick until the mission failed; a Phase 3 integration test
caught it.

Phase 8 debounced it: the route must read as blocked on three consecutive
refreshes. On the full stack a single-frame detector false positive beside
a victim had the victim abandoned (details in
[phase8-integration.md](phase8-integration.md)).

### The port changed (ADR 0003)

`fuse()` was `(detections, behaviour, local_decision)`, with detections in
pixels. Whether debris is on the tile ahead depends on the camera's window
— clamped at map edges, so the vehicle is not always at its centre — and
on the vehicle's heading, and the signature carried neither. It is now
`fuse(observation, sightings, behaviour, local_decision)`: sightings are
Phase 3's `WorldDetection` s, already in map tiles, and the observation says
where the vehicle is and which way it faces. `WorldDetection` moved into
`interfaces/perception.py` so the port does not import an adapter.

### The label: the DQN's move, made safe

Every tick the DQN proposes an action on the lagging map. The label is that
action — unless it would move the vehicle into a tile that is *really*
impassable, in which case STOP. Fusion therefore learns one precise thing:
keep the DQN's driving, veto the moves the camera shows are unsafe.

Data is recorded with the DQN driving and the fusion slot filled by a
recorder that passes the DQN through, so the DQN really does drive into the
debris — those are the samples that matter. 150 training missions give
24,304 decisions, 406 of them unsafe; 50 validation missions give 7,958 and 82.

### Features, egocentric, 31 of them

| group | width | content |
|---|---|---|
| dqn | 5 | softmax of the Q-values |
| lstm | 4 | predicted behaviour, one-hot x confidence |
| camera | 15 | per class (victim, fire, obstacle), highest confidence on the tiles one and two ahead, behind, left and right |
| observation | 7 | the DQN's own egocentric features |

"Ahead" follows the vehicle's heading, and a tile off the map's edge reads
as empty rather than crashing — `Position` rightly refuses negative
coordinates, so tiles are compared as tuples.

The MLP is 31 → 64 → 64 → 5 with ReLU and Dropout 0.2, trained with Adam and
square-root class weights (the Phase 5 lesson). The single-signal baselines
are the same network with the other groups masked to zero — a registered
buffer, so a test can prove a masked feature cannot move the output.

### The LSTM was retrained on the DQN's driving

Phase 5's LSTM learned the waypoint follower. The DQN drives differently —
it reverses and pre-turns — so before fusion reads its predictions, it was
retrained on DQN trajectories (`record_trajectories.py --dqn`). M5 still
holds: macro-F1 0.866 against baselines of 0.20 and 0.24; the DQN pauses more,
so HOLD is now a measurable class (F1 0.85, n = 209).

### A hand-written rule is reported next to the network

`CameraVetoFusion` is the obvious rule: take the DQN's action unless it
drives into a tile where the camera sees fire or debris at 0.5 confidence
or more; then stop. If the MLP could not beat it, the network would have
earned its place only as a Unit I exercise. The mission results show why
it does not.

## Results

CPU. Detector: YOLOv8n trained on degraded frames (Phase 4's `cpu20`,
mAP50 0.990); DQN: Phase 6; LSTM: retrained on the DQN.

### Per decision — M7 (7,958 held-out decisions, 82 unsafe)

| | accuracy | macro-F1 | unsafe moves vetoed | needless stops |
|---|---|---|---|---|
| DQN as-is | 0.990 | 0.969 | 0% | 0% |
| camera-veto rule | 0.986 | 0.968 | 100% | 1.4% |
| **MLP fusion** | **0.998** | **0.993** | **100%** | **0.3%** |
| DQN signal only | 0.985 | 0.955 | 0% | 0.4% |
| LSTM only | 0.733 | 0.409 | 34% | 26.2% |
| camera only | 0.816 | 0.500 | 100% | 18.3% |

**M7 is met**: 0.993 against the best single signal's 0.955. Each signal
fails alone for its own reason — the DQN never sees the debris, the camera
does not know where the route goes, the LSTM predicts behaviour rather than
obstacles. Only together do they make the decision.

### Per mission (40 new missions, never used for training or selection)

Command-center map 30 refreshes behind reality. Re-measured after Phase 8
debounced the route re-check (see
[phase8-integration.md](phase8-integration.md)); fusion's numbers moved by
one tick and 0.1% damage, the follower's collisions fell from 164 to 154.

| | rescued | lost | collisions | vehicle damage | completed | ticks |
|---|---|---|---|---|---|---|
| waypoint follower | 143 | 13 | 154 | 21.4% | 100% | 170 |
| DQN alone | 143 | 13 | 150 | 20.1% | 100% | 165 |
| DQN + camera-veto rule | **101** | **39** | 0 | 10.4% | 88% | 215 |
| **DQN + MLP fusion** | **143** | **13** | **13** | **2.9%** | **100%** | **166** |

* **Fusion avoids 91% of collisions and 86% of vehicle damage, at no cost.**
  Same rescues, same losses, same mission time as the DQN alone.
* **The rule is a trap.** Zero collisions — and 42 fewer people rescued, three
  times as many lost. Its 1.4% needless stops are not harmless one-tick
  pauses: when the detector wrongly sees debris on a clear road, the rule
  stops, and keeps stopping, because the command center's map says the road
  is clear and nothing ever replans. The vehicle freezes while victims run
  out of time, and takes fire damage while it waits. The MLP learned when a
  camera warning is real and when to override it.
* **Fusion is not perfect.** It vetoed every unsafe move in the recorded
  validation set but still collided 13 times live: once fusion drives, the
  vehicle reaches situations the DQN-driven recording never did.
* **Avoided crashes did not raise rescues.** On this map a crash costs 5%
  of the vehicle's health, not the mission. The value shows in damage — on
  a longer mission or a harsher disaster it would decide whether the
  vehicle comes back.

## Still open

* **Record with fusion driving (DAgger).** The 13 live collisions come from
  states the DQN-driven recording never visited. Recording a second round
  with fusion in control and retraining is the standard fix.
* **Stop is the only veto.** Fusion waits for the map to catch up. Turning
  away, or reporting the sighting to the command center so it replans at
  once, would be faster; both need the camera's sightings to reach the
  map, which is Phase 8's integration.
* **The onboard detector is the Phase 4 CPU model on degraded frames.** The
  full-length GPU `labelfix` model, or the denoised pipeline, should be
  swapped in with `--detector`; the recording must then be redone, since
  fusion learns that detector's particular mistakes.
* **One lag, one map.** Thirty refreshes on the shipped city. The effect of
  shorter and longer lags on the fusion's benefit is not yet measured.
