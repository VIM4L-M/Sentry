# Phase 8 — end-to-end integration: design notes

**Status:** complete. Every trained model from Phases 3-7 now drives one
mission together, with no human input and no ground truth anywhere in the
loop: the command center plans on the map its CCTV cameras build, and the
vehicle acts on what its own camera sees. M8 is met (see Results).

## The stack

```
CCTV x4 -> degrade -> denoise (AE, Ph 4) -> YOLO, one batch (Ph 3) -> merge -> belief grid
                                                                                |
                                          A* command center (Ph 2) <------------+
                                                | route
onboard camera -> degrade -> YOLO (Ph 4) -> sightings -+
DQN local controller (Ph 6) ---------------------------+--> MLP fusion (Ph 7) -> action
state history -> LSTM (Ph 5) --------------------------+
```

The vehicle moves through the real city; nothing downstream of the cameras
ever reads it. Frames are degraded (smoke, blur, sensor noise) before any
model sees them, exactly as in training.

## What it adds

| piece | where |
|---|---|
| `AutonomyStack` — loads every model once, wires a fresh mission per call | `autonomy/stack.py` |
| `Profiler` and `Timed*` wrappers — wall-clock time per stage | `autonomy/profiling.py` |
| `MissionRecord` — one mission as plain JSON: outcome, path, timings, events | `autonomy/record.py` |
| `ThrottledGridSource` — re-perceive the city every N refreshes | `simulation/grid_source.py` |
| `IVisionDetector.detect_many` — batched detection (the port default loops) | `interfaces/perception.py` |
| route re-check debounce, `mission.block_confirm_refreshes` | `simulation/mission.py` |
| config | `configs/autonomy.yaml` |
| entry points | `scripts/run_autonomous.py`, `scripts/evaluate_autonomy.py` |

```bash
python scripts/run_autonomous.py                 # the window; TAB switches to manual
python scripts/run_autonomous.py --headless      # one mission, then the profile
python scripts/evaluate_autonomy.py              # M8: 20 new missions vs a reference
```

Every mission writes a record to `runs/missions/` (git-ignored). Phase 9's
dashboard reads those files; it never runs the models itself.

## Decisions

### A composition root, not a new layer

Nothing in `autonomy/` makes a decision. The adapters from Phases 3-7 are
used unchanged, and `build_mission` is the same factory every earlier
script calls; the stack only chooses which grid source and which
controller to hand it. That is why no port had to change except by one
optional method (below), and why the live window and the headless run are
the same code — the window passes `wrap=build_mode_switch` to add the
keyboard override.

The stack imports nothing from `training/`. The first draft pulled in a
checkpoint-fingerprint helper and the device resolver; both were removed
(the device choice is four lines, inlined) so the live system does not
depend on the offline one.

### Timing every stage

Each expensive call is wrapped in a `Timed*` decorator implementing the
same port, so the profile is taken on the real wiring, not a benchmark
beside it. Measured on the cloud CPU (4 cores, no GPU), shipped map,
perception every tick:

| stage | ms per tick |
|---|---|
| CCTV denoise (4 frames) | 357 |
| CCTV detect (one batch of 4) | 32 |
| onboard detect | 20 |
| LSTM | 0.8 |
| DQN | 0.4 |
| fusion | 0.3 |
| **whole tick** | **~426** |

The tick is 100 ms of simulated time, so on CPU the stack runs at about
0.24x real time. **The denoiser is 84% of it** — a full-resolution
convolutional autoencoder on four 256 px frames. The decision models
together cost under 2 ms. On the target RTX laptop GPU the convolutional
stages are the ones that accelerate; run `run_autonomous.py --headless` there
to fill in this table.

### Batching the cameras

The four CCTV frames used to go through YOLO one at a time.
`IVisionDetector.detect_many` sends them as one batch; the port's default
still loops over `detect`, so a detector that cannot batch needs no change.
Checked over 40 frames: identical boxes, largest confidence difference
4.8e-7 (float noise).

### Throttling perception

`ThrottledGridSource` re-perceives the city every `perception_every`
refreshes and hands back the last map in between — the knob to trade
freshness for speed on slow hardware. A real CCTV network does not
re-report every 100 ms either. At `perception_every: 4` the CCTV cost per
tick falls by roughly 4x; the command center then sees changes up to 0.3 s
late, well inside the 3 s lag Phase 7 was trained against.

### A phantom obstacle cost a victim — the route re-check is now debounced

The first full-stack run on the shipped map rescued 3 and lost 1 — where the
same fused controller on a ground-truth map rescues all 4. The event log
said "route blocked on the latest map — replanning", then "no route to
victim_03 — abandoned". Diffing the camera-built map tick by tick found
one false DEBRIS cell at (8, 17), beside victim_03 at (7, 17), present for
a single tick.

Phase 7's re-check (drop the route when a fresh map shows it blocked)
reacted to that one frame; the replan, on the same wrong map, found no
way in and wrote victim_03 off. On a ground-truth map a single frame is
never wrong, so Phase 7 could not have seen this.

A detector flickers; a collapse persists. The route must now read as
blocked on `mission.block_confirm_refreshes` (3) consecutive refreshes
before it is dropped. With it, the shipped-map full-stack mission rescues
4, loses 0, with no collisions, in 179 ticks. Tests pin both sides: a
blockage that persists for three refreshes replans; a one-refresh phantom
leaves the route alone.

The Phase 7 evaluation was re-run with the debounce: fusion's numbers are
unchanged within noise (13 collisions, damage 2.8% -> 2.9%); the waypoint
follower's collisions fell from 164 to 154. The old test for the re-check
relied on a blocked reading one tick before a hazard replanned anyway; it
was replaced by the two deterministic tests above.

## Results — M8

20 new missions (seeds 7000-7019, random starts), CPU, perception every
tick, no map lag. Each is driven twice: by the full stack, and by the
reference — the waypoint follower on the ground-truth map, the best this
city allows with perfect knowledge.

| | rescued | lost | collisions | vehicle damage | completed | ticks |
|---|---|---|---|---|---|---|
| **full stack (camera-built map)** | **72** | **2** | **0** | **0.0%** | **100%** | **163** |
| reference (ground-truth map) | 77 | 1 | 0 | 1.2% | 100% | 172 |

**M8 is met**: every mission ran from spawn to the hospital with no human
input (threshold: at least 90% completed), rescuing 94% of what perfect
knowledge rescues (threshold: at least 90%).

Measured cost across all 20 missions: 413 ms per tick, of which the
denoiser is 345 ms, CCTV detection 31, the onboard camera 20, and the three
decision models 1.5 together.

**Where the five missing rescues went.** Almost all of them are one victim:
victim_03 at (7, 17), at the end of a narrow pocket. In 7 of
the 20 missions the camera-built map shows that pocket sealed — a fire or
debris cell the detector reports on its only way in — and the command
center writes the victim off as unreachable, where the ground-truth map
may still show a path. The debounce stops a single frame doing this; a
misreading that persists for three frames still does. The vehicle never
crashes and never takes damage: when the camera map is wrong, it is wrong
on the cautious side.

## Still open

* **victim_03's pocket.** The main loss above is perception, not decision:
  the detector's readings around (7, 17) wall the pocket off. Targeted
  training frames of that spot, or letting the planner try a route through
  low-confidence obstacles before abandoning anyone, are the two fixes.

* **GPU profile.** The table above is CPU. The same command on the RTX
  laptop gives the number that matters for the target hardware.
* **The onboard detector and fusion were trained together.** Swapping in the
  GPU `labelfix` detector means re-recording fusion's data
  (`train_fusion.py --record --detector ...`): fusion learns its detector's
  particular mistakes.
* **Sightings do not reach the map.** Fusion can only stop; the command
  center hears about debris only when the CCTV sees it. Feeding the
  vehicle's sightings into the belief grid would let it replan at once.
