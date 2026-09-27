# SENTRY AI

**Autonomous Emergency Rescue Vehicle for Disaster Zones** — a simulation-based Applied
Deep Learning project. See [`PROJECT.md`](PROJECT.md) for the full architecture,
diagrams, phase roadmap, and API contracts — it is also where the working rules this
codebase follows are written down (§2, §12, §16).

**Status:** Phases 1, 2 and 3 complete. A full autonomous rescue mission runs end to end
(occupancy grid → A\* routing → mission control → live HUD) in a city that changes
underneath it: fire spreads, buildings collapse into the streets, and the command
center replans around both. A four-camera CCTV network feeds a YOLOv8n detector
whose output builds the occupancy grid the planner reasons over — the vehicle
can complete a rescue on what it *sees* rather than on ground truth. Phase 4
(denoising autoencoder) is complete: the denoiser slots in front of the detector
with `--denoiser`, and at double the smoke it lifts victim recall from 0.86 to 0.99
(M4). Phase 5 (behaviour LSTM) meets M5, and Phase 6 (DQN driving policy) meets M6. See PROJECT.md §10-11 for the phase breakdown and
milestones.

## Quick Start

Requires Python 3.11+ (developed and tested on 3.14; see "Environment Notes" below).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the sentry_ai package in editable mode
```

Run the test suite:

```bash
pytest                                          # unit + integration tests
pytest --cov=sentry_ai --cov-report=term-missing  # with coverage
mypy src scripts                                 # type checking
ruff check src tests scripts                      # linting
```

Run a rescue mission (the vehicle plans routes and drives itself):

```bash
python scripts/run_simulation.py                          # windowed, ground-truth map
python scripts/run_simulation.py --headless               # no window, prints the outcome
python scripts/run_simulation.py --perception --device cuda   # plan on what the cameras see
python scripts/run_simulation.py --perception --denoiser models/autoencoder/sentry/best.pt \
    --weights models/yolo/denoised/weights/best.pt      # see Phase 4 below
```

| Key | Action |
|---|---|
| `Escape` | quit |
| `Space` | pause / resume |
| `R` | restart the mission |
| `Tab` | toggle autonomous ↔ manual driving |
| `G` | toggle the occupancy-grid debug view |
| `C` | toggle the camera panel |
| arrows / `WASD` | drive (manual mode) |

The window is laid out so a mission can be followed without narration:

```
┌────────────────────────────────┬──────────────┐
│                                │  cctv_nw     │
│   the city — terrain, glyphs,  │  cctv_ne     │  what the cameras see,
│   CCTV footprints, and the     │  cctv_sw     │  with ground-truth boxes
│   planned route (the route it  │  cctv_se     │  (YOLO output in Phase 3)
│   replaced stays greyed out)   │  onboard     │
├────────────────────────────────┴──────────────┤
│ phase │ counters │ event log │ battery/health │
└───────────────────────────────────────────────┘
```

`R` restarts without closing the window. It rebuilds the mission from config
rather than rewinding, so it replays *identically* — the hazards are seeded. For a
different disaster on the same map, change `hazards.seed` in
`configs/simulation.yaml`.

Press `G` and the city is replaced by the occupancy grid the planner
actually reasons over — the 0-6 codes, colour-coded and numbered. Without
`--perception` it matches the world exactly. With it, you are looking at a
*belief* assembled from YOLO detections, and this is how you see where it is
wrong.

Capture the synthetic perception dataset (what Phases 3 and 4 train on):

```bash
python scripts/capture_dataset.py --output data/synthetic
```

This runs a full mission and writes, for every camera and every Nth tick:

| Path | Contents |
|---|---|
| `images/` | degraded frames — smoke, blur, sensor noise (detector input) |
| `labels/` | YOLO labels for those frames (`class cx cy w h`, normalized) |
| `clean/` | the matching clean frames — paired with `images/`, this is the denoising autoencoder's training set |
| `classes.txt` | class id → name (`victim`, `fire`, `obstacle`) |

Render the static disaster city instead (Phase 1's walking skeleton — no movement):

```bash
python scripts/run_preview.py
# or with a custom config:
python scripts/run_preview.py --config configs/app.yaml
```

Press `Escape` or close the window to exit.

From Phase 3 onward you'll also need the heavier ML stack:

```bash
pip install -r requirements-ml.txt
```

**For GPU training**, that installs a CPU-only Torch. Replace it with a CUDA
build matching your driver — on an RTX 40-series with a recent driver:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130     torch torchvision
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`. The
torch wheel is ~1.9 GB; if the download times out, fetch it with
`curl -C -` (which resumes) and `pip install` the local file.

## Phase 3 — Detection (Unit II)

```bash
python scripts/fetch_pretrained.py     # once: COCO nano checkpoint
python scripts/build_dataset.py        # ~2 min: 1965 labelled frames
python scripts/train_yolo.py           # fine-tune YOLOv8n
python scripts/evaluate_yolo.py        # per-class metrics
```

The dataset is split **by mission**, not by frame: validation missions are
disasters the model has seen no frame of. Splitting frames at random would put
near-duplicates on both sides and turn mAP into a memorisation score.

**One tile yields one annotation, and a victim outranks the debris pinning
them.** A victim trapped in rubble is still *painted* over the debris — a brown
ring around a pink core — but only the victim is labelled. Annotating both left
the detector choosing between a 13 px obstacle box and the 8 px victim box
inside it, at twelve obstacles to every victim; it answered `OBSTACLE` every
time, which is an impassable code, so a trapped victim became a wall the planner
routed around. This is the same precedence `OccupancyGrid` already applies to the
true world state. Measured victim recall was exactly the share of victims *not*
pinned in rubble.

To run a camera experiment without touching the shipped config, point the builder
at another sensors file — marker sizes live in `sensors.markers`:

```bash
python scripts/build_dataset.py --sensors configs/sensors_v75.yaml --output data/v75
```

Final detector: victim recall **1.000**, mAP50 **0.991**. The experiment log,
including the hypotheses that were measured and rejected, is in
[`docs/architecture/phase3-detection.md`](docs/architecture/phase3-detection.md).

**3.2 — merging across cameras.** `DetectionMerger` turns per-camera detections
into one world-space belief, so a victim seen by two overlapping cameras is one
victim. Detections project to the *footprint* of tiles their box covers, not a
centre tile, because a fire straddling a camera seam is clipped differently by
each camera and only its footprints meet. The merge rule is asymmetric on
purpose: fire merges on adjacency as well as overlap, victims and debris merge
only on overlap — over-merging a hazard costs nothing, over-merging victims
erases a person.

**3.3 — building the map.** `OccupancyGridBuilder` turns merged detections into
the `OccupancyGrid` A\* plans over, replacing `OccupancyGrid.from_city_map` as
the producer; the ground-truth version stays as the answer key, and
`GridComparison` marks the paper.

```bash
python scripts/evaluate_grid.py --device cuda   # score the real detector's grid
python scripts/evaluate_grid.py --perfect       # isolate the pipeline from the weights
```

The static street plan is copied from the surveyed map and only `FIRE`,
`DEBRIS` and `VICTIM` come from the cameras — so a building that collapses
mid-mission has to be *detected*, because the survey is deliberately never
refreshed.

Cell accuracy is the wrong headline here: the grid is ~94% road and building,
so a belief that detected nothing would still score in the nineties. What gets
reported instead is the three failure modes, as tile sets rather than counts —
a **missed hazard** drives the vehicle into a fire, a **phantom obstacle** costs
a detour, and a **missed victim** means nobody is ever dispatched.

Fire was where this bit. A camera reports a box; ground truth stamps a disc,
and marking the whole box doubled the believed footprint. Those invented tiles
are impassable, so they walled off open streets — measured, that cost two
victims and failed a mission. Fire now projects by tile *centre* rather than
any pixel overlap, and a square footprint is read back through the same
`tiles_within` that drew it. Victims are excluded from both rules on purpose:
an over-claimed tile costs a detour, a dropped victim costs a life.

**3.4 / 3.5 — the mission runs on it.** `MissionController` now asks an
injected `IOccupancyGridSource` for its map, so ground truth and perception are
interchangeable and nothing downstream can tell which it got:

```bash
python scripts/run_simulation.py --perception --device cuda
```

Press `G` and the grid view is a *belief* rather than the world. Against real
`labelfix` weights on degraded frames, across six hazard seeds, missions
complete with **4 rescued, 0 lost, 0 unreachable** — identical to ground truth.
On the shipped map the pipeline is lossless: the grid built from YOLO output
matches `from_city_map` cell for cell, which a test asserts as an equality.
The caveats, and the failure modes that remain, are in
[`docs/architecture/phase3-detection.md`](docs/architecture/phase3-detection.md).

## Phase 4 — Denoising Autoencoder (Unit IV)

```bash
python scripts/train_autoencoder.py --device cuda     # ~3M-parameter conv autoencoder
python scripts/evaluate_denoiser.py                   # PSNR, input vs denoised
python scripts/build_denoised_dataset.py              # dataset for the detector behind it
python scripts/train_yolo.py --dataset data/denoised --name denoised --device cuda
python scripts/evaluate_denoiser.py --detector models/yolo/labelfix/weights/best.pt \
    --denoised-detector models/yolo/denoised/weights/best.pt   # the M4 comparison
```

`ConvDenoisingAutoencoder` implements `IDenoiser` and sits between the degrader
and the detector:

```
SensorRig -> FrameDegrader -> IDenoiser -> YoloDetector -> DetectionMerger -> ...
```

**Training pairs are made on the fly.** Each sample is a random crop of a
*clean* frame, rotated or mirrored, then corrupted at a random severity between
0.5x and 2x the shipped smoke/blur/noise. Every epoch sees new smoke, and half
of what it trains on is worse than anything the detector was trained on — which
is where a denoiser can actually help. Validation uses the stored degraded
frames, the exact input the detector gets.

**Skip connections are on by default** (`--no-skip` trains without them for
comparison). Without them an 8 px victim has to squeeze through a 1/8-resolution
bottleneck, and comes back as a smudge.

**The denoiser and the detector are a pair.** Put in front of the detector
trained on smoky frames, the denoiser made detection *worse* — victim recall fell
from 0.996 to 0.347 — because that detector learned what a victim looks like
through smoke. Retrained on denoised frames, the pair wins where it should:

| corruption | mAP50, no denoiser → with | victim recall, no denoiser → with |
|---|---|---|
| none (clear day) | 0.983 → **0.992** | 1.000 → 1.000 |
| 1.0x (trained on) | 0.992 → **0.993** | 1.000 → 1.000 |
| 1.5x | 0.977 → **0.982** | 1.000 → 1.000 |
| 2.0x | 0.880 → **0.938** | 0.861 → **0.990** |

Full length on an RTX 3050 Laptop GPU: 40 autoencoder epochs, 60 per detector.

**PSNR is always reported next to the input's own PSNR.** And PSNR is not the
goal: milestone M4 is about *detection*. `evaluate_denoiser.py --detector` scores
the same detector on corrupted and on denoised frames, at severities 1.0, 1.5
and 2.0. Design notes and results are in
[`docs/architecture/phase4-denoising.md`](docs/architecture/phase4-denoising.md).

## Phase 5 — Behaviour LSTM (Unit III)

```bash
python scripts/record_trajectories.py   # ~40 s: 200 missions, each from a random road tile
python scripts/train_lstm.py            # ~1-2 min, CPU is fine
python scripts/evaluate_lstm.py         # vs baselines: all held-out windows, then novel only
```

`LstmMotionPredictor` implements `IMotionPredictor`: from the last 8 ticks of
vehicle state it predicts what the vehicle does over the next 4 — **advance**,
**retreat**, **hold** or **divert** — judged against the way it was facing. The
same rule labels the training data and runs the "keep doing the same" baseline,
so model and baseline answer exactly the same question.

**Scored by macro-F1, not accuracy.** Seven windows in ten are "advance", so
"always advance" gets 0.69 accuracy while never predicting a turn; macro-F1 gives
it 0.20.

| held-out windows | best baseline macro-F1 | LSTM macro-F1 | LSTM accuracy |
|---|---|---|---|
| all (7,279) | 0.205 | **0.802** | 0.942 |
| never seen in training (178) | 0.349 | **0.663** | 0.820 |

**The first result was memorisation.** With every mission starting on the same
tile, 99.4% of held-out windows were exact copies of training windows and the
model scored 0.964 by remembering routes. Starts are now randomised, and
`evaluate_lstm.py` also scores only the windows that repeat nothing — the second
row. Details in
[`docs/architecture/phase5-sequence.md`](docs/architecture/phase5-sequence.md).

## Phase 6 — DQN Driving Policy (Unit V)

```bash
python scripts/train_dqn.py                 # ~15 min on CPU; a GPU buys little here
python scripts/evaluate_dqn.py              # 40 held-out missions vs the waypoint follower
python scripts/evaluate_dqn.py --simulation configs/simulation_stress.yaml
python scripts/run_simulation.py --dqn models/dqn/sentry/best.zip   # watch it drive
```

`SentryEnv` turns a whole mission into a Gymnasium environment: the A\* command
center still plans every route, and a Stable-Baselines3 DQN replaces the waypoint
follower for the tick-by-tick driving — forward, reverse, turn, stop. It sees seven
**egocentric** numbers (where the waypoint is ahead/right, blocked ahead, fire,
battery), never its absolute position, so it learns to drive rather than to
memorise the city.

**Judged on missions, not reward.** M6 was fixed before training: rescue at least
90% of what the follower rescues on the same held-out missions.

| 40 held-out missions | rescued | lost | collisions | completed |
|---|---|---|---|---|
| waypoint follower | 143 | 5 | 0 | 95% |
| DQN | **149** | 5 | 0 | **100%** |

The honest reading is **parity with a speed edge**: the six extra rescues are one
situation, where the DQN was two tiles further on when a fire sealed a street. Two
things were caught on the way — an early policy drove 14% of its tiles *backwards*
(the physics made reversing as cheap as driving; a reverse penalty fixed it), and
with a perfect map the planner absorbs every surprise, so the learned driver's real
test is on the camera-built map in Phase 8. Details in
[`docs/architecture/phase6-reinforcement.md`](docs/architecture/phase6-reinforcement.md).

## Project Layout

```
configs/     All tunables — YAML, loaded into typed dataclasses (see config/schema.py)
src/sentry_ai/
  common/      Logging, exceptions, shared type aliases
  config/      Typed config schema + the only code that reads YAML
  domain/      Entities, enums, CityMap, OccupancyGrid — pure Python, no frameworks
  interfaces/  Ports (ABCs): perception, sequence, navigation, world, decision fusion
  navigation/  A* global route planner (classical, not learned)
  perception/  YOLO adapter, denoising autoencoder, cross-camera merger,
               occupancy-grid builder + scoring
  simulation/  Tick engine, mission state machine, vehicle physics, hazards
  sensors/     Synthetic cameras: rasterizer, ground-truth labels, frame degradation
  sequence/    Behaviour classes and the LSTM motion predictor
  decision/    The DQN local controller (Phase 6); fusion comes in Phase 7
  training/    Offline only: dataset builder, YOLO + autoencoder + LSTM training,
               trajectory recording, seeding, metric logs, checkpoints
  rendering/   Pygame map renderer, HUD, keyboard input, mission window
scripts/     Composition roots / CLI entry points
tests/       unit / integration / e2e, mirroring src/
docs/        Phase design notes and ADRs
```

**How navigation is split:** A\* plans the route across the city from an occupancy
grid; a learned local controller (Phase 6 DQN, today a deterministic waypoint
follower) decides each tick's move. The reasoning is in
[`docs/adr/0002-two-tier-navigation-and-command-center.md`](docs/adr/0002-two-tier-navigation-and-command-center.md).

**How the two hazards differ:** fire only takes hold on flammable terrain, so it
threatens victims and raises route costs but never severs the road network; debris
collapses into clear streets, which is what actually forces replanning. Keeping them
separate is what stops a mission becoming a coin flip — see
[`docs/architecture/phase2-dynamic-world-and-sensors.md`](docs/architecture/phase2-dynamic-world-and-sensors.md).

**Why the cameras have their own palette:** `configs/render.yaml` styles the map a
human reads; `configs/sensors.yaml` styles what the detector is trained on. Separating
them means restyling the operator display cannot silently invalidate a trained model.

See PROJECT.md §4 for the full target folder structure (including packages not yet
built, owned by later phases).

## Environment Notes

- Vanilla `pygame` has no prebuilt wheel for CPython 3.14 and fails to build from
  source without system SDL dev headers. This project uses **`pygame-ce`** instead —
  an actively maintained, drop-in-compatible fork (`import pygame` is unchanged).
- Target hardware: an RTX-class laptop GPU with 4-8GB VRAM and 16GB system RAM — every
  model choice (YOLOv8**n**, modest LSTM/MLP sizes, DQN) is made with that ceiling in
  mind (see PROJECT.md §18/Environment Notes).

## Documentation

- [`PROJECT.md`](PROJECT.md) — architecture, diagrams, roadmap, contracts (canonical, kept current)
- [`docs/architecture/`](docs/architecture/) — per-phase design notes
  ([Phase 1](docs/architecture/phase1-foundation.md),
  [Phase 2 — mission loop](docs/architecture/phase2-simulation.md),
  [Phase 2 — dynamic world & sensors](docs/architecture/phase2-dynamic-world-and-sensors.md),
  [Phase 3 — detection](docs/architecture/phase3-detection.md),
  [Phase 4 — denoising](docs/architecture/phase4-denoising.md),
  [Phase 5 — sequence](docs/architecture/phase5-sequence.md),
  [Phase 6 — reinforcement](docs/architecture/phase6-reinforcement.md))
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
  ([0001 config](docs/adr/0001-config-driven-yaml-dataclasses.md),
  [0002 two-tier navigation](docs/adr/0002-two-tier-navigation-and-command-center.md))
