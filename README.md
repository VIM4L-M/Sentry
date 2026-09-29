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
(M4). Phase 5 (behaviour LSTM) meets M5, and Phase 6 (DQN driving policy) meets M6.
Phase 7's fusion MLP meets M7, Phase 8 runs all five models live (M8), and Phase 9
adds a mission-control screen with live ablation keys, real cities from
OpenStreetMap, a 3D view, and a one-command demo. See PROJECT.md §10-11 for the phase
breakdown and milestones.

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

## Phase 7 — Decision Fusion MLP (Unit I)

```bash
python scripts/train_fusion.py --detector models/yolo/denoised/weights/best.pt \
    --denoiser models/autoencoder/sentry/best.pt --device cuda     # ~25 min collect + train
python scripts/evaluate_fusion.py --detector models/yolo/denoised/weights/best.pt \
    --denoiser models/autoencoder/sentry/best.pt --device cuda     # 40 whole missions
```

`MlpFusion` (Linear -> ReLU -> Dropout, trained with Adam) takes three inputs and
picks the action the vehicle executes:
- the DQN's Q-values,
- the LSTM's behaviour prediction,
- what the vehicle's own camera sees, projected egocentrically (debris *ahead*,
  fire on the *left*...).

**Fusion needs something to fix.** With a current map the planner routes around every
hazard and the DQN never crashes, so fusion is judged where real systems need it: the
command center's map **lags the world by 3 s**. The DQN, driving on that stale map,
crashes into collapses the map has not heard of. The onboard camera sees them. Labels
are exact and free: each tick the DQN is also asked what it would do on the *true*
map.

| held-out ticks | accuracy | DQN mistakes caught | correct choices wrongly changed |
|---|---|---|---|
| DQN alone | 0.992 | 0% | 0% |
| **fusion** | 0.985 | **89.5%** | 1.4% |
| trained on camera only | 0.752 | 50.9% | 24.6% |
| trained on LSTM only | 0.516 | 15.8% | 48.1% |

Only the combination does both: it catches the mistakes (the camera) and leaves the
correct choices alone (the DQN). That is milestone M7.

| 40 whole missions, 3 s map lag | rescued | lost | collisions | damage | completed |
|---|---|---|---|---|---|
| fresh map, DQN (ceiling) | 150 | 4 | 0 | 28 | 40/40 |
| lagged map, DQN | 141 | 10 | 71 | 423 | 38/40 |
| **lagged map, fusion** | **138** | **9** | **9** | **127** | **37/40** |

Collisions fall 87% and damage 70%, for 3 fewer rescues. Fire is projected by tile
centre, as in the CCTV grid, so a fire *beside* the street is not mistaken for one
ahead. An optional mode also reports what the camera saw to the command center's
map: 0 collisions, but 26 of 40 missions completed. On real streets it was never
trained on, reports pay off when the camera is right; details in the Phase 7 notes. With a perfect camera, fusion matched
the fresh-map ceiling. Full story, including the metric that turned out to be wrong,
in [`docs/architecture/phase7-fusion.md`](docs/architecture/phase7-fusion.md).

## Phase 8 — Full Autonomy (all five models)

```bash
python scripts/run_simulation.py --full --device cuda
```

`--full` composes every model behind the one `ILocalController` port the engine
already had, so the engine itself is unchanged:

```
onboard camera -> FrameDegrader -> autoencoder -> YOLOv8n -> SceneEvidence --+
vehicle states  -> LSTM behaviour predictor -----------------------------------+--> fusion MLP --> action
stale-map observation -> DQN Q-values -----------------------------------------+
```

`FusedLocalController` runs the DQN, the LSTM and the camera each tick, and
the fusion network makes the final call. A 3 s map lag is on from the start.
Individual models can also be given explicitly: `--dqn`, `--lstm`, `--fusion`,
`--onboard-weights`, `--denoiser`, `--lag`.

## Phase 9 — Mission Control, Real Cities, 3D, Demo

**Mission control.** With `--full`, a strip on the right of the window shows,
live, every input to the vehicle's decision:

- the DQN's Q-value for each action,
- the LSTM's predicted behaviour,
- what the onboard camera sees around the vehicle (class x ahead/left/right/behind),
- the fusion network's final action, flagged `OVERRIDE` when it overrules the DQN,
- running counts of overrides and collisions.

**Ablation keys** switch models off and on while the mission runs:

| Key | Model |
|---|---|
| `1` | onboard camera + YOLO |
| `2` | LSTM |
| `3` | fusion MLP (off = the DQN drives unchecked) |
| `4` | denoising autoencoder |
| `5` | DQN (off = the hand-written waypoint follower) |
| `L` | the command-center map lag |
| `V` | top-down / isometric 3D view |
| `F` | drive view: follows the vehicle, map turns with it (city-sized maps open in it) |
| `6` | emergency brake for cars and pedestrians (maps with traffic) |
| `[` / `]` | halve / double the simulation speed (0.1x to 2x) |
| `S` | satellite imagery / drawn map (imported maps) |
| `P` | street-photo column (maps with photos fetched) |

**Real cities from OpenStreetMap.** Any place can become a disaster city:

```bash
python scripts/import_osm_map.py --place "Kattankulathur, Chengalpattu" --name ktr
python scripts/run_simulation.py --map configs/maps/osm_ktr.yaml --full
```

The importer does five things:

1. Fetches the streets and buildings around the place from OpenStreetMap
   (cached under `data/maps/osm_cache`).
2. Rasterises them onto the shipped 30x20 grid at 15 m per tile, about
   450 m x 300 m of real city, so every trained model and camera works on it
   unchanged.
3. Walls off courtyards no road reaches.
4. Stages a seeded disaster: collapsed buildings, victims, fires, and the
   hospital on OSM's own hospital when the area has one.
5. Writes `configs/maps/osm_<name>.yaml`, with a check that every victim is
   reachable.

A mission then loads the file offline. `configs/maps/osm_tnagar.yaml`
(T. Nagar, Chennai) ships as an example.

On that map, no model had seen the layout before:
- the DQN completed its mission (3 rescued, 1 lost to the victim clock, 0 collisions);
- the CCTV + YOLO pipeline built a grid that gave the same result as ground truth.

Map data © OpenStreetMap contributors (ODbL).

**Satellite view.** For an imported map, fetch the real aerial imagery once:

```bash
python scripts/fetch_satellite.py --map configs/maps/osm_cit.yaml
```

The window then opens on the satellite picture of the area, with fire, victims,
rubble, the vehicle and its route drawn on top. Press `S` to switch back to the
drawn map. The simulated cameras and every model still see the rendered city
they were trained on: the imagery is for the people watching. The college map
(`configs/maps/osm_cit.yaml`, around Chennai Institute of Technology) uses real
OpenStreetMap streets. Most of its buildings are inferred, because OSM has few
building outlines there; the file header says so. Imagery © Esri, Maxar,
Earthstar Geographics.

**Street photos (Mapillary).** Real dashcam photos along the streets, shown live in
a column beside mission control: the photo nearest the vehicle, facing its way, above
what the AI actually sees. Needs a free Mapillary client token in `MAPILLARY_TOKEN`,
used only to fetch; the photos are then cached for offline use:

```bash
python scripts/fetch_street_photos.py --map configs/maps/osm_annanagar.yaml
python scripts/run_simulation.py --full --map configs/maps/osm_annanagar.yaml --hazard-seed 1
```

Coverage decides where this works. Scanning Chennai, Anna Nagar had the best:
photos within 40 m of 45% of its road tiles (607 photos), against 17% around the
college and none in Kundrathur. Elsewhere the panel says how far the nearest photo
is, or that there is none. Photos © Mapillary contributors, CC BY-SA 4.0. `P`
toggles the column.

On Anna Nagar the battery budget is tight: with the default hazard seed, fusion's
extra waiting ran the battery out after 2 rescues where the DQN alone finished. With
`--hazard-seed 1` both complete with 4 rescued, fusion with 0 collisions.

**3D view.** Press `V` for an isometric view of the same mission:
- buildings extruded as blocks,
- collapsed buildings as stumps,
- fire burning on the rooftops,
- the vehicle as a small truck.

It is pure pygame with no 3D-engine dependency, so it runs anywhere the 2D view does.

**Smooth, slow driving.** In the top-down view the vehicle glides from tile to tile
and swings through its turns, like a car in a game, instead of hopping
(`rendering/motion.py`). Only the drawn position is smoothed; the simulation and
every model still move tile by tile.

`--speed 0.5` plays the mission in slow motion, and `[` / `]` change the speed live
(0.1x to 2x). The window and the demo start at 0.3x, three moves a second. Speed changes only how fast ticks
happen, never what a tick does, so results are identical at any speed.

**A whole district: Chicago and the drive view.** The importer is not limited to
the 30x20 grid. `configs/maps/osm_chicago.yaml` is 4 km x 4 km of Chicago's West
Side (Humboldt Park, Ukrainian Village, West Town), 200x200 tiles of 20 m (40,000
tiles, 67 times the shipped city), with 6 victims, 4 fires and 12 collapses:

```bash
python scripts/import_osm_map.py --lat 41.8960 --lon -87.6850 --name chicago \
    --tile-metres 20 --width 200 --height 200 --victims 6 --fires 4 --collapses 12 \
    --min-distance 15 --max-distance 70 --min-street-share 0.12 --seed 1
python scripts/fetch_satellite.py --map configs/maps/osm_chicago.yaml
python scripts/run_simulation.py --config configs/app_city.yaml --full --speed 0.5
```

Chicago rather than Barcelona, which was tried first: Barcelona's Eixample grid
runs at 45 degrees to north, and on a north-aligned tile grid its streets become
staircases the vehicle turns on at every tile. Chicago's grid runs exactly
north-south, so the vehicle drives long straight streets, and the grid continues
for kilometres in every direction.

`configs/app_city.yaml` pairs the map with a battery sized for kilometres
(`vehicle_city.yaml`) and victims that hold out longer (`simulation_city.yaml`).
Nothing was retrained: the DQN drives on 7 egocentric numbers, so the city's size
never reaches it.

On a map this size the window opens in the **drive view** (`rendering/drive_view.py`,
key `F`), laid out like a car's autopilot display:
- the camera follows the vehicle and the map turns with it, so the road ahead runs
  up the screen;
- the A* route is a blue ribbon to the current goal;
- corner brackets mark what the onboard model detects this tick, with its confidence;
- a banner shows the mission phase, the goal and the distance left;
- a minimap in the corner shows the whole city.

**Traffic, pedestrians and the emergency brake.** City configs add road users
(`traffic:` in the simulation config, `simulation/traffic.py`). Chicago has 350 cars
and 450 pedestrians. **Chennai has Indian traffic** (`simulation_chennai.yaml`):
150 cars, 150 autorickshaws, 250 two-wheelers, 450 pedestrians and 60 cows. That is
the unstructured road the [India Driving Dataset](https://idd.insaan.iiit.ac.in/)
(IIIT Hyderabad) documents, where every kind of road user shares one street.
- Vehicles drive the roads, going straight and turning at junctions, each kind at
  its own pace.
- Pedestrians walk the pavements and sometimes step out to cross; cows wander the
  road and the verge.
- Every road user gives way to the rescue vehicle's own tile. One boxed in beside
  it steps onto the pavement or leaves the street, so traffic never deadlocks.
- Road users are never on the command center's map.

Three layers keep the ambulance from hurting anyone:
- **A driver trained among traffic** (reinforcement learning, Unit V). The DQN was
  retrained with two more inputs, a road user on the tile ahead and two tiles ahead,
  and a reward that makes hitting any car or person cost as much as failing the
  whole mission (`road_user_hit: -10`). It learned to slow down, wait and give way
  from the reward alone (`configs/training/dqn_traffic.yaml`, `models/dqn/traffic`).
- **Surround sensors.** Four cameras, front, left, right and rear, in the column
  beside the drive view (`rendering/surround_cameras.py`). Each one turns with the
  vehicle and counts the vehicles and people in its quarter. Road users within 5
  tiles are outlined, and the nearest three are tagged with their distance.
- **The emergency brake** (`decision/emergency_brake.py`, key `6`). A rule-based
  safety layer around whatever controller drives, the same split a real car makes
  between its learned planner and its AEB. If the next move would enter a road
  user's tile, it stops instead. A red "EMERGENCY BRAKE" banner shows who for. The
  AI panel counts brakes, and the hits made while the brake was off.

Driving into a road user is a collision like driving into debris
(`TrafficAwarePhysics`). **What the reinforcement learning did**, on 40 held-out
missions on the standard map with 10 cars and 14 pedestrians, brake off:

| Driver | Rescued | Completed | Road users hit |
|---|---|---|---|
| Rule-based waypoint follower | 128 | 31/40 | 423 |
| DQN without traffic training | 146 | 39/40 | 361 |
| **DQN trained among traffic** | **147** | **40/40** | **4** |

Hits fell by 99% with no rescues lost. Whole city missions, full stack:

| City, hazard seed | Driver | Rescued | Brake stops | Road users hit |
|---|---|---|---|---|
| Chennai, 2 | old DQN + brake | 5 of 6 | 74 | 0 |
| Chennai, 2 | **traffic DQN + brake** | 5 of 6 | **10** | 0 |
| Chennai, 2 | traffic DQN, brake off | 5 of 6 | 0 | 4 |
| Chennai, 3 | old DQN + brake | 4 of 6 | 56 | 0 |
| Chennai, 3 | **traffic DQN + brake** | 4 of 6 | **1** | 0 |
| Chicago, 2 | old DQN + brake | 4 of 6 | 92 | 0 |
| Chicago, 2 | **traffic DQN, brake off** | 4 of 6 | 0 | **0** |

With the trained driver the brake is a backup, not the thing doing the work. The
city demos run the traffic DQN with its own fusion model
(`--fusion-config configs/training/fusion_veto_traffic.yaml`), retrained on that
DQN's Q-values.

**Chennai at district scale.** `configs/app_chennai.yaml`: 4 km x 4 km around Anna
Nagar (`osm_chennai.yaml`), satellite imagery, and real Mapillary street photos for
24% of the mission area's road tiles (6,847 photos). Chicago has 61% (21,507 photos).

Chicago measured headless with the full stack before traffic was added, 5 hazard seeds:
every mission completed, 4-5 of 6 rescued, 0 collisions. Three changes made the
size practical:
- the occupancy grid's terrain layer is cached until the city changes
  (`TerrainLayer.version`), and built with numpy;
- A* reads a precomputed traversability mask instead of asking tile by tile, which
  cut one plan from 0.4 s to a few ms, with identical routes;
- the drive view rotates with `rotate`, not `rotozoom`, and culls far road users.

A tick went from 0.29 s to about 0.05 s. A frame of the whole window, with
traffic and the four cameras, draws in about 25 ms on the 4060 laptop.

**The review demo, one command:**

```bash
python scripts/demo.py --device cuda     # every scene in order; Escape moves on
python scripts/demo.py --list            # the running order
python scripts/demo.py --scene 5         # start from a scene
```

Each scene prints what to point out while it runs. A scene whose model is
not trained yet is skipped with a message rather than crashing.

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
  decision/    The DQN local controller (Phase 6), the fusion MLP and the fused
               controller that runs every model together (Phases 7-8)
  mapping/     OpenStreetMap import: real streets -> a disaster-city map
  training/    Offline only: dataset builder, YOLO + autoencoder + LSTM training,
               trajectory recording, seeding, metric logs, checkpoints
  rendering/   Pygame map renderer, isometric 3D view, vehicle glide, HUD, mission-control
               strip, keyboard input, mission window
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
  [Phase 6 — reinforcement](docs/architecture/phase6-reinforcement.md),
  [Phase 7 — fusion](docs/architecture/phase7-fusion.md))
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
  ([0001 config](docs/adr/0001-config-driven-yaml-dataclasses.md),
  [0002 two-tier navigation](docs/adr/0002-two-tier-navigation-and-command-center.md),
  [0003 scene evidence and map lag](docs/adr/0003-egocentric-scene-evidence-and-map-lag.md))
