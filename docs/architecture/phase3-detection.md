# Phase 3 — perception: design notes

**Status:** complete, 3.1 through 3.5. A full mission runs on an occupancy grid
built from YOLO detections, with `navigation/` and the mission state machine
unchanged. Records decisions made *during* implementation that PROJECT.md does
not specify.

Phase 3 is split into five steps ([PROJECT.md §10](../../PROJECT.md)), and the
split is the design: the detector never learns anything about maps.

| step | what it adds | testable without |
|---|---|---|
| 3.1 | image → boxes (`YoloDetector`) | maps, cameras |
| 3.2 | boxes → one world belief (`DetectionMerger`) | a model |
| 3.3 | belief → occupancy grid (`OccupancyGridBuilder`, `GridComparison`) | a model, a mission |
| 3.4 | the grid producer becomes injectable (`IOccupancyGridSource`) | a model |
| 3.5 | a whole mission runs on it | — |

Sections below follow that order. 3.1 is the longest because it is where the
labelling defect was found.

## What runs today

```
python scripts/fetch_pretrained.py                 # once
python scripts/build_dataset.py                    # ~2 min, 1965 frames
python scripts/train_yolo.py
python scripts/evaluate_yolo.py
```

## Decisions

### The split is by mission, not by frame

This is the decision most likely to be got wrong, and the one that would be
hardest to notice.

Consecutive frames from the same mission are nearly identical: the same city,
the same fires, the same camera, one second apart. Splitting those at random
puts near-duplicates on both sides of the train/validation boundary. The model
then scores well by recognising images it has effectively already seen, and the
resulting mAP measures memorisation.

So whole missions go to one split or the other. A validation mission is a
disaster the model has seen no frame of. `YoloDatasetBuilder._plan` assigns
consecutive hazard seeds to splits and there is a test asserting no seed appears
in both.

The same reasoning sets where variety comes from: re-seeding the hazards, not
sampling more densely. A hundred frames of one fire teach less than ten frames
of ten different fires.

### The class balance is reported, because it is bad

A build prints its instance counts. The shipped configuration produces roughly:

| class | instances |
|---|---|
| obstacle | 8655 |
| fire | 2685 |
| victim | 744 |

Twelve obstacles per victim. A model that never detected a victim at all would
still post a respectable overall mAP, which is why `evaluate_yolo.py` reports
per class and calls out victim recall specifically. Victims are the rarest
class, the smallest on screen, and the only one the mission actually depends on.

### One tile yields one answer, and a victim outranks the debris pinning them

This one was found by measurement, after three plausible theories turned
out to be wrong. It is the most useful thing in this document.

The first trained model scored 54.5% victim recall against 99.8% for
obstacles, with victim *precision* of 1.000. The obvious reading was size:
a victim marker is 8 px, which is about one cell on YOLOv8's finest
(stride-8) detection head, while an obstacle is 13 px and about 1.67 cells.

That reading was wrong. Breaking the misses down by camera:

| camera | victims nested inside an obstacle box | victim recall |
|---|---|---|
| cctv_ne | 0.000 | 1.000 |
| cctv_sw | 0.000 | 1.000 |
| cctv_nw | 1.000 | 0.000 |
| cctv_se | 1.000 | 0.000 |
| onboard | 0.632 | 0.368 |
| **all** | **0.454** | **0.546** |

Recall was exactly one minus the share of victims pinned under debris, to
three decimals. Missed victims and found victims were both 8.00 px. Two
cameras scored a perfect 1.000 on the very 8 px targets the size theory
said were unresolvable.

The cause: a victim trapped in rubble was annotated twice — once as an
8 px victim, once as the 13 px obstacle box containing it. At twelve
obstacles to every victim, the detector learned to answer `OBSTACLE` every
time, and did so at 0.87-0.90 confidence.

That answer is worse than a miss. `OBSTACLE` maps to `OccupancyCode.DEBRIS`,
which is in `_IMPASSABLE_CODES`. A trapped victim did not merely go
undetected — they became a wall the planner routed around.

The fix makes the label obey a precedence the domain had already written
down. From `domain/occupancy.py`:

> Layering order matters — later writes win: terrain, then fires, then
> victims, then the hospital, then the vehicle. A victim standing in debris
> must read as `VICTIM` so the planner can route *to* it.

`OccupancyGrid` builds a grid from the true world with that rule. The
Phase 3 labels contradicted it. Now `_paint_obstacles` still *paints* debris
under a victim — the rubble is really there, and the victim's smaller marker
over it leaves a brown ring around a pink core, which is what "trapped in
rubble" looks like from above — but does not *annotate* it.

Two lessons worth keeping:

* **Per-class metrics were not enough.** "Victim recall 54.5%" pointed at
  the model. Only slicing by camera showed a bimodal 1.000/0.000 split,
  which no amount of training would have produced.
* **Two hypotheses were tested and killed before the real one was found.**
  Disabling `mosaic` and `scale` moved recall from 0.5448 to 0.5460 — a
  clean negative that ruled out augmentation. Measuring marker contrast
  against the clean frames showed victims were the *highest*-contrast class
  after degradation (84.5 versus 23.3 for obstacles), ruling out smoke.

### Marker sizes are configuration

`sensors.markers` sets how much of a tile each entity fills. These decide
the pixel size of every training target, so they were the first suspects
for low victim recall and had to be adjustable without a source edit.
`configs/sensors_v75.yaml` and `scripts/build_dataset.py --sensors` exist so
a camera experiment is one command.

They did not turn out to be the answer here, but the measurement they were
built to support is what ruled size out.

### Hue and saturation augmentation are switched off

Standard YOLO recipes jitter hue and saturation freely. Here they must not:
class identity is largely *carried* by colour. Fire is orange, victims are pink,
debris is brown. Shifting hue does not produce a harder example of the same
class — it produces a mislabelled one.

Flips are the opposite case and are turned up. The city is viewed from directly
overhead, so a mirrored frame is a perfectly plausible city, the label geometry
stays valid, and it doubles the effective data for free. Vertical flips too,
which would be wrong for almost any ground-level dataset.

### Boxes are covering, and tiny boxes survive

`_clamped_box` floors the minimum and ceils the maximum, so the integer box
contains the float one and rounding never shaves a pixel off a detection. A box
that rounds to sub-pixel is widened to one pixel rather than dropped.

That asymmetry is deliberate. A victim marker is 8 px in a 256x160 frame, and
smaller again after augmentation scaling; discarding a marginal victim costs
more than keeping a marginal false positive, because the command center can
investigate a false victim but cannot rescue one it never heard about.

A box falling entirely outside the frame *is* dropped — that is garbage, not a
small detection.

### The pretrained checkpoint is fetched explicitly

Ultralytics will happily download `yolov8n.pt` the first time it is asked for.
`scripts/fetch_pretrained.py` does it beforehand instead, and the config points
at the local file. A training run should not have a hidden network dependency
that can kill it twenty minutes in — which is exactly what happened during
development.

### Ultralytics stays behind the port

`YoloDetector` implements the Phase 1 `IVisionDetector` and is the only module
that imports `ultralytics`. Everything downstream — the mission loop, the
occupancy grid, every test that fakes a detector — stays free of a dependency
that pulls in Torch.

The adapter's whole job is translation: Ultralytics speaks in tensors, `xyxy`
floats and class indices; this project speaks in `Detection`, `EntityKind` and
integer pixel boxes.

### One indexing trap, tested

Ultralytics reports `p`, `r` and `ap50` as arrays ordered by `ap_class_index` —
only the classes that actually appeared in the split — while `maps` is indexed
by class id across all classes. Reading both the same way silently attributes
one class's numbers to another the moment a class is absent from validation,
which on this dataset is entirely possible for victims. `_report_from` handles
the two differently and `test_yolo_evaluation.py` pins it down.

## 3.2 — merging across cameras

`DetectionMerger` takes what each camera reported and answers one question:
*are these two boxes the same object seen twice, or two objects?* It never
touches pixels, never loads a model, and never builds a grid, so it can be
tested with hand-built detections and no Torch in sight.

### Footprints, not centres

A detection projects to the set of world tiles its box covers
(`CameraView.tiles_of_box`), not to a single centre tile.

Centres would be enough for victims and debris — both are painted inside one
tile, so two cameras seeing one victim agree on that tile. They are wrong for
fire. A fire is annotated across its whole visible disc, and a blaze
straddling a camera seam is *clipped differently by each camera*: the two
views' box centres land on different tiles, while their footprints meet at
the seam. Centre-matching would report one fire as two.

The footprint is also what 3.3 needs. A grid builder must mark every burning
cell, not the one under the centre of the box.

### The merge rule is asymmetric between people and hazards

Two sightings merge when they share a label and their footprints meet. What
"meet" means depends on the class, and the asymmetry is deliberate:

| class | merges on | why |
|---|---|---|
| victim, obstacle | overlap only | two victims on neighbouring tiles are two people; merging them erases one |
| fire | overlap **or** edge adjacency | one blaze clipped by a camera seam becomes two disjoint halves that touch |

Over-merging fire costs nothing — adjacent burning tiles are one impassable
region to the planner either way. Over-merging victims loses a person, which
is the worst error this system can make. So hazards merge greedily and people
do not.

This is not configurable. It follows from what the classes mean, not from a
threshold worth tuning.

Grouping is transitive: if A meets B and B meets C, all three are one object
even when A and C do not touch. That is what a fire spanning three camera
footprints looks like.

### Confidence is the best view, not the average

A merged detection carries the highest confidence of its contributing
sightings, and the set of `camera_ids` that saw it. A camera with a clear
line of sight should not be dragged down by one looking through smoke, and
`corroborated` lets downstream code tell a two-camera confirmation from a
lone sighting.

### Tested against the real rig with a perfect detector

`tests/integration/test_detection_merge_pipeline.py` runs the merger over the
actual four-camera network from `configs/sensors.yaml` and the actual city,
feeding the rasterizer's *ground-truth* annotations in place of a model's
predictions.

That substitution isolates the step. If a victim standing in the two-tile
overlap comes back as two victims there, the fault is in the merging, because
a perfect detector is exactly what is being fed in.

## Experiment log

Every change to the detector recorded with the number that justified it, so
the reasoning survives after the weights are regenerated. All runs: 60
epochs, YOLOv8n from COCO, imgsz 256, on an RTX 4060 laptop GPU.

| run | change | victim recall | victim precision | mAP50 |
|---|---|---|---|---|
| `sentry` | baseline | 0.545 | 1.000 | 0.843 |
| `noaug` | `mosaic 0.0`, `scale 0.1` | 0.546 | 0.999 | 0.842 |
| `labelfix` | one annotation per tile, victim outranks debris | **1.000** | 1.000 | **0.991** |

`noaug` is the useful negative: augmentation was not shrinking victims below
detectability, and knowing that is what forced the per-camera breakdown that
found the real cause.

Full per-class figures for `labelfix`:

| class | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|
| victim | 1.000 | 1.000 | 0.995 | 0.854 |
| fire | 0.947 | 0.953 | 0.984 | 0.930 |
| obstacle | 0.999 | 0.998 | 0.995 | 0.928 |

Overall mAP50 0.9914, mAP50-95 0.9040, against 0.8428 / 0.7792 for the
baseline. Validation instances fell 3212 to 3128 — exactly the 84 nested
obstacle labels the fix removed from the split, which is the arithmetic
closing on the diagnosis.

**What 1.000 does and does not mean.** The city map is fixed and the four
victims sit at static tiles; only the hazards vary between missions. The
detector is recognising a distinctive marker at familiar locations, not
solving a hard vision problem. The defensible claim is that the labelling
defect is fixed and the detector is no longer the bottleneck — not that the
detector is strong. A harder test would randomise victim placement, and the
number would drop.

Fire is now the weakest class at 0.947 precision, down from 0.970 in the
baseline. It is the only class with a fuzzy boundary — a gradient disc whose
painted extent scales with intensity — so slightly over-wide boxes are the
expected failure. Not blocking, and worth revisiting if 3.3 shows the grid
over-marking fire.

## 3.3 — building the map

`OccupancyGridBuilder` turns merged world-space detections into the
`OccupancyGrid` A\* plans over, replacing `OccupancyGrid.from_city_map` as
the *producer*. The ground-truth method stays, and `GridComparison` scores
one against the other.

```
SensorRig -> FrameDegrader -> YoloDetector -> DetectionMerger
          -> OccupancyGridBuilder -> GridComparison vs from_city_map
```

### Two sources, one grid

The static street plan is copied from the surveyed map; only `FIRE`,
`DEBRIS` and `VICTIM` come from the cameras. That is the decision PROJECT.md
already recorded for Phase 3 — the simulator generated the layout, so
training a detector to rediscover roads and buildings would cost parameters
and add label noise to learn nothing.

The consequence is deliberate and is the reason the split needs a test of its
own: `OccupancyGridBuilder.terrain` is surveyed **once** and never refreshed.
When a building collapses into a street mid-mission, the static half of the
grid is stale, and the debris has to arrive through the cameras or not at
all. `test_a_collapse_after_the_survey_is_still_found` asserts exactly that,
and the evolving-mission run below exercises it twice for real.

### Precedence is copied from the answer key, not reinvented

Detections are stamped lowest-precedence-first — obstacle, then fire, then
victim — mirroring `from_city_map`'s terrain → fires → victims order. If the
two disagreed about a victim pinned in burning rubble, `GridComparison` would
be measuring the disagreement rather than the detector. The hospital is
restored last, so a detection on the drop-off point cannot erase the only
place victims can be delivered.

This is the same precedence rule as 3.1's labelling fix, now applied a third
time. It is stated once per layer because each layer resolves a contested
tile independently.

### Cell accuracy is the wrong headline

The grid is ~94% road and building. A belief that detected *nothing* would
still score in the nineties. So `GridComparison` reports the three failure
modes separately, because they cost completely different things:

| failure | meaning | cost |
|---|---|---|
| missed hazard | believed clear, actually fire/debris | the vehicle drives into it |
| phantom obstacle | believed blocked, actually clear | a detour; at worst an unreachable victim |
| missed victim | never marked `VICTIM` | nobody is dispatched — the person is not rescued |

Each is reported as the *set of tiles*, not a count, because "which tiles"
is what you need to debug a mission that went wrong.

### Fire: a judgement that was wrong, and how it was caught

This section records a decision, its falsification, and its reversal, because
the reasoning that produced the wrong answer was not obviously bad.

Ground truth stamps a fire as a Euclidean **disc** (`mark_radius`); a camera
reports an axis-aligned **box**, and a disc's bounding box includes its
corners. Fed a *perfect* detector, the first version of 3.3 scored fire at
recall 1.000 and precision **0.529** — every burning tile believed burning,
plus a ring that was not.

**The original decision was to accept it.** The argument: trimming the ring
means removing tiles from a hazard footprint on an estimate, and a wrong
estimate turns a harmless detour into a vehicle driving into a fire. Over-
marking a hazard is free; under-marking one is not.

**The measurement that killed it.** Running a real mission on the belief grid:

```
ground truth : Mission completed  — 4 rescued, 0 lost, 0 unreachable
perception   : Mission failed     — 2 rescued, 1 unreachable, hospital unreachable
```

The belief held **132 fire tiles against 66 in truth — exactly double**.
Patching only the phantom fire tiles back to their true values restored the
route. The inflated footprint was the sole cause.

So "over-marking a hazard is free" was simply false. Fire is impassable, a
doubled footprint is a doubled wall, and enough of them seal a city. The cost
was not a detour; it was two victims and a failed mission.

**Two changes recover the disc exactly.** Both are asymmetric between hazards
and people, for the same reason the merge rule is:

1. **Fire projects by tile centre, not by any pixel overlap.**
   `tiles_of_box` claims a tile on one pixel of contact. That is right for a
   marker painted well inside a tile and wrong for anything spanning several:
   the detector's boxes run about half a tile wide, which claims a whole extra
   ring. `tiles_centred_in_box` requires the tile's centre, absorbing up to
   half a tile of regression error. Victims are excluded deliberately — their
   marker fills a quarter of a tile, so the strict rule could yield *no* tile
   and delete a person. A hazard reported half a tile off falls back to the
   generous rule rather than vanishing.

2. **A square footprint is read back as the disc it bounds**, through the same
   `tiles_within` that drew it — so this inverts the drawing rule rather than
   guessing a shape. It applies only where the reconstruction is exact (an
   odd-sided square, clear of the map border) and only ever by *intersection*,
   so it can remove a corner but never invent a tile. Anything else keeps the
   raw footprint: over-marked, which is the direction that cannot strand the
   vehicle in a fire.

The safety property the original argument was protecting is kept — fire recall
stays 1.000, and no test permits a missed hazard — while the precision that
was actually costing missions is recovered.

### Measured against the real detector

`scripts/evaluate_grid.py` runs the whole chain on the real city. `--perfect`
substitutes ground-truth annotations, isolating projection and merging from
the weights entirely.

| | perfect detector | `labelfix` weights, degraded frames |
|---|---|---|
| cell agreement | **1.0000** | **1.0000** |
| traversability agreement | 1.0000 | 1.0000 |
| missed hazards | 0 | 0 |
| missed victims | 0 | 0 |
| phantom obstacles | 0 | 0 |
| fire IoU | 1.000 | 1.000 |
| victim IoU | 1.000 | 1.000 |
| victims reachable by A\* | 4/4 | 4/4 |

The pipeline is **lossless** on the shipped map: cameras, projection, merging
and precedence together reproduce `from_city_map` cell for cell, from real
YOLO output on degraded frames. `test_a_perfect_detector_reproduces_the_world_exactly`
asserts the equality, so any future change that costs a single cell fails.

Two cautions on reading that. It is one instant of one fixed map, and the
detector is recognising distinctive markers at familiar locations — the same
caveat 3.1 records about victim recall. And "identical to ground truth" is a
statement about *this* city's fire geometry: isolated discs, which the
reconstruction handles exactly. It degrades as soon as they stop being
isolated, which the next table shows.

### Across an evolving disaster

Scored every 25 ticks of a real 184-tick mission (hazard seed 7, real
detector, fresh degradation noise each sample, terrain surveyed only at
tick 0):

| tick | cells | traversability | missed hazards | phantom | missed victims | reachable |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 0 | 0 | 0 | 4/4 |
| 25 | 1.0000 | 1.0000 | 0 | 0 | 0 | 3/3 |
| 50 | 1.0000 | 1.0000 | 0 | 0 | 0 | 2/2 |
| 75 | 1.0000 | 1.0000 | 0 | 0 | 0 | 2/2 |
| 100 | 0.9433 | 0.9583 | 0 | 25 | 0 | 2/2 |
| 125 | 0.9400 | 0.9483 | 0 | 31 | 0 | 1/1 |
| 150 | 0.9333 | 0.9433 | 0 | 34 | 0 | 0/0 |
| 175 | 0.9333 | 0.9433 | 0 | 34 | 0 | 0/0 |

Exact for the first 75 ticks, then decaying — and the decay is *by design*.
As fires spread they overlap and merge into irregular blobs, whose footprints
are no longer odd-sided squares, so the reconstruction correctly declines to
act and the raw over-marking returns. The guard is doing its job: it would
rather waste tiles than guess a centre it cannot recover. Phantom obstacles at
tick 150 fell from 61 before the fix to 34 after.

Two buildings collapsed during the run, at (23, 16) and (23, 5). Both were
detected despite the survey being stale — the terrain/detection split works.

Never a missed hazard, never a missed victim, every remaining trapped victim
reachable at every sample.

## 3.4 / 3.5 — the mission runs on it

The Phase 3 deliverable here is the *absence* of new code. `AStarPlanner`,
`MissionController`, `VehicleController` and `WaypointFollower` are unchanged
Phase 2 logic; swapping the grid's producer is a constructor argument.

### One seam, injected

`MissionController.refresh_grid` called `OccupancyGrid.from_city_map`
directly. It now asks an injected
[`IOccupancyGridSource`](../../src/sentry_ai/interfaces/world.py):

| implementation | reads |
|---|---|
| `GroundTruthGridSource` | the simulated world — Phase 2 behaviour, and the answer key |
| `DetectedGridSource` | the city *through cameras* — rig, degrader, detector, merger, builder |

That is the only line of the mission loop Phase 3 touches, and it is the seam
ADR 0002 drew deliberately. The proof it was drawn in the right place: adding
the port and defaulting it to ground truth left **all 646 existing tests
passing with no test changed at all**.

`IFrameObserver` sits inside `DetectedGridSource` so "run the model"
(`ModelObserver`) and "use the answer key" (`GroundTruthObserver`) are the
same shape. That is what makes a failure attributable — a mission that fails
with a perfect observer has a wiring bug, not a weights problem — and it is
shared with `evaluate_grid.py --perfect` rather than duplicated.

### Results

Real `labelfix` weights on degraded frames, driving the mission, against the
same map and hazard seed on ground truth:

| seed | ground truth | YOLO perception |
|---|---|---|
| 1 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |
| 3 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |
| 5 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |
| 7 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |
| 11 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |
| 13 | completed — 4 rescued, 0 lost | completed — 4 rescued, 0 lost |

This is the **mission-success metric** that recall alone could never give:
recall counts victims per frame, but a mission only needs each victim found in
*one* of the many frames taken while the vehicle drives there.

Run it either way, including in the live window:

```bash
python scripts/run_simulation.py --perception --device cuda
```

Press `G` with `--perception` on and the grid view is a *belief* rather than
the world.

## Still open

- **A recall floor worth defending.** `evaluate_yolo.py` uses 60% victim
  recall as a provisional bar. It was set before there were any numbers to
  argue with, and the baseline's 54.5% turned out to measure a labelling bug
  rather than the detector, so the floor still has not been tested against a
  genuine limit.
- **Merged fires are still over-marked.** The disc reconstruction only fires
  on an isolated blaze, so once two fires overlap into an irregular blob the
  raw bounding footprint returns — 34 phantom obstacles by tick 150 above.
  That is the safe direction and it did not block a route here, but it is the
  same class of defect that failed a mission before the fix, just smaller. A
  larger map or a wider-spreading fire model could still seal a corridor. The
  failure would be loud (`unreachable`) rather than silent, and no test yet
  searches for the point where it happens.
- **Everything is measured on one map.** Six hazard seeds vary the disaster
  but not the street plan, the camera layout, or where the victims start. The
  claim "the pipeline is lossless" is a claim about *this* city — in
  particular about fire geometry being isolated discs, which is exactly the
  case the reconstruction handles exactly.
- **The static survey has no expiry.** Terrain is read once at mission start.
  Correct for this simulation, where the only terrain change is a collapse the
  cameras also see. A hazard altering terrain *outside* camera coverage would
  go unnoticed forever, and nothing asserts coverage stays at 1.0 for the life
  of a mission.
- **Confidence thresholding is untested as a control.** `min_confidence`
  exists on the builder and defaults to 0.0 (trust the detector). Sweeping the
  detector's own threshold from 0.25 to 0.7 changed fire precision not at all
  — the boxes are confident and merely wide — so the knob has never been shown
  to do anything useful.
