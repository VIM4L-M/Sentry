# Phase 3.1 — YOLOv8n detection: design notes

**Status:** Pipeline complete and verified end to end; the production training
run is pending. Records decisions made *during* implementation that PROJECT.md
does not specify.

Phase 3 is split into five steps ([PROJECT.md §10](../../PROJECT.md)). This
document covers 3.1 — getting a detector that turns an image into boxes, and
nothing else. Merging those boxes across cameras (3.2) and building an occupancy
grid from them (3.3) are separate steps precisely so the detector never learns
anything about maps.

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

### Fire is over-marked, and that is the safe direction

The one disagreement a perfect detector cannot remove. Ground truth stamps a
fire as a Euclidean **disc** (`mark_radius`); a camera reports an axis-aligned
**box**, and a disc's bounding box includes its corners. With ground-truth
annotations fed in as a perfect detector, fire recall is 1.000 and precision
0.529 — every burning tile believed burning, plus a ring that is not.

This closes the question 3.1 left open. Fire was flagged there as the only
class with a fuzzy painted boundary, "worth revisiting if 3.3 shows the grid
over-marking fire". It does, and the cause is not the weights: it is
information loss inherent to bounding boxes, present at 0.529 precision
before the model is involved at all.

**The builder deliberately does not re-inscribe a disc.** It could estimate a
centre and radius and carve the corners back off, and that would raise the
number. It would also mean *removing* tiles from a hazard footprint on a
guess — converting a harmless detour into a vehicle driving into a fire the
moment the radius estimate is wrong. Over-marking a hazard is free; under-
marking one is not. The asymmetry decides it, the same way it decided the
merge rule in 3.2.

### Measured against the real detector

`scripts/evaluate_grid.py` runs the whole chain on the real city. `--perfect`
substitutes ground-truth annotations, which isolates the projection and merge
steps from the weights entirely.

| | perfect detector | `labelfix` weights, degraded frames |
|---|---|---|
| cell agreement | 0.9733 | 0.9433 |
| traversability agreement | 0.9783 | 0.9483 |
| **missed hazards** | **0** | **0** |
| **missed victims** | **0** | **0** |
| phantom obstacles | 13 | 31 |
| victim IoU | 1.000 | 1.000 |
| debris IoU | 1.000 | 1.000 |
| fire precision | 0.529 | 0.346 |
| victims reachable by A\* | 4/4 | 4/4 |

The real detector produces a **mission-equivalent** grid. Every difference
from the perfect run lands in the fire class — real boxes are looser, so the
bloat roughly doubles. Victims and debris are pixel-exact in both, which
answers the question 3.2 left open: a real box drifting a pixel or two still
projects to the same tile, because those markers are painted well inside one.

### Across an evolving disaster

Scored every 25 ticks of a real 184-tick mission (hazard seed 7, fresh
degradation noise each sample, terrain surveyed only at tick 0):

| tick | cells | traversability | missed hazards | phantom | missed victims | reachable |
|---|---|---|---|---|---|---|
| 0 | 0.9433 | 0.9483 | 0 | 31 | 0 | 4/4 |
| 25 | 0.9533 | 0.9583 | 0 | 25 | 0 | 3/3 |
| 50 | 0.9150 | 0.9350 | 0 | 39 | 0 | 2/2 |
| 75 | 0.9033 | 0.9300 | 0 | 42 | 0 | 2/2 |
| 100 | 0.9150 | 0.9383 | 0 | 37 | 0 | 2/2 |
| 125 | 0.8967 | 0.9133 | 0 | 52 | 0 | 1/1 |
| 150 | 0.8800 | 0.8983 | 0 | 61 | 0 | 0/0 |
| 175 | 0.9100 | 0.9250 | 0 | 45 | 0 | 0/0 |

Two buildings collapsed during the run, at (23, 16) and (23, 5). Both were
detected despite the survey being stale — the split works.

Never a missed hazard, never a missed victim, every remaining trapped victim
reachable at every sample. Agreement decays from 0.948 to 0.898 as fires
spread, and every point of that decay is fire bloat: more fire means more
bounding-box corners.

**What this does not yet show.** The mission above drove on the *ground-truth*
grid; the belief grid was scored alongside it, not steering. Swapping the
producer inside `MissionController` is 3.4/3.5, and until that runs there is
no claim here that a mission *succeeds* on detector-derived perception — only
that the map it would have been given was good enough to have succeeded.

## Still open

- **A recall floor worth defending.** `evaluate_yolo.py` uses 60% victim
  recall as a provisional bar. It was set before there were any numbers to
  argue with, and the baseline's 54.5% turned out to measure a labelling bug
  rather than the detector, so the floor still has not been tested against a
  genuine limit.
- **Mission success is not the same metric as recall.** Recall counts
  victims per *frame*; a mission only needs each victim found in *one* frame
  of the many a camera takes while it drives there. 3.3 supplies the missing
  half — "every victim on the map, every victim reachable" is measurable now,
  and holds at every sample of an evolving mission. The number that is still
  missing is the one where the belief grid actually *steers*, which is 3.4.
- **Phases 3.4-3.5.** Proving a full mission runs on a detector-derived grid
  without `navigation/` or `simulation/` changing. `MissionController.
  refresh_grid` currently calls `OccupancyGrid.from_city_map` directly, so
  the swap needs a grid-source port injected there — the one seam Phase 3
  legitimately has to touch, and the reason ADR 0002 drew it where it did.
- **Fire's bounding-box bloat is bounded but unmeasured at scale.** Phantom
  obstacles grew from 31 to 61 as fires spread across one mission and never
  blocked a route. A larger map, or a fire model that spreads further, could
  seal a corridor. The failure would be loud (`unreachable`), not silent, but
  no test currently searches for the point where it happens.
- **The static survey has no expiry.** Terrain is read once at mission start.
  That is correct for this simulation, where the only terrain change is a
  collapse the cameras also see. A hazard that altered terrain *outside*
  camera coverage would go unnoticed forever, and nothing currently asserts
  that camera coverage stays at 1.0 for the life of a mission.
