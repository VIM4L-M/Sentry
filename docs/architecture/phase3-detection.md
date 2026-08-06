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
| obstacle | 8973 |
| fire | 2685 |
| victim | 744 |

Twelve obstacles per victim. A model that never detected a victim at all would
still post a respectable overall mAP, which is why `evaluate_yolo.py` reports
per class and calls out victim recall specifically. Victims are the rarest
class, the smallest on screen, and the only one the mission actually depends on.

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

## Still open

- **The production training run.** The pipeline is verified — 3 epochs train and
  validate in about 7 seconds on CPU — but the real 60-epoch run has not been
  done yet.
- **A recall floor worth defending.** `evaluate_yolo.py` uses 60% victim recall
  as a provisional bar. It is a guess until there are real numbers to argue with.
- **Phases 3.2-3.5.** Merging detections across the overlapping CCTV footprints,
  building the occupancy grid from them, and proving a full mission runs on a
  detector-derived grid without `navigation/` or `simulation/` changing.
