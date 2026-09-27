# Phase 4 — representation learning: design notes

**Status:** complete. The denoiser is built, tested, trained, and wired
into the camera pipeline, and milestone M4 — *the denoiser measurably
improves detection under injected noise* — is met, confirmed at full
training length on the target GPU. It holds **only once the detector is
retrained on denoised frames**: dropped in front of the existing detector,
the denoiser made detection worse. Both results are below; the headline
numbers are the GPU run, and the CPU runs that preceded it are kept because
they are where the failure mode was found.

Records decisions made *during* implementation that PROJECT.md does not
specify.

## What it adds

```
SensorRig -> FrameDegrader -> IDenoiser -> IVisionDetector -> DetectionMerger -> ...
                               ^^^^^^^^^ Phase 4
```

| piece | where |
|---|---|
| `DenoisingAutoencoderNet`, `ConvDenoisingAutoencoder` (the `IDenoiser`) | `perception/autoencoder.py` |
| dataset, trainer, PSNR evaluation | `training/denoising.py` |
| shared training infrastructure | `training/seed.py`, `metrics.py`, `checkpoint.py` |
| config | `configs/training/autoencoder.yaml` |
| entry points | `scripts/train_autoencoder.py`, `scripts/evaluate_denoiser.py`, `scripts/build_denoised_dataset.py` |
| pipeline switch | `--denoiser` on `run_simulation.py` and `evaluate_grid.py` |

## What runs today

```bash
python scripts/build_dataset.py                 # if not already built (~20 s)
python scripts/train_autoencoder.py             # ~3M params; use --device cuda
python scripts/evaluate_denoiser.py             # PSNR at severity 1.0 / 1.5 / 2.0
python scripts/build_denoised_dataset.py        # the detector's real input, on disk
python scripts/train_yolo.py --dataset data/denoised --name denoised
python scripts/evaluate_denoiser.py --detector models/yolo/labelfix/weights/best.pt \
    --denoised-detector models/yolo/denoised/weights/best.pt
python scripts/run_simulation.py --perception --denoiser models/autoencoder/sentry/best.pt \
    --weights models/yolo/denoised/weights/best.pt
```

The denoiser and the detector are a **pair**: run `--denoiser` with a
detector trained on denoised frames. With the smoky-trained detector it is
worse than no denoiser at all — see Results.

## Decisions

### Training pairs are made on the fly

The dataset already stores a degraded copy of every frame, so the obvious
training set is `images/train` → `clean/train`. It is the wrong one, for two
reasons.

* **One smoke pattern per frame.** Each stored frame was corrupted once. A
  network trained on it for forty epochs sees the same smoke on the same
  street forty times, and can learn that pattern instead of learning smoke.
* **One strength.** Every stored frame is at exactly `sensors.degradation`.
  The denoiser would learn to undo that amount and nothing else — which is
  the one amount the detector has *already* been trained to cope with.

So `CorruptedCropDataset` reads only the clean frames, takes a random
128 px crop, applies a random quarter-turn and mirror, and then corrupts it
with a `FrameDegrader` at a severity drawn from `[0.5, 2.0]` times the
shipped config. Every epoch sees new smoke, and half the training
distribution is *worse* than anything in the dataset.

`FrameDegrader` gained `degrade_pixels` for this, since a crop is no longer
any camera's frame; `degrade` now calls it, and a test asserts the two agree
for the same seed. `DegradationConfig.scaled(severity)` defines what
"twice as bad" means: smoke opacity and noise multiply (smoke saturates at
fully opaque), blur passes round half up, and the smoke colour never
changes.

Crops also solve a shape problem: CCTV frames are 256x160 and onboard
frames 144x144, and a batch has to be one shape.

### Validation uses the stored pairs

Training draws its own corruption; validation deliberately does not.
`images/val` holds exactly the degraded frames the detector is scored on,
so validating against them measures the denoiser on the pipeline's real
input rather than on its own training distribution. Validation missions are
held out whole, as they are for the detector: no validation scene was ever
cropped for training.

Validation frames are run whole and one at a time, since they come in two
sizes.

### Every PSNR is reported next to its baseline

The trainer reports the PSNR of the degraded frames themselves alongside the
denoiser's, every epoch, and `metrics.csv` carries both columns. "31 dB" is
not a result; "31 dB, up from 18" is. The baseline is computed once, from
the same pairs, in the same units.

### Skip connections are on — and still a switch

A pure autoencoder forces everything through the bottleneck. At depth 3 the
bottleneck is 1/8 resolution, so an 8 px victim marker has to survive being
represented by about one cell, and it comes back as a pink smudge. That is
fatal here: the detector's hardest, rarest class is the one a bottleneck
erases first.

With skip connections each decoder level also receives the matching encoder
level's features (the symmetric-skip design of RED-Net, Mao et al. 2016, a
convolutional denoising autoencoder). The bottleneck still learns the
low-frequency part of the problem — smoke is soft and large — while fine
structure has a path around it.

It stays a config switch (`skip_connections`, or `--no-skip`) so the
pure-bottleneck variant can be measured rather than argued about. The
skip-free network is 7% smaller.

### L1, not MSE

Squared error is dominated by the few pixels that are badly wrong, and the
cheapest way to reduce it is a safe grey average — blur, which is what
erases small markers. L1 penalises every error in proportion and is the
usual choice for restoration for that reason. `loss: mse` exists for the
comparison.

### The decoder upsamples to a size, not by a factor

`functional.interpolate(size=skip.shape)` rather than a fixed ×2 means any
frame size works with no padding, including dimensions that do not divide by
`2 ** depth`. A test runs a 37x51 frame through every configuration.

### The checkpoint carries its architecture

A `.pt` file holds `architecture`, `state_dict`, `format` and metadata, and
the adapter rebuilds the network from the file alone. The detector has a
"`image_size` must match training or every box lands in the wrong place"
trap; the denoiser does not get an equivalent one. `format` is versioned so
an old file fails with a clear message.

The format lives in `perception/`, not `training/`, because the live
pipeline reads it and `training/` is never imported at runtime.

### Shared training infrastructure, built now because it is needed now

PROJECT.md §14 planned `training/seed.py`, `metrics.py` and `checkpoint.py`.
Phase 3 did not need them — Ultralytics owns the detector's loop — so they
were not built. The autoencoder is the first hand-written training loop, so
they arrive with it, generic enough for the LSTM, DQN and fusion trainers:

* `seed_everything` seeds `random`, NumPy and Torch together, and
  `resolve_device` answers `"auto"` in one place.
* `CsvMetricLogger` writes one row per epoch and refuses a row whose
  columns changed mid-run.
* `save_checkpoint` writes `<name>.pt` and `<name>.json` side by side; the
  JSON records a fingerprint of the config dataclass, the metrics, the time
  and the Torch version.

Every crop and every corruption is seeded by `(seed, epoch, index)`, so a run
replays exactly regardless of how many DataLoader workers fetch samples. A
test trains twice and asserts identical validation loss.

## Results — full length, RTX 3050 Laptop GPU (4 GB)

The four commands under "What runs today", unmodified: autoencoder 40
epochs, both detectors 60 epochs. The smoky-trained detector is a fresh
`labelfix` run on the same machine (mAP50 0.992, mAP50-95 0.933, victim
recall 1.000 on its own validation split). Each pipeline is scored with the
detector trained on its own input, on the 495 held-out frames:

| severity | mAP50, smoky pipeline | mAP50, denoised pipeline | victim recall, smoky | victim recall, denoised |
|---|---|---|---|---|
| 0 (clean) | 0.983 | **0.992** | 1.000 | 1.000 |
| 1.0 | 0.992 | **0.993** | 1.000 | 1.000 |
| 1.5 | 0.977 | **0.982** | 1.000 | 1.000 |
| **2.0** | 0.880 | **0.938** | 0.861 | **0.990** |

At twice the shipped corruption the smoky pipeline misses 14% of victims;
the denoised pipeline misses 1%. mAP50-95 at 2.0 goes from 0.630 to 0.768 —
the boxes are not only found but placed much more tightly, which is what
the occupancy grid consumes. The denoised pipeline is at least as good at
every severity; the small mAP dip the CPU runs showed at 1.5 is gone at full
length. **M4 is met.**

## Results — short CPU runs (where the failure mode was found)

All runs on a 4-core CPU, same dataset (16 missions, 1965 frames, split by
mission), 495 held-out validation frames. Severity is a multiple of the
shipped `sensors.degradation`; 1.0 is what the dataset was captured at.

### The denoiser: +7.5 dB at the shipped corruption, +11.2 dB at double

Shipped config (skips on, L1, 2.9M parameters), cut short at 12 epochs of
the configured 40 for CPU time. It had not stopped improving: validation
PSNR rose every epoch but one and the best checkpoint was the last.

| epoch | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| validation PSNR (input 22.42 dB) | 25.86 | 27.36 | 28.64 | 29.51 | **29.87** |

Across severities (`evaluate_denoiser.py`, every frame at each strength):

| severity | input | denoised | gain |
|---|---|---|---|
| 0 (clean) | — | 30.15 dB | n/a |
| 1.0 | 22.41 dB | 29.86 dB | +7.45 |
| 1.5 | 19.23 dB | 28.98 dB | +9.75 |
| 2.0 | 16.86 dB | 28.03 dB | +11.17 |

The output barely degrades as the input gets worse: from severity 1.0 to
2.0 the input loses 5.6 dB and the output loses 1.8. That is the
severity-range training doing its job. On a *clean* frame it scores
30.15 dB rather than infinity — it slightly alters frames that need no
help, since training never went below severity 0.5.

### Detection, first attempt: the denoiser made things worse

The obvious deployment — insert the denoiser in front of the existing
detector — failed, clearly:

| severity | smoky frames | denoised frames | victim recall, smoky | victim recall, denoised |
|---|---|---|---|---|
| 0 | 0.923 | 0.910 | 0.978 | **0.069** |
| 1.0 | 0.990 | 0.895 | 0.996 | **0.347** |
| 1.5 | 0.977 | 0.808 | 0.993 | 0.502 |
| 2.0 | 0.862 | 0.735 | 0.868 | 0.678 |

(mAP50 in the first two columns; the detector is YOLOv8n trained 20 CPU
epochs on the smoky frames, mAP50 0.990 and victim recall 1.000 on its own
validation split — the `labelfix` result reproduced.)

The first row is the diagnosis. Severity 0 hands the detector *perfectly
clean* frames and it still drops to 0.923 — it learned what a victim looks
like **through smoke**, and a clean frame is out of its distribution. The
denoiser's output is cleaner still than it is accurate, and victim recall
collapses. Higher PSNR, worse detection: exactly the trap the "Measuring
M4" section below warns about.

### Detection, with a detector trained on denoised frames: M4 met

`build_denoised_dataset.py` ran every frame of the dataset through the
denoiser, labels and mission split unchanged, and a second detector was
trained on it with the same recipe (20 CPU epochs; mAP50 0.989 on its own
validation). Each pipeline is then scored with the detector trained on its
own input:

| severity | mAP50, smoky pipeline | mAP50, denoised pipeline | victim recall, smoky | victim recall, denoised |
|---|---|---|---|---|
| 0 | 0.923 | **0.989** | 0.978 | **1.000** |
| 1.0 | 0.990 | 0.989 | 0.996 | **1.000** |
| 1.5 | 0.977 | 0.972 | 0.993 | **1.000** |
| 2.0 | 0.862 | **0.890** | 0.868 | **0.950** |

* **At severity 2.0 both numbers improve**: mAP50 +0.028, victim recall
  +0.083 — about one missed victim in twelve recovered. That is M4.
* **Victim recall improves at every severity**, and is perfect up to 1.5.
  Victims are the class the mission depends on.
* **mAP50 at 1.0 and 1.5 is a wash** (−0.001, −0.005). At the corruption
  the smoky detector was trained on, there is nothing for a denoiser to
  win, and the result says so rather than claiming a gain.
* **Severity 0 is the robustness story.** The denoised pipeline handles a
  clear day (0.989); the smoky one does not (0.923). A detector trained on
  denoised frames sees the same thing whatever the weather did.

### A mission runs on it

```bash
python scripts/run_simulation.py --headless --perception \
    --denoiser models/autoencoder/cpu12/best.pt --weights models/yolo/denoised20/weights/best.pt
```

Shipped map and hazard seed, CPU: **completed — 4 rescued, 0 lost, 0
unreachable**, 6 replans, 0 collisions; 1 min 19 s wall clock for the whole
mission on four CPU cores.

Across the six hazard seeds Phase 3 used, each against ground truth on the
same seed:

| seed | ground truth | denoiser + denoised-trained detector |
|---|---|---|
| 1 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |
| 3 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |
| 5 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |
| 7 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |
| 11 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |
| 13 | 4 rescued, 0 lost, 0 unreachable | 4 rescued, 0 lost, 0 unreachable |

Inserting the denoiser costs the mission nothing on any seed.

The denoiser does not make the detector better at the conditions it was
trained for. It makes the *system* hold up when conditions get worse than
that — and, unexpectedly, when they get better.

## Measuring M4

M4 says the denoiser *measurably improves detection*, not PSNR. PSNR can be
raised by smoothing, and smoothing is exactly what erases a victim, so a
PSNR gain on its own proves nothing about the mission.

`evaluate_denoiser.py --detector <weights>` answers the actual question. For
each severity it corrupts every clean validation frame, scores the detector
on the corrupted frames and on the denoised ones, and prints mAP50 and victim
recall side by side. Each frame keeps the same smoke pattern at every
severity, so the rows differ by strength alone.

`--denoised-detector` scores the denoised arm with a different detector, so
each pipeline gets the detector trained on its own input — the comparison
the Results above rest on. Without it, both arms use one detector, which is
the first, failed experiment.

Severities above 1.0 are where M4 is decided: the smoky detector has never
seen smoke that heavy, and the denoiser has.

## Still open

* **One seed per run.** The GPU run confirmed the CPU result with a larger
  margin, which is two independent trainings agreeing; a formal multi-seed
  study has not been done.
* **The smoky detector was trained at severity 1.0 only.** A fairer
  baseline would train it with severity augmentation too; if that closes
  the 2.0 gap on its own, the denoiser's case rests on the victim-recall
  and clear-day results rather than on mAP.
* **The skip-connection and loss ablations have not been run.** The
  switches exist (`--no-skip`, `loss: mse`); the numbers do not.
* **Missions ran at the shipped corruption level only.** The six-seed
  check above is at severity 1.0; nothing yet drives a whole mission with
  the degrader turned up to 2.0, which is where the denoiser earns its
  place.
* **Latency.** The grid is rebuilt every tick, so the denoiser runs on four
  frames per tick next to YOLO. Fine on a GPU; noticeably slower on CPU.
  Not profiled yet — that belongs to Phase 8's hardware pass.
* **The onboard camera is in the training set but not the pipeline.** Its
  frames add variety to training; the grid source only denoises CCTV frames,
  because only CCTV frames reach the detector.
