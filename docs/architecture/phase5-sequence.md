# Phase 5 — sequence modelling: design notes

**Status:** core complete. An LSTM implements `IMotionPredictor`, trained on
recorded missions, and milestone M5 — *beats a naive baseline on held-out
trajectories* — is met, including on the held-out windows that repeat
nothing in training. Nothing consumes the prediction yet; that is Phase 7's
fusion network. Trained on CPU in about a minute; no GPU run is needed.

Records decisions made *during* implementation that PROJECT.md does not
specify.

## What it adds

| piece | where |
|---|---|
| what ADVANCE / RETREAT / HOLD / DIVERT mean | `sequence/behaviour.py` |
| `BehaviourLstmNet`, `LstmMotionPredictor` (the `IMotionPredictor`) | `sequence/lstm_predictor.py` |
| trajectory recording and storage | `training/trajectories.py` |
| windows, baselines, classification report, trainer | `training/motion.py` |
| config | `configs/training/lstm.yaml` |
| entry points | `scripts/record_trajectories.py`, `train_lstm.py`, `evaluate_lstm.py` |

## What runs today

```bash
python scripts/record_trajectories.py   # ~40 s: 150 train + 50 held-out missions
python scripts/train_lstm.py            # ~1-2 min on CPU; prints the baselines too
python scripts/evaluate_lstm.py         # all held-out windows, then novel ones only
```

## Decisions

### The four classes needed a definition first

`BehaviourClass` has existed since Phase 1 with four names and no meaning.
`sequence/behaviour.py` is the meaning, and it is kinematic and
egocentric — judged against where the vehicle was facing, not against a
compass or a goal:

| class | net displacement over the horizon |
|---|---|
| HOLD | none |
| ADVANCE | within 30 degrees of straight ahead |
| RETREAT | within 30 degrees of straight behind |
| DIVERT | anything else — a corner, a swerve |

Egocentric for the same reason the action space is: "about to turn left"
means the same heading north as heading west. Goal-relative classes
("advancing on the victim") were considered and rejected, because
`VehicleState` does not carry a goal, and adding one is an interface
change that would need an ADR.

**One rule labels the training data and runs the persistence baseline.**
If they were written separately, the model and the baseline could quietly
be answering different questions.

### Horizon 4, window 8 — chosen by measurement

A probe over 20 missions measured the class balance at several horizons:

| horizon | advance / divert / retreat / hold | persistence acc. | "always advance" acc. |
|---|---|---|---|
| 2 | 81 / 13 / 0 / 5 % | 0.683 | 0.813 |
| **4** | **71 / 24 / 5 / 0 %** | **0.521** | **0.708** |
| 6 | 53 / 41 / 6 / 1 % | 0.312 | 0.527 |

At 2 ticks there is almost nothing to predict. At 4 a corner fits inside
the horizon (turning costs a tick, then the vehicle moves) and all the
interesting classes occur. Longer horizons make the answer depend on
decisions the planner has not taken yet.

The probe also showed that **persistence is weaker than "always advance"**
here: a vehicle mid-turn looks like it is diverting, so "keep doing that"
guesses a turn on the straight that follows. M5 therefore has to beat the
stronger of the two, not the one named "naive".

### Macro-F1 is the headline, not accuracy

Seven windows in ten are ADVANCE, so "always advance" scores 0.70 accuracy
while never once predicting a turn. Macro-F1 — per-class F1, averaged over
the classes that occur — gives that predictor 0.20. Accuracy is still
reported, because the model must not buy macro-F1 by getting worse at the
common case.

### The first result was memorisation, and the evaluation was rebuilt around it

The first training run scored **macro-F1 0.964** on held-out missions.
That was too good, so before reporting it, the held-out windows were
compared against training: **99.4% were exact copies** of a training
window, position for position and heading for heading. Every mission
started on the same tile and drove to the same four victims; only the
hazards differed, and they rarely change the route. The score measured how
well the model had memorised the routes.

Two changes followed.

* **Randomised starts.** `record_trajectories.py` now starts each mission on
  a road tile drawn from its seed (`start_positions`; the map's own
  validation still applies). `--fixed-start` reproduces the original set.
* **A second score on novel windows only.** Randomising starts cut the copy
  rate only to 97.6%, and that residue is structural: the city is 30x20
  with a handful of streets, and every mission drives some of them.
  `evaluate_lstm.py` therefore also scores just the held-out windows whose
  exact sequence never occurred in training — the honest test of
  generalisation — and prints how many there are.

Knowing the map is a legitimate part of predicting behaviour *in this
city*, which is the only city the system runs in. The two scores separate
"good at this city" from "learned how vehicles behave".

### Square-root class weighting — changed after a validation result

Unweighted, the cheapest low loss is "always advance". Plain inverse
frequency was tried first. It failed on the randomised data, where HOLD
appeared: 26 training windows in 22,000, so each weighed as much as ~500
ADVANCE windows (212 against 0.4). The model called "hold" whenever unsure —
86% HOLD recall at **6% precision**, macro-F1 0.69.

Square-root inverse frequency keeps the ordering and caps the distortion
(HOLD ~15). Macro-F1 went to 0.80 and accuracy from 0.90 to 0.94. This
choice was made *after* seeing validation numbers, which is recorded here
because it is a mild form of tuning on the validation set; it was a fix
for an obvious pathology rather than a search over settings, and nothing
else was tuned.

### Seven features, and why position is one of them

Per tick: `x`, `y` scaled to 0-1, heading as `sin`/`cos` (so 350 and 10
degrees are neighbours), battery 0-1, and the step just taken. Position is
included on purpose — on one map, "about to turn" is largely a fact about
where the corners are. The novel-window score is what keeps that honest.

Short histories — a mission's first ticks — are padded by repeating the
oldest state, which reads as "was standing still": the honest assumption
about ticks nobody recorded. The checkpoint carries the window length, the
map size used for scaling, and the behaviour definition (horizon, cone), so
a predictor cannot be run with the wrong scale or asked the wrong question.

### Selection by validation macro-F1, scored through the port

The best checkpoint is the one with the highest validation macro-F1, not
the lowest loss. `evaluate_lstm.py` scores it through `IMotionPredictor` —
one window at a time, exactly as the live pipeline will call it — and
reproduces the trainer's batched numbers to four decimals, which a test
also asserts.

## Results

CPU, 150 training missions / 50 held-out missions, randomised starts;
30 epochs, best at 22; 52k parameters.

**All held-out windows** (7,279):

| | accuracy | macro-F1 |
|---|---|---|
| always advance | 0.691 | 0.204 |
| persistence | 0.501 | 0.205 |
| **LSTM** | **0.942** | **0.802** |

Per class: advance F1 0.960, retreat 0.930, divert 0.903, hold 0.414
(n = 7 — too few to measure; see below).

**Novel windows only** (178, 2.4% of held-out):

| | accuracy | macro-F1 |
|---|---|---|
| always advance | 0.719 | 0.279 |
| persistence | 0.607 | 0.349 |
| **LSTM** | **0.820** | **0.663** |

**M5 is met on both.** On situations it has genuinely never seen the margin
is smaller — 0.663 against 0.349 — but it is the same direction and large.

For comparison, the fixed-start run that turned out to be memorisation
scored 0.970 / 0.964; that number is not a result.

## Still open

* **Nothing consumes the prediction yet.** `LstmMotionPredictor.probabilities`
  returns the full distribution for Phase 7's fusion network; until then
  the LSTM runs offline only.
* **HOLD and RETREAT are thinly measured.** Seven HOLD windows and seven
  novel RETREAT windows cannot support a per-class claim. More missions, or
  more hazards that force the vehicle to stop and back out, would.
* **One city.** The novel-window score is the best available proxy for
  generalisation, but a second map is the real test. A procedurally
  generated map (PROJECT.md §18) would settle it.
* **The waypoint follower is the only driver.** The LSTM has learned how
  *this* controller drives. Once Phase 6's DQN drives instead, its
  trajectories must be recorded and the LSTM retrained, or it will be
  predicting a driver that no longer exists.
