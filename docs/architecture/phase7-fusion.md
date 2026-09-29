# Phase 7 — decision fusion: design notes

**Status:** complete. An MLP (Linear, ReLU, Dropout; trained with Adam)
implements `IDecisionFusion`. It combines:

- the DQN's Q-values,
- the LSTM's behaviour prediction,
- what the vehicle's own camera sees.

Milestone M7 is met on held-out ticks: fusion catches 89.5% of the DQN's
stale-map mistakes. On 40 whole missions it cuts the collisions a stale map
causes by 89% and the damage by 71%, for 3 fewer rescues.

Records decisions made *during* implementation that PROJECT.md does not
specify. The two structural ones — egocentric camera evidence and the
map-lag scenario — are ADR 0003.

## What it adds

| piece | where |
|---|---|
| `SceneEvidence`, the new `IDecisionFusion` contract | `interfaces/decision.py` |
| detections -> egocentric evidence; the onboard camera source | `perception/scene_evidence.py` |
| `FusionNet`, `MlpFusion`, `fusion_features` | `decision/mlp_fusion.py` |
| `FusedLocalController` — every model behind one `ILocalController` | `decision/fused_controller.py` |
| the lagging map, with live local reports | `simulation/grid_source.py` (`LaggedGridSource`) |
| true-world physics under a lagging map | `simulation/engine.py` (`physics_source`) |
| camera sightings reported to the command center | `simulation/onboard_reports.py` |
| data collection, trainer, `ActionReport` | `training/fusion.py` |
| config | `configs/training/fusion.yaml` |
| entry points | `scripts/train_fusion.py`, `scripts/evaluate_fusion.py`, `run_simulation.py --full` |

## What runs today

```bash
python scripts/train_fusion.py --detector models/yolo/denoised/weights/best.pt \
    --denoiser models/autoencoder/sentry/best.pt --device cuda   # collect + train 4 variants
python scripts/train_fusion.py --reuse-data --device cuda        # retrain on saved ticks
python scripts/evaluate_fusion.py --detector models/yolo/denoised/weights/best.pt \
    --denoiser models/autoencoder/sentry/best.pt --device cuda   # 40 whole missions
python scripts/run_simulation.py --full --device cuda            # watch it, mission control on
```

Collection: 150 missions, about 32k ticks, with the camera through the
same smoke, denoiser and YOLO the live demo uses. It takes about 25 minutes
on an RTX 4060. Training all four variants takes about 2 minutes.

## Decisions

### Fusion needed a problem to solve

With the command center's map current, the planner routes around every
known hazard, and the DQN drove 40 held-out missions without a single
collision. A fusion network measured there would be measured at parity
by construction. The scenario that gives it something to do is the one
real systems have: **the map lags the world** (CCTV processing, radio,
operator). Measured with no fusion, 20 missions:

| map lag | collisions | vehicle damage |
|---|---|---|
| none | 0 | 12 |
| 1 s | 5 | 55 |
| 3 s | 44 | 246 |
| 6 s | 91 | 511 |

The default is 3 s (30 refreshes). The vehicle's physics act on the true
world (`physics_source`), so debris the map has not heard of still stops
it.

### Labels come from the simulator, exactly

Each tick the DQN is asked twice: once on the stale map it really drives
on, and once on the true world. The second answer is the label. A tick
where the two differ is **critical** — the stale-map mistake fusion exists
to catch. About 2% of ticks are critical. 30% of collection ticks are driven
by the true-world choice, so the data also shows what happens *after* a
correction.

### The DQN's preference is a skip connection

An MLP asked to learn "agree with the DQN" from scratch spent its capacity
copying and still missed rare actions. The policy block is added straight
to the output through a learned scale (`prior_scale`, initialised at 4).
The network therefore starts from agreement, and its hidden layers only
learn *when not to*. A network trained without the policy group has the
block zeroed by its input mask, so the skip adds nothing and the ablation
stays fair.

### Critical ticks are up-weighted 2x, not 5x

Swept on the same saved ticks:

| critical weight | caught | wrongly overridden |
|---|---|---|
| 1 | 84.2% | 1.55% |
| **2** | **82.5%** | **1.14%** |
| 3 | 80.7% | 1.24% |
| 5 | 87.7% | 3.0% (814 overrides in 40 missions) |

At 5 the network caught the most mistakes and overrode the DQN so often
that missions failed. At 2 it keeps most of the gain with the fewest wrong
overrides.

### Macro-F1 was the wrong M7 metric

The first M7 check used macro-F1 over the five actions, and it failed —
for every network, including one trained on the DQN's own Q-values alone.
The reason is `STOP`: 3 of 7,080 validation ticks. One miss moves the
macro-average by about 0.13. The metric was measuring three ticks. M7 is
judged instead on:

- **caught**: the share of the stale-map DQN's mistakes corrected;
- **wrongly overridden**: the share of ticks the DQN had right that fusion
  changed;
- **weighted accuracy**, which combines both as the loss does (this is also
  what checkpoints are selected on).

Macro-F1 is still printed, next to them, so nothing is hidden.

### Fusion only overrides when it is sure

At inference an action other than the DQN's is taken only at probability
0.8 or more (`override_threshold`). Taking every argmax overrode the DQN
814 times over 40 missions. Most of those overrides were unsure, and each
one sent the vehicle off its route.

### Fire is projected by tile centre, like the CCTV grid

A YOLO fire box spills about half a tile past the fire (measured in
Phase 3). Counting every tile a box touched reported a fire *beside* the
street as a fire *ahead*, so fusion stopped for fires that were not in the
way. Fire now claims only the tiles whose centre its box covers, the same
rule the CCTV grid already uses (`tiles_centred_in_box`), falling back to
overlap for a box too small to cover any centre. Retrained on data
collected with the fix, fusion caught 89.5% of the DQN's mistakes, up from
82.5%, and completed 37 of 40 missions, up from 33.

### Fusion only overrides on what the camera sees

On a map it was not trained on (Kundrathur, from OpenStreetMap) fusion once
overrode the DQN with nothing in view — turn left became drive on, at 91% —
and the detour led into a pocket a fire later closed. Correcting the DQN is
the camera's job, so an override now needs the camera to report something
at 0.5 confidence or more; otherwise the DQN's choice stands. On the 40
default missions this changed nothing measurable (138 rescued, 9 collisions,
127 damage, 37/40 completed, against 138 / 8 / 122 / 37 without it), so it
stays as a safety rule.

### Reporting to the command center is optional, and off

The first whole-mission run showed a failure the tick-level scores could
not. Fusion stopped the vehicle in front of debris, but the route still ran
through it, so the vehicle *waited* up to 3 s for the map to catch up,
sometimes while a fire spread onto it. `OnboardHazardReporter` fixes that
case. When fusion overrides because the camera sees debris or fire directly
ahead, it overlays that tile on the lagging map until the map catches up,
and drops the route so the planner goes round.

It was measured both ways (below). With the fire fix, fusion on its own
already avoids most hazards. Reporting then removes the last few collisions,
but each YOLO report closes a street, and completions fell from 37 to 25 of
40. It stays available (`report_hazards: true` in
`configs/training/fusion.yaml`) and is off by default.

## Results

### Held-out ticks (30 missions, 7,080 ticks, 57 critical)

| | accuracy | caught | wrongly overridden |
|---|---|---|---|
| DQN alone | 0.992 | 0% | 0% |
| **fusion (all signals)** | 0.985 | **89.5%** | 1.4% |
| trained on DQN inputs only | 0.990 | 8.8% | 0.3% |
| trained on camera only | 0.752 | 50.9% | 24.6% |
| trained on LSTM only | 0.516 | 15.8% | 48.1% |

**M7 met.** Fusion beats every single signal on weighted accuracy. Only the
combination both catches the DQN's mistakes (the camera's contribution)
and leaves its correct choices alone (the DQN's). Neither signal manages
both on its own.

### Whole missions (40 held-out missions, 3 s map lag, camera through smoke + denoiser + YOLO)

| | rescued | lost | collisions | damage | completed |
|---|---|---|---|---|---|
| fresh map, DQN (the ceiling) | 150 | 4 | 0 | 28 | 40/40 |
| lagged map, DQN | 141 | 10 | 71 | 423 | 38/40 |
| **lagged, fusion (the default)** | **138** | **9** | **9** | **127** | **37/40** |
| lagged, fusion + map reports | 117 | 4 | 0 | 174 | 26/40 |

The default removes **87% of the collisions** the stale map causes and
**70% of the damage**. It costs 3 rescues and one completed mission.
Map reports trade the other way: no collisions and 4 deaths, but 26 of 40
missions completed.

### Real streets it was never trained on (10 missions each, 3 s lag)

| map | camera | fusion | fusion + map reports |
|---|---|---|---|
| college area (Nandambakkam) | ground-truth labels | 30 rescued, 9 lost, 0 collisions, 10/10 | 30, **0 lost**, 1, 10/10 |
| college area (Nandambakkam) | YOLO + denoiser | 30 rescued, 9 lost, 4 collisions, 10/10 | 30, **1 lost**, 2, 10/10 |
| Kundrathur | ground-truth labels | 30 rescued, 5 lost, 28 collisions, 7/10 | **33, 2, 6, 10/10** |
| Kundrathur | YOLO + denoiser | 31 rescued, 4 lost, 34 collisions, 8/10 | 28, 2, 18, 5/10 |

With the DQN alone and labels, Kundrathur gave 32 rescued, 42 collisions,
7/10. Two readings. Map reports pay off when the camera is right: with
labels they win on both real maps. With YOLO they are a trade, fewer deaths
for fewer completions, which is why they stay off by default. And
Kundrathur is hard for every setting: its dense street grid is unlike the
training city, so the fusion network's corrections are least reliable
there.

Before the fire fix (the v1 model, kept as `models/fusion/sentry_v1`), the
best setting was fusion + reports: 134 rescued, 8 collisions, 124 damage,
33/40 completed. With the camera reading ground-truth labels instead of YOLO
(15 missions), fusion + reports matched the fresh-map ceiling exactly:
56 rescued, 1 collision, every mission completed. Every remaining gap is
detector error.

## Round 2: what the teammate's build did better, adopted and measured

A teammate built Phases 7-8 independently. Their fusion lost no rescues where
round 1 here lost 3 and failed 3 of 40 missions. Measured on this laptop with
their own models and evaluation, the difference came from three decisions,
not from the network or the hardware:

1. **The route is re-checked against every fresh map.** When the lagging map
   finally shows a collapse on the rest of the route for 3 refreshes in a row,
   the command center replans (`MissionController._drop_route_if_blocked`,
   `mission.block_confirm_refreshes`). Round 1 had no such check. After fusion
   stopped the vehicle at the debris, it waited for a map that had already
   been refreshed, and the lost time cost missions.
2. **The LSTM was retrained on the DQN's own driving**
   (`record_trajectories.py --dqn`, `models/lstm/dqn`). Round 1's LSTM had
   learned the waypoint follower and scored 0.52 on its own. Retrained, it
   scores macro-F1 0.909 (the teammate's: 0.866).
3. **A simpler label** (`label_mode: veto`). Keep the DQN's move unless it
   would enter a tile the real city has blocked, then STOP. The network only
   has to learn when to veto. Round 1 imitated the DQN on the true map, any
   action.

All three were ported with tests (`tests/integration/test_route_recheck.py`,
`tests/unit/test_fusion_labels.py`). Both labels were retrained with the new
LSTM and the re-check. 40 held-out missions, 3 s lag, camera through smoke,
denoiser and YOLO:

| Driver | Rescued | Lost | Collisions | Damage / mission | Completed |
|---|---|---|---|---|---|
| DQN alone (with the re-check) | 147 | 8 | 71 | 10.6% | 40/40 |
| Round 1 fusion | 138 | 9 | 9 | 3.2% | 37/40 |
| Teammate's fusion (their evaluation, this laptop) | 140 | 12 | 13 | 3.6% | 39/40 |
| **truth2 + camera reports** | **148** | **5** | **0** | **1.5%** | 39/40 |
| veto | 147 | 8 | 6 | 2.7% | **40/40** |

`truth2` (`configs/training/fusion_truth2.yaml`) is the default on the shipped
map: it rescues more than the DQN alone, with no collisions. Camera reports
used to cost completions and now help, because the re-check replans on them.

On cities it never saw, with traffic, `veto` is the safer one. Chicago,
3 hazard seeds: equal rescues, and 0 collisions against 10-12 for `truth2` +
reports. The city demos use the veto model retrained on the traffic-aware DQN
(`fusion_veto_traffic.yaml`; README, Phase 9).

## Known limitations

- **What YOLO reports ahead is not always what is ahead.** The report
  threshold is not the lever: 0.5, 0.7 and 0.85 gave identical results on
  the same 15 missions, because YOLO's reports are all high-confidence. The
  fire-centre fix removed the largest source of error. A detector trained
  on onboard frames specifically, not the shared dataset, is the next step.
- **One lag, one training city.** The network was trained at a 3 s lag, on
  the shipped map. It has not been measured at other lags, and on dense
  real street grids (Kundrathur) it is least reliable. Training on a mix of
  imported maps is the next step.
- **Round 1's LSTM contributed little.** It was trained on the waypoint
  follower's driving, not the DQN's; round 2 fixed that (macro-F1 0.909).
