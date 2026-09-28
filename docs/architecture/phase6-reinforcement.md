# Phase 6 — reinforcement learning: design notes

**Status:** core complete. A Stable-Baselines3 DQN implements
`ILocalController`, trained in `SentryEnv` — whole missions, the real
physics — and milestone M6 is met: on 40 held-out missions it rescues 104%
of what the waypoint follower rescues. Read "Results" before quoting that
number; the honest summary is *parity, with a speed edge*. Trained on CPU in
about 16 minutes; a GPU is not needed at this network size.

Records decisions made *during* implementation that PROJECT.md does not
specify.

## What it adds

| piece | where |
|---|---|
| egocentric policy features, `DqnLocalController` (the `ILocalController`) | `decision/dqn_controller.py` |
| `SentryEnv` — a mission as a Gymnasium environment | `training/sentry_env.py` |
| missions varied by seed, with random starts (shared with Phase 5) | `training/missions.py` |
| trainer, mission-level comparison against the follower | `training/dqn.py` |
| config, including every reward weight | `configs/training/dqn.yaml` |
| a harsher disaster, for robustness evaluation only | `configs/simulation_stress.yaml` |
| entry points | `scripts/train_dqn.py`, `scripts/evaluate_dqn.py`, `run_simulation.py --dqn` |

## What runs today

```bash
python scripts/train_dqn.py                 # ~15 min on CPU, 300k steps
python scripts/evaluate_dqn.py              # 40 held-out missions vs the follower
python scripts/evaluate_dqn.py --simulation configs/simulation_stress.yaml
python scripts/run_simulation.py --dqn models/dqn/sentry/best.zip   # watch it drive
```

## Decisions

### M6 is parity, stated before training

The roadmap's M6 is "reaches a defined rescue-rate threshold". The
threshold was fixed before any training: **the DQN must rescue at least
90% of what the waypoint follower rescues on the same held-out missions.**

"Beat the follower" was not on offer. The follower turns toward the next
waypoint and drives at it; on this map, with a ground-truth grid, it
rescues everyone on every mission where rescue is possible. Given the same
inputs, the best a learned policy can do is match it plus whatever edge
cases the rules miss. M6 asks a network told nothing about driving but a
reward to match a hand-written driver that is already perfect at the job.

### One episode is one real mission

`SentryEnv` does not simulate a simplified task. An episode is a complete
mission: hazards spread and collapse, the A\* command center plans and
replans, `VehicleController` applies the same physics the live window uses.
The agent replaces only the waypoint follower — the environment installs a
controller that returns whatever the agent chose, so the engine's ordering
(hazards, decision, physics, mission) is untouched. A policy cannot learn
against rules the deployed system does not enforce.

`SimulationEngine.observe()` became public for this: the agent needs the
observation *before* the tick that acts on it.

### Egocentric features, no absolute position

The policy sees seven numbers — the waypoint's offset *ahead* and *to the
right*, whether there is a waypoint and whether it is standing on it, the
tile ahead blocked, fire proximity, battery — not
`LocalObservation.as_array()`'s world-frame layout. In world coordinates a
network must learn a rotation before it can learn to drive; egocentrically,
"the waypoint is ahead" means one thing in every heading.

Absolute position is excluded on purpose: Phase 5 showed that a model which
knows where it is memorises the city. A test asserts that the same
situation at two different places produces identical features.

### The reward pays for mission facts, and the checkpoint is chosen on missions

Every reward term is a weight on something the physics or the mission
already reports: progress toward the waypoint that was active before the
tick (moving away costs the same, so there is nothing to farm), pickups,
deliveries, collisions, damage, fire proximity, time, battery, reversing,
completion, failure. Signs are enforced by the config.

The *return* is how the agent learns; it is never how the result is
judged. During training the policy drives 20 held-out missions every 25k
steps through `DqnLocalController` — the live adapter — and the checkpoint
kept is the one that rescues most, then completes most, then collides
least, then finishes fastest. Before training started, the reward was
checked to rank driving correctly: the follower driven through the
environment earns +150 to +163 per mission, random actions −163.

### The first policy drove backwards, and the reward was changed

A 20k-step run matched the follower and was faster — suspicious for a
minute of training, so it was checked before being believed. An untrained
network in the same slot rescues nobody, so the DQN really is driving. It
agreed with the follower on 87% of ticks; almost all the rest was one
trick: where the follower U-turns (two turns, then drive), the DQN
**reversed**. The physics charge a reverse exactly what a forward move
costs, so driving backwards was the cheapest way to change direction — 14%
of its tiles, in stretches of up to ten.

The reward allowed it; the design does not want it. ADR 0002 has the local
controller reverse to back out of a dead end, not to cross town. A
`reverse` penalty of −0.2 per reverse makes a U-turn (≈0.12 in time and
battery) the cheaper way round. Reversing fell to 3% of moves in the final
policy; streaks of up to nine tiles remain, so the penalty discourages the
habit without eliminating it. `reverse: 0.0` restores the original
behaviour.

The other disagreement is benign: where the follower *stops* — standing on
a reached waypoint before the next is set — the DQN turns toward where it
is going next. That, and reversing, are where its speed comes from.

## Results

CPU, 300k steps; held-out missions never trained on; random starts; both
drivers on identical missions (same hazard seed, same start tile).

**Shipped disaster** (40 missions):

| | rescued | lost | collisions | completed | mean ticks |
|---|---|---|---|---|---|
| waypoint follower | 143 | 5 | 0 | 95% | 156 |
| **DQN** | **149** | 5 | 0 | **100%** | 161 |

**M6 is met — 104%.** But the six extra rescues are one situation, not a
general superiority. Both of the follower's failures (seeds 3400 and 3401)
start on the same tile, (3, 19), whose route begins with a U-turn. At tick
30 a fire spread sealed the area around (7, 17): the follower was standing
there and was trapped ("hospital unreachable"); the DQN, having reversed out
instead of turning round, was two tiles on at (9, 17) and escaped. Two
ticks of speed, at the one moment it mattered. That is a real mechanism —
exposure time is risk in a spreading disaster — and it is also partly
timing. On the 38 missions both completed, the two are level. (Mean ticks
is higher for the DQN only because it finished the two long missions the
follower abandoned at tick 30.)

**Harsh disaster** (`configs/simulation_stress.yaml`: collapses three times
as often with a 20-collapse budget, fire twice as fast and further; 40
missions, never trained on):

| | rescued | lost | collisions | completed | mean ticks |
|---|---|---|---|---|---|
| waypoint follower | 135 | 14 | 0 | 98% | 164 |
| **DQN** | **136** | 13 | 0 | 98% | 160 |

Parity again, a little faster. Neither driver ever collides.

**Why neither ever collides — and what that means for the design.** With a
ground-truth occupancy grid, every collapse is known the instant it happens
and A\* replans before the vehicle arrives. The planner absorbs every
surprise, and the local controller is left with nothing unexpected to
handle. The case ADR 0002 built the local tier for — an obstacle the
command center *does not know about* — does not occur here. It will once
the grid comes from the cameras (Phase 3's `DetectedGridSource`), which is
wrong in the ways docs/architecture/phase3-detection.md lists. That is where
a learned local driver can earn more than parity, and it belongs to
Phase 8's end-to-end runs.

## Still open

* **Train and evaluate on the perceived grid.** The DQN has only ever seen
  a perfect map. Phase 7 measured what that costs: with the command
  center's map three seconds behind reality (and the vehicle now moving
  through the real city, ADR 0003) the DQN collides 150 times in 40
  missions. Decision fusion with the onboard camera brings that to 13 —
  see [phase7-fusion.md](phase7-fusion.md). Training the DQN itself on a
  lagging or perceived map is still open.
* **The observation has no side view.** `LocalObservation` reports only the
  tile ahead. Swerving around an unexpected obstacle needs to know whether
  left and right are clear; adding that is a change to a port's value type,
  and so an ADR, not a quiet edit.
* **Reversing is discouraged, not gone.** 3% of moves, streaks up to nine.
  Whether a rescue vehicle may reverse at all is a product decision; the
  weight is one line of config.
* **One seed.** The training curve is flat at parity from 25k steps to
  300k, with the extra rescues appearing at two checkpoints, so the result
  is stable — but it is still one training run.
* **Phase 5's LSTM learned the follower's driving.** The DQN drives
  slightly differently (it reverses and pre-turns), so the LSTM should be
  retrained on DQN trajectories before Phase 7 fuses the two.
