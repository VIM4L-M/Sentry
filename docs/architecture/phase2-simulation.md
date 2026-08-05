# Phase 2 — Simulation Engine, Occupancy Grid & Global Planning: design notes

**Status:** Complete. This document covers the mission loop; the hazards and sensors
that finished the phase are in
[phase2-dynamic-world-and-sensors.md](phase2-dynamic-world-and-sensors.md).

Records decisions made *during* implementation that PROJECT.md does not specify, per
CLAUDE.md's documentation rules. The architectural split this phase rests on is
recorded separately in [ADR 0002](../adr/0002-two-tier-navigation-and-command-center.md).

## What runs today

`python scripts/run_simulation.py` opens the disaster city and drives a complete
autonomous rescue mission: the command center builds an occupancy grid, A* routes to
the nearest victim, the vehicle follows the route tile by tile, picks victims up,
delivers them to the hospital, and reports the outcome. `--headless` runs the same
mission with no window.

On the shipped 30x20 map that is 3 of 4 victims rescued in 142 ticks with zero
collisions. The fourth (`victim_03`) is sealed inside a building block on all four
sides — the planner reports it unreachable and the mission controller abandons that
objective rather than looping. This is deliberate: an unreachable objective is an
ordinary mission state that the system must handle, and the shipped map exercises it.

## Decisions

### The tick is the unit of everything

`SimulationEngine.tick()` is one indivisible step: observe → decide → act →
adjudicate, always in that order. Simulated time advances at
`simulation.tick_rate_hz`, deliberately decoupled from `render.target_fps` — the
window uses a fixed-timestep accumulator, so a slow machine drops frames instead of
slowing the mission. A mission therefore replays identically given identical inputs,
which Phase 6 needs and which makes bug reports reproducible now.

### The occupancy grid is a belief, and it is rebuilt every tick

`MissionController._sync_grid` rebuilds the grid from the world each tick rather than
patching it incrementally. At 600 cells that costs nothing, and it removes a whole
class of stale-state bugs. When Phase 3 replaces the ground-truth producer with
projected detections, the rebuild becomes a merge — the call site does not move.

Only `TRAPPED` victims are stamped onto the grid: once someone is aboard, their old
tile must revert to plain terrain or the planner keeps routing to where they used to
be.

### Fire is a cost, not a wall

`OccupancyCode.FIRE` is impassable to the planner, but the tiles *around* a fire
carry a distance-weighted `fire_risk_penalty` instead of being blocked. The vehicle
skirts heat when there is room and drives past when there is not, which is what a
rescue vehicle should do. Set the penalty to `0.0` for pure shortest-path behaviour.

### Physics live in exactly one place

`VehicleController` is the only code that may move the vehicle, spend battery, or
apply damage. Manual keyboard driving, the Phase 2 waypoint follower, and the Phase 6
DQN all go through it, so a policy cannot learn against rules the live simulation
does not enforce.

It returns a `MoveOutcome` rather than mutating silently: Phase 6's reward function is
defined in terms of exactly those facts (did we advance, did we hit something, what
did it cost), and the HUD reports them live.

### `Position` forbids negative coordinates, so edges are checked before construction

`Position.__post_init__` rejects negatives (a Phase 1 decision). Movement and
neighbour-generation code therefore checks `x < 0 or y < 0` on raw integers *before*
building a `Position` — driving off the map edge must read as a collision, not raise
a `DomainValidationError`. `VehicleController` and `AStarPlanner._neighbours` both do
this, and there is a regression test for it.

### Manual mode is an adapter, not a mode flag

`KeyboardController` implements `ILocalController` and lives in `rendering/` — the
application layer must not import a UI toolkit. `ModeSwitchController` composes the
autonomous and manual controllers behind one port, so `Tab` swaps drivers without the
engine knowing either exists. A human drives through exactly the same action space and
physics as the DQN, which makes manual mode a fair human baseline as well as a
debugging tool.

## What came next

The three items this document originally listed as open were built in the second half
of Phase 2 and are documented in
[phase2-dynamic-world-and-sensors.md](phase2-dynamic-world-and-sensors.md):

- **Dynamic hazards** — `FireSpreadProcess` and `DebrisCollapseProcess` behind the
  `IWorldProcess` port. The replanning path this document anticipated is now pulled by
  hazards as well as collisions.
- **CCTV rig** — `sensors.rig.SensorRig`, which also produces the ground-truth labels
  Phase 3 trains against.
- **Weather** — still not started, still optional per the specification.

Two decisions here were revised by that work, both noted in the newer document: fire
now outranks the vehicle marker on the occupancy grid (so `fire_damage_per_tick` is
actually reachable), and `AStarPlanner` no longer refuses to plan from an impassable
start (so a vehicle engulfed by spreading fire is given a route out).

**Streamlit dashboard** remains Phase 9. The HUD covers the same telemetry in-window
for now, and `MissionStats.as_display_rows` is the shared surface both will render.
