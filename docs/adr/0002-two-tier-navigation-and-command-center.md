# ADR 0002: Two-tier navigation — A* globally, DQN locally

**Status:** Accepted
**Phase:** 2
**Supersedes:** the single-policy navigation contract defined in Phase 1's
`interfaces/decision.py`

## Context

Phase 1 shipped `INavigationPolicy`, a single port whose Deep Q-Network
implementation was to choose *every* movement, including the city-scale route from
the safe zone to a victim. The revised project specification splits navigation
explicitly:

> Global navigation: use A\*. Input: occupancy grid map. Output: waypoints.
> Local navigation: use DQN. … DQN should NOT generate the entire city route.

It also introduces a command center that builds a **global occupancy grid** from
multiple CCTV feeds, plans routes on it, and monitors the mission — a layer Phase 1
had no equivalent of.

Beyond following the spec, the split is the better engineering:

- **A DQN is a poor global planner.** Learning shortest paths across a 30x20 city
  means learning something Dijkstra already computes exactly, in under a
  millisecond, with a proof of optimality. Every training epoch spent on it is spent
  not learning the thing only RL can do.
- **Replanning must be instant and explainable.** When a building collapses across
  the route, the command center must produce a new route immediately and a human
  operator must be able to see why it goes where it goes. A* gives both; a policy
  network gives neither.
- **RL earns its place locally.** Reacting to a newly-detected obstacle, deciding
  whether to wait for a fire to die down or back out and go around, trading battery
  against time — these are exactly the sequential, partially-observed decisions with
  delayed payoff that a DQN is for, and they have no closed-form solution.

Options considered:

1. **Keep one learned policy for everything.** Contradicts the specification, wastes
   training capacity on a solved subproblem, and makes replanning opaque.
2. **A\* only, no RL.** Deterministic and simple, but drops Unit V from the project
   entirely and cannot react to anything the grid does not already know.
3. **Two tiers with separate ports** — A* over the occupancy grid for routing, a
   learned controller for per-tick execution.

## Decision

Adopt option 3.

`interfaces/navigation.py` defines both tiers:

- `IRoutePlanner.plan(grid, start, goal) -> Route` — implemented by
  `navigation.astar.AStarPlanner`. Deliberately **not** learned.
- `ILocalController.decide(observation) -> LocalDecision` — implemented in Phase 2
  by `simulation.waypoint_follower.WaypointFollower` (deterministic baseline) and in
  Phase 6 by a Stable-Baselines3 DQN.

The action space becomes **egocentric** — `MOVE_FORWARD`, `REVERSE`, `TURN_LEFT`,
`TURN_RIGHT`, `STOP` — replacing Phase 1's absolute `MOVE_NORTH`/`MOVE_SOUTH`/… so a
learned policy generalizes across approach directions instead of memorizing one
behaviour per heading. `Vehicle` gains a `heading` field to make that space
meaningful.

`VehicleAction`, `PolicyOutput`, and `INavigationPolicy` are removed from
`interfaces/decision.py`, which now holds only the fusion contract. Nothing had
implemented them yet — Phase 1 defined contracts only — so no working code was
rewritten to make this change.

`domain/occupancy.py` adds the `OccupancyGrid` the planner consumes, with the
specification's integer codes (`0` road, `1` building, `2` fire, `3` debris,
`4` victim, `5` hospital, `6` vehicle) as an `IntEnum`, because those integers are
the contract once the grid becomes a network input.

## Consequences

- The command center owns *where*, the vehicle owns *how*. `MissionController`
  chooses objectives and routes; `ILocalController` executes one tick at a time.
  Neither can be tested only through the other.
- Phase 6's reward function shrinks to local concerns (progress along the route,
  collisions, fire proximity, battery), which is a far easier credit-assignment
  problem than rewarding a whole cross-city journey.
- `WaypointFollower` is not throwaway scaffolding: it is the **baseline the DQN must
  beat**. A learned controller that cannot outperform greedy waypoint-chasing around
  dynamic obstacles has not earned its place in the pipeline.
- The grid is a *belief*, not truth. `OccupancyGrid.from_city_map` fills it from
  ground truth today; Phase 3 replaces that producer with projected YOLO detections
  and nothing downstream changes. The seam is already load-bearing —
  `MissionController.invalidate_route` is the hook a newly-detected obstacle pulls.
- A* is unaware of anything the grid does not record. Fire *risk* is modelled as a
  configurable cost penalty (`PlannerConfig.fire_risk_penalty`) rather than a hard
  wall, so the planner skirts heat when it can and drives through when there is no
  alternative.
- If a later phase needs continuous (non-tile) movement, both the action space and
  the planner's 4-connected assumption must be revisited in a new ADR rather than
  bent here.
