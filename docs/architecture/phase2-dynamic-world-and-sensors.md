# Phase 2, part 2 — the dynamic world and the sensor layer: design notes

**Status:** Complete. This closes Phase 2; only weather (optional in the
specification) is unbuilt.

Records decisions made *during* implementation that PROJECT.md does not specify.
The first part of Phase 2 is documented in
[phase2-simulation.md](phase2-simulation.md); the navigation split both rest on is
[ADR 0002](../adr/0002-two-tier-navigation-and-command-center.md).

## What this adds

The city used to be a still photograph the vehicle drove across. Now it burns and
collapses while the mission runs, and four cameras plus one onboard view report what
it looks like — with labels.

`python scripts/run_simulation.py --headless` on the shipped map: 3 of 4 victims
rescued, ~5 fires ignited by spread, ~5 collapses, at least one route cut and
replanned, zero collisions. `python scripts/capture_dataset.py --output data/synthetic`
writes the perception dataset from that same mission.

## Decisions

### Fire threatens victims; debris threatens routes

Both hazards could have blocked roads. Only one does, and that is deliberate.

`FireSpreadProcess` ignites only flammable terrain — buildings, trees, rubble — never
roads or open ground. Fire therefore spreads *through the blocks* where victims are
trapped, raising the planner's risk cost near them and eventually reaching them, but
it can never sever the road network. `DebrisCollapseProcess` does the opposite: it
drops rubble into clear streets beside standing buildings, which is precisely the
event that invalidates a plan.

The alternative — one hazard that does both — made missions a coin flip. A fire that
crosses tarmac can wall the vehicle off from the hospital through no decision it made,
which is not a lesson any policy can learn from. Splitting the two gives the DQN in
Phase 6 a stable road network to learn on and a genuine source of surprise on it.

### Hazards report changes; they do not decide what changes mean

`IWorldProcess.advance` returns a `WorldChange` — a set of tiles and a description —
rather than mutating quietly or calling the mission controller itself. The engine
merges the changes and hands them to `MissionController.invalidate_route_if_affected`,
which cares about exactly one question: did this land on the part of the route we have
not driven yet? Debris behind the vehicle costs nothing. Debris ahead costs the plan.

That split means a new hazard type never touches mission code, and the replanning rule
lives in one place with one test.

### Every hazard is seeded, and each owns its generator

`build_world_processes` derives an independent `random.Random` per process from one
config seed instead of sharing one. Sharing would couple them: turning fire off would
shift every debris draw and silently produce a different mission from the "same" seed.
Phase 6 will compare training runs against each other, and that comparison is only
meaningful if disabling one hazard leaves the others identical. There is a test for it.

### Fire outranks the vehicle marker on the occupancy grid

The specification's grid stores one code per cell, so "vehicle standing in fire" is
unrepresentable. It used to resolve as `VEHICLE`, which meant `fire_damage_per_tick`
could never be charged — the vehicle controller reads the grid, and the grid never
said `FIRE` where the vehicle was.

`OccupancyGrid.mark_vehicle` now leaves a burning cell alone. Fire wins because fire is
the fact that changes decisions. Nothing is lost visually: the renderer draws the
vehicle from the `CityMap`, never from the grid.

That change required a second one. `AStarPlanner` used to reject an impassable *start*
outright, which would have stranded a vehicle the moment fire reached it. It no longer
does: being already somewhere is not a reason to refuse to leave. The search still
refuses to *enter* impassable tiles, so a route out of a fire never drives back
through one. Both behaviours have tests.

### The camera produces labels, not just pixels

`FrameRasterizer` annotates every victim, fire, and obstacle *in the same pass that
paints it*, from the geometry that painted it. The alternative — render, then walk the
`CityMap` again to work out where things ended up — has two sources of truth that
drift the first time a drawing rule changes.

Annotations are `interfaces.perception.Detection` values with confidence `1.0`: the
identical type the Phase 3 detector will emit. Prediction and ground truth are
therefore directly comparable with no adapter in between.

Victims aboard the vehicle or already delivered are not painted and not labelled.
Labelling them at their old tiles would train the detector to hallucinate.

### The camera palette is not the operator's theme

`configs/render.yaml` styles the human-facing map; `configs/sensors.yaml` styles what
the cameras see. Keeping them separate costs one config section and buys an important
guarantee: restyling the operator display cannot silently invalidate a trained model.
Only the sensor palette is part of the detector's contract.

Both now share `common.color.Color`, which moved out of `rendering/` so the sensors
package would not have to import the UI layer to name a colour.

### Texture is a requirement, not decoration

Flat blocks of uniform colour make a detection problem solvable by reading one pixel.
Each camera therefore carries a fixed per-pixel grain, seeded by CRC32 of its id —
CRC32 rather than `hash()`, because Python salts string hashes per process and a
salted seed would make every run produce different imagery.

Fixed, not per-frame: a static camera watching an unchanged scene must return
byte-identical pixels, so that "did the world change?" is a question about the world
and not about the renderer. Variation between samples comes from the degrader, which
is where variation belongs.

### Degradation is separate from capture

`SensorRig` returns clean frames. `FrameDegrader` corrupts them. The rig has no
degrader and no `degraded=True` flag, because Unit IV needs *both* halves at once and
`degrade_pair` returning `(corrupted, clean)` is the natural shape for that. Perfect
pixel alignment between the pair is the whole reason for corrupting synthetically
rather than filming something dirty.

Corruption order is blur → smoke → noise: optics first, then electronics. Noise before
blur would smear the noise, which is not what a sensor does and would teach the
autoencoder the wrong inverse.

### CCTV cameras overlap

Four cameras at 16x10 tiles cover the 30x20 city with two columns of overlap rather
than tiling it exactly. Real networks overlap, and more usefully it means some victims
are seen by two cameras at once — which is the duplicate the command center's merge
step has to cope with in Phase 3. Building the merge against a network that never
produces duplicates would be building it against a fiction.

Frames come out 256x160, both multiples of 32, which is what YOLO's stride wants.
The onboard camera is 9x9 tiles at 16 px — odd-sized so the vehicle sits dead centre,
and clamped at the map edges so the frame never changes shape. A convolutional network
cannot accept a frame that shrinks in a corner.

## Presentation

A separate, later pass, prompted by a review observing that the engineering was ahead
of the visuals: a reviewer should be able to watch for thirty seconds and understand
the story without narration. These decisions are about legibility, and none of them
touched the architecture.

### Glyphs are drawn, not loaded

Victims, fire, debris, trees, the vehicle, and the hospital are Pygame primitives in
`rendering/glyphs.py` rather than sprite assets. Image files would mean binaries in the
repo, an asset loader, a search path, a scaling policy, and a licence question — a lot
of machinery for what is fundamentally "draw a stick figure". Every glyph scales from
the rect it is handed, so changing `render.tile_size_px` needs no new artwork.

Drawing them surfaced two bugs that flat discs had hidden: rescued victims were still
being drawn at the tile they were rescued from, and *trees were drawn as rubble* — they
are both `Obstacle` entities, but a park is not a collapsed building.

### Fire flickers on the map and never in a camera frame

The operator display animates; `sensors/rasterizer.py` does not. That split already
existed for a different reason — the detector needs stable imagery — and it is what
makes animation free here. A flicker that reached the camera would inject variance into
every training sample.

The flicker only ever *shrinks* the flame. Scaling above full height pushed the tip into
the tile above, which misreports where the fire is; there is a test asserting no glyph
escapes its own tile at any phase.

### The map and the camera palettes were aligned

They remain separate files for the reason recorded above, but the entity colours in
`configs/render.yaml` were moved to match `configs/sensors.yaml`. Both views are now on
screen simultaneously, and a victim that was yellow on the map and pink in the camera
thumbnail read as two different things.

### The abandoned route stays on screen

`MissionController.previous_route` keeps the plan a replan replaced, drawn greyed out
beneath the live one. "Routes cut: 1" is a number; a grey path forking away from a blue
one is the story. This is the cheapest possible way to make the system's most
interesting behaviour visible.

### The event log is a second audience, not a duplicate of the logger

`simulation/events.py` holds a bounded `EventLog` fed by `MissionController.record`,
which writes to the logger *and* the log in one call. The log file is for grep; the
on-screen panel is for someone watching. Messages are phrased for the screen and
ellipsized to fit, measured against the real font rather than an assumed character
width.

### `G` shows the belief, not the world

`rendering/grid_overlay.py` draws the occupancy grid as colour-coded, numbered cells.
Today it matches the world exactly and looks redundant. From Phase 3 it will be built
from detections and will be wrong in interesting ways, and this is the view that will
show *where* — the difference between "the vehicle drove somewhere strange" and "the
grid believed that tile was debris".

### The camera panel is Phase-3 furniture, built early

It renders `CameraFrame`s with their detections boxed. Those detections are ground truth
today and YOLO output later, and because a frame carries its own annotations, nothing
about the panel changes when the producer does.

## Still open

- **Weather** — listed as optional in the specification, not started. It would enter
  as another `IWorldProcess` plus a degradation preset, with no new seams needed.
- **Live perception in the loop** — frames are captured and written to disk, but
  nothing consumes them during a mission yet. That is Phase 3's job, and the seam it
  plugs into (`OccupancyGrid` built from projected detections rather than ground
  truth) is already load-bearing.
