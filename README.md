# SENTRY AI

**Autonomous Emergency Rescue Vehicle for Disaster Zones** — a simulation-based Applied
Deep Learning project. See [`PROJECT.md`](PROJECT.md) for the full architecture,
diagrams, phase roadmap, and API contracts. See [`CLAUDE.md`](CLAUDE.md) for the working
rules this codebase follows.

**Status:** Phases 1 and 2 complete. A full autonomous rescue mission runs end to end
(occupancy grid → A\* routing → mission control → live HUD) in a city that changes
underneath it: fire spreads, buildings collapse into the streets, and the command
center replans around both. A four-camera CCTV network plus the vehicle's onboard view
produce labelled frames for the AI phases. Phase 3 (YOLOv8n detection) is next. See
PROJECT.md §10-11 for the phase breakdown and milestones.

## Quick Start

Requires Python 3.11+ (developed and tested on 3.14; see "Environment Notes" below).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the sentry_ai package in editable mode
```

Run the test suite:

```bash
pytest                                          # unit + integration tests
pytest --cov=sentry_ai --cov-report=term-missing  # with coverage
mypy src scripts                                 # type checking
ruff check src tests scripts                      # linting
```

Run a rescue mission (Phase 2 — the vehicle plans routes and drives itself):

```bash
python scripts/run_simulation.py              # windowed
python scripts/run_simulation.py --headless   # no window, prints the outcome
```

| Key | Action |
|---|---|
| `Escape` | quit |
| `Space` | pause / resume |
| `Tab` | toggle autonomous ↔ manual driving |
| `G` | toggle the occupancy-grid debug view |
| `C` | toggle the camera panel |
| arrows / `WASD` | drive (manual mode) |

The window is laid out so a mission can be followed without narration:

```
┌────────────────────────────────┬──────────────┐
│                                │  cctv_nw     │
│   the city — terrain, glyphs,  │  cctv_ne     │  what the cameras see,
│   CCTV footprints, and the     │  cctv_sw     │  with ground-truth boxes
│   planned route (the route it  │  cctv_se     │  (YOLO output in Phase 3)
│   replaced stays greyed out)   │  onboard     │
├────────────────────────────────┴──────────────┤
│ phase │ counters │ event log │ battery/health │
└───────────────────────────────────────────────┘
```

Press `G` and the city is replaced by the occupancy grid the planner
actually reasons over — the 0-6 codes, colour-coded and numbered. Today it
matches the world exactly; from Phase 3 it will be built from detections and
will be wrong in interesting ways, and this is how you will see that.

Capture the synthetic perception dataset (what Phases 3 and 4 train on):

```bash
python scripts/capture_dataset.py --output data/synthetic
```

This runs a full mission and writes, for every camera and every Nth tick:

| Path | Contents |
|---|---|
| `images/` | degraded frames — smoke, blur, sensor noise (detector input) |
| `labels/` | YOLO labels for those frames (`class cx cy w h`, normalized) |
| `clean/` | the matching clean frames — paired with `images/`, this is the denoising autoencoder's training set |
| `classes.txt` | class id → name (`victim`, `fire`, `obstacle`) |

Render the static disaster city instead (Phase 1's walking skeleton — no movement):

```bash
python scripts/run_preview.py
# or with a custom config:
python scripts/run_preview.py --config configs/app.yaml
```

Press `Escape` or close the window to exit.

From Phase 3 onward you'll also need the heavier ML stack:

```bash
pip install -r requirements-ml.txt
```

## Project Layout

```
configs/     All tunables — YAML, loaded into typed dataclasses (see config/schema.py)
src/sentry_ai/
  common/      Logging, exceptions, shared type aliases
  config/      Typed config schema + the only code that reads YAML
  domain/      Entities, enums, CityMap, OccupancyGrid — pure Python, no frameworks
  interfaces/  Ports (ABCs): perception, sequence, navigation, world, decision fusion
  navigation/  A* global route planner (classical, not learned)
  simulation/  Tick engine, mission state machine, vehicle physics, hazards
  sensors/     Synthetic cameras: rasterizer, ground-truth labels, frame degradation
  rendering/   Pygame map renderer, HUD, keyboard input, mission window
scripts/     Composition roots / CLI entry points
tests/       unit / integration / e2e, mirroring src/
docs/        Phase design notes and ADRs
```

**How navigation is split:** A\* plans the route across the city from an occupancy
grid; a learned local controller (Phase 6 DQN, today a deterministic waypoint
follower) decides each tick's move. The reasoning is in
[`docs/adr/0002-two-tier-navigation-and-command-center.md`](docs/adr/0002-two-tier-navigation-and-command-center.md).

**How the two hazards differ:** fire only takes hold on flammable terrain, so it
threatens victims and raises route costs but never severs the road network; debris
collapses into clear streets, which is what actually forces replanning. Keeping them
separate is what stops a mission becoming a coin flip — see
[`docs/architecture/phase2-dynamic-world-and-sensors.md`](docs/architecture/phase2-dynamic-world-and-sensors.md).

**Why the cameras have their own palette:** `configs/render.yaml` styles the map a
human reads; `configs/sensors.yaml` styles what the detector is trained on. Separating
them means restyling the operator display cannot silently invalidate a trained model.

See PROJECT.md §4 for the full target folder structure (including packages not yet
built, owned by later phases).

## Environment Notes

- Vanilla `pygame` has no prebuilt wheel for CPython 3.14 and fails to build from
  source without system SDL dev headers. This project uses **`pygame-ce`** instead —
  an actively maintained, drop-in-compatible fork (`import pygame` is unchanged).
- Target hardware: an RTX-class laptop GPU with 4-8GB VRAM and 16GB system RAM — every
  model choice (YOLOv8**n**, modest LSTM/MLP sizes, DQN) is made with that ceiling in
  mind (see PROJECT.md §18/Environment Notes).

## Documentation

- [`PROJECT.md`](PROJECT.md) — architecture, diagrams, roadmap, contracts (canonical, kept current)
- [`docs/architecture/`](docs/architecture/) — per-phase design notes
  ([Phase 1](docs/architecture/phase1-foundation.md),
  [Phase 2 — mission loop](docs/architecture/phase2-simulation.md),
  [Phase 2 — dynamic world & sensors](docs/architecture/phase2-dynamic-world-and-sensors.md))
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
  ([0001 config](docs/adr/0001-config-driven-yaml-dataclasses.md),
  [0002 two-tier navigation](docs/adr/0002-two-tier-navigation-and-command-center.md))
