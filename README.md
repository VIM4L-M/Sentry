# SENTRY AI

**Autonomous Emergency Rescue Vehicle for Disaster Zones** — a simulation-based Applied
Deep Learning project. See [`PROJECT.md`](PROJECT.md) for the full architecture,
diagrams, phase roadmap, and API contracts. See [`CLAUDE.md`](CLAUDE.md) for the working
rules this codebase follows.

**Status:** Phase 1 complete. Phase 2 core complete — a full autonomous rescue mission
runs end to end (occupancy grid → A\* routing → mission control → live HUD). Dynamic
hazards and the CCTV sensor rig are the next increment. See PROJECT.md §10-11 for the
phase breakdown and milestones.

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
| arrows / `WASD` | drive (manual mode) |

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
  interfaces/  Ports (ABCs): perception, sequence, navigation, decision fusion
  navigation/  A* global route planner (classical, not learned)
  simulation/  Tick engine, mission state machine, vehicle physics
  rendering/   Pygame map renderer, HUD, keyboard input, mission window
scripts/     Composition roots / CLI entry points
tests/       unit / integration / e2e, mirroring src/
docs/        Phase design notes and ADRs
```

**How navigation is split:** A\* plans the route across the city from an occupancy
grid; a learned local controller (Phase 6 DQN, today a deterministic waypoint
follower) decides each tick's move. The reasoning is in
[`docs/adr/0002-two-tier-navigation-and-command-center.md`](docs/adr/0002-two-tier-navigation-and-command-center.md).

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
  ([Phase 1](docs/architecture/phase1-foundation.md), [Phase 2](docs/architecture/phase2-simulation.md))
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
  ([0001 config](docs/adr/0001-config-driven-yaml-dataclasses.md),
  [0002 two-tier navigation](docs/adr/0002-two-tier-navigation-and-command-center.md))
