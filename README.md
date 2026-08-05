# SENTRY AI

**Autonomous Emergency Rescue Vehicle for Disaster Zones** — a simulation-based Applied
Deep Learning project. See [`PROJECT.md`](PROJECT.md) for the full architecture,
diagrams, phase roadmap, and API contracts. See [`CLAUDE.md`](CLAUDE.md) for the working
rules this codebase follows.

**Status:** Phase 1 (Foundation & Core Architecture) complete. See PROJECT.md §10-11 for
the phase breakdown and milestones.

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

Render the disaster city (Phase 1's walking skeleton — opens a window, no AI, no
movement yet):

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
src/sentry_ai/  The package: common, config, domain, interfaces, rendering, ...
scripts/     Composition roots / CLI entry points
tests/       unit / integration / e2e, mirroring src/
docs/        Phase design notes and ADRs
```

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
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
