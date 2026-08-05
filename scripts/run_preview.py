#!/usr/bin/env python3
"""Composition root: load config, build the disaster city map, render it.

This is Phase 1's end-to-end sanity check — it wires a ``ConfigLoader``,
the ``CityMap`` domain aggregate, a ``Theme``, and the ``PreviewApp``
together the same way every later phase's entry point will, just with a
static map instead of a running simulation.

Usage:
    python scripts/run_preview.py [--config configs/app.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.app import PreviewApp
from sentry_ai.rendering.theme import Theme

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the SENTRY AI disaster city map.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to app.yaml, relative to the project root (default: configs/app.yaml).",
    )
    args = parser.parse_args()

    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app_config = loader.load_app_config(args.config)
    setup_logging(app_config.logging_config_path)

    logger = get_logger(__name__)
    logger.info("Loading map from %s", app_config.map_config_path)
    map_data = loader.load_yaml(app_config.map_config_path)
    city_map = CityMap.from_config(map_data)
    logger.info(
        "Map loaded: %dx%d, %d victim(s), %d fire(s), %d obstacle(s)",
        city_map.width,
        city_map.height,
        len(city_map.victims),
        len(city_map.fires),
        len(city_map.obstacles),
    )

    theme = Theme.from_config(loader, app_config.render.palette_config_path)
    app = PreviewApp(city_map=city_map, render_config=app_config.render, theme=theme)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
