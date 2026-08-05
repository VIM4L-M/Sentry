"""Structured logging setup.

SENTRY AI never calls ``print()`` for anything other than a CLI script's
final user-facing output — every library module logs through the standard
``logging`` module via :func:`get_logger`. The actual handlers/formatters
are configured once, from a YAML file, by :func:`setup_logging`.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.types import PathLike

_REQUIRED_KEYS = {"version", "handlers"}


def setup_logging(config_path: PathLike) -> None:
    """Configure the logging system from a ``logging.config.dictConfig`` YAML file.

    Args:
        config_path: Path to a YAML file whose contents follow the
            standard library's ``dictConfig`` schema (see
            ``configs/logging.yaml`` for the project's default).

    Raises:
        ConfigurationError: If the file is missing, is not valid YAML, does
            not describe a mapping, or is missing keys ``dictConfig``
            requires to do anything useful.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigurationError(f"Logging config not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Logging config is not valid YAML: {path}") from exc

    if not isinstance(config, dict):
        raise ConfigurationError(f"Logging config must be a YAML mapping: {path}")

    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ConfigurationError(
            f"Logging config {path} is missing required key(s): {sorted(missing)}"
        )

    try:
        logging.config.dictConfig(config)
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        raise ConfigurationError(f"Invalid logging configuration in {path}: {exc}") from exc


def get_logger(name: str) -> logging.Logger:
    """Return the module-scoped logger for ``name``.

    Thin wrapper around ``logging.getLogger`` so every module obtains its
    logger the same documented way, and so call sites are easy to grep for.
    Safe to call before :func:`setup_logging` — it will simply use the
    logging module's default configuration until setup runs.
    """
    return logging.getLogger(name)
