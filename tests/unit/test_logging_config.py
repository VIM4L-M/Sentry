"""Unit tests for sentry_ai.common.logging_config."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.common.logging_config import get_logger, setup_logging


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("sentry_ai.some.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "sentry_ai.some.module"


def test_setup_logging_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        setup_logging(tmp_path / "missing.yaml")


def test_setup_logging_invalid_yaml_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("version: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        setup_logging(bad_file)


def test_setup_logging_non_mapping_raises(tmp_path: Path) -> None:
    list_file = tmp_path / "list.yaml"
    list_file.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        setup_logging(list_file)


def test_setup_logging_missing_required_key_raises(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("version: 1\n", encoding="utf-8")  # no 'handlers'
    with pytest.raises(ConfigurationError):
        setup_logging(incomplete)


def test_setup_logging_applies_valid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "logging.yaml"
    config_file.write_text(
        """
version: 1
disable_existing_loggers: false
formatters:
  standard:
    format: "%(levelname)s:%(name)s:%(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard
loggers:
  sentry_ai_test:
    level: DEBUG
    handlers: [console]
    propagate: false
""",
        encoding="utf-8",
    )
    setup_logging(config_file)
    logger = get_logger("sentry_ai_test")
    assert logger.level == logging.DEBUG


def test_real_logging_config_applies_cleanly(project_root: Path) -> None:
    """The actual configs/logging.yaml shipped in the repo must be valid."""
    setup_logging(project_root / "configs" / "logging.yaml")
    logger = get_logger("sentry_ai")
    assert logger.level == logging.INFO
