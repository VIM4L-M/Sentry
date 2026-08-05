"""Project-wide exception hierarchy.

Every SENTRY AI error inherits from :class:`SentryAIError` so callers can
catch the whole family with one clause when that's appropriate, while still
being able to catch a specific failure mode (a bad config file vs. an
invalid map layout) when they need to react differently.
"""

from __future__ import annotations


class SentryAIError(Exception):
    """Base class for every exception raised by SENTRY AI code.

    Third-party exceptions (``yaml.YAMLError``, ``pygame.error``, ...) are
    caught at the boundary that uses that library and re-raised as one of
    this hierarchy's subclasses, so calling code never has to know which
    external library produced a failure.
    """


class ConfigurationError(SentryAIError):
    """A configuration file is missing, unreadable, or malformed YAML."""


class ConfigValidationError(SentryAIError):
    """A configuration file parsed correctly but failed schema validation.

    Raised for problems like a negative tile size, an out-of-range battery
    limit, or a required field left out of a YAML document.
    """


class AssetNotFoundError(SentryAIError):
    """A referenced file (config, map, model weight, ...) does not exist."""


class DomainValidationError(SentryAIError):
    """A domain entity or aggregate was constructed in an invalid state.

    Examples: a :class:`~sentry_ai.domain.entities.Position` outside the map
    bounds, two entities occupying the same blocking tile, or a
    :class:`~sentry_ai.domain.map.CityMap` with no vehicle start position.
    """
