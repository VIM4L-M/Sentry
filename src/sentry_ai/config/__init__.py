"""Typed configuration schema and loader.

Nothing outside this package reads a YAML config file directly — every
module that needs a tunable value gets it from a dataclass produced by
:func:`sentry_ai.config.loader.load_app_config`. See PROJECT.md §13.
"""
