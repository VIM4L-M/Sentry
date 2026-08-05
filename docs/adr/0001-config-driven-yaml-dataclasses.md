# ADR 0001: Configuration via YAML + typed dataclasses, loaded through one `ConfigLoader`

**Status:** Accepted
**Phase:** 1

## Context

CLAUDE.md mandates "configuration driven" behavior and forbids magic numbers and
hardcoded paths. The project also targets "industrial grade" structure evaluated as
production software, and needs configs for very different concerns over its lifetime:
window/render settings (Phase 1), simulation tuning (Phase 2), and five separate model
hyperparameter sets (Phases 3-7).

Options considered:

1. **Bare YAML dicts passed around** — fast to write, but no validation, no
   autocomplete/type-checking, errors surface deep inside business logic instead of at
   load time.
2. **Pydantic models** — strong validation and parsing for free, but adds a dependency
   not in the originally specified tech stack, and its validation-error messages are
   harder to map to the project's own exception hierarchy.
3. **Plain dataclasses + a small hand-written loader, YAML via PyYAML** — no new
   dependency, validation logic is explicit and lives in `__post_init__` (easy to read,
   easy to test), and every config error becomes one of the project's own
   `ConfigurationError` / `ConfigValidationError` types.

## Decision

Use option 3. Every config file has a corresponding `@dataclass` in `config/schema.py`
(or, for domain-specific files like the map, a `from_config` classmethod on the owning
domain class). A single `ConfigLoader`, constructed with an explicit `project_root`
(dependency injection, never `Path.cwd()`), is the only code allowed to call
`yaml.safe_load` directly — see PROJECT.md §13.

## Consequences

- Adding a new config file means: add a dataclass, add a loader method (or a
  `from_config` on the domain class it feeds), add a default YAML file. No new
  third-party dependency per config surface.
- Validation errors are raised at load time, with a message naming the exact field —
  they cannot silently propagate as a `None` deep inside a render loop or training run.
- `ConfigLoader` being rooted at an explicit path (not `cwd()`) means scripts, tests, and
  (Phase 9) the Streamlit app can all load the same configs regardless of their working
  directory.
- If config validation needs grow substantially more complex (cross-field constraints,
  nested unions) in a later phase, this decision should be revisited in a new ADR rather
  than organically growing `__post_init__` methods past readability.
