"""Saving trained weights with a record of how they were made.

Every checkpoint is two files side by side:

* ``<name>.pt`` — whatever the model needs to rebuild itself, via
  ``torch.save``.
* ``<name>.json`` — human-readable metadata: a fingerprint of the config
  that produced it, its metrics, when it was written, and the Torch version.

The metadata is what makes a file in ``models/`` answerable months later:
*which config made this, and how good was it?* (PROJECT.md §14, step 7).
Its config fingerprint is a hash of the validated dataclass, so two runs
from the same config share one, and changing any hyperparameter changes it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import torch

from sentry_ai.common.exceptions import AssetNotFoundError


def config_fingerprint(config: Any) -> str:
    """A short, stable hash of a config dataclass.

    Paths are hashed as written, so the same config loaded from two
    checkouts in different directories gives two fingerprints — which is
    correct, since they point at different data. They are always written
    with forward slashes, so the same config on Windows and Linux shares
    one fingerprint.

    Raises:
        TypeError: If ``config`` is not a dataclass instance.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"config_fingerprint expects a dataclass instance, got {type(config)}")
    canonical = json.dumps(asdict(config), sort_keys=True, default=_portable)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def save_checkpoint(
    path: Path,
    payload: dict[str, Any],
    config: Any,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Write ``payload`` to ``path`` and its metadata beside it.

    Args:
        path: The ``.pt`` file to write. Parent directories are created.
        payload: What ``torch.save`` stores — typically a model's
            ``to_checkpoint()`` output.
        config: The validated config dataclass that produced the model.
        metrics: Numbers worth keeping with the weights.

    Returns:
        The metadata file's path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

    metadata = {
        "weights": path.name,
        "config_fingerprint": config_fingerprint(config),
        "config": json.loads(json.dumps(asdict(config), default=_portable)),
        "metrics": dict(metrics or {}),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }
    metadata_path = metadata_path_for(path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_metadata(path: Path) -> dict[str, Any]:
    """Read the metadata written alongside the checkpoint at ``path``.

    Raises:
        AssetNotFoundError: If there is no metadata file.
    """
    metadata_path = metadata_path_for(path)
    if not metadata_path.is_file():
        raise AssetNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    loaded: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return loaded


def metadata_path_for(path: Path) -> Path:
    """The metadata file that belongs to the checkpoint at ``path``."""
    return path.with_suffix(".json")


def _portable(value: Any) -> str:
    """JSON form for values json cannot encode: paths with ``/`` on every OS.

    ``str(Path)`` gives backslashes on Windows, which would make one config
    fingerprint two ways and write metadata that reads differently per OS.
    """
    if isinstance(value, PurePath):
        return value.as_posix()
    return str(value)
