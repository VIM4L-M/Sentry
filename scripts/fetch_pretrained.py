#!/usr/bin/env python3
"""Downloads the pretrained checkpoints the training scripts transfer from.

Run once, before the first training run. Ultralytics will happily fetch
``yolov8n.pt`` itself the first time it is asked for, but doing it here
instead means a long training run has no hidden network dependency and
cannot die twenty minutes in because a download timed out.

Usage:
    python scripts/fetch_pretrained.py
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Checkpoint name -> download URL. Pinned to a release tag rather than
#: "latest" so a rebuild months from now transfers from the same weights.
CHECKPOINTS: dict[str, str] = {
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    ),
}

#: Anything smaller than this is an error page, not a checkpoint.
MIN_BYTES = 100_000

logger = get_logger(__name__)


def main() -> int:
    """Download every checkpoint that is not already present."""
    destination = PROJECT_ROOT / "models" / "pretrained"
    destination.mkdir(parents=True, exist_ok=True)

    for name, url in CHECKPOINTS.items():
        path = destination / name
        if path.is_file() and path.stat().st_size >= MIN_BYTES:
            print(f"{name:<16}already present ({path.stat().st_size / 1e6:.1f} MB)")
            continue
        _download(url, path)
        print(f"{name:<16}downloaded ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nCheckpoints in {destination}")
    return 0


def _download(url: str, path: Path) -> None:
    """Fetch one checkpoint, failing loudly rather than leaving a stub.

    Raises:
        AssetNotFoundError: If the download fails or returns something too
            small to be a real checkpoint — usually an error page.
    """
    logger.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, path)  # noqa: S310 - URL is a pinned constant
    except (urllib.error.URLError, OSError) as exc:
        raise AssetNotFoundError(f"Could not download {url}: {exc}") from exc

    if path.stat().st_size < MIN_BYTES:
        path.unlink(missing_ok=True)
        raise AssetNotFoundError(f"Download from {url} was too small to be a checkpoint")


if __name__ == "__main__":
    raise SystemExit(main())
