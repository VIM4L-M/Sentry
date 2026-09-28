"""Fetching OpenStreetMap data: geocoding and the Overpass API.

Both public OSM services ask for an identifying User-Agent and gentle use;
every response is cached to disk, so re-running an import makes no request.
Data is © OpenStreetMap contributors, licensed under the ODbL — the
generated map file carries that attribution.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger

logger = get_logger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
#: Public Overpass servers, tried in order; the main one is often busy.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
#: Rounds over all servers, and the wait between them, when every one is busy.
_ATTEMPTS = 3
_BACKOFF_SECONDS = 20.0

USER_AGENT = "sentry-ai-student-project/0.1 (educational disaster-response simulation)"


@dataclass(frozen=True)
class BoundingBox:
    """A latitude/longitude rectangle."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (self.south < self.north and self.west < self.east):
            raise ValueError(f"degenerate bounding box {self}")


class OsmClient:
    """Geocodes places and downloads the ways inside a bounding box."""

    def __init__(self, cache_dir: Path, timeout_seconds: float = 60.0) -> None:
        """Create a client that caches every response under ``cache_dir``."""
        self._cache_dir = cache_dir
        self._timeout = timeout_seconds

    def geocode(self, place: str) -> tuple[float, float]:
        """The ``(lat, lon)`` of the best match for ``place``.

        Raises:
            AssetNotFoundError: If nothing matches.
        """
        query = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
        results = self._cached(f"geocode:{place}", lambda: self._get(f"{NOMINATIM_URL}?{query}"))
        if not results:
            raise AssetNotFoundError(f"OpenStreetMap has no place matching {place!r}")
        return float(results[0]["lat"]), float(results[0]["lon"])

    def ways(self, box: BoundingBox) -> list[dict[str, Any]]:
        """Every road, building, park and hospital way in ``box``, with geometry."""
        area = f"({box.south},{box.west},{box.north},{box.east})"
        query = (
            f"[out:json][timeout:{int(self._timeout)}];("
            f'way["highway"]{area};way["building"]{area};'
            f'way["leisure"="park"]{area};way["landuse"="grass"]{area};'
            f'way["amenity"="hospital"]{area};node["amenity"="hospital"]{area};'
            ");out geom;"
        )
        payload = urllib.parse.urlencode({"data": query}).encode()
        response = self._cached(f"overpass:{query}", lambda: self._post_any(payload))
        return list(response.get("elements", []))

    def _cached(self, key: str, fetch: Any) -> Any:
        path = self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        logger.info("Fetching from OpenStreetMap: %s", key[:80])
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _get(self, url: str) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read())

    def _post_any(self, payload: bytes) -> Any:
        """POST to each Overpass server in turn until one answers, retrying with backoff.

        Public servers answer 504 when busy; waiting a little usually clears it.

        Raises:
            OSError: The last server's error, if none answers.
        """
        error: OSError | None = None
        for attempt in range(_ATTEMPTS):
            for url in OVERPASS_URLS:
                try:
                    return self._post(url, payload)
                except OSError as exc:  # HTTPError and URLError are OSErrors
                    logger.warning("Overpass server %s failed: %s", url, exc)
                    error = exc
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        assert error is not None
        raise error

    def _post(self, url: str, payload: bytes) -> Any:
        request = urllib.request.Request(url, data=payload, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read())
