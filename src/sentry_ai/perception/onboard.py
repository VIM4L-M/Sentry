"""What the vehicle's own camera sees, as map-space sightings (Phase 7).

The CCTV pipeline (``grid_source.py``) turns four fixed cameras into the
command center's map. This is the same chain for the one camera the vehicle
carries — capture, corrupt, optionally denoise, detect, project — but its
output is not a map. It is a short list of
:class:`~sentry_ai.interfaces.perception.WorldDetection` s around the vehicle,
handed straight to decision fusion every tick.

The difference matters because of timing. The command center's map can lag
reality; the onboard camera cannot, because it is looking at the street the
vehicle is on right now.

A sensing object is attached to one mission's city with :meth:`attach`
after the mission is built — the controller that uses it has to exist
before the mission does, since the mission is built around its controller.
"""

from __future__ import annotations

from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.perception import IDenoiser, WorldDetection
from sentry_ai.perception.grid_source import IFrameObserver
from sentry_ai.perception.merger import DetectionMerger
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.rig import SensorRig


class OnboardSensing:
    """Onboard camera -> degrader -> [denoiser] -> observer -> map-space sightings."""

    def __init__(
        self,
        rig: SensorRig,
        observer: IFrameObserver,
        degrader: FrameDegrader | None = None,
        denoiser: IDenoiser | None = None,
    ) -> None:
        """Compose the chain.

        Args:
            rig: Supplies the onboard camera.
            observer: The detector (``ModelObserver``) or the answer key
                (``GroundTruthObserver``) — the same control condition the
                CCTV pipeline uses.
            degrader: Smoke, blur and noise; ``None`` for clean frames.
            denoiser: Phase 4's autoencoder, if the detector expects it.
        """
        self._rig = rig
        self._observer = observer
        self._degrader = degrader
        self._denoiser = denoiser
        self._merger = DetectionMerger()
        self._city_map: CityMap | None = None

    def attach(self, city_map: CityMap) -> None:
        """Point the camera at a mission's city."""
        self._city_map = city_map

    def sightings(self) -> list[WorldDetection]:
        """What the onboard camera sees this instant, projected onto map tiles.

        Raises:
            RuntimeError: Before :meth:`attach`.
        """
        if self._city_map is None:
            raise RuntimeError("attach() a city before asking what the camera sees")
        frame = self._rig.capture_onboard(self._city_map)
        if self._degrader is not None:
            frame = self._degrader.degrade(frame)
        if self._denoiser is not None:
            frame = frame.with_pixels(self._denoiser.denoise(frame.pixels))
        return self._merger.merge(self._observer.observe([frame]))
