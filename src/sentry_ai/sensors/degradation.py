"""Smoke, blur, and sensor noise applied to a clean camera frame.

This module exists to make Unit IV's denoising autoencoder trainable. An
autoencoder that removes smoke needs (corrupted, clean) *pairs*, and the
only way to get a perfectly aligned pair is to corrupt a known-clean frame
rather than to try to clean a naturally dirty one.

Corruption is applied in the order a real camera suffers it: the scene is
obscured by smoke and defocused first (optics), then quantised by an
imperfect sensor (electronics). Applying noise before blur would smear the
noise, which is not what a real sensor does and would teach the autoencoder
the wrong inverse.

Nothing here touches a frame's annotations. Smoke changes what the camera
can see; it does not move the victim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sentry_ai.config.schema import DegradationConfig
from sentry_ai.sensors.frame import CameraFrame

#: Resolution of the smoke field before it is upsampled, as a fraction of
#: the frame. Low-frequency by construction: smoke drifts in large soft
#: banks, and per-pixel randomness would just be more noise.
_SMOKE_FIELD_DIVISOR = 8


class FrameDegrader:
    """Corrupts clean frames into the kind a camera in a fire actually returns."""

    def __init__(self, config: DegradationConfig, rng: np.random.Generator | None = None) -> None:
        """Create a degrader.

        Args:
            config: Smoke, blur, and noise strengths.
            rng: Generator for smoke placement and sensor noise. Pass a
                seeded one to make a dataset reproducible; the default is
                seeded from the OS, which is what you want when generating
                many independent samples of the same scene.
        """
        self._config = config
        self._rng = rng if rng is not None else np.random.default_rng()

    def degrade(self, frame: CameraFrame) -> CameraFrame:
        """Return a corrupted copy of ``frame``, annotations untouched."""
        return frame.with_pixels(self.degrade_pixels(frame.pixels))

    def degrade_pixels(self, pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Corrupt a bare ``(h, w, 3)`` image. The input is not modified.

        The same corruption as :meth:`degrade`, for callers that hold pixels
        without a camera — the autoencoder's training crops, which are cut
        out of a frame and so no longer match any camera's geometry.
        """
        corrupted = pixels.astype(np.float32)
        corrupted = self._blur(corrupted)
        corrupted = self._add_smoke(corrupted)
        corrupted = self._add_noise(corrupted)
        return np.clip(corrupted, 0, 255).astype(np.uint8)

    def degrade_pair(self, frame: CameraFrame) -> tuple[CameraFrame, CameraFrame]:
        """A ``(corrupted, clean)`` pair — the training sample for Unit IV."""
        return self.degrade(frame), frame

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _blur(self, pixels: NDArray[np.float32]) -> NDArray[np.float32]:
        """Defocus the frame with a separable box blur.

        Repeated 3-tap averaging approximates a Gaussian well enough at
        these radii and needs nothing beyond numpy — this project cannot
        take a SciPy dependency for one convolution.
        """
        radius = self._config.blur_radius
        blurred = pixels
        for _ in range(radius):
            blurred = _average_3tap(_average_3tap(blurred, axis=0), axis=1)
        return blurred

    def _add_smoke(self, pixels: NDArray[np.float32]) -> NDArray[np.float32]:
        """Blend the frame toward flat grey under a soft, drifting smoke field."""
        density = self._config.smoke_density
        if density <= 0.0:
            return pixels

        height, width = pixels.shape[0], pixels.shape[1]
        field = self._smoke_field(height, width) * density
        grey = np.full_like(pixels, float(self._config.smoke_grey))
        return np.asarray(pixels * (1.0 - field) + grey * field, dtype=np.float32)

    def _smoke_field(self, height: int, width: int) -> NDArray[np.float32]:
        """A soft 0-1 opacity field, low-frequency and smoothly varying."""
        coarse_h = max(2, height // _SMOKE_FIELD_DIVISOR)
        coarse_w = max(2, width // _SMOKE_FIELD_DIVISOR)
        coarse = self._rng.random((coarse_h, coarse_w), dtype=np.float32)

        upsampled = np.repeat(
            np.repeat(coarse, height // coarse_h + 1, axis=0),
            width // coarse_w + 1,
            axis=1,
        )[:height, :width]
        smoothed = _average_3tap(_average_3tap(upsampled, axis=0), axis=1)
        return smoothed[:, :, np.newaxis]

    def _add_noise(self, pixels: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add zero-mean Gaussian sensor noise."""
        if self._config.noise_std <= 0.0:
            return pixels
        noise = self._rng.normal(0.0, self._config.noise_std, size=pixels.shape)
        return pixels + noise.astype(np.float32)


def _average_3tap(values: NDArray[np.float32], axis: int) -> NDArray[np.float32]:
    """Average each element with its two neighbours along ``axis``.

    Edges repeat the border value, so the frame does not darken at its
    margins the way zero-padding would.
    """
    forward = np.roll(values, 1, axis=axis)
    backward = np.roll(values, -1, axis=axis)
    _copy_border(forward, axis, source=1, target=0)
    _copy_border(backward, axis, source=-2, target=-1)
    result: NDArray[np.float32] = (values + forward + backward) / 3.0
    return result


def _copy_border(values: NDArray[np.float32], axis: int, source: int, target: int) -> None:
    """Repair the wrapped edge left behind by ``np.roll``."""
    if axis == 0:
        values[target] = values[source]
    else:
        values[:, target] = values[:, source]
