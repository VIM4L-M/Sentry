"""Synthetic cameras: the sensor layer every later AI phase consumes.

Nothing downstream of Phase 2 reads the :class:`~sentry_ai.domain.map.CityMap`
directly. The detector, the denoiser, and the local controller all see
*images* — which means the simulation has to produce them, together with
the ground truth needed to train against.

That is what this package is:

* :mod:`~sentry_ai.sensors.camera` — where a camera is pointed, and the
  projection between frame pixels and world tiles that turns a detection
  back into a map coordinate.
* :mod:`~sentry_ai.sensors.rasterizer` — the city painted into an RGB
  array, with a ground-truth ``Detection`` for every victim, fire, and
  piece of debris in shot.
* :mod:`~sentry_ai.sensors.degradation` — smoke, blur, and sensor noise.
  Corrupting a clean frame is what produces the (noisy, clean) pairs the
  Unit IV autoencoder trains on.
* :mod:`~sentry_ai.sensors.rig` — the CCTV network plus the vehicle's
  onboard camera, captured together.
"""
