"""Concrete adapters for the perception ports.

Phase 3 fills :class:`~sentry_ai.interfaces.perception.IVisionDetector` with
a fine-tuned YOLOv8n; Phase 4 fills
:class:`~sentry_ai.interfaces.perception.IDenoiser` with a convolutional
autoencoder.

Everything here works in **image space only**. A detector returns boxes in
frame pixels and has no idea what a map tile is — turning a box back into a
world coordinate is
:meth:`~sentry_ai.sensors.frame.CameraFrame.world_position_of`'s job, and
merging those coordinates into a belief about the city is the command
center's. Keeping that boundary sharp is what lets the detector be swapped,
retrained, or scored in isolation.
"""
