"""Dataset assembly, training loops, and metrics — everything offline.

Nothing in this package runs during a mission. It exists to turn the
simulation into training material and trained weights into files on disk;
the live pipeline reads those files through the ports in
:mod:`sentry_ai.interfaces` and never imports anything from here.
"""
