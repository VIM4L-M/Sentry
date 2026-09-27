"""One call that seeds every random number generator a training run touches.

A run is only reproducible if *all* of them are seeded: Python's ``random``,
NumPy's legacy global generator, and Torch's CPU and CUDA generators. Missing
any one of them gives a run that is nearly the same every time, which is
worse than obviously different — it hides the fact that it is not.

Torch is optional here. Seeding is useful before any model exists (dataset
generation, for one), and importing this module must not pull in Torch.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed ``random``, NumPy, and — when installed — Torch (CPU and CUDA).

    Args:
        seed: Any non-negative integer.

    Raises:
        ValueError: If ``seed`` is negative, which NumPy would reject anyway
            but with a less helpful message.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    """The device to run on, resolving ``"auto"`` against what exists.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"``, or a device index.
            Anything but ``"auto"`` is returned unchanged.
    """
    if requested != "auto":
        return requested

    import torch  # noqa: PLC0415 - only needed to answer this question

    return "cuda" if torch.cuda.is_available() else "cpu"
