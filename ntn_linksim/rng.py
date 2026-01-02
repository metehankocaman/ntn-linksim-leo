"""Seeded RNG helpers."""

from __future__ import annotations

import numpy as np


def seeded_rng(seed: int) -> np.random.Generator:
    """Return a deterministic NumPy RNG for a given seed."""
    return np.random.default_rng(int(seed))
