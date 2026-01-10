"""CFO/Doppler impairment model."""

from __future__ import annotations

import numpy as np


def apply_cfo(x: np.ndarray, fs_hz: float, cfo_hz: float) -> np.ndarray:
    """Apply CFO/Doppler as a complex exponential rotation in baseband.

    In complex baseband, carrier frequency offset and Doppler shift are
    equivalent to a sample-wise phase rotation.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D complex array")
    if not np.iscomplexobj(x):
        raise ValueError("x must be a 1-D complex array")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")

    x = x.astype(np.complex128, copy=False)
    n = np.arange(x.size, dtype=np.float64)
    phasor = np.exp(1j * 2.0 * np.pi * cfo_hz * n / fs_hz).astype(
        np.complex128
    )
    return (x * phasor).astype(np.complex128)
