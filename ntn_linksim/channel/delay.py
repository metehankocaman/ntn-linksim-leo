"""Propagation delay / timing offset impairment model."""

from __future__ import annotations

import math

import numpy as np


def apply_integer_delay(x: np.ndarray, delay: int) -> np.ndarray:
    """Shift signal right by *delay* samples: zero-pad front, truncate tail.

    Length is preserved.  A delay of 0 returns a copy.

    Args:
        x: 1-D complex signal.
        delay: Non-negative integer sample delay.

    Returns:
        Delayed signal (same length as *x*), complex128.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if delay == 0:
        return x.copy()
    if delay >= x.size:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[delay:] = x[: x.size - delay]
    return out


def apply_fractional_delay(x: np.ndarray, frac_delay: float) -> np.ndarray:
    """Apply a fractional sample delay via frequency-domain linear phase.

    Multiplies the spectrum by ``exp(-j*2*pi*k*frac_delay/N)`` where *k* is
    the DFT bin index and *N* is the signal length.

    Args:
        x: 1-D complex signal.
        frac_delay: Fractional part of the delay in samples (|frac_delay| < 1).

    Returns:
        Delayed signal (same length), complex128.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if abs(frac_delay) < 1e-12:
        return x.copy()

    n = x.size
    X = np.fft.fft(x)
    k = np.arange(n, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * frac_delay * k / n).astype(np.complex128)
    return np.fft.ifft(X * phase).astype(np.complex128)


def apply_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a (possibly fractional) sample delay to *x*.

    Splits into integer + fractional parts and applies both.

    Args:
        x: 1-D complex signal.
        delay_samples: Total delay in samples (must be >= 0).

    Returns:
        Delayed signal (same length), complex128.

    Raises:
        ValueError: If *delay_samples* < 0.
    """
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")

    int_delay = int(math.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    out = apply_integer_delay(x, int_delay)
    if abs(frac_delay) > 1e-12:
        out = apply_fractional_delay(out, frac_delay)
    return out
