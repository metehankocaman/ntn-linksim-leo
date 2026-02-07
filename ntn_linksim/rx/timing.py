"""Timing offset estimation and compensation utilities."""

from __future__ import annotations

import numpy as np


def estimate_timing_offset_cp(
    rx: np.ndarray,
    n_fft: int,
    cp_len: int,
    n_symbols: int,
) -> int:
    """Estimate integer timing offset using CP sliding correlation.

    For each candidate offset *d*, the metric sums the correlation between
    each symbol's CP region and the corresponding tail across all symbols.
    The offset that maximises |metric| is returned.

    The search range is ``[0, 2*cp_len]`` — enough to cover delays within
    and slightly beyond the CP length.

    Args:
        rx: 1-D complex received samples.
        n_fft: FFT size.
        cp_len: Cyclic prefix length in samples.
        n_symbols: Number of OFDM symbols.

    Returns:
        Estimated integer sample delay (>= 0).
    """
    rx = np.asarray(rx, dtype=np.complex128)
    if rx.ndim != 1:
        raise ValueError("rx must be a 1-D complex array")
    if not np.iscomplexobj(rx):
        raise ValueError("rx must be complex")
    if n_fft <= 0 or cp_len <= 0 or n_symbols <= 0:
        raise ValueError("n_fft, cp_len, n_symbols must be positive")

    sym_len = n_fft + cp_len
    max_delay = min(2 * cp_len, rx.size // sym_len - 1) if rx.size > sym_len else 0

    best_d = 0
    best_metric = -1.0

    for d in range(max_delay + 1):
        metric = 0.0 + 0.0j
        for s in range(n_symbols):
            start = d + s * sym_len
            if start + sym_len > rx.size:
                break
            cp_part = rx[start : start + cp_len]
            tail_part = rx[start + n_fft : start + sym_len]
            metric += np.sum(tail_part * np.conjugate(cp_part))
        mag = abs(metric)
        if mag > best_metric:
            best_metric = mag
            best_d = d

    return best_d


def compensate_integer_delay(x: np.ndarray, delay: int) -> np.ndarray:
    """Shift signal left by *delay* samples to undo a timing offset.

    Samples shifted past the beginning are discarded; the tail is zero-padded
    so the output length matches the input.

    Args:
        x: 1-D complex signal.
        delay: Non-negative integer delay to compensate.

    Returns:
        Re-aligned signal (same length), complex128.
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
    out[: x.size - delay] = x[delay:]
    return out
