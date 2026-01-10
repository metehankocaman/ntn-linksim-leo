"""CFO estimation and compensation utilities."""

from __future__ import annotations

import numpy as np


def estimate_cfo_from_cp(
    rx: np.ndarray, n_fft: int, cp_len: int, fs_hz: float
) -> float:
    """Estimate CFO using CP correlation on a single OFDM symbol with CP.

    Uses tail * conj(cp) so positive CFO matches apply_cfo() convention.
    """
    rx = np.asarray(rx)
    if rx.ndim != 1:
        raise ValueError("rx must be a 1-D complex array")
    if not np.iscomplexobj(rx):
        raise ValueError("rx must be a 1-D complex array")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if cp_len <= 0:
        raise ValueError("cp_len must be positive")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    expected_len = n_fft + cp_len
    if rx.size != expected_len:
        raise ValueError("rx length must be n_fft + cp_len")

    rx = rx.astype(np.complex128, copy=False)
    p = np.sum(rx[n_fft : n_fft + cp_len] * np.conjugate(rx[0:cp_len]))
    eps_hat = np.angle(p) / n_fft
    return float(eps_hat * fs_hz / (2.0 * np.pi))


def compensate_cfo(x: np.ndarray, fs_hz: float, cfo_hz: float) -> np.ndarray:
    """Apply CFO compensation as a complex exponential derotation."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D complex array")
    if not np.iscomplexobj(x):
        raise ValueError("x must be a 1-D complex array")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")

    x = x.astype(np.complex128, copy=False)
    n = np.arange(x.size, dtype=np.float64)
    phasor = np.exp(-1j * 2.0 * np.pi * cfo_hz * n / fs_hz).astype(
        np.complex128
    )
    return (x * phasor).astype(np.complex128)
