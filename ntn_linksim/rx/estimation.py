"""LS channel estimation, interpolation, and zero-forcing equalization."""

from __future__ import annotations

import numpy as np


def ls_estimate_at_pilots(
    rx_pilots: np.ndarray,
    tx_pilots: np.ndarray,
) -> np.ndarray:
    """Least-squares channel estimate at pilot positions.

    ``H_ls[k] = Y[k] / X[k]`` for each pilot subcarrier.

    Args:
        rx_pilots: Received pilot symbols, shape ``(n_sym, n_pilots)``.
        tx_pilots: Transmitted pilot symbols, shape ``(n_pilots,)`` or
            ``(n_sym, n_pilots)``.

    Returns:
        Complex128 array of shape ``(n_sym, n_pilots)``.
    """
    rx_pilots = np.asarray(rx_pilots, dtype=np.complex128)
    tx_pilots = np.asarray(tx_pilots, dtype=np.complex128)
    if rx_pilots.ndim != 2:
        raise ValueError("rx_pilots must be 2-D")
    if tx_pilots.ndim == 1:
        tx_pilots = tx_pilots[np.newaxis, :]
    return rx_pilots / tx_pilots


def interpolate_ls(
    h_ls: np.ndarray,
    pilot_idx: np.ndarray,
    n_used: int,
) -> np.ndarray:
    """Interpolate LS estimates from pilot positions to all used subcarriers.

    Uses linear interpolation of real and imaginary parts independently.

    Args:
        h_ls: LS estimates at pilot positions, shape ``(n_sym, n_pilots)``.
        pilot_idx: Pilot subcarrier indices, shape ``(n_pilots,)``.
        n_used: Total number of used subcarriers.

    Returns:
        Complex128 array of shape ``(n_sym, n_used)``.
    """
    h_ls = np.asarray(h_ls, dtype=np.complex128)
    pilot_idx = np.asarray(pilot_idx)
    if h_ls.ndim != 2:
        raise ValueError("h_ls must be 2-D")
    if h_ls.shape[1] != len(pilot_idx):
        raise ValueError("h_ls columns must match len(pilot_idx)")

    x_all = np.arange(n_used, dtype=np.float64)
    x_pilot = pilot_idx.astype(np.float64)
    n_sym = h_ls.shape[0]
    out = np.empty((n_sym, n_used), dtype=np.complex128)

    for s in range(n_sym):
        re = np.interp(x_all, x_pilot, h_ls[s].real)
        im = np.interp(x_all, x_pilot, h_ls[s].imag)
        out[s] = re + 1j * im

    return out


def zf_equalize(
    rx_data: np.ndarray,
    h_est_data: np.ndarray,
) -> np.ndarray:
    """Zero-forcing equalization: divide received data by channel estimate.

    Args:
        rx_data: Received data symbols, shape ``(n_sym, n_data)``.
        h_est_data: Channel estimate at data positions, shape ``(n_sym, n_data)``.

    Returns:
        Equalized symbols, same shape as *rx_data*.
    """
    rx_data = np.asarray(rx_data, dtype=np.complex128)
    h_est_data = np.asarray(h_est_data, dtype=np.complex128)
    if rx_data.shape != h_est_data.shape:
        raise ValueError("rx_data and h_est_data must have the same shape")
    return rx_data / h_est_data
