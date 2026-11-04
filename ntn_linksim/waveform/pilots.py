"""Comb-type pilot insertion and extraction for OFDM."""

from __future__ import annotations

import numpy as np


def pilot_indices(n_used: int, pilot_spacing: int) -> np.ndarray:
    """Return indices of pilot subcarriers within the used subcarrier block.

    Args:
        n_used: Number of used subcarriers.
        pilot_spacing: Place a pilot every *pilot_spacing* subcarriers.

    Returns:
        Sorted 1-D int array of pilot positions, e.g. ``[0, 4, 8, ...]``.

    Raises:
        ValueError: If *pilot_spacing* is not in ``[1, n_used]``.
    """
    if pilot_spacing < 1 or pilot_spacing > n_used:
        raise ValueError(
            f"pilot_spacing must be in [1, {n_used}], got {pilot_spacing}"
        )
    return np.arange(0, n_used, pilot_spacing, dtype=np.intp)


def data_indices(n_used: int, pilot_spacing: int) -> np.ndarray:
    """Return indices of data subcarriers (complement of pilot indices).

    Args:
        n_used: Number of used subcarriers.
        pilot_spacing: Pilot spacing (same as :func:`pilot_indices`).

    Returns:
        Sorted 1-D int array of data positions.
    """
    p_idx = pilot_indices(n_used, pilot_spacing)
    all_idx = np.arange(n_used, dtype=np.intp)
    return np.setdiff1d(all_idx, p_idx)


def pilot_symbols(n_pilots: int) -> np.ndarray:
    """Return constant pilot symbols (all 1+0j).

    Args:
        n_pilots: Number of pilots.

    Returns:
        Complex128 array of ones with shape ``(n_pilots,)``.
    """
    return np.ones(n_pilots, dtype=np.complex128)


def insert_pilots(
    data_symbols: np.ndarray,
    n_used: int,
    pilot_spacing: int,
) -> np.ndarray:
    """Insert pilot symbols into each OFDM symbol row.

    Args:
        data_symbols: Data symbols, shape ``(n_sym, n_data)``.
        n_used: Total number of used subcarriers.
        pilot_spacing: Pilot spacing.

    Returns:
        Combined grid of shape ``(n_sym, n_used)`` with pilots and data.

    Raises:
        ValueError: If *data_symbols* has wrong number of data subcarriers.
    """
    data_symbols = np.asarray(data_symbols, dtype=np.complex128)
    p_idx = pilot_indices(n_used, pilot_spacing)
    d_idx = data_indices(n_used, pilot_spacing)

    if data_symbols.ndim != 2:
        raise ValueError("data_symbols must be 2-D")
    if data_symbols.shape[1] != len(d_idx):
        raise ValueError(
            f"data_symbols has {data_symbols.shape[1]} columns, "
            f"expected {len(d_idx)} data subcarriers"
        )

    n_sym = data_symbols.shape[0]
    grid = np.empty((n_sym, n_used), dtype=np.complex128)
    grid[:, p_idx] = pilot_symbols(len(p_idx))[np.newaxis, :]
    grid[:, d_idx] = data_symbols
    return grid


def extract_pilots(
    rx_used: np.ndarray,
    n_used: int,
    pilot_spacing: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split received used subcarriers into pilot and data parts.

    Args:
        rx_used: Received symbols on used subcarriers, shape ``(n_sym, n_used)``.
        n_used: Number of used subcarriers.
        pilot_spacing: Pilot spacing.

    Returns:
        ``(rx_pilots, rx_data)`` with shapes ``(n_sym, n_pilots)`` and
        ``(n_sym, n_data)`` respectively.

    Raises:
        ValueError: If *rx_used* shape doesn't match *n_used*.
    """
    rx_used = np.asarray(rx_used, dtype=np.complex128)
    if rx_used.ndim != 2:
        raise ValueError("rx_used must be 2-D")
    if rx_used.shape[1] != n_used:
        raise ValueError(
            f"rx_used has {rx_used.shape[1]} columns, expected {n_used}"
        )

    p_idx = pilot_indices(n_used, pilot_spacing)
    d_idx = data_indices(n_used, pilot_spacing)
    return rx_used[:, p_idx], rx_used[:, d_idx]
