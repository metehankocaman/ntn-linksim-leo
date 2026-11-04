"""Frequency-selective multi-tap fading channel model.

Generates an L-tap channel impulse response per OFDM symbol with an
exponential power delay profile.  The CP absorbs ISI when L ≤ cp_len.

When combined with Rician fading, tap 0 gets the LoS component;
taps 1..L-1 are pure Rayleigh scatter.
"""

from __future__ import annotations

import numpy as np


def exponential_pdp(n_taps: int, tau_rms: float) -> np.ndarray:
    """Normalized exponential power delay profile.

    ``p(l) = exp(-l / tau_rms) / sum``, sums to 1.0.

    Args:
        n_taps: Number of channel taps (≥ 1).
        tau_rms: RMS delay spread in tap units (> 0).

    Returns:
        1-D float64 array of length *n_taps*, sums to 1.0.

    Raises:
        ValueError: If *n_taps* < 1 or *tau_rms* ≤ 0.
    """
    if n_taps < 1:
        raise ValueError("n_taps must be >= 1")
    if tau_rms <= 0:
        raise ValueError("tau_rms must be > 0")

    taps = np.arange(n_taps, dtype=np.float64)
    pdp = np.exp(-taps / tau_rms)
    pdp /= pdp.sum()
    return pdp


def generate_channel_taps(
    n_symbols: int,
    n_taps: int,
    tau_rms: float,
    rng: np.random.Generator,
    *,
    rician_k_db: float | None = None,
) -> np.ndarray:
    """Draw i.i.d. multi-tap channel coefficients per symbol.

    Returns ``(n_symbols, n_taps)`` complex128.  The scatter component is
    drawn first (fixed RNG consumption regardless of Rician flag), then
    tap 0 is post-processed with a LoS component when *rician_k_db* is set.

    Total average power is 1.0 (unit power channel).

    Args:
        n_symbols: Number of OFDM symbols.
        n_taps: Number of channel taps.
        tau_rms: RMS delay spread (passed to :func:`exponential_pdp`).
        rng: NumPy Generator for reproducibility.
        rician_k_db: If set, add LoS component to tap 0.

    Returns:
        Complex128 array of shape ``(n_symbols, n_taps)``.
    """
    pdp = exponential_pdp(n_taps, tau_rms)

    # Draw all scatter at once — fixed RNG consumption
    scatter_real = rng.standard_normal((n_symbols, n_taps))
    scatter_imag = rng.standard_normal((n_symbols, n_taps))
    scatter = (scatter_real + 1j * scatter_imag).astype(np.complex128) / np.sqrt(2.0)

    # Scale each tap by sqrt(pdp) so E[|h_l|^2] = pdp[l]
    h = scatter * np.sqrt(pdp)[np.newaxis, :]

    if rician_k_db is not None:
        k_lin = 10.0 ** (rician_k_db / 10.0)
        p0 = pdp[0]
        # LoS amplitude for tap 0 (deterministic part)
        los_amp = np.sqrt(k_lin / (k_lin + 1.0) * p0)
        # Scale down scatter for tap 0 so scatter + LoS gives p0 total power
        nlos_scale = np.sqrt(1.0 / (k_lin + 1.0))
        h[:, 0] = los_amp + scatter[:, 0] * np.sqrt(p0) * nlos_scale

    return h


def channel_freq_response(h_taps: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute per-symbol frequency response from tap coefficients.

    Args:
        h_taps: Complex array of shape ``(n_symbols, n_taps)``.
        n_fft: FFT size (number of subcarriers).

    Returns:
        Complex128 array of shape ``(n_symbols, n_fft)``.
    """
    h_taps = np.asarray(h_taps, dtype=np.complex128)
    if h_taps.ndim != 2:
        raise ValueError("h_taps must be 2-D")
    return np.fft.fft(h_taps, n=n_fft, axis=1).astype(np.complex128)


def apply_multipath_channel(
    tx_with_cp: np.ndarray,
    h_taps: np.ndarray,
) -> np.ndarray:
    """Apply per-symbol multi-tap fading via convolution.

    Each symbol row is convolved with its tap vector and truncated to the
    original symbol length (the CP absorbs the tail).

    Args:
        tx_with_cp: 2-D complex array ``(n_symbols, symbol_len)``.
        h_taps: 2-D complex array ``(n_symbols, n_taps)``.

    Returns:
        Faded signal, same shape as *tx_with_cp*.

    Raises:
        ValueError: If shapes are incompatible, inputs are not 2-D,
            or not complex.
    """
    tx_with_cp = np.asarray(tx_with_cp)
    h_taps = np.asarray(h_taps)

    if tx_with_cp.ndim != 2:
        raise ValueError("tx_with_cp must be 2-D")
    if h_taps.ndim != 2:
        raise ValueError("h_taps must be 2-D")
    if not np.iscomplexobj(tx_with_cp):
        raise ValueError("tx_with_cp must be complex")
    if not np.iscomplexobj(h_taps):
        raise ValueError("h_taps must be complex")

    n_symbols, symbol_len = tx_with_cp.shape
    n_taps = h_taps.shape[1]

    if h_taps.shape[0] != n_symbols:
        raise ValueError("h_taps and tx_with_cp must have the same number of symbols")
    if n_taps > symbol_len:
        raise ValueError("n_taps must be <= symbol_len")

    tx_with_cp = tx_with_cp.astype(np.complex128, copy=False)
    h_taps = h_taps.astype(np.complex128, copy=False)

    out = np.empty_like(tx_with_cp)
    for s in range(n_symbols):
        conv = np.convolve(tx_with_cp[s], h_taps[s], mode="full")
        out[s] = conv[:symbol_len]
    return out
