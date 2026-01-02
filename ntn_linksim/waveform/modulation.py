"""QPSK modulation and hard-decision demodulation."""

from __future__ import annotations

import numpy as np


def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """Map bits to unit-power QPSK symbols.

    Bits are mapped as (b0, b1) -> (1-2*b0) + j(1-2*b1), normalized by sqrt(2).
    """
    bits = np.asarray(bits, dtype=np.int8)
    if bits.ndim != 1:
        raise ValueError("bits must be a 1-D array")
    if bits.size % 2 != 0:
        raise ValueError("bits length must be even for QPSK")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be 0 or 1")

    pairs = bits.reshape(-1, 2)
    re = 1 - 2 * pairs[:, 0]
    im = 1 - 2 * pairs[:, 1]
    symbols = (re + 1j * im) / np.sqrt(2.0)
    return symbols.astype(np.complex128)


def qpsk_demod_hard(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation returning bits."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    bits_re = (np.real(symbols) < 0).astype(np.int8)
    bits_im = (np.imag(symbols) < 0).astype(np.int8)
    bits = np.stack([bits_re, bits_im], axis=1).reshape(-1)
    return bits
