"""OFDM waveform helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OfdmParams:
    """OFDM parameter bundle."""

    n_fft: int
    n_used: int
    cp_len: int
    n_symbols: int

    def validate(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.n_used <= 0 or self.n_used >= self.n_fft:
            raise ValueError("n_used must be in (0, n_fft)")
        if self.n_used % 2 != 0:
            raise ValueError("n_used must be even")
        if self.cp_len < 0 or self.cp_len >= self.n_fft:
            raise ValueError("cp_len must be in [0, n_fft)")
        if self.n_symbols <= 0:
            raise ValueError("n_symbols must be positive")


def used_subcarrier_indices(n_fft: int, n_used: int) -> np.ndarray:
    """Return FFT bin indices for used subcarriers [-K..-1, +1..+K]."""
    if n_used % 2 != 0:
        raise ValueError("n_used must be even")
    if n_used >= n_fft:
        raise ValueError("n_used must be less than n_fft")
    k = n_used // 2
    neg = np.arange(n_fft - k, n_fft)
    pos = np.arange(1, k + 1)
    return np.concatenate([neg, pos])


def tx_grid(symbols: np.ndarray, params: OfdmParams) -> np.ndarray:
    """Map QPSK symbols into an OFDM frequency grid."""
    params.validate()
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.shape != (params.n_symbols, params.n_used):
        raise ValueError("symbols shape must be (n_symbols, n_used)")
    grid = np.zeros((params.n_symbols, params.n_fft), dtype=np.complex128)
    idx = used_subcarrier_indices(params.n_fft, params.n_used)
    grid[:, idx] = symbols
    return grid


def ifft_symbols(grid: np.ndarray) -> np.ndarray:
    """IFFT across subcarriers to generate time-domain symbols."""
    grid = np.asarray(grid, dtype=np.complex128)
    return np.fft.ifft(grid, axis=1).astype(np.complex128)


def add_cp(time_symbols: np.ndarray, cp_len: int) -> np.ndarray:
    """Add cyclic prefix to each OFDM symbol."""
    time_symbols = np.asarray(time_symbols, dtype=np.complex128)
    if cp_len < 0 or cp_len >= time_symbols.shape[1]:
        raise ValueError("cp_len must be in [0, n_fft)")
    if cp_len == 0:
        return time_symbols
    cp = time_symbols[:, -cp_len:]
    return np.concatenate([cp, time_symbols], axis=1)


def serialize_symbols(symbols_with_cp: np.ndarray) -> np.ndarray:
    """Serialize a 2-D symbol array into 1-D samples."""
    symbols_with_cp = np.asarray(symbols_with_cp, dtype=np.complex128)
    return symbols_with_cp.reshape(-1)


def deserialize_symbols(samples: np.ndarray, params: OfdmParams) -> np.ndarray:
    """Reshape 1-D samples into OFDM symbols with CP."""
    params.validate()
    samples = np.asarray(samples, dtype=np.complex128)
    sym_len = params.n_fft + params.cp_len
    expected = params.n_symbols * sym_len
    if samples.size != expected:
        raise ValueError("sample length does not match OFDM params")
    return samples.reshape(params.n_symbols, sym_len)


def remove_cp(symbols_with_cp: np.ndarray, cp_len: int) -> np.ndarray:
    """Remove cyclic prefix from each OFDM symbol."""
    symbols_with_cp = np.asarray(symbols_with_cp, dtype=np.complex128)
    if cp_len < 0 or cp_len >= symbols_with_cp.shape[1]:
        raise ValueError("cp_len must be in [0, n_fft)")
    return symbols_with_cp[:, cp_len:]


def fft_symbols(time_symbols: np.ndarray) -> np.ndarray:
    """FFT across time to recover frequency-domain grid."""
    time_symbols = np.asarray(time_symbols, dtype=np.complex128)
    return np.fft.fft(time_symbols, axis=1).astype(np.complex128)


def extract_used(grid: np.ndarray, params: OfdmParams) -> np.ndarray:
    """Extract used subcarriers from the frequency grid."""
    params.validate()
    grid = np.asarray(grid, dtype=np.complex128)
    if grid.shape != (params.n_symbols, params.n_fft):
        raise ValueError("grid shape must be (n_symbols, n_fft)")
    idx = used_subcarrier_indices(params.n_fft, params.n_used)
    return grid[:, idx]
