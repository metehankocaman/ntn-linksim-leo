"""Tests for CP-based timing offset estimation."""

import numpy as np

from ntn_linksim.channel.delay import apply_integer_delay
from ntn_linksim.rx.timing import estimate_timing_offset_cp
from ntn_linksim.waveform.modulation import qpsk_mod
from ntn_linksim.waveform.ofdm import (
    OfdmParams,
    add_cp,
    ifft_symbols,
    serialize_symbols,
    tx_grid,
)


def _make_tx_samples(
    seed: int = 0, n_symbols: int = 10
) -> tuple[np.ndarray, OfdmParams]:
    """Generate clean serialized TX samples for testing."""
    params = OfdmParams(n_fft=64, n_used=52, cp_len=16, n_symbols=n_symbols)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=params.n_used * params.n_symbols * 2, dtype=np.int8)
    symbols = qpsk_mod(bits).reshape(params.n_symbols, params.n_used)
    grid = tx_grid(symbols, params)
    time_symbols = ifft_symbols(grid)
    tx_with_cp = add_cp(time_symbols, params.cp_len)
    return serialize_symbols(tx_with_cp), params


def test_no_delay_returns_zero() -> None:
    """With no delay applied, estimator should return 0."""
    tx, params = _make_tx_samples()
    offset = estimate_timing_offset_cp(
        tx, n_fft=params.n_fft, cp_len=params.cp_len, n_symbols=params.n_symbols
    )
    assert offset == 0


def test_known_delay_8() -> None:
    """Delay of 8 samples should be estimated correctly (±1 tolerance)."""
    tx, params = _make_tx_samples(seed=1, n_symbols=20)
    rx = apply_integer_delay(tx, 8)
    offset = estimate_timing_offset_cp(
        rx, n_fft=params.n_fft, cp_len=params.cp_len, n_symbols=params.n_symbols
    )
    assert abs(offset - 8) <= 1, f"expected ~8, got {offset}"


def test_known_delay_4() -> None:
    """Delay of 4 samples should be estimated correctly (±1 tolerance)."""
    tx, params = _make_tx_samples(seed=2, n_symbols=20)
    rx = apply_integer_delay(tx, 4)
    offset = estimate_timing_offset_cp(
        rx, n_fft=params.n_fft, cp_len=params.cp_len, n_symbols=params.n_symbols
    )
    assert abs(offset - 4) <= 1, f"expected ~4, got {offset}"
