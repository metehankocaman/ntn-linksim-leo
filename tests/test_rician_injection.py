"""Unit tests for apply_rician_fading()."""

import numpy as np
import pytest

from ntn_linksim.channel.rician import apply_rician_fading


def _make_tx(n_symbols: int = 10, symbol_len: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_symbols, symbol_len))
            + 1j * rng.standard_normal((n_symbols, symbol_len))).astype(np.complex128)


def test_rician_output_shape() -> None:
    """Output shape equals input shape."""
    tx = _make_tx(n_symbols=8, symbol_len=80)
    rng = np.random.default_rng(42)
    out = apply_rician_fading(tx, rician_k_db=10.0, rng=rng)
    assert out.shape == tx.shape


def test_rician_high_k_near_identity() -> None:
    """K=40 dB -> LoS dominant, output ~= input."""
    tx = _make_tx(n_symbols=20, symbol_len=80)
    rng = np.random.default_rng(42)
    out = apply_rician_fading(tx, rician_k_db=40.0, rng=rng)
    np.testing.assert_allclose(np.abs(out), np.abs(tx), rtol=0.02)


def test_rician_low_k_causes_variation() -> None:
    """K=0 dB -> significant amplitude spread across symbols."""
    tx = _make_tx(n_symbols=100, symbol_len=80)
    rng = np.random.default_rng(42)
    out = apply_rician_fading(tx, rician_k_db=0.0, rng=rng)
    # Per-symbol power should vary
    powers_in = np.mean(np.abs(tx) ** 2, axis=1)
    powers_out = np.mean(np.abs(out) ** 2, axis=1)
    ratios = powers_out / powers_in
    # At K=0 dB the gain varies substantially
    assert np.std(ratios) > 0.1


def test_rician_deterministic() -> None:
    """Same RNG seed -> identical output."""
    tx = _make_tx(n_symbols=10, symbol_len=80)
    out1 = apply_rician_fading(tx, rician_k_db=5.0, rng=np.random.default_rng(99))
    out2 = apply_rician_fading(tx, rician_k_db=5.0, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)


def test_rician_unit_power() -> None:
    """E[|h|^2] ~ 1.0 over many symbols (tolerance 5%)."""
    n_symbols = 10000
    tx = np.ones((n_symbols, 80), dtype=np.complex128)
    rng = np.random.default_rng(7)
    out = apply_rician_fading(tx, rician_k_db=3.0, rng=rng)
    # Each symbol row is multiplied by h[s], so |out[s,0]|^2 = |h[s]|^2
    h_sq = np.abs(out[:, 0]) ** 2
    mean_h_sq = np.mean(h_sq)
    assert abs(mean_h_sq - 1.0) < 0.05, f"E[|h|^2] = {mean_h_sq}, expected ~1.0"


def test_rician_1d_raises() -> None:
    """ValueError on 1-D input."""
    x = np.ones(100, dtype=np.complex128)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2-D"):
        apply_rician_fading(x, rician_k_db=10.0, rng=rng)
