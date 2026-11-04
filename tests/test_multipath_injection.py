"""Unit tests for multipath channel functions."""

from __future__ import annotations

import numpy as np
import pytest

from ntn_linksim.channel.multipath import (
    apply_multipath_channel,
    channel_freq_response,
    exponential_pdp,
    generate_channel_taps,
)

# ---------- exponential_pdp ----------


def test_exponential_pdp_sums_to_one() -> None:
    """PDP weights sum to 1.0 for various tap counts."""
    for n_taps in (1, 4, 8, 16):
        pdp = exponential_pdp(n_taps, tau_rms=3.0)
        assert pdp.shape == (n_taps,)
        np.testing.assert_allclose(pdp.sum(), 1.0)


def test_exponential_pdp_invalid_raises() -> None:
    """n_taps < 1 or tau_rms <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="n_taps"):
        exponential_pdp(0, tau_rms=3.0)
    with pytest.raises(ValueError, match="tau_rms"):
        exponential_pdp(8, tau_rms=0.0)
    with pytest.raises(ValueError, match="tau_rms"):
        exponential_pdp(8, tau_rms=-1.0)


# ---------- generate_channel_taps ----------


def test_generate_taps_shape() -> None:
    """Output shape is (n_symbols, n_taps)."""
    h = generate_channel_taps(20, 8, 3.0, np.random.default_rng(0))
    assert h.shape == (20, 8)
    assert h.dtype == np.complex128


def test_generate_taps_unit_power() -> None:
    """Mean total power across taps ≈ 1.0 (10k symbols, 5% tol)."""
    h = generate_channel_taps(10_000, 8, 3.0, np.random.default_rng(42))
    mean_power = np.mean(np.sum(np.abs(h) ** 2, axis=1))
    assert abs(mean_power - 1.0) < 0.05, f"E[||h||^2] = {mean_power}"


def test_generate_taps_rician_first_tap() -> None:
    """K=40 dB → tap 0 nearly deterministic across symbols."""
    h = generate_channel_taps(
        100, 4, 3.0, np.random.default_rng(7), rician_k_db=40.0
    )
    # At K=40 dB the LoS dominates, so tap 0 magnitudes should cluster tightly
    mags = np.abs(h[:, 0])
    assert np.std(mags) / np.mean(mags) < 0.05


def test_generate_taps_deterministic() -> None:
    """Same seed → identical output."""
    h1 = generate_channel_taps(10, 8, 3.0, np.random.default_rng(99))
    h2 = generate_channel_taps(10, 8, 3.0, np.random.default_rng(99))
    np.testing.assert_array_equal(h1, h2)


# ---------- channel_freq_response ----------


def test_freq_response_shape() -> None:
    """Output is (n_symbols, n_fft)."""
    h = generate_channel_taps(5, 4, 3.0, np.random.default_rng(0))
    H = channel_freq_response(h, n_fft=64)
    assert H.shape == (5, 64)
    assert H.dtype == np.complex128


def test_freq_response_flat_single_tap() -> None:
    """Single-tap channel → H constant across subcarriers."""
    h = generate_channel_taps(10, 1, 1.0, np.random.default_rng(0))
    H = channel_freq_response(h, n_fft=64)
    # All subcarriers should have the same value per symbol
    for s in range(10):
        np.testing.assert_allclose(H[s], H[s, 0], atol=1e-12)


def test_freq_response_varies_multi_tap() -> None:
    """Multi-tap channel → H varies across subcarriers."""
    h = generate_channel_taps(100, 8, 3.0, np.random.default_rng(42))
    H = channel_freq_response(h, n_fft=64)
    # Average std of |H| across subcarriers should be significant
    mean_std = np.mean(np.std(np.abs(H), axis=1))
    assert mean_std > 0.01


# ---------- apply_multipath_channel ----------


def _make_tx(n_symbols: int = 10, symbol_len: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_symbols, symbol_len))
        + 1j * rng.standard_normal((n_symbols, symbol_len))
    ).astype(np.complex128)


def test_apply_output_shape() -> None:
    """Output shape matches input shape."""
    tx = _make_tx(n_symbols=10, symbol_len=80)
    h = generate_channel_taps(10, 8, 3.0, np.random.default_rng(0))
    out = apply_multipath_channel(tx, h)
    assert out.shape == tx.shape


def test_apply_single_tap_matches_scalar() -> None:
    """Single-tap convolution is equivalent to scalar multiplication."""
    tx = _make_tx(n_symbols=5, symbol_len=80)
    h = generate_channel_taps(5, 1, 1.0, np.random.default_rng(0))
    out = apply_multipath_channel(tx, h)
    expected = tx * h  # h is (5, 1), broadcasts to (5, 80)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_apply_invalid_raises() -> None:
    """Wrong shapes and non-complex inputs raise ValueError."""
    tx = _make_tx(n_symbols=5, symbol_len=80)
    h = generate_channel_taps(5, 8, 3.0, np.random.default_rng(0))

    # 1-D input
    with pytest.raises(ValueError, match="2-D"):
        apply_multipath_channel(tx.reshape(-1), h)

    # Mismatched symbol count
    with pytest.raises(ValueError, match="same number"):
        apply_multipath_channel(tx, h[:3])

    # Non-complex tx
    with pytest.raises(ValueError, match="complex"):
        apply_multipath_channel(np.ones((5, 80)), h)

    # Non-complex h
    with pytest.raises(ValueError, match="complex"):
        apply_multipath_channel(tx, np.ones((5, 8)))

    # n_taps > symbol_len
    h_too_long = np.ones((5, 100), dtype=np.complex128)
    with pytest.raises(ValueError, match="n_taps must be <= symbol_len"):
        apply_multipath_channel(tx, h_too_long)
