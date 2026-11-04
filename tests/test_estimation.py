"""Tests for LS channel estimation, interpolation, and integration."""

from __future__ import annotations

import numpy as np

from ntn_linksim.rx.estimation import interpolate_ls, zf_equalize
from ntn_linksim.sim import SimConfig, run_once
from ntn_linksim.waveform.pilots import pilot_indices


def test_ls_noiseless_perfect() -> None:
    """At infinite SNR (noiseless), H_ls should equal H_true at pilot positions."""
    cfg = SimConfig(
        seed=1,
        n_symbols=10,
        snr_db=200.0,  # effectively noiseless
        enable_pilots=True,
        pilot_spacing=4,
        enable_multipath=True,
        multipath_n_taps=4,
        multipath_tau_rms=2.0,
    )
    result = run_once(cfg)
    assert result.h_ls is not None
    assert result.h_true is not None
    assert result.pilot_idx is not None

    # Extract H_true at used subcarrier positions, then at pilot positions
    from ntn_linksim.waveform.ofdm import used_subcarrier_indices

    used_idx = used_subcarrier_indices(cfg.n_fft, cfg.n_used)
    h_true_used = result.h_true[:, used_idx]
    h_true_at_pilots = h_true_used[:, result.pilot_idx]

    np.testing.assert_allclose(result.h_ls, h_true_at_pilots, atol=1e-6)


def test_interpolate_identity_at_pilots() -> None:
    """Interpolated values at pilot positions should match the LS estimates."""
    rng = np.random.default_rng(42)
    p_idx = pilot_indices(52, 4)
    n_sym = 5
    h_ls = rng.standard_normal((n_sym, len(p_idx))) + 1j * rng.standard_normal(
        (n_sym, len(p_idx))
    )
    h_interp = interpolate_ls(h_ls, p_idx, 52)
    np.testing.assert_allclose(h_interp[:, p_idx], h_ls, atol=1e-12)


def test_interpolate_shape() -> None:
    """Interpolation output should have shape (n_sym, n_used)."""
    p_idx = pilot_indices(52, 4)
    h_ls = np.ones((3, len(p_idx)), dtype=np.complex128)
    h_interp = interpolate_ls(h_ls, p_idx, 52)
    assert h_interp.shape == (3, 52)


def test_ls_high_snr_close() -> None:
    """Via run_once at 40 dB, MSE between H_ls_interp and H_true should be small."""
    cfg = SimConfig(
        seed=7,
        n_symbols=50,
        snr_db=40.0,
        enable_pilots=True,
        pilot_spacing=4,
        enable_multipath=True,
        multipath_n_taps=4,
        multipath_tau_rms=2.0,
    )
    result = run_once(cfg)
    assert result.h_ls_interp is not None
    assert result.h_true is not None

    from ntn_linksim.waveform.ofdm import used_subcarrier_indices

    used_idx = used_subcarrier_indices(cfg.n_fft, cfg.n_used)
    h_true_used = result.h_true[:, used_idx]

    mse = np.mean(np.abs(result.h_ls_interp - h_true_used) ** 2)
    assert mse < 0.01  # Should be very low at 40 dB


def test_zf_equalize_identity() -> None:
    """ZF equalization with identity channel returns input."""
    rx = np.array([[1 + 1j, 2 + 0j], [0.5 - 0.5j, 3 + 1j]])
    h = np.ones_like(rx)
    eq = zf_equalize(rx, h)
    np.testing.assert_allclose(eq, rx)


def test_result_fields_none_when_pilots_disabled() -> None:
    """SimResult estimation fields should be None when pilots are off."""
    cfg = SimConfig(seed=1, n_symbols=10, enable_pilots=False)
    result = run_once(cfg)
    assert result.h_ls is None
    assert result.h_ls_interp is None
    assert result.pilot_idx is None
    assert result.rx_data is None
    # bits_tx is always populated now
    assert result.bits_tx is not None


def test_result_fields_none_when_no_multipath() -> None:
    """Estimation fields None when pilots=True but multipath=False."""
    cfg = SimConfig(seed=1, n_symbols=10, enable_pilots=True, enable_multipath=False)
    result = run_once(cfg)
    assert result.h_ls is None
    assert result.h_ls_interp is None
    assert result.pilot_idx is None
    assert result.rx_data is None
