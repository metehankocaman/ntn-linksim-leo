"""Integration tests for pilot insertion in the simulation pipeline."""

from __future__ import annotations

from dataclasses import replace

from ntn_linksim.sim import SimConfig, run_once


def test_pilots_disabled_backward_compat() -> None:
    """Pilots disabled should give identical BER as before."""
    cfg = SimConfig(seed=7, n_symbols=200, snr_db=10.0, enable_pilots=False)
    r1 = run_once(cfg)
    r2 = run_once(cfg)
    assert r1.ber == r2.ber
    assert r1.n_bits == 200 * 52 * 2


def test_pilots_fewer_bits() -> None:
    """With pilots enabled, fewer data bits per symbol (39 vs 52 data subcarriers)."""
    cfg_no = SimConfig(seed=1, n_symbols=100, enable_pilots=False)
    cfg_yes = SimConfig(seed=1, n_symbols=100, enable_pilots=True, pilot_spacing=4)
    r_no = run_once(cfg_no)
    r_yes = run_once(cfg_yes)
    # 52*2 = 104 bits/sym vs 39*2 = 78 bits/sym
    assert r_no.n_bits == 100 * 52 * 2
    assert r_yes.n_bits == 100 * 39 * 2
    assert r_yes.n_bits < r_no.n_bits


def test_pilots_reproducible() -> None:
    """Same seed, same config -> identical result."""
    cfg = SimConfig(seed=42, n_symbols=50, snr_db=15.0, enable_pilots=True)
    r1 = run_once(cfg)
    r2 = run_once(cfg)
    assert r1.ber == r2.ber
    assert r1.n_bits == r2.n_bits


def test_pilots_awgn_only() -> None:
    """Pilots work without multipath (AWGN-only channel)."""
    cfg = SimConfig(
        seed=3,
        n_symbols=200,
        snr_db=20.0,
        enable_pilots=True,
        enable_multipath=False,
    )
    result = run_once(cfg)
    # At 20 dB SNR in AWGN, BER should be very low
    assert result.ber < 0.01
    assert result.n_bits == 200 * 39 * 2


def test_pilots_with_multipath() -> None:
    """Pilots + multipath runs without error (BER will be high without equalization)."""
    cfg = SimConfig(
        seed=5,
        n_symbols=50,
        snr_db=20.0,
        enable_pilots=True,
        enable_multipath=True,
        multipath_n_taps=4,
        multipath_tau_rms=2.0,
    )
    result = run_once(cfg)
    assert result.n_bits == 50 * 39 * 2
    assert 0.0 <= result.ber <= 1.0


def test_pilots_ber_improves_with_snr() -> None:
    """BER should decrease as SNR increases, even with pilots."""
    base = SimConfig(seed=10, n_symbols=400, enable_pilots=True)
    r_low = run_once(replace(base, snr_db=0.0))
    r_high = run_once(replace(base, snr_db=15.0))
    assert r_high.ber < r_low.ber
