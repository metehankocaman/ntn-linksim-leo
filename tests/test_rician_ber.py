"""Integration tests for Rician fading via run_once()."""

from dataclasses import replace

from ntn_linksim.sim import SimConfig, run_once


def test_rician_disabled_backward_compat() -> None:
    """enable_rician=False -> identical BER to old baseline (no RNG change)."""
    cfg_old = SimConfig(seed=1, snr_db=10.0, n_symbols=200)
    cfg_new = replace(cfg_old, enable_rician=False, rician_k_db=10.0)
    r_old = run_once(cfg_old)
    r_new = run_once(cfg_new)
    assert r_old.ber == r_new.ber


def test_rician_degrades_ber() -> None:
    """Rician K=5 dB -> higher BER than pure AWGN at same SNR."""
    cfg_awgn = SimConfig(seed=7, snr_db=12.0, n_symbols=200)
    cfg_rician = replace(cfg_awgn, enable_rician=True, rician_k_db=5.0)
    ber_awgn = run_once(cfg_awgn).ber
    ber_rician = run_once(cfg_rician).ber
    assert ber_rician > ber_awgn


def test_higher_k_better_ber() -> None:
    """K=15 dB -> lower BER than K=0 dB (stronger LoS helps)."""
    base = SimConfig(seed=7, snr_db=12.0, n_symbols=200, enable_rician=True)
    ber_k0 = run_once(replace(base, rician_k_db=0.0)).ber
    ber_k15 = run_once(replace(base, rician_k_db=15.0)).ber
    assert ber_k15 < ber_k0


def test_rician_very_high_k_near_awgn() -> None:
    """K=40 dB -> BER within 0.01 of AWGN."""
    cfg_awgn = SimConfig(seed=3, snr_db=10.0, n_symbols=200)
    cfg_rician = replace(cfg_awgn, enable_rician=True, rician_k_db=40.0)
    ber_awgn = run_once(cfg_awgn).ber
    ber_rician = run_once(cfg_rician).ber
    assert abs(ber_rician - ber_awgn) < 0.01


def test_rician_reproducible() -> None:
    """Same config -> identical BER."""
    cfg = SimConfig(
        seed=42, snr_db=10.0, n_symbols=200,
        enable_rician=True, rician_k_db=5.0,
    )
    r1 = run_once(cfg)
    r2 = run_once(cfg)
    assert r1.ber == r2.ber
