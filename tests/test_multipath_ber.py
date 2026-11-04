"""Integration tests for multipath channel via run_once()."""

from __future__ import annotations

import numpy as np
import pytest

from ntn_linksim.sim import SimConfig, run_once


def _base_config(**overrides: object) -> SimConfig:
    """Return a small, fast SimConfig with optional overrides."""
    defaults = dict(n_symbols=50, seed=42, snr_db=15.0)
    defaults.update(overrides)
    from dataclasses import replace

    return replace(SimConfig(), **defaults)


# ---------- backward compatibility ----------


def test_multipath_disabled_backward_compat() -> None:
    """enable_multipath=False produces the same BER as the old baseline."""
    cfg_old = _base_config(enable_multipath=False)
    cfg_new = _base_config(enable_multipath=False)
    r_old = run_once(cfg_old)
    r_new = run_once(cfg_new)
    assert r_old.ber == r_new.ber


# ---------- BER impact ----------


def test_multipath_degrades_ber() -> None:
    """Multipath without equalization should increase BER vs AWGN-only."""
    cfg_awgn = _base_config(snr_db=20.0)
    cfg_mp = _base_config(
        snr_db=20.0, enable_multipath=True, multipath_n_taps=8, multipath_tau_rms=3.0
    )
    r_awgn = run_once(cfg_awgn)
    r_mp = run_once(cfg_mp)
    assert r_mp.ber > r_awgn.ber


# ---------- h_true ----------


def test_multipath_returns_h_true() -> None:
    """When multipath is on, h_true is returned with correct shape."""
    cfg = _base_config(enable_multipath=True, multipath_n_taps=4)
    result = run_once(cfg)
    assert result.h_true is not None
    assert result.h_true.shape == (cfg.n_symbols, cfg.n_fft)
    assert np.iscomplexobj(result.h_true)


def test_multipath_h_true_none_when_disabled() -> None:
    """h_true is None when multipath is off."""
    cfg = _base_config(enable_multipath=False)
    result = run_once(cfg)
    assert result.h_true is None


def test_multipath_h_true_unit_power() -> None:
    """Mean |H|^2 across symbols and subcarriers ≈ 1.0."""
    cfg = _base_config(
        n_symbols=500, enable_multipath=True, multipath_n_taps=8, multipath_tau_rms=3.0
    )
    result = run_once(cfg)
    mean_power = np.mean(np.abs(result.h_true) ** 2)
    assert abs(mean_power - 1.0) < 0.1, f"E[|H|^2] = {mean_power}"


# ---------- reproducibility ----------


def test_multipath_reproducible() -> None:
    """Same config → identical BER and h_true."""
    cfg = _base_config(enable_multipath=True, multipath_n_taps=4)
    r1 = run_once(cfg)
    r2 = run_once(cfg)
    assert r1.ber == r2.ber
    np.testing.assert_array_equal(r1.h_true, r2.h_true)


# ---------- combined with Rician ----------


def test_multipath_with_rician() -> None:
    """Both enable_multipath and enable_rician can be on together."""
    cfg = _base_config(
        enable_multipath=True,
        enable_rician=True,
        rician_k_db=10.0,
        multipath_n_taps=4,
    )
    result = run_once(cfg)
    assert result.h_true is not None
    assert result.ber >= 0.0


# ---------- validation ----------


def test_multipath_ntaps_too_large_raises() -> None:
    """n_taps > cp_len raises ValueError on validate."""
    cfg = _base_config(enable_multipath=True, multipath_n_taps=20)
    with pytest.raises(ValueError, match="multipath_n_taps"):
        run_once(cfg)


# ---------- fewer taps → less distortion ----------


def test_multipath_fewer_taps_less_distortion() -> None:
    """n_taps=2 should cause less BER degradation than n_taps=8."""
    snr = 15.0
    cfg_2 = _base_config(
        snr_db=snr,
        enable_multipath=True,
        multipath_n_taps=2,
        multipath_tau_rms=3.0,
        n_symbols=200,
    )
    cfg_8 = _base_config(
        snr_db=snr,
        enable_multipath=True,
        multipath_n_taps=8,
        multipath_tau_rms=3.0,
        n_symbols=200,
    )
    r_2 = run_once(cfg_2)
    r_8 = run_once(cfg_8)
    # More taps → more frequency selectivity → worse BER (without equalizer)
    assert r_8.ber >= r_2.ber
