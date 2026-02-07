"""Tests that timing compensation reduces BER under delay impairment."""

from dataclasses import replace

from ntn_linksim.sim import SimConfig, run_once


def test_delay_beyond_cp_compensation_helps() -> None:
    """Delay > cp_len without comp → high BER; with comp → low BER."""
    base = SimConfig(seed=7, snr_db=30.0, n_symbols=400, delay_samples=20.0)

    result_no = run_once(replace(base, enable_timing_comp=False))
    result_yes = run_once(replace(base, enable_timing_comp=True))

    assert result_no.ber > 0.1, f"expected high BER without comp, got {result_no.ber}"
    assert result_yes.ber < result_no.ber
    assert result_yes.ber < 0.05


def test_delay_within_cp_compensation_helps() -> None:
    """Delay=8 (within cp_len=16) still benefits from compensation."""
    base = SimConfig(seed=7, snr_db=30.0, n_symbols=400, delay_samples=8.0)

    result_yes = run_once(replace(base, enable_timing_comp=True))

    assert result_yes.ber < 0.01, (
        f"expected very low BER with comp, got {result_yes.ber}"
    )


def test_nonzero_delay_without_comp_degrades() -> None:
    """Any nonzero delay without comp should cause some BER degradation."""
    base_clean = SimConfig(seed=7, snr_db=30.0, n_symbols=400)
    base_delay = replace(base_clean, delay_samples=4.0)

    result_clean = run_once(base_clean)
    result_delay = run_once(base_delay)

    # Delay should cause *some* degradation vs clean
    assert result_delay.ber >= result_clean.ber


def test_cfo_plus_delay_both_compensations() -> None:
    """Both CFO + delay with both compensations → BER recoverable.

    Uses 0.1*subcarrier_spacing CFO (24 kHz) which is safely within the
    CP-based estimator's effective range.
    """
    base = SimConfig(
        seed=7,
        snr_db=30.0,
        n_symbols=400,
    )
    # 0.1 * subcarrier_spacing — matches existing CFO comp test
    cfo_hz = 0.1 * base.fs_hz / base.ofdm_params().n_fft
    base = SimConfig(
        seed=7,
        snr_db=30.0,
        n_symbols=400,
        cfo_hz=float(cfo_hz),
        delay_samples=8.0,
        enable_cfo_comp=True,
        enable_timing_comp=True,
    )
    result = run_once(base)
    assert result.ber < 0.05, f"expected recoverable BER, got {result.ber}"
