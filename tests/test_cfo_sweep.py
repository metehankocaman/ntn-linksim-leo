"""Tests for CFO sweep functionality."""

from ntn_linksim.experiments.sweep import sweep_ber_vs_cfo
from ntn_linksim.sim import SimConfig


def test_cfo_sweep_deterministic() -> None:
    """Same seed produces identical CFO sweep results."""
    config = SimConfig(seed=42, snr_db=20.0, n_symbols=100)
    cfo_hz_list = [0.0, 10000.0, 20000.0]

    ber1 = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=False)
    ber2 = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=False)

    assert ber1 == ber2


def test_cfo_sweep_degradation_trend() -> None:
    """BER increases with larger CFO (no compensation)."""
    config = SimConfig(seed=7, snr_db=25.0, n_symbols=200)
    # Use CFO values within unambiguous range but enough to show degradation
    cfo_hz_list = [0.0, 20000.0, 40000.0, 60000.0]

    ber_list = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=False)

    # BER at zero CFO should be lowest
    assert ber_list[0] < ber_list[-1]
    # Monotonic increase (or at least non-decreasing trend)
    for i in range(len(ber_list) - 1):
        assert ber_list[i] <= ber_list[i + 1] + 0.01  # Small tolerance for noise


def test_cfo_sweep_compensation_helps() -> None:
    """CFO compensation reduces BER across the sweep."""
    config = SimConfig(seed=7, snr_db=25.0, n_symbols=200)
    # Use moderate CFO values where compensation should help
    cfo_hz_list = [10000.0, 20000.0, 30000.0]

    ber_no_comp = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=False)
    ber_with_comp = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=True)

    # Compensation should reduce BER for non-zero CFO
    for i, cfo in enumerate(cfo_hz_list):
        if cfo > 0:
            assert ber_with_comp[i] < ber_no_comp[i]
