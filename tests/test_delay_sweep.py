"""Tests for delay sweep functionality."""

from ntn_linksim.experiments.sweep import sweep_ber_vs_delay
from ntn_linksim.sim import SimConfig


def test_delay_sweep_deterministic() -> None:
    """Same seed produces identical delay sweep results."""
    config = SimConfig(seed=42, snr_db=20.0, n_symbols=100)
    delay_list = [0.0, 4.0, 8.0, 16.0]

    ber1 = sweep_ber_vs_delay(config, delay_list, enable_comp=False)
    ber2 = sweep_ber_vs_delay(config, delay_list, enable_comp=False)

    assert ber1 == ber2


def test_delay_sweep_degradation_trend() -> None:
    """Any nonzero delay destroys BER without compensation.

    Without per-subcarrier equalization, even a 1-sample FFT window shift
    creates phase rotations across all subcarriers, pushing BER near 0.5.
    """
    config = SimConfig(seed=7, snr_db=25.0, n_symbols=200)
    delay_list = [0.0, 4.0, 8.0, 16.0, 24.0]

    ber_list = sweep_ber_vs_delay(config, delay_list, enable_comp=False)

    # Zero delay → near-zero BER
    assert ber_list[0] < 0.01
    # Any nonzero delay → high BER (near random guessing)
    for i in range(1, len(delay_list)):
        assert ber_list[i] > 0.3, f"delay={delay_list[i]}: BER={ber_list[i]}"


def test_delay_sweep_compensation_helps() -> None:
    """Timing compensation reduces BER across the sweep."""
    config = SimConfig(seed=7, snr_db=25.0, n_symbols=200)
    # Use delay values where compensation should help
    delay_list = [4.0, 8.0, 16.0, 24.0]

    ber_no_comp = sweep_ber_vs_delay(config, delay_list, enable_comp=False)
    ber_with_comp = sweep_ber_vs_delay(config, delay_list, enable_comp=True)

    # Compensation should reduce BER for non-zero delay
    for i, delay in enumerate(delay_list):
        if delay > 0:
            assert ber_with_comp[i] <= ber_no_comp[i] + 0.001
