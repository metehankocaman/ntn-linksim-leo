"""Tests for Rician K-factor sweep functionality."""

from ntn_linksim.experiments.sweep import sweep_ber_vs_rician_k
from ntn_linksim.sim import SimConfig


def test_rician_sweep_deterministic() -> None:
    """Same seed produces identical Rician sweep results."""
    config = SimConfig(seed=42, snr_db=15.0, n_symbols=100)
    k_db_list = [-3.0, 0.0, 5.0, 10.0, 20.0]

    ber1 = sweep_ber_vs_rician_k(config, k_db_list)
    ber2 = sweep_ber_vs_rician_k(config, k_db_list)

    assert ber1 == ber2


def test_rician_sweep_k_trend() -> None:
    """BER decreases monotonically with increasing K (stronger LoS)."""
    config = SimConfig(seed=7, snr_db=12.0, n_symbols=200)
    k_db_list = [-3.0, 0.0, 5.0, 10.0, 20.0]

    ber_list = sweep_ber_vs_rician_k(config, k_db_list)

    for i in range(1, len(ber_list)):
        assert ber_list[i] <= ber_list[i - 1] + 0.005, (
            f"BER did not decrease: K={k_db_list[i-1]} dB -> {ber_list[i-1]}, "
            f"K={k_db_list[i]} dB -> {ber_list[i]}"
        )
