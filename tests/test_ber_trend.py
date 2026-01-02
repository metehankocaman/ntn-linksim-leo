from dataclasses import replace

from ntn_linksim.sim import SimConfig, run_once


def test_ber_trend() -> None:
    base = SimConfig(seed=7, n_symbols=400)
    result_low = run_once(replace(base, snr_db=0.0))
    result_high = run_once(replace(base, snr_db=12.0))
    assert result_high.ber < result_low.ber
