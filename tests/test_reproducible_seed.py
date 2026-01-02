import numpy as np

from ntn_linksim.sim import SimConfig, run_once


def test_reproducible_seed() -> None:
    config = SimConfig(seed=123, snr_db=5.0, n_symbols=50)
    result_a = run_once(config)
    result_b = run_once(config)
    assert result_a.n_bits == result_b.n_bits
    assert np.isclose(result_a.ber, result_b.ber, atol=0.0)
