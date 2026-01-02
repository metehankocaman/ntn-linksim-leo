import numpy as np

from ntn_linksim.channel.awgn import add_awgn
from ntn_linksim.rng import seeded_rng


def test_awgn_calibration() -> None:
    rng = seeded_rng(2024)
    x = (
        rng.standard_normal(200000).astype(np.float64)
        + 1j * rng.standard_normal(200000).astype(np.float64)
    )
    x = x.astype(np.complex128)
    snr_db = 10.0
    y = add_awgn(x, snr_db, rng)
    noise_power_est = float(np.mean(np.abs(y - x) ** 2))
    signal_power = float(np.mean(np.abs(x) ** 2))
    expected = signal_power / (10 ** (snr_db / 10.0))
    rel_err = abs(noise_power_est - expected) / expected
    assert rel_err < 0.15
