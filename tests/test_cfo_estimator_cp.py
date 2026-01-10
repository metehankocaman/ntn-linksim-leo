import numpy as np

from ntn_linksim.channel.cfo import apply_cfo
from ntn_linksim.rx.cfo import estimate_cfo_from_cp
from ntn_linksim.waveform.modulation import qpsk_mod
from ntn_linksim.waveform.ofdm import (
    OfdmParams,
    add_cp,
    ifft_symbols,
    serialize_symbols,
    tx_grid,
)


def test_estimate_cfo_from_cp() -> None:
    fs_hz = 15.36e6
    params = OfdmParams(n_fft=64, n_used=52, cp_len=16, n_symbols=1)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=params.n_used * 2, dtype=np.int8)
    symbols = qpsk_mod(bits).reshape(1, params.n_used)

    grid = tx_grid(symbols, params)
    time_symbols = ifft_symbols(grid)
    tx_with_cp = add_cp(time_symbols, params.cp_len)
    tx_samples = serialize_symbols(tx_with_cp)

    cfo_hz = 1200.0
    rx_samples = apply_cfo(tx_samples, fs_hz=fs_hz, cfo_hz=cfo_hz)
    cfo_hat = estimate_cfo_from_cp(rx_samples, params.n_fft, params.cp_len, fs_hz)
    assert abs(cfo_hat - cfo_hz) < 5.0
