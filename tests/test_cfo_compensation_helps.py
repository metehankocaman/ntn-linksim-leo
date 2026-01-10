from dataclasses import replace

from ntn_linksim.sim import SimConfig, run_once


def test_cfo_compensation_helps() -> None:
    base = SimConfig(
        seed=7,
        snr_db=30.0,
        n_symbols=400,
    )

    params = base.ofdm_params()
    # Subcarrier spacing Δf = fs / NFFT. CP-based estimator is unambiguous
    # for |CFO| < Δf/2.
    # Safely within Δf/2.
    cfo_hz = 0.1 * base.fs_hz / params.n_fft

    base = replace(base, cfo_hz=float(cfo_hz))

    result_no = run_once(replace(base, enable_cfo_comp=False))
    result_yes = run_once(replace(base, enable_cfo_comp=True))

    assert result_yes.ber < result_no.ber
    assert result_yes.ber < 0.1
