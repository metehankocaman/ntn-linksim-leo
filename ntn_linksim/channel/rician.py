"""Single-tap block Rician fading channel model."""

from __future__ import annotations

import numpy as np


def apply_rician_fading(
    tx_with_cp: np.ndarray,
    rician_k_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply single-tap block Rician fading to OFDM symbols.

    One i.i.d. complex coefficient per OFDM symbol (flat fading).
    All subcarriers within a symbol see the same gain.

    The channel coefficient for symbol *s* is:

        h[s] = sqrt(K/(K+1)) + sqrt(1/(K+1)) * CN(0,1) / sqrt(2)

    where K = 10^(k_db/10).  E[|h|^2] = 1 by construction, so average
    signal power (and therefore SNR meaning) is preserved.

    Args:
        tx_with_cp: 2-D complex array (n_symbols, n_fft + cp_len).
        rician_k_db: Rician K-factor in dB.  Typical LEO LoS: 10 dB.
        rng: NumPy Generator for reproducibility.

    Returns:
        Faded signal, same shape and dtype as input.

    Raises:
        ValueError: If *tx_with_cp* is not 2-D or not complex.
    """
    tx_with_cp = np.asarray(tx_with_cp)
    if tx_with_cp.ndim != 2:
        raise ValueError("tx_with_cp must be a 2-D array")
    if not np.iscomplexobj(tx_with_cp):
        raise ValueError("tx_with_cp must be complex")

    tx_with_cp = tx_with_cp.astype(np.complex128, copy=False)
    n_symbols = tx_with_cp.shape[0]

    k_lin = 10.0 ** (rician_k_db / 10.0)
    los_amp = np.sqrt(k_lin / (k_lin + 1.0))
    nlos_amp = np.sqrt(1.0 / (k_lin + 1.0))

    # CN(0,1) / sqrt(2) has unit total variance split across real/imag
    scatter = (
        rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
    ).astype(np.complex128) / np.sqrt(2.0)

    h = los_amp + nlos_amp * scatter  # shape (n_symbols,)

    return tx_with_cp * h[:, np.newaxis]
