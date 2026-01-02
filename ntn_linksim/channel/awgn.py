"""AWGN channel model."""

from __future__ import annotations

import numpy as np


def add_awgn(
    samples: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add complex AWGN to samples for a target SNR in dB.

    SNR is defined as signal_power / noise_power with signal_power = mean(|x|^2).
    """
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.size == 0:
        raise ValueError("samples must be non-empty")
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = np.mean(np.abs(samples) ** 2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (
        rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
    )
    return samples + noise.astype(np.complex128)
