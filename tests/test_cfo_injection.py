import numpy as np

from ntn_linksim.channel.cfo import apply_cfo


def _estimate_cfo(y: np.ndarray, fs_hz: float) -> float:
    n = np.arange(y.size, dtype=np.float64)
    phi = np.unwrap(np.angle(y))
    n_centered = n - np.mean(n)
    phi_centered = phi - np.mean(phi)
    slope = float(np.dot(n_centered, phi_centered) / np.dot(n_centered, n_centered))
    return slope * fs_hz / (2.0 * np.pi)


def test_apply_cfo_estimation() -> None:
    fs_hz = 15.36e6
    n_samples = 8192
    x = np.ones(n_samples, dtype=np.complex128)

    for cfo_hz in (1200.0, -750.0):
        y = apply_cfo(x, fs_hz=fs_hz, cfo_hz=cfo_hz)
        cfo_est = _estimate_cfo(y, fs_hz=fs_hz)
        assert abs(cfo_est - cfo_hz) < 1.0
