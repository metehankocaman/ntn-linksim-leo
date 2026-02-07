"""Tests for delay / timing offset impairment injection."""

import numpy as np
import pytest

from ntn_linksim.channel.delay import (
    apply_delay,
    apply_fractional_delay,
    apply_integer_delay,
)


def test_zero_delay_identity() -> None:
    """Zero integer delay returns an identical copy."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    y = apply_integer_delay(x, 0)
    np.testing.assert_array_equal(y, x)


def test_integer_delay_known_shift() -> None:
    """Known shift: first *delay* samples are zero, rest matches input."""
    x = np.arange(1, 11, dtype=np.complex128)
    y = apply_integer_delay(x, 3)
    assert y.size == x.size
    np.testing.assert_array_equal(y[:3], 0.0)
    np.testing.assert_array_equal(y[3:], x[:7])


def test_integer_delay_length_preserved() -> None:
    """Output length always equals input length."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    for d in (0, 1, 10, 255, 256, 300):
        y = apply_integer_delay(x, d)
        assert y.size == x.size


def test_fractional_near_zero_identity() -> None:
    """Fractional delay near zero is approximately identity."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    y = apply_fractional_delay(x, 1e-14)
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_negative_delay_raises() -> None:
    """Negative delay raises ValueError."""
    x = np.ones(64, dtype=np.complex128)
    with pytest.raises(ValueError, match="non-negative"):
        apply_integer_delay(x, -1)
    with pytest.raises(ValueError, match="non-negative"):
        apply_delay(x, -0.5)


def test_apply_delay_integer_matches() -> None:
    """apply_delay with integer value matches apply_integer_delay."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    y1 = apply_integer_delay(x, 5)
    y2 = apply_delay(x, 5.0)
    np.testing.assert_allclose(y2, y1, atol=1e-10)


def test_apply_delay_fractional_roundtrip() -> None:
    """Fractional delay of 0.5 produces a signal different from the original."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    y = apply_delay(x, 0.5)
    assert y.size == x.size
    # Should differ from the original
    assert not np.allclose(y, x)


def test_full_delay_returns_zeros() -> None:
    """Delay >= signal length produces all zeros."""
    x = np.ones(64, dtype=np.complex128)
    y = apply_integer_delay(x, 64)
    np.testing.assert_array_equal(y, 0.0)
