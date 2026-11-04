"""Unit tests for waveform.pilots module."""

from __future__ import annotations

import numpy as np
import pytest

from ntn_linksim.waveform.pilots import (
    data_indices,
    extract_pilots,
    insert_pilots,
    pilot_indices,
    pilot_symbols,
)


def test_pilot_indices_spacing_4() -> None:
    idx = pilot_indices(52, 4)
    expected = np.arange(0, 52, 4)  # [0, 4, 8, ..., 48] -> 13 pilots
    np.testing.assert_array_equal(idx, expected)
    assert len(idx) == 13


def test_data_indices_complement() -> None:
    p_idx = pilot_indices(52, 4)
    d_idx = data_indices(52, 4)
    # Union should be full set
    combined = np.sort(np.concatenate([p_idx, d_idx]))
    np.testing.assert_array_equal(combined, np.arange(52))
    # No overlap
    assert len(np.intersect1d(p_idx, d_idx)) == 0
    assert len(d_idx) == 39


def test_pilot_symbols_value() -> None:
    ps = pilot_symbols(13)
    np.testing.assert_array_equal(ps, np.ones(13, dtype=np.complex128))
    assert ps.dtype == np.complex128


def test_insert_extract_roundtrip() -> None:
    rng = np.random.default_rng(42)
    n_sym = 10
    n_used = 52
    spacing = 4
    n_data = len(data_indices(n_used, spacing))

    data = (rng.standard_normal((n_sym, n_data))
            + 1j * rng.standard_normal((n_sym, n_data)))

    grid = insert_pilots(data, n_used, spacing)
    assert grid.shape == (n_sym, n_used)

    rx_pilots, rx_data = extract_pilots(grid, n_used, spacing)
    # Data roundtrips exactly
    np.testing.assert_allclose(rx_data, data)
    # Pilots are all 1+0j
    np.testing.assert_allclose(rx_pilots, 1.0 + 0j)


def test_pilot_invalid_spacing_raises() -> None:
    with pytest.raises(ValueError, match="pilot_spacing"):
        pilot_indices(52, 0)
    with pytest.raises(ValueError, match="pilot_spacing"):
        pilot_indices(52, 53)


def test_insert_wrong_data_shape_raises() -> None:
    data = np.ones((5, 52), dtype=np.complex128)  # 52 != 39 data subcarriers
    with pytest.raises(ValueError, match="data subcarriers"):
        insert_pilots(data, 52, 4)


def test_extract_wrong_shape_raises() -> None:
    rx = np.ones((5, 40), dtype=np.complex128)  # 40 != 52
    with pytest.raises(ValueError, match="expected 52"):
        extract_pilots(rx, 52, 4)
