"""Tests for scenario YAML loading and config building."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ntn_linksim.scenarios import load_scenario, scenario_to_config


def _write_yaml(tmp: Path, data: dict, name: str = "test.yaml") -> Path:
    p = tmp / name
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return p


def test_load_awgn_scenario() -> None:
    """Parse minimal YAML, check keys."""
    data = {
        "name": "AWGN",
        "config": {"seed": 1, "n_symbols": 50},
        "sweep": {"type": "snr", "snr_db": [0, 5, 10]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(Path(tmp), data)
        scenario = load_scenario(path)
        assert scenario["name"] == "AWGN"
        assert scenario["sweep"]["type"] == "snr"
        assert scenario["sweep"]["snr_db"] == [0, 5, 10]


def test_scenario_to_config() -> None:
    """Produces valid SimConfig with Rician fields."""
    data = {
        "name": "Rician test",
        "config": {
            "seed": 42,
            "snr_db": 15.0,
            "enable_rician": True,
            "rician_k_db": 5.0,
        },
        "sweep": {"type": "rician_k", "k_db": [0, 10]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(Path(tmp), data)
        scenario = load_scenario(path)
        cfg = scenario_to_config(scenario)
        assert cfg.seed == 42
        assert cfg.snr_db == 15.0
        assert cfg.enable_rician is True
        assert cfg.rician_k_db == 5.0


def test_missing_sweep_type_raises() -> None:
    """ValueError when sweep section has no 'type'."""
    data = {
        "name": "bad",
        "config": {},
        "sweep": {"snr_db": [0, 5]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(Path(tmp), data)
        with pytest.raises(ValueError, match="sweep.type"):
            load_scenario(path)


def test_unknown_sweep_type_raises() -> None:
    """ValueError on unrecognized sweep type."""
    data = {
        "name": "bad",
        "config": {},
        "sweep": {"type": "foobar"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(Path(tmp), data)
        with pytest.raises(ValueError, match="Unknown sweep type"):
            load_scenario(path)
