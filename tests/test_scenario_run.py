"""Tests for scenario execution (run_scenario)."""

import tempfile
from pathlib import Path

import yaml

from ntn_linksim.scenarios import load_scenario, run_scenario


def _write_yaml(tmp: Path, data: dict, name: str = "test.yaml") -> Path:
    p = tmp / name
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return p


def test_run_snr_scenario() -> None:
    """Mini SNR scenario produces sweep.json + ber_vs_snr.png."""
    data = {
        "name": "AWGN mini",
        "config": {"seed": 1, "n_symbols": 50},
        "sweep": {"type": "snr", "snr_db": [0, 5, 10]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_yaml(tmp_path, data)
        scenario = load_scenario(path)
        out = tmp_path / "out"
        run_scenario(scenario, out)
        assert (out / "sweep.json").exists()
        assert (out / "ber_vs_snr.png").exists()


def test_run_rician_scenario() -> None:
    """Rician scenario completes and produces artifacts."""
    data = {
        "name": "Rician mini",
        "config": {"seed": 1, "n_symbols": 50, "snr_db": 15.0},
        "sweep": {"type": "rician_k", "k_db": [0, 10, 20]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_yaml(tmp_path, data)
        scenario = load_scenario(path)
        out = tmp_path / "out"
        run_scenario(scenario, out)
        assert (out / "sweep_rician.json").exists()
        assert (out / "ber_vs_rician_k.png").exists()
