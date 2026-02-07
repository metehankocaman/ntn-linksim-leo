"""Scenario loader, config builder, and runner for YAML-driven experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from ntn_linksim.experiments.sweep import (
    save_sweep,
    save_sweep_cfo,
    save_sweep_delay,
    save_sweep_rician,
    sweep_ber,
    sweep_ber_vs_cfo,
    sweep_ber_vs_delay,
    sweep_ber_vs_rician_k,
)
from ntn_linksim.sim import SimConfig

_VALID_SWEEP_TYPES = {"snr", "cfo", "delay", "rician_k"}


def load_scenario(path: str | Path) -> dict:
    """Parse and validate a scenario YAML file.

    Args:
        path: Path to a ``.yaml`` scenario file.

    Returns:
        Parsed scenario dict with keys ``name``, ``config``, ``sweep``.

    Raises:
        ValueError: If required keys are missing or sweep type is unknown.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "sweep" not in data:
        raise ValueError(f"Scenario {path.name} missing 'sweep' section")
    sweep = data["sweep"]
    if "type" not in sweep:
        raise ValueError(f"Scenario {path.name} missing 'sweep.type'")
    if sweep["type"] not in _VALID_SWEEP_TYPES:
        raise ValueError(
            f"Unknown sweep type '{sweep['type']}' in {path.name}. "
            f"Valid: {sorted(_VALID_SWEEP_TYPES)}"
        )
    return data


def scenario_to_config(scenario: dict) -> SimConfig:
    """Build a SimConfig from the ``config`` section of a scenario.

    Unrecognized keys in the config section are silently ignored.

    Args:
        scenario: Parsed scenario dict.

    Returns:
        A SimConfig with fields set from the YAML config section.
    """
    cfg_data = scenario.get("config", {})
    base = SimConfig()
    # Only pass keys that SimConfig actually has
    valid_fields = {f.name for f in base.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg_data.items() if k in valid_fields}
    return replace(base, **filtered)


def run_scenario(scenario: dict, out_dir: str | Path) -> None:
    """Dispatch a scenario to the appropriate sweep + save function.

    Args:
        scenario: Parsed scenario dict (from :func:`load_scenario`).
        out_dir: Directory for output artifacts.
    """
    config = scenario_to_config(scenario)
    sweep = scenario["sweep"]
    sweep_type = sweep["type"]

    if sweep_type == "snr":
        snr_db_list = sweep["snr_db"]
        ber_list = sweep_ber(config, snr_db_list)
        save_sweep(out_dir, snr_db_list, ber_list)

    elif sweep_type == "cfo":
        cfo_hz_list = sweep["cfo_hz"]
        enable_comp = sweep.get("enable_comp", False)
        ber_no_comp = sweep_ber_vs_cfo(config, cfo_hz_list, enable_comp=False)
        ber_with_comp = None
        if enable_comp:
            ber_with_comp = sweep_ber_vs_cfo(
                config, cfo_hz_list, enable_comp=True
            )
        save_sweep_cfo(
            out_dir, cfo_hz_list, ber_no_comp, ber_with_comp, snr_db=config.snr_db
        )

    elif sweep_type == "delay":
        delay_list = sweep["delay_samples"]
        enable_comp = sweep.get("enable_comp", False)
        ber_no_comp = sweep_ber_vs_delay(config, delay_list, enable_comp=False)
        ber_with_comp = None
        if enable_comp:
            ber_with_comp = sweep_ber_vs_delay(
                config, delay_list, enable_comp=True
            )
        save_sweep_delay(
            out_dir, delay_list, ber_no_comp, ber_with_comp, snr_db=config.snr_db
        )

    elif sweep_type == "rician_k":
        k_db_list = sweep["k_db"]
        ber_list = sweep_ber_vs_rician_k(config, k_db_list)
        save_sweep_rician(out_dir, k_db_list, ber_list, snr_db=config.snr_db)


def reproduce_all(scenario_dir: str | Path, out_dir: str | Path) -> None:
    """Run all YAML scenarios in a directory and save artifacts.

    Each scenario's output goes to ``out_dir/<scenario_stem>/``.

    Args:
        scenario_dir: Directory containing ``.yaml`` scenario files.
        out_dir: Root output directory.
    """
    scenario_dir = Path(scenario_dir)
    out_dir = Path(out_dir)
    yaml_files = sorted(scenario_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {scenario_dir}")
    for yaml_path in yaml_files:
        scenario = load_scenario(yaml_path)
        dest = out_dir / yaml_path.stem
        run_scenario(scenario, dest)
