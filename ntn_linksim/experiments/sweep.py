"""SNR, CFO, and delay sweep utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ntn_linksim.sim import SimConfig, run_once


def sweep_ber(config: SimConfig, snr_db_list: Iterable[float]) -> list[float]:
    """Run a BER sweep across SNR points."""
    ber_list = []
    for snr_db in snr_db_list:
        result = run_once(replace(config, snr_db=float(snr_db)))
        ber_list.append(result.ber)
    return ber_list


def save_sweep(
    out_dir: str | Path,
    snr_db_list: Iterable[float],
    ber_list: list[float],
) -> None:
    """Save sweep JSON and BER vs SNR plot."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snr_values = [float(x) for x in snr_db_list]
    payload = {"snr_db": snr_values, "ber": ber_list}

    json_path = out_path / "sweep.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    plt.figure(figsize=(6, 4))
    plt.plot(snr_values, ber_list, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR (AWGN)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path / "ber_vs_snr.png", dpi=150)
    plt.close()


def sweep_ber_vs_cfo(
    config: SimConfig,
    cfo_hz_list: Iterable[float],
    enable_comp: bool = False,
) -> list[float]:
    """Sweep CFO at fixed SNR, return BER list.

    Args:
        config: Base simulation config (snr_db used as the fixed SNR point).
        cfo_hz_list: CFO values in Hz to sweep.
        enable_comp: Whether to enable CFO compensation.

    Returns:
        List of BER values corresponding to each CFO point.
    """
    ber_list = []
    for cfo_hz in cfo_hz_list:
        cfg = replace(config, cfo_hz=float(cfo_hz), enable_cfo_comp=enable_comp)
        result = run_once(cfg)
        ber_list.append(result.ber)
    return ber_list


def save_sweep_cfo(
    out_dir: str | Path,
    cfo_hz_list: Iterable[float],
    ber_no_comp: list[float],
    ber_with_comp: list[float] | None = None,
    snr_db: float | None = None,
) -> None:
    """Save CFO sweep JSON and BER vs CFO plot.

    Args:
        out_dir: Output directory for artifacts.
        cfo_hz_list: CFO values in Hz that were swept.
        ber_no_comp: BER values without compensation.
        ber_with_comp: BER values with compensation (optional).
        snr_db: Fixed SNR used for the sweep (for labeling).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfo_values = [float(x) for x in cfo_hz_list]
    payload: dict = {
        "cfo_hz": cfo_values,
        "ber_no_comp": ber_no_comp,
    }
    if ber_with_comp is not None:
        payload["ber_with_comp"] = ber_with_comp
    if snr_db is not None:
        payload["snr_db"] = snr_db

    json_path = out_path / "sweep_cfo.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # Convert Hz to kHz for plotting
    cfo_khz = [x / 1000.0 for x in cfo_values]

    plt.figure(figsize=(6, 4))
    plt.plot(cfo_khz, ber_no_comp, marker="o", linestyle="-", label="No compensation")
    if ber_with_comp is not None:
        plt.plot(
            cfo_khz,
            ber_with_comp,
            marker="s",
            linestyle="--",
            label="With compensation",
        )
        plt.legend()
    plt.xlabel("CFO (kHz)")
    plt.ylabel("BER")
    title = "BER vs CFO"
    if snr_db is not None:
        title += f" at SNR = {snr_db} dB"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path / "ber_vs_cfo.png", dpi=150)
    plt.close()


def sweep_ber_vs_delay(
    config: SimConfig,
    delay_list: Iterable[float],
    enable_comp: bool = False,
) -> list[float]:
    """Sweep timing offset at fixed SNR, return BER list.

    Args:
        config: Base simulation config (snr_db used as the fixed SNR point).
        delay_list: Delay values in samples to sweep.
        enable_comp: Whether to enable timing compensation.

    Returns:
        List of BER values corresponding to each delay point.
    """
    ber_list = []
    for delay in delay_list:
        cfg = replace(
            config,
            delay_samples=float(delay),
            enable_timing_comp=enable_comp,
        )
        result = run_once(cfg)
        ber_list.append(result.ber)
    return ber_list


def save_sweep_delay(
    out_dir: str | Path,
    delay_list: Iterable[float],
    ber_no_comp: list[float],
    ber_with_comp: list[float] | None = None,
    snr_db: float | None = None,
) -> None:
    """Save delay sweep JSON and BER vs delay plot.

    Args:
        out_dir: Output directory for artifacts.
        delay_list: Delay values in samples that were swept.
        ber_no_comp: BER values without compensation.
        ber_with_comp: BER values with compensation (optional).
        snr_db: Fixed SNR used for the sweep (for labeling).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    delay_values = [float(x) for x in delay_list]
    payload: dict = {
        "delay_samples": delay_values,
        "ber_no_comp": ber_no_comp,
    }
    if ber_with_comp is not None:
        payload["ber_with_comp"] = ber_with_comp
    if snr_db is not None:
        payload["snr_db"] = snr_db

    json_path = out_path / "sweep_delay.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    plt.figure(figsize=(6, 4))
    plt.plot(
        delay_values, ber_no_comp, marker="o", linestyle="-", label="No compensation"
    )
    if ber_with_comp is not None:
        plt.plot(
            delay_values,
            ber_with_comp,
            marker="s",
            linestyle="--",
            label="With compensation",
        )
        plt.legend()
    plt.xlabel("Delay (samples)")
    plt.ylabel("BER")
    title = "BER vs Timing Offset"
    if snr_db is not None:
        title += f" at SNR = {snr_db} dB"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path / "ber_vs_delay.png", dpi=150)
    plt.close()
