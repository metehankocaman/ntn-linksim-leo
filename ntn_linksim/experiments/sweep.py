"""SNR sweep utilities."""

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
