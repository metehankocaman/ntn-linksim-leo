"""Simulation entry points and dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ntn_linksim.channel.awgn import add_awgn
from ntn_linksim.channel.cfo import apply_cfo
from ntn_linksim.channel.delay import apply_delay
from ntn_linksim.channel.rician import apply_rician_fading
from ntn_linksim.rng import seeded_rng
from ntn_linksim.rx.cfo import compensate_cfo, estimate_cfo_from_cp
from ntn_linksim.rx.timing import compensate_integer_delay, estimate_timing_offset_cp
from ntn_linksim.waveform.modulation import qpsk_demod_hard, qpsk_mod
from ntn_linksim.waveform.ofdm import (
    OfdmParams,
    add_cp,
    deserialize_symbols,
    extract_used,
    fft_symbols,
    ifft_symbols,
    remove_cp,
    serialize_symbols,
    tx_grid,
)


@dataclass(frozen=True)
class SimConfig:
    """Simulation configuration for an OFDM AWGN run."""

    n_fft: int = 64
    n_used: int = 52
    cp_len: int = 16
    n_symbols: int = 200
    snr_db: float = 10.0
    seed: int = 1
    fs_hz: float = 15.36e6
    cfo_hz: float = 0.0
    enable_cfo_comp: bool = False
    delay_samples: float = 0.0
    enable_timing_comp: bool = False
    enable_rician: bool = False
    rician_k_db: float = 10.0

    def validate(self) -> None:
        params = OfdmParams(
            n_fft=self.n_fft,
            n_used=self.n_used,
            cp_len=self.cp_len,
            n_symbols=self.n_symbols,
        )
        params.validate()
        if self.fs_hz <= 0:
            raise ValueError("fs_hz must be positive")

    def ofdm_params(self) -> OfdmParams:
        return OfdmParams(
            n_fft=self.n_fft,
            n_used=self.n_used,
            cp_len=self.cp_len,
            n_symbols=self.n_symbols,
        )


@dataclass(frozen=True)
class SimResult:
    """Simulation outputs."""

    ber: float
    n_bits: int
    snr_db: float


def run_once(config: SimConfig) -> SimResult:
    """Run a single OFDM AWGN simulation and return BER results."""
    config.validate()
    rng = seeded_rng(config.seed)
    params = config.ofdm_params()

    n_bits = params.n_symbols * params.n_used * 2
    bits_tx = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    symbols = qpsk_mod(bits_tx).reshape(params.n_symbols, params.n_used)

    grid = tx_grid(symbols, params)
    time_symbols = ifft_symbols(grid)
    tx_with_cp = add_cp(time_symbols, params.cp_len)
    if config.enable_rician:
        tx_with_cp = apply_rician_fading(tx_with_cp, config.rician_k_db, rng)
    tx_samples = serialize_symbols(tx_with_cp)
    if config.cfo_hz != 0.0:
        tx_samples = apply_cfo(tx_samples, fs_hz=config.fs_hz, cfo_hz=config.cfo_hz)
    if config.delay_samples != 0.0:
        tx_samples = apply_delay(tx_samples, config.delay_samples)

    rx_samples = add_awgn(tx_samples, config.snr_db, rng)

    # Timing compensation first (must align symbol boundaries before CFO est.)
    if config.enable_timing_comp:
        delay_hat = estimate_timing_offset_cp(
            rx_samples,
            n_fft=params.n_fft,
            cp_len=params.cp_len,
            n_symbols=params.n_symbols,
        )
        rx_samples = compensate_integer_delay(rx_samples, delay_hat)

    if config.enable_cfo_comp:
        symbol_len = params.n_fft + params.cp_len
        rx0 = rx_samples[:symbol_len]
        cfo_hat = estimate_cfo_from_cp(
            rx0,
            n_fft=params.n_fft,
            cp_len=params.cp_len,
            fs_hz=config.fs_hz,
        )
        rx_samples = compensate_cfo(rx_samples, fs_hz=config.fs_hz, cfo_hz=cfo_hat)

    rx_with_cp = deserialize_symbols(rx_samples, params)
    rx_no_cp = remove_cp(rx_with_cp, params.cp_len)
    rx_grid = fft_symbols(rx_no_cp)
    rx_used = extract_used(rx_grid, params)

    bits_rx = qpsk_demod_hard(rx_used.reshape(-1))
    n_errors = int(np.sum(bits_rx != bits_tx))
    ber = n_errors / n_bits
    return SimResult(ber=ber, n_bits=n_bits, snr_db=config.snr_db)


def save_run(out_dir: str | Path, config: SimConfig, result: SimResult) -> Path:
    """Save a single-run JSON artifact and return its path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "result": asdict(result),
    }
    dest = out_path / "run.json"
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return dest
