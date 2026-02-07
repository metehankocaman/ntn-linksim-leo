# ntn-linksim-leo

Simulation-first, reproducible OFDM link-level simulator for NTN/LEO research.

**Completed milestones:**
- **M0**: AWGN baseline with QPSK-OFDM, BER measurement, CLI, deterministic artifacts
- **M1**: CFO (Doppler) impairment + CP-based estimation + compensation
- **M2**: Delay / timing offset impairment + CP-based timing estimation + compensation

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## CLI Commands

**SNR sweep** (AWGN baseline):
```bash
ntnls simulate --snr-db 0 5 10 --seed 1 --out results/
```

**CFO sweep** (Doppler/compensation analysis):
```bash
ntnls cfo-sweep --cfo-hz 0 15000 30000 45000 60000 --snr-db 20 --seed 42 --out results_cfo/
```

**Delay sweep** (timing offset/compensation analysis):
```bash
ntnls delay-sweep --delay-samples 0 4 8 12 16 20 24 --snr-db 20 --seed 1 --out results_delay/
```

## Artifacts

| Command | Outputs |
|---------|---------|
| `simulate` | `sweep.json`, `ber_vs_snr.png` |
| `cfo-sweep` | `sweep_cfo.json`, `ber_vs_cfo.png` |
| `delay-sweep` | `sweep_delay.json`, `ber_vs_delay.png` |

## Example outputs

### BER vs SNR (AWGN baseline)
![BER vs SNR](docs/ber_vs_snr.png)

### BER vs CFO (Doppler compensation)
![BER vs CFO](docs/ber_vs_cfo.png)

The CFO sweep demonstrates that without compensation, even moderate Doppler shifts
($\geq$ 15 kHz) push BER to ~0.5 (random guessing). CP-based CFO estimation and
compensation recover the signal, keeping BER usable up to 60 kHz CFO.

> **Note**: The CP-based estimator is fractional-only with unambiguous range $|CFO|\lt \frac{\Delta f}{2}$
> ($\pm$ 120 kHz for default parameters).

### BER vs Delay (Timing compensation)
![BER vs Delay](docs/ber_vs_delay.png)

The delay sweep shows that even delays within the CP length (16 samples) cause BER
degradation due to FFT window misalignment (per-subcarrier phase rotations). Delays
beyond the CP length produce severe ISI. CP-based timing estimation + compensation
realigns the FFT window, keeping BER near zero across all tested delays.

> **Note**: The timing estimator uses CP sliding correlation for integer-sample
> offset detection. Fractional delay injection is supported but compensation is
> integer-only.

## Next milestones

- **M3**: Rician fading + scenario harness
