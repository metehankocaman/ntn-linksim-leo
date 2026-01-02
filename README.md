# ntn-linksim-leo

Simulation-first, reproducible OFDM link-level simulator for NTN/LEO research.
M0 delivers an AWGN baseline with QPSK-OFDM, BER measurement, CLI entrypoint,
and deterministic artifacts.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ntnls simulate --snr-db 0 5 10 --seed 1 --out results/
```

## Artifacts

- `results/run.json` for single-SNR runs
- `results/sweep.json` and `results/ber_vs_snr.png` for SNR sweeps

## Next milestones

- Doppler and delay models for LEO dynamics
- CFO and timing recovery
- Channel estimation and equalization
