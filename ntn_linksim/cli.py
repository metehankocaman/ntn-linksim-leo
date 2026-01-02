"""Command-line interface for ntn-linksim-leo."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ntn_linksim.experiments.sweep import save_sweep, sweep_ber
from ntn_linksim.sim import SimConfig, run_once, save_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ntnls")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Run OFDM AWGN simulation")
    sim_parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        required=True,
        help="SNR points in dB (one or more)",
    )
    sim_parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    sim_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for artifacts",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.command == "simulate":
        out_dir = Path(args.out)
        config = SimConfig(seed=args.seed)
        if len(args.snr_db) == 1:
            config = replace(config, snr_db=args.snr_db[0])
            result = run_once(config)
            save_run(out_dir, config, result)
        else:
            ber_list = sweep_ber(config, args.snr_db)
            save_sweep(out_dir, args.snr_db, ber_list)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
