"""Command-line interface for ntn-linksim-leo."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ntn_linksim.experiments.sweep import (
    save_sweep,
    save_sweep_cfo,
    sweep_ber,
    sweep_ber_vs_cfo,
)
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

    cfo_parser = subparsers.add_parser(
        "cfo-sweep",
        help="Sweep CFO at fixed SNR to produce BER vs CFO plot",
    )
    cfo_parser.add_argument(
        "--cfo-hz",
        nargs="+",
        type=float,
        required=True,
        help=(
            "CFO points in Hz (one or more). "
            "CP-based estimator is unambiguous for |CFO| < subcarrier_spacing/2 "
            "(~120 kHz for default params)."
        ),
    )
    cfo_parser.add_argument(
        "--snr-db",
        type=float,
        default=20.0,
        help="Fixed SNR in dB (default: 20.0)",
    )
    cfo_parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    cfo_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for artifacts",
    )
    cfo_parser.add_argument(
        "--no-comp",
        action="store_true",
        help="Skip compensation curve (only plot no-compensation BER)",
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

    if args.command == "cfo-sweep":
        out_dir = Path(args.out)
        config = SimConfig(seed=args.seed, snr_db=args.snr_db)
        ber_no_comp = sweep_ber_vs_cfo(config, args.cfo_hz, enable_comp=False)
        ber_with_comp = None
        if not args.no_comp:
            ber_with_comp = sweep_ber_vs_cfo(config, args.cfo_hz, enable_comp=True)
        save_sweep_cfo(
            out_dir,
            args.cfo_hz,
            ber_no_comp,
            ber_with_comp,
            snr_db=args.snr_db,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
