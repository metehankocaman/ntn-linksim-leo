"""Command-line interface for ntn-linksim-leo."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

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
from ntn_linksim.scenarios import load_scenario, reproduce_all, run_scenario
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

    delay_parser = subparsers.add_parser(
        "delay-sweep",
        help="Sweep timing offset at fixed SNR to produce BER vs delay plot",
    )
    delay_parser.add_argument(
        "--delay-samples",
        nargs="+",
        type=float,
        required=True,
        help="Delay points in samples (one or more, must be >= 0)",
    )
    delay_parser.add_argument(
        "--snr-db",
        type=float,
        default=20.0,
        help="Fixed SNR in dB (default: 20.0)",
    )
    delay_parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    delay_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for artifacts",
    )
    delay_parser.add_argument(
        "--no-comp",
        action="store_true",
        help="Skip compensation curve (only plot no-compensation BER)",
    )

    rician_parser = subparsers.add_parser(
        "rician-sweep",
        help="Sweep Rician K-factor at fixed SNR to produce BER vs K plot",
    )
    rician_parser.add_argument(
        "--k-db",
        nargs="+",
        type=float,
        required=True,
        help="Rician K-factor values in dB (one or more)",
    )
    rician_parser.add_argument(
        "--snr-db",
        type=float,
        default=15.0,
        help="Fixed SNR in dB (default: 15.0)",
    )
    rician_parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    rician_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for artifacts",
    )

    scenario_parser = subparsers.add_parser(
        "run-scenario",
        help="Run a single scenario from a YAML file",
    )
    scenario_parser.add_argument(
        "scenario",
        type=str,
        help="Path to a scenario YAML file",
    )
    scenario_parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for artifacts",
    )

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Run all scenarios in a directory",
    )
    reproduce_parser.add_argument(
        "--scenario-dir",
        type=str,
        default="scenarios",
        help="Directory containing scenario YAML files",
    )
    reproduce_parser.add_argument(
        "--out",
        type=str,
        default="docs",
        help="Root output directory for artifacts",
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

    if args.command == "delay-sweep":
        out_dir = Path(args.out)
        config = SimConfig(seed=args.seed, snr_db=args.snr_db)
        ber_no_comp = sweep_ber_vs_delay(
            config, args.delay_samples, enable_comp=False
        )
        ber_with_comp = None
        if not args.no_comp:
            ber_with_comp = sweep_ber_vs_delay(
                config, args.delay_samples, enable_comp=True
            )
        save_sweep_delay(
            out_dir,
            args.delay_samples,
            ber_no_comp,
            ber_with_comp,
            snr_db=args.snr_db,
        )
        return 0

    if args.command == "rician-sweep":
        out_dir = Path(args.out)
        config = SimConfig(seed=args.seed, snr_db=args.snr_db)
        ber_list = sweep_ber_vs_rician_k(config, args.k_db)
        save_sweep_rician(out_dir, args.k_db, ber_list, snr_db=args.snr_db)
        return 0

    if args.command == "run-scenario":
        scenario = load_scenario(args.scenario)
        run_scenario(scenario, args.out)
        return 0

    if args.command == "reproduce":
        reproduce_all(args.scenario_dir, args.out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
